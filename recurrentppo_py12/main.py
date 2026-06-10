from enum import IntEnum

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.utils import set_random_seed

import config
from algo import build_model


class Action(IntEnum):
    """行動の定義（B-7: コメント不一致を解消し一元管理）。"""

    LONG = 0
    FLAT = 1
    SHORT = 2


# ──────────────────────────────
# 1. 改善版 環境 (NikkeiEnv) — RecurrentPPO 版
class NikkeiEnv(gym.Env):
    """
    日経225のトレーディング環境（RecurrentPPO 用）。

    qrdqn_py12 からの変更点:
    - 観測（G-1）: 「window×特徴」の2次元から **1日1ベクトル** に変更。
      時系列の記憶は RecurrentPPO の内蔵LSTMに任せる。
      観測 = 正規化済み特徴(29) ＋ 現在ポジションの one-hot(3)。
      （取引コストがあるため最適行動は現在ポジションに依存する。LSTMは自分の行動を
        観測できないので、ポジションを明示的に観測へ入れる。）
    - 正規化（B-3）: ウィンドウ内MinMax を廃止。
      ①リターン化/相対化で定常化 → ②ローリング・ロバスト z-score（中央値/MAD,
      因果的=未来は見ない）→ ③±OBS_CLIP クリップ。
      絶対水準の破壊・ウィンドウ間の基準ズレ（分布シフト）を解消する。
    - trade_end_date（G-3）: 複数OOS窓評価のため、窓の終了日でエピソードを打ち切れる。

    ・行動：Action enum（0: LONG, 1: FLAT, 2: SHORT）
    ・取引手数料：前回ポジションと異なる場合、現在の残高に対して transaction_cost % の費用
    ・報酬：差分シャープ等（config.REWARD_TYPE で切替, 手数料考慮済み）
    ・エピソード終了：データ終了（or trade_end_date 到達）、資産が初期× risk_limit 未満
    """

    metadata = {"render_modes": ["human"]}

    # 特徴量ごとの定常化方法（B-3）。
    # "logdiff": 対数差分 / "rel_open": Open比-1 / "div_open": Open割り /
    # "log": 対数 / "scale100": 1/100 / "raw": そのまま → いずれも後段でロバストz。
    FEATURE_TRANSFORMS = {
        "Open": "logdiff",
        "SMA_5": "rel_open",
        "SMA_25": "rel_open",
        "SMA_75": "rel_open",
        "Upper_3σ": "rel_open",
        "Upper_2σ": "rel_open",
        "Upper_1σ": "rel_open",
        "Lower_3σ": "rel_open",
        "Lower_2σ": "rel_open",
        "Lower_1σ": "rel_open",
        "偏差値25": "raw",
        "Upper2_3σ": "rel_open",
        "Upper2_2σ": "rel_open",
        "Upper2_1σ": "rel_open",
        "Lower2_3σ": "rel_open",
        "Lower2_2σ": "rel_open",
        "Lower2_1σ": "rel_open",
        "偏差値75": "raw",
        "RSI_14": "scale100",
        "RSI_22": "scale100",
        "MACD": "div_open",
        "MACD_signal": "div_open",
        "Japan_10Y_Rate": "raw",
        "US_10Y_Rate": "raw",
        "ATR_5": "div_open",
        "ATR_25": "div_open",
        "RCI_9": "scale100",
        "RCI_26": "scale100",
        "VIX": "log",
    }

    def __init__(
        self,
        df,
        window_size=config.WINDOW_SIZE,
        transaction_cost=config.TRANSACTION_COST,
        risk_limit=config.RISK_LIMIT,
        trade_start_date=None,
        trade_end_date=None,
    ):
        super().__init__()

        df = df.dropna()
        # 日付は常に保持（trade_start/end の解決と複数OOS窓評価に使う）
        self.dates = pd.to_datetime(df.index)
        df = df.reset_index(drop=True)
        self.df = df
        self.feature_cols = list(self.FEATURE_TRANSFORMS.keys())

        self.open_prices = self.df["Open"].values.astype(np.float64)
        self.n = len(self.df)
        self.window_size = window_size  # 観測ではなく warmup バー数（G-1）

        # 観測特徴を事前計算（B-3: リターン化＋ロバストz。因果的なので一括計算可）
        self.features = self._precompute_features()
        self.num_features = self.features.shape[1]

        # トレード開始位置（検証/テストで warmup後の特定日から始めるため）
        if trade_start_date is not None:
            idx = int(np.asarray(self.dates >= pd.Timestamp(trade_start_date)).argmax())
            self.trade_start = max(idx, window_size)
        else:
            self.trade_start = window_size
        # トレード終了位置（G-3: 複数OOS窓評価用。None ならデータ終端）
        if trade_end_date is not None:
            within = np.asarray(self.dates <= pd.Timestamp(trade_end_date))
            self.end_step = max(int(within.sum()) - 1, self.trade_start + 1)
        else:
            self.end_step = self.n - 1
        self.current_step = self.trade_start

        # 行動空間（0:Long, 1:Flat, 2:Short → Action enum と対応）
        self.action_space = spaces.Discrete(3)
        # 観測 = 特徴ベクトル(±OBS_CLIP) + ポジション one-hot(0/1)
        self.observation_space = spaces.Box(
            low=-config.OBS_CLIP,
            high=config.OBS_CLIP,
            shape=(self.num_features + 3,),
            dtype=np.float32,
        )

        # 資産関係の初期設定
        self.initial_balance = config.INITIAL_BALANCE
        self.balance = self.initial_balance
        self.equity_curve = [self.balance]
        self.sum_reward = 0
        self.num_step = 0

        self.transaction_cost = transaction_cost  # 例：0.001 → 0.1%
        self.risk_limit = risk_limit  # 資金が初期の risk_limit 未満なら終了

        self.trade_count = 0  # 累積の取引回数

        # 報酬設計（G-3-1: 差分シャープレシオ）
        self.reward_type = config.REWARD_TYPE
        self.dsr_eta = config.DSR_ETA
        # 数値安定化パラメータ（G-3-2）
        self.dsr_warmup = config.DSR_WARMUP        # 序盤は報酬0（分母が信用できない）
        self.dsr_var_floor = config.DSR_VAR_FLOOR  # 分散の床（日次リターンのスケール）
        self.dsr_clip = config.DSR_CLIP            # 1ステップ報酬のクリップ幅
        self.benchmark_weight = config.BENCHMARK_WEIGHT  # excess系: 部分ベンチマーク係数 β
        # リターンの1次/2次モーメントのEMA（差分シャープ計算用）
        self.dsr_A = 0.0
        self.dsr_B = 0.0
        # 差分下方偏差レシオ用: 1次モーメント A と下方2次モーメント DD²（min(r,0)²）のEMA
        self.ddr_A = 0.0
        self.ddr_DD2 = 0.0

        # エピソード開始時のポジション
        self.prev_action = int(Action.FLAT)

    # ── 観測（B-3: リターン化＋ローリング・ロバスト z-score）──
    def _precompute_features(self):
        """全ステップ分の正規化済み特徴ベクトルを一括計算。

        手順:
        1. 定常化: 価格水準系は Open 比/対数差分に変換（絶対水準とスケールの混在を解消）
        2. ローリング・ロバスト z-score: (x − rolling中央値) / (1.4826·MAD)。
           窓 Z_WINDOW・最低 Z_MIN_PERIODS。過去のみ参照する因果的計算（リーク無し）。
        3. ±OBS_CLIP でクリップ。warmup 不足の先頭行は 0 埋め
           （トレード開始は window_size 以降なので学習では参照されない）。
        戻り値 shape: (n, num_features) float32
        """
        open_ = self.df["Open"].astype(np.float64)
        feat = pd.DataFrame(index=self.df.index)
        for col, mode in self.FEATURE_TRANSFORMS.items():
            x = self.df[col].astype(np.float64)
            if mode == "logdiff":
                feat[col] = np.log(x).diff()
            elif mode == "rel_open":
                feat[col] = x / open_ - 1.0
            elif mode == "div_open":
                feat[col] = x / open_
            elif mode == "log":
                feat[col] = np.log(x.clip(lower=1e-6))
            elif mode == "scale100":
                feat[col] = x / 100.0
            else:  # raw
                feat[col] = x

        med = feat.rolling(config.Z_WINDOW, min_periods=config.Z_MIN_PERIODS).median()
        mad = (feat - med).abs().rolling(
            config.Z_WINDOW, min_periods=config.Z_MIN_PERIODS
        ).median()
        z = (feat - med) / (1.4826 * mad + 1e-8)
        z = z.clip(-config.OBS_CLIP, config.OBS_CLIP).fillna(0.0)
        return z.to_numpy(dtype=np.float32)

    def obs_at(self, idx, prev_action=None):
        """任意ステップの観測を返す（評価時のLSTM warmupにも使う）。"""
        if prev_action is None:
            prev_action = int(Action.FLAT)
        idx = min(max(idx, 0), self.n - 1)
        pos = np.zeros(3, dtype=np.float32)
        pos[int(prev_action)] = 1.0
        return np.concatenate([self.features[idx], pos])

    def _get_observation(self):
        # B-5: 終端でも有効な観測を返すため obs_at 側で clamp する
        return self.obs_at(self.current_step, self.prev_action)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.trade_start
        self.balance = self.initial_balance
        self.equity_curve = [self.balance]
        self.trade_count = 0  # 取引数もリセット
        self.prev_action = int(Action.FLAT)
        self.sum_reward = 0
        self.num_step = 0
        # 差分シャープのモーメントもリセット
        self.dsr_A = 0.0
        self.dsr_B = 0.0
        # 差分下方偏差のモーメントもリセット
        self.ddr_A = 0.0
        self.ddr_DD2 = 0.0
        return self._get_observation(), {}

    def _differential_sharpe(self, r):
        """差分シャープレシオ（Moody & Saffell 1998）を1ステップ分計算（G-3-1）。

        D_t = (B_{t-1}·ΔA - 0.5·A_{t-1}·ΔB) / (B_{t-1} - A_{t-1}^2)^{3/2}
            ΔA = r - A_{t-1},  ΔB = r^2 - B_{t-1}
        A,B はリターンの1次/2次モーメントのEMA。リスク調整後の改善度を即時報酬にする。

        数値安定化（G-3-2）:
        - エピソード開始直後は A,B,var が極小で分母 var^1.5 が爆発するため、
          (1) 最初の DSR_WARMUP ステップは統計だけ更新して報酬0、
          (2) 分散を日次リターンのスケールでフロア、(3) 報酬をクリップ。
        """
        eta = self.dsr_eta
        A_prev, B_prev = self.dsr_A, self.dsr_B
        dA = r - A_prev
        dB = r * r - B_prev
        # 分散は理論上非負だが浮動小数で負になりうるので max(0,·) → 現実的な床を張る
        var = max(B_prev - A_prev * A_prev, 0.0)
        var = max(var, self.dsr_var_floor)

        # ウォームアップ中は分母が信用できないので報酬0（統計だけ更新）
        if self.num_step <= self.dsr_warmup:
            dsr = 0.0
        else:
            dsr = (B_prev * dA - 0.5 * A_prev * dB) / (var ** 1.5)
            dsr = float(np.clip(dsr, -self.dsr_clip, self.dsr_clip))

        # モーメントを更新（D_t を計算した後に行う）
        self.dsr_A = A_prev + eta * dA
        self.dsr_B = B_prev + eta * dB
        return float(dsr)

    def _differential_downside(self, r):
        """差分下方偏差レシオ（Differential Downside Deviation Ratio; Moody&Saffell 1998）。

        DSR が全分散で割るのに対し、DDR は下方偏差 DD（負リターンのみの2乗平均）で割る
        Sortino 型（G-3-4）。上昇方向のボラは罰さないので「上昇を取りに行く」方向に
        インセンティブが働き、DSR の様子見(Flat)偏りを緩和する狙い。

        A   : リターンの1次モーメントのEMA
        DD² : min(r,0)² のEMA（下方2次モーメント）
        D_t = dDDR/dη|_{η=0}（Moody&Saffell の閉形式）:
            r > 0 : (r − A/2) / DD
            r ≤ 0 : (DD²·(r − A/2) − A·r²/2) / DD³
        数値安定化は DSR と同様（warmup で報酬0 / DD² にフロア / 報酬クリップ）。
        """
        eta = self.dsr_eta
        A_prev, DD2_prev = self.ddr_A, self.ddr_DD2
        downside = min(r, 0.0)
        # 下方2次モーメントにフロア（序盤 DD²≈0 で DD³ が爆発するのを防ぐ）
        DD2_floored = max(DD2_prev, self.dsr_var_floor)
        DD_prev = DD2_floored ** 0.5

        # ウォームアップ中は分母が信用できないので報酬0（統計だけ更新）
        if self.num_step <= self.dsr_warmup:
            ddr = 0.0
        elif r > 0.0:
            ddr = (r - 0.5 * A_prev) / DD_prev
        else:
            ddr = (DD2_floored * (r - 0.5 * A_prev) - 0.5 * A_prev * r * r) / (DD_prev ** 3)
        if self.num_step > self.dsr_warmup:
            ddr = float(np.clip(ddr, -self.dsr_clip, self.dsr_clip))

        # モーメントを更新（D_t を計算した後に行う）
        self.ddr_A = A_prev + eta * (r - A_prev)
        self.ddr_DD2 = DD2_prev + eta * (downside * downside - DD2_prev)
        return float(ddr)

    def step(self, action):
        action = int(action)
        old_balance = float(self.balance)
        self.num_step += 1

        # 当日と翌日の株価（ここではOpen値）を取得
        price_today = self.open_prices[self.current_step]
        if self.current_step + 1 < self.n:
            price_tomorrow = self.open_prices[self.current_step + 1]
        else:
            price_tomorrow = price_today

        ret = (price_tomorrow - price_today) / price_today

        # 保有ポジションごとに資産を更新（B-4: daily_inflation のデッドコードは削除）
        if action == Action.LONG:
            self.balance *= 1 + ret
        elif action == Action.SHORT:
            self.balance *= 1 - ret
        # Action.FLAT: 資産は変化しない

        # 前回のポジションと異なる場合は手数料を引く（A-3: trade_penalty は廃止）
        if action != self.prev_action:
            cost = self.balance * self.transaction_cost
            self.balance -= cost
            self.trade_count += 1

        # 1日分の対数リターン（手数料込み）。
        # B-1: 単純/対数の混在を解消し対数で統一。
        # B-2: 未来3日を覗く中期報酬シェイピングは TD 学習を壊すため削除。
        step_log_return = float(np.log(self.balance / old_balance))

        # 報酬: 差分シャープ / 差分下方偏差 / 超過リターン / 対数リターン（config で切替）
        if self.reward_type == "dsr":
            reward = self._differential_sharpe(step_log_return)
        elif self.reward_type == "ddr":
            reward = self._differential_downside(step_log_return)
        elif self.reward_type == "excess":
            # 超過リターン（対B&H）: 戦略の対数リターン − β·市場(Long固定)の対数リターン（G-3-5/G-3-7）
            market_log_return = float(np.log1p(ret))
            reward = step_log_return - self.benchmark_weight * market_log_return
        elif self.reward_type == "excess_dsr":
            # 差分インフォメーションレシオ（G-3-6）: DSRを部分超過リターンの系列に対して計算する
            market_log_return = float(np.log1p(ret))
            excess = step_log_return - self.benchmark_weight * market_log_return
            reward = self._differential_sharpe(excess)
        else:
            reward = step_log_return

        self.prev_action = action
        self.sum_reward += reward

        # 終了条件（B-6: terminated = リスク失格 / truncated = データ終端 or 窓終了）
        terminated = bool(self.balance <= 0 or self.balance < self.initial_balance * self.risk_limit)
        truncated = bool(self.current_step >= self.end_step)

        if terminated or truncated:
            print(
                f"アクション[0:買,1:待,2:売]:{action}, ステップ:{self.num_step}, "
                f"累積リワード:{self.sum_reward:.4f}, 資産:{int(self.balance)}, リターン:{ret}, "
                f"トレード回数:{self.trade_count}, 明日: {int(price_tomorrow)},株価:{int(price_today)}"
            )

        self.equity_curve.append(float(self.balance))
        self.current_step += 1
        # B-5: 終端でも None ではなく有効な観測を返す
        obs = self._get_observation()
        info = {"trade_count": self.trade_count}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # 必要に応じて可視化ロジックを実装可能
        pass

    def get_equity_curve(self):
        return self.equity_curve


# ──────────────────────────────
# 2. データ準備・学習/評価ユーティリティ
def make_env(df, trade_start_date=None, trade_end_date=None):
    """A-4: コスト・リスク制限は config に統一。trade_start/end_date で窓を指定可。"""
    return NikkeiEnv(
        df,
        window_size=config.WINDOW_SIZE,
        transaction_cost=config.TRANSACTION_COST,
        risk_limit=config.RISK_LIMIT,
        trade_start_date=trade_start_date,
        trade_end_date=trade_end_date,
    )


def warmup_lstm_state(model, env, n_warmup=None, deterministic=True):
    """トレード開始前の観測を LSTM に流して隠れ状態を温める（G-1）。

    qrdqn版は「直近130日のwindow」が観測に入っていたが、RecurrentPPO は履歴を
    LSTM の隠れ状態として持つ。評価でトレード開始位置へジャンプする場合、
    ゼロ状態のままだと文脈が無いので、開始前 n_warmup 本（既定 window_size）の
    観測（ポジションは FLAT）を流して状態を作ってから取引を始める。
    戻り値: (lstm_states, episode_start) — そのまま model.predict に渡せる。
    """
    if n_warmup is None:
        n_warmup = env.window_size
    state = None
    episode_start = np.ones(1, dtype=bool)
    start = max(env.trade_start - n_warmup, 0)
    for i in range(start, env.trade_start):
        obs = env.obs_at(i)
        _, state = model.predict(
            obs, state=state, episode_start=episode_start, deterministic=deterministic
        )
        episode_start = np.zeros(1, dtype=bool)
    return state, episode_start


def rollout(model, env, deterministic=True, lstm_warmup=True):
    """env を reset して1エピソード実行（RecurrentPPO の状態管理込み）。

    戻り値: (actions, equity_curve)
    PPO（非リカレント）でも model.predict が state を無視するのでそのまま動く。
    """
    obs, _ = env.reset()
    if lstm_warmup:
        state, episode_start = warmup_lstm_state(model, env, deterministic=deterministic)
    else:
        state, episode_start = None, np.ones(1, dtype=bool)
    actions = []
    done = False
    while not done:
        a, state = model.predict(
            obs, state=state, episode_start=episode_start, deterministic=deterministic
        )
        episode_start = np.zeros(1, dtype=bool)
        a = int(a)
        actions.append(a)
        obs, _, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
    return actions, env.get_equity_curve()


def prepare_train_val_data(full=None):
    """in-sampleデータを 学習(VAL_START未満) と 検証(warmup付き) に分割。

    full を渡せば再DLしない（キャッシュ利用）。
    検証はテスト(2024-)に触れず、VAL_START→TRAIN_END だけをトレードする。
    """
    if full is None:
        from data import generate_env_data
        full = generate_env_data(config.TRAIN_START, config.TRAIN_END, ticker=config.TICKER)
    train_df = full[full.index < pd.Timestamp(config.VAL_START)]
    val_df = full[full.index >= pd.Timestamp(config.VAL_WARMUP_START)]
    return train_df, val_df


def make_eval_callback(val_df, best_dir):
    """検証envでのEvalCallback(+頭打ち早期停止)。検証ベストを best_dir/best_model.zip に保存。

    報酬がDSRなので、1エピソードのDSR合計 ≒ 検証期間のシャープに相当し、
    これを最大化するモデルを「検証ベスト」として選ぶ＝過学習前で止める。
    （EvalCallback の evaluate_policy は RecurrentPPO の LSTM状態を内部で扱える）
    """
    import os
    os.makedirs(best_dir, exist_ok=True)
    val_env = DummyVecEnv([lambda: make_env(val_df, trade_start_date=config.VAL_START)])
    stop = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config.EARLY_STOP_PATIENCE,
        min_evals=config.EARLY_STOP_MIN_EVALS,
        verbose=1,
    )
    return EvalCallback(
        val_env,
        best_model_save_path=best_dir,
        log_path=best_dir,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=1,
        deterministic=True,
        callback_after_eval=stop,
        verbose=1,
    )


def save_best_or_last(model, best_dir, out_path):
    """検証ベスト(best_model.zip)があればそれを out_path にコピー、無ければ最終モデルを保存。"""
    import os, shutil
    best = os.path.join(best_dir, "best_model.zip")
    if os.path.exists(best):
        shutil.copy(best, out_path + ".zip")
        print(f"検証ベストを {out_path}.zip に保存")
    else:
        model.save(out_path)
        print(f"検証ベストが無いため最終モデルを {out_path}.zip に保存")


if __name__ == "__main__":
    import os
    import sys

    # C-2: 再現性のためシード固定。ただしアンサンブル用に実行ごとに変えられるよう、
    # 引数 or 環境変数 SEED で上書き可（例: `python main.py 3` / `SEED=3 python main.py`）。
    seed = config.SEED
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    elif os.environ.get("SEED"):
        seed = int(os.environ["SEED"])
    set_random_seed(seed)
    print(f"seed = {seed}")

    print("データ準備中（学習/検証分割）...")
    train_df, val_df = prepare_train_val_data()
    print(
        f"学習 {len(train_df)}行 (〜{config.VAL_START}) / "
        f"検証 {len(val_df)}行 (warmup {config.VAL_WARMUP_START}〜, トレード {config.VAL_START}〜{config.TRAIN_END})"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_env = DummyVecEnv([lambda: make_env(train_df)])

    # G-1: config.ALGO に応じて RecurrentPPO / PPO（観測は1日1ベクトル＋LSTM）
    model = build_model(train_env, device, seed=seed)
    print(f"新たにモデルを作成しました（algo={config.ALGO}, lstm={config.LSTM_HIDDEN}x{config.LSTM_LAYERS}, seed={seed}）。")

    # シード別に保存先を分け、複数runが互いに上書きしないようにする
    prefix = f"{config.model_name()}_seed{seed}"
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=config.MODEL_DIR, name_prefix=prefix
    )
    best_dir = f"best_seed{seed}"
    eval_callback = make_eval_callback(val_df, best_dir)  # 検証スコアで早期停止

    print("エージェントの学習開始（検証スコアで早期停止）...")
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )
    print("学習完了！")
    save_best_or_last(model, best_dir, prefix)
