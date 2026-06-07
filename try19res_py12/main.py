from enum import IntEnum

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

import config


class Action(IntEnum):
    """行動の定義（B-7: コメント不一致を解消し一元管理）。"""

    LONG = 0
    FLAT = 1
    SHORT = 2


# ──────────────────────────────
# 1. 改善版 環境 (NikkeiEnv)
class NikkeiEnv(gym.Env):
    """
    日経225の終値・出来高データを用いたシンプルなトレーディング環境
    ・観測：直近 window_size 日間の各種特徴（例：始値、出来高）を、それぞれウィンドウ初日を基準に正規化
    ・行動：Action enum（0: LONG, 1: FLAT, 2: SHORT）
    ・取引手数料：前回ポジションと異なる場合、現在の残高に対して transaction_cost % の費用がかかる
    ・報酬：1日分の相対的な対数リターン（手数料考慮済み）
    ・エピソード終了：データ終了、あるいは資産残高が初期資産の risk_limit 未満になった場合
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size=config.WINDOW_SIZE,
        transaction_cost=config.TRANSACTION_COST,
        risk_limit=config.RISK_LIMIT,
    ):
        super().__init__()

        # 既存の初期化処理…
        df = df.dropna().reset_index(drop=True)
        self.df = df
        self.feature_cols = [
            "Open",
            # "Volume",
            "SMA_5",
            "SMA_25",
            "SMA_75",
            "Upper_3σ",
            "Upper_2σ",
            "Upper_1σ",
            "Lower_3σ",
            "Lower_2σ",
            "Lower_1σ",
            "偏差値25",
            "Upper2_3σ",
            "Upper2_2σ",
            "Upper2_1σ",
            "Lower2_3σ",
            "Lower2_2σ",
            "Lower2_1σ",
            "偏差値75",
            "RSI_14",
            "RSI_22",
            "MACD",
            "MACD_signal",
            "Japan_10Y_Rate",
            "US_10Y_Rate",
            "ATR_5",
            "ATR_25",
            "RCI_9",
            "RCI_26",
            "VIX",
        ]

        self.data = {col: self.df[col].values for col in self.feature_cols}
        self.open_prices = self.df["Open"].values.astype(np.float64)
        self.n = len(self.df)
        self.window_size = window_size
        self.current_step = window_size  # 最初の window_size 日は観測用

        # 行動空間（0:Long, 1:Flat, 2:Short → Action enum と対応）
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(window_size, len(self.feature_cols)),
            dtype=np.float32,
        )

        # 観測（MinMax正規化）を事前計算してキャッシュ（B-3 / F-5）。
        # 毎ステップ window×features を再計算していたCPUボトルネックを解消する。
        self._obs_cache = self._precompute_observations()

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
        # リターンの1次/2次モーメントのEMA（差分シャープ計算用）
        self.dsr_A = 0.0
        self.dsr_B = 0.0

        # エピソード開始時のポジション
        self.prev_action = int(Action.FLAT)

    def _precompute_observations(self):
        """全ステップ分の MinMax 正規化済み観測を一括計算（B-3 / F-5）。

        元実装と同じ「各ウィンドウ内の min/max で正規化」を、
        sliding_window_view でベクトル化して O(1) 参照にする。
        戻り値 shape: (num_windows, window_size, num_features)
        idx = current_step - window_size で参照する。
        """
        raw = np.stack(
            [self.data[col].astype(np.float64) for col in self.feature_cols], axis=1
        )  # (n, F)
        # (num_windows, F, window) → (num_windows, window, F)
        sw = sliding_window_view(raw, self.window_size, axis=0).transpose(0, 2, 1)
        mn = sw.min(axis=1, keepdims=True)
        mx = sw.max(axis=1, keepdims=True)
        rng = mx - mn
        norm = np.where(rng == 0, 0.0, (sw - mn) / rng)
        return norm.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.equity_curve = [self.balance]
        self.trade_count = 0  # 取引数もリセット
        self.prev_action = int(Action.FLAT)
        self.sum_reward = 0
        self.num_step = 0
        # 差分シャープのモーメントもリセット
        self.dsr_A = 0.0
        self.dsr_B = 0.0
        return self._get_observation(), {}

    def _get_observation(self):
        # 事前計算済みキャッシュから参照（B-5: 終端でも有効な観測を返すため範囲をclamp）
        idx = self.current_step - self.window_size
        idx = min(max(idx, 0), len(self._obs_cache) - 1)
        return self._obs_cache[idx]

    def _differential_sharpe(self, r):
        """差分シャープレシオ（Moody & Saffell 1998）を1ステップ分計算（G-3-1）。

        D_t = (B_{t-1}·ΔA - 0.5·A_{t-1}·ΔB) / (B_{t-1} - A_{t-1}^2)^{3/2}
            ΔA = r - A_{t-1},  ΔB = r^2 - B_{t-1}
        A,B はリターンの1次/2次モーメントのEMA。リスク調整後の改善度を即時報酬にする。
        """
        eta = self.dsr_eta
        A_prev, B_prev = self.dsr_A, self.dsr_B
        dA = r - A_prev
        dB = r * r - B_prev
        var = B_prev - A_prev * A_prev
        # 分散がほぼ0（初期数ステップ等）は未定義 → 0 を返してから統計だけ更新
        if var > 1e-12:
            dsr = (B_prev * dA - 0.5 * A_prev * dB) / (var ** 1.5)
        else:
            dsr = 0.0
        # モーメントを更新（D_t を計算した後に行う）
        self.dsr_A = A_prev + eta * dA
        self.dsr_B = B_prev + eta * dB
        return float(dsr)

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

        # 報酬（G-3-1）: 差分シャープレシオ or 対数リターン（config で切替）
        if self.reward_type == "dsr":
            reward = self._differential_sharpe(step_log_return)
        else:
            reward = step_log_return

        self.prev_action = action
        self.sum_reward += reward

        # 終了条件（B-6: terminated = リスク失格 / truncated = データ終端）
        terminated = bool(self.balance <= 0 or self.balance < self.initial_balance * self.risk_limit)
        truncated = bool(self.current_step >= self.n - 1)

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


class ResNetFeatures(BaseFeaturesExtractor):
    """
    1D ResNet ベースの特徴抽出器：
    入力時系列（window_size × input_dim）に対して1次元畳み込みと残差接続を使用
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim=128,
                 num_blocks=3):
        """
        Args:
            observation_space: 観測空間
            features_dim: 出力特徴量の次元数
            num_blocks: ResNetブロックの数
        """
        super(ResNetFeatures, self).__init__(observation_space, features_dim=features_dim)

        self.window_size = observation_space.shape[0]  # 時系列長
        self.input_dim = observation_space.shape[1]    # 入力特徴数

        # 入力層: (batch, window_size, input_dim) -> (batch, features_dim, window_size)
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, features_dim),
            nn.ReLU()
        )

        # 残差ブロック
        self.res_blocks = nn.ModuleList([
            ResidualBlock(features_dim) for _ in range(num_blocks)
        ])

        # グローバル平均プーリング
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, observations):
        # observations の shape: (batch, window_size, input_dim)
        batch_size = observations.size(0)

        # 特徴量次元に射影
        x = self.input_projection(observations)  # (batch, window_size, features_dim)
        x = x.transpose(1, 2)  # (batch, features_dim, window_size) に変換

        # 残差ブロックを通す
        for block in self.res_blocks:
            x = block(x)

        # グローバル平均プーリング
        x = self.pool(x).view(batch_size, -1)  # (batch, features_dim)
        return x


def _gn_groups(channels, max_groups=8):
    """channels を割り切れる最大のグループ数を返す（GroupNorm 用）。"""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResidualBlock(nn.Module):
    """
    1D ResNet の残差ブロック
    """

    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        # C-1: BatchNorm は DQN（単一サンプル推論・ターゲットネット）と相性が悪いため GroupNorm に置換
        self.bn1 = nn.GroupNorm(_gn_groups(channels), channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.bn2 = nn.GroupNorm(_gn_groups(channels), channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # 残差接続
        out = self.relu(out)

        return out

# ──────────────────────────────
# 2. 改善版 Transformer を用いた特徴抽出器
class TransformerFeatures(BaseFeaturesExtractor):
    """
    カスタム特徴抽出器：入力時系列 (window_size × input_dim) を線形変換し、
    学習可能な位置エンコーディングを加えた上で Transformer Encoder で変換し、
    時系列方向に平均プーリングして最終特徴量 (model_dim 次元) を得る。
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        model_dim=128,
        nhead=4,
        num_layers=2,
        dropout_rate=0.1,
    ):
        # 最終的な特徴次元は model_dim
        super(TransformerFeatures, self).__init__(
            observation_space, features_dim=model_dim
        )
        self.window_size = observation_space.shape[0]  # 時系列長
        self.input_dim = observation_space.shape[
            1
        ]  # 入力特徴数（例: 終値と出来高の場合は2）
        self.model_dim = model_dim

        # 入力を model_dim 次元に射影
        self.input_proj = nn.Linear(self.input_dim, model_dim)
        # 学習可能な位置エンコーディング（初期化に Xavier Uniform を使用）
        self.pos_emb = nn.Parameter(torch.zeros(1, self.window_size, model_dim))
        nn.init.xavier_uniform_(self.pos_emb)

        # Transformer Encoder の定義（dropout_rate を導入）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 時系列方向での平均プーリング（出力 shape: (batch, model_dim)）
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, observations):
        # observations の shape: (batch, window_size, input_dim)
        x = self.input_proj(observations)  # → (batch, window_size, model_dim)
        x = x + self.pos_emb  # 位置エンコーディングの加算
        x = x.transpose(0, 1)  # Transformer 用に (window_size, batch, model_dim) に変形
        x = self.transformer_encoder(x)  # Transformer Encoder 層
        x = x.transpose(0, 1)  # → (batch, window_size, model_dim)
        x = x.transpose(1, 2)  # → (batch, model_dim, window_size) に変形して
        x = self.pool(x).squeeze(-1)  # 時系列方向で平均プーリング → (batch, model_dim)
        return x


from stable_baselines3.common.vec_env import DummyVecEnv
# ──────────────────────────────
# 3. データのダウンロードと環境の作成
if __name__ == "__main__":
    from data import generate_env_data

    # C-2: 再現性のためシード固定
    set_random_seed(config.SEED)

    print("データをダウンロード中...")
    # Yahoo Finance から日経225 (^N225) のヒストリカルデータを取得
    start = config.TRAIN_START
    end = config.TRAIN_END
    train_data = generate_env_data(start, end, ticker=config.TICKER)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_env():
        # A-4: コスト・リスク制限は config に統一（学習と評価で同じ値）
        return NikkeiEnv(
            train_data,
            window_size=config.WINDOW_SIZE,
            transaction_cost=config.TRANSACTION_COST,
            risk_limit=config.RISK_LIMIT,
        )

    train_env = DummyVecEnv([make_env])

    policy_kwargs = dict(
        features_extractor_class=ResNetFeatures,
        features_extractor_kwargs=dict(
            features_dim=config.FEATURES_DIM,
            num_blocks=config.NUM_BLOCKS,
        ),
    )

    model = DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        exploration_final_eps=config.EXPLORATION_FINAL_EPS,
        exploration_fraction=config.EXPLORATION_FRACTION,
        learning_rate=config.LEARNING_RATE,
        buffer_size=config.BUFFER_SIZE,   # F-2: メモリ警告を根本解消
        batch_size=config.BATCH_SIZE,     # F-5: GPU を使い切る
        train_freq=config.TRAIN_FREQ,
        gradient_steps=config.GRADIENT_STEPS,
        seed=config.SEED,                 # C-2
        tensorboard_log=config.TENSORBOARD_LOG,  # C-3
        verbose=1,
        device=device,
    )
    print("新たにモデルを作成しました。")

    # チェックポイントコールバックの作成
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=config.MODEL_DIR,
        name_prefix=config.model_name(),
    )

    print("エージェントの学習開始...")
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    print("学習完了！")
    model.save(config.model_name())
