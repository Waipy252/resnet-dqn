import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback


# ──────────────────────────────
# 1. 改善版 環境 (NikkeiEnv)
class NikkeiEnv(gym.Env):
    """
    日経225の終値・出来高データを用いたシンプルなトレーディング環境
    ・観測：直近 window_size 日間の各種特徴（例：始値、出来高）を、それぞれウィンドウ初日を基準に正規化
    ・行動：2: ショート, 1: フラット, 0: ロング
    ・取引手数料：前回ポジションと異なる場合、現在の残高に対して transaction_cost % の費用がかかる
    ・報酬：1日分の相対的な対数リターン（手数料考慮済み）
    ・エピソード終了：データ終了、あるいは資産残高が初期資産の risk_limit 未満になった場合
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        window_size=30,
        transaction_cost=0.001,
        risk_limit=0.5,
        trade_penalty=0.002,
    ):
        super(NikkeiEnv, self).__init__()

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
        self.window_size = window_size
        self.current_step = window_size  # 最初の window_size 日は観測用

        # 行動空間（例：0→Long, 1→Flat）
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(self.feature_cols)),
            dtype=np.float32,
        )

        # 資産関係の初期設定
        self.initial_balance = 1_000_000
        self.balance = self.initial_balance
        self.equity_curve = [self.balance]
        self.sum_reward = 0
        self.num_step = 0

        self.transaction_cost = transaction_cost  # 例：0.001 → 0.1%
        self.risk_limit = risk_limit  # 資金が初期の risk_limit 未満なら終了

        # 新たに取引ペナルティと取引数を管理する変数を設定
        self.trade_penalty = trade_penalty  # 1回の取引ごとに与える追加ペナルティ
        self.trade_count = 0  # 累積の取引回数

        # エピソード開始時のポジション
        self.prev_action = 1  # 例：フラット（何もしない）

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.equity_curve = [self.balance]
        self.trade_count = 0  # 取引数もリセット
        self.prev_action = 1
        self.sum_reward = 0
        self.num_step = 0
        return self._get_observation()

    def _get_observation(self):
        # MIXMAX法
        obs = []
        for col in self.feature_cols:
            # 現在のウィンドウの値を取得
            window = self.data[col][
                self.current_step - self.window_size : self.current_step
            ]
            # ウィンドウ内の最小値・最大値を計算：　MinMax法
            min_val = np.min(window)
            max_val = np.max(window)
            # ゼロ除算を防ぐため、最大値と最小値が等しい場合の処理
            if max_val - min_val == 0:
                norm = np.zeros_like(window)
            else:
                norm = (window - min_val) / (max_val - min_val)
            # shape (window_size, 1) に整形してリストに追加
            obs.append(norm.reshape(-1, 1))
        # 各特徴量ごとの正規化済みデータを連結 → shape: (window_size, len(feature_cols))
        observation = np.concatenate(obs, axis=1).astype(np.float32)
        return observation

    def step(self, action):
        old_balance = float(self.balance)
        self.num_step += 1

        # 当日と翌日の株価（ここではOpen値）を取得
        price_today = self.data["Open"][self.current_step]
        if self.current_step + 1 < len(self.data["Open"]):
            price_tomorrow = self.data["Open"][self.current_step + 1]
        else:
            price_tomorrow = price_today

        ret = (price_tomorrow - price_today) / price_today

        # 保有ポジションごとに資産を更新（0: ロング, 1: フラット, 2: ショート）
        if action == 0:  # ロングの場合
            self.balance *= 1 + ret
        elif action == 2:  # ショートの場合
            self.balance *= 1 - ret
        elif action == 1:  # フラットの場合（インフレペナルティなどがあれば適用）
            daily_inflation = 0.0 / 365
            self.balance *= 1 - daily_inflation

        # 前回のポジションと異なる場合は手数料を引く
        if action != self.prev_action:
            cost = self.balance * self.transaction_cost
            self.balance -= cost
            self.trade_count += 1

        # 1日分のリワード：対数リターン
        reward = np.log(self.balance / old_balance)
        # ── ここから中期的視点の報酬追加 ──
        # 3日後の株価を使用（もし存在しない場合は末尾の値を使用）
        discount_factor = 0.3
        days = 3
        if self.current_step + days < len(self.data["Open"]):
            price_future = self.data["Open"][self.current_step + days]
        else:
            price_future = self.data["Open"][-1]

        # 行動に応じた中期リターンの計算
        if action == 0:  # ロングの場合
            mid_return = np.log(price_future / price_today)
        elif action == 2:  # ショートの場合
            mid_return = -np.log(price_future / price_today)
        else:  # フラットの場合：中期的なリターンは0とする
            mid_return = 0.0

        # 割引率を考慮して中期的な報酬を追加
        reward += discount_factor * mid_return
        # ── ここまで中期的視点の報酬追加 ──

        # エピソード終了条件判定
        done = False
        self.prev_action = action
        self.sum_reward += reward

        if (
            (self.current_step >= len(self.data["Open"]) - 1)
            or (self.balance < self.initial_balance * self.risk_limit)
            or (self.balance <= 0)
        ):
            print(
                f"アクション[0:買,1:待,2:売]:{action}, ステップ:{self.num_step}, "
                f"累積リワード:{self.sum_reward:.4f}, 資産:{int(self.balance)}, リターン:{ret}, "
                f"トレード回数:{self.trade_count}, 明日: {int(price_tomorrow)},株価:{int(price_today)}"
            )
            done = True

        obs = self._get_observation() if not done else None
        info = {"trade_count": self.trade_count}
        self.equity_curve.append(float(self.balance))
        self.current_step += 1
        return obs, reward, done, info

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
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)

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
    print("データをダウンロード中...")
    # Yahoo Finance から日経225 (^N225) のヒストリカルデータを取得
    start = "1997-01-01"
    end = "2024-01-01"
    # train_data = generate_env_data(start, end, ticker="^GSPC")
    # train_data = generate_env_data(start, end, ticker="JPY=X")
    train_data = generate_env_data(start, end, ticker="^N225")

    # 窓長（直近○日分のデータを入力とする）
    window_size = 130
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_envs = 1
    def make_env():
        return NikkeiEnv(
            train_data,
            window_size=window_size,
            transaction_cost=0.0001,
            risk_limit=0.6,
            trade_penalty=0.000000,
        )
    env_fns = [lambda: make_env() for _ in range(num_envs)]
    train_env = DummyVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=ResNetFeatures,
        features_extractor_kwargs=dict(
            features_dim=158,#128,
            num_blocks=3  # ResNetの残差ブロック数
        ),
    )

    model = DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        exploration_final_eps=0.05,
        exploration_fraction=0.2,
        learning_rate=1e-4,
        verbose=1,
        device=device
    )
    print("新たにモデルを作成しました。")

    # チェックポイントコールバックの作成
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./",  # モデルを保存するディレクトリ（存在するか、事前に作成してください）
        name_prefix=f"nikkei_cp_{start}_{end}",
    )

    print("エージェントの学習開始...")
    model.learn(total_timesteps=350000, callback=checkpoint_callback, progress_bar=True)
    print("学習完了！")
    model.save(f"nikkei_cp_{start}_{end}")
