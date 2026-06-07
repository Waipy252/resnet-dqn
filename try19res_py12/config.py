"""学習・評価で共有する設定（A-4 / C-4: ハイパーパラメータの一元管理）。

main.py / eval.py / run_simulation.py はここを import して同じ値を使う。
学習と評価でコストやリスク制限がズレないようにするのが目的。
"""

# ── データ ──────────────────────────────
TICKER = "^N225"
TRAIN_START = "1997-01-01"
TRAIN_END = "2024-01-01"

# ── 環境 ────────────────────────────────
WINDOW_SIZE = 130
INITIAL_BALANCE = 1_000_000
TRANSACTION_COST = 0.001  # 学習・評価で統一（往復スプレッド+手数料想定, A-4）
RISK_LIMIT = 0.5          # 初期資産の RISK_LIMIT 未満で終了（学習・評価で統一）

# ── 報酬設計（G-3-1）──────────────────────
# "dsr"   : 差分シャープレシオ（Moody&Saffell）。リスク調整後の改善を即時報酬に。
#           生の対数リターンより学習が安定し、ドローダウンを抑えて一貫した勝ち方に。
# "logret": 1日分の対数リターン（従来）。比較用に残す。
REWARD_TYPE = "dsr"
DSR_ETA = 0.01            # 差分シャープの適応率（リターン統計のEMA係数）

# ── モデル（ResNet特徴抽出器）──────────────
FEATURES_DIM = 128        # 158→128 に整理（C-4）。GroupNorm の分割も素直になる
NUM_BLOCKS = 3

# ── 学習（DQN）─────────────────────────────
SEED = 42                 # 再現性（C-2）
TOTAL_TIMESTEPS = 350_000
LEARNING_RATE = 1e-4
EXPLORATION_FRACTION = 0.2
EXPLORATION_FINAL_EPS = 0.05
BUFFER_SIZE = 200_000     # 100万→20万。観測が大きくメモリ警告が出ていた（F-2）
BATCH_SIZE = 256          # 32→256。GPU を使い切るため（F-5）
GRADIENT_STEPS = 1
TRAIN_FREQ = 4

# ── パス ────────────────────────────────
MODEL_DIR = "."           # 既存どおりカレントに保存
TENSORBOARD_LOG = "./tb/"  # 学習曲線の記録（C-3）


def model_name(steps=None):
    """チェックポイント/最終モデルのファイル名プレフィックスを生成。"""
    base = f"nikkei_cp_{TRAIN_START}_{TRAIN_END}"
    if steps is None:
        return base
    return f"{base}_{steps}_steps"
