"""学習・評価で共有する設定（A-4 / C-4: ハイパーパラメータの一元管理）。

main.py / eval.py / run_simulation.py はここを import して同じ値を使う。
学習と評価でコストやリスク制限がズレないようにするのが目的。
"""

# ── データ ──────────────────────────────
TICKER = "^N225"
TRAIN_START = "1997-01-01"
TRAIN_END = "2024-01-01"  # in-sample データの終端（テストはこれ以降）＝モデル名にも使用

# 学習/検証 分割（早期停止用）。学習はVAL_STARTまで、検証はVAL_START→TRAIN_END。
# テスト(2024-01-01以降)には一切触れない。
VAL_START = "2022-06-01"        # ここから先は学習に使わず検証に回す
VAL_WARMUP_START = "2021-01-01"  # 検証env の正規化warmup用にこの辺りからデータを渡す

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
# 数値安定化（G-3-2）: エピソード序盤の var≈0 で分母 var^1.5 が爆発し、
# eval報酬が±1500フリップ→checkpoint選抜/シード不安定の主因だったため追加。
DSR_WARMUP = 100          # 最初の N ステップは報酬0（≒1/DSR_ETA。統計だけ温める）
DSR_VAR_FLOOR = 1e-4      # 分散の下限（日次対数リターン std≈0.01 → var≈1e-4 のスケール）
DSR_CLIP = 1.0            # 1ステップDSR報酬のクリップ幅 [-1,1]

# ── モデル（ResNet特徴抽出器）──────────────
FEATURES_DIM = 128        # 158→128 に整理（C-4）。GroupNorm の分割も素直になる
NUM_BLOCKS = 3

# ── 学習アルゴリズム（G-1）────────────────────
# "qrdqn": 分布型DQN（sb3-contrib）。報酬ノイズに強く DQN より安定。env はそのまま。
# "dqn"  : 通常のDQN（比較用）。
ALGO = "qrdqn"
N_QUANTILES = 50          # QR-DQN の分位点数（デフォルト200。50で十分かつ軽量）

# ── 学習（共通ハイパーパラメータ）──────────────
SEED = 42                 # 再現性（C-2）
TOTAL_TIMESTEPS = 350_000
LEARNING_RATE = 1e-4
EXPLORATION_FRACTION = 0.2
EXPLORATION_FINAL_EPS = 0.05
BUFFER_SIZE = 200_000     # 100万→20万。観測が大きくメモリ警告が出ていた（F-2）
BATCH_SIZE = 256          # 32→256。GPU を使い切るため（F-5）
GRADIENT_STEPS = 1
TRAIN_FREQ = 4

# ── 早期停止（過学習対策）──────────────────
# 検証スコア(EvalCallback)が頭打ちになったら学習を止め、検証ベスト版を保存する。
EVAL_FREQ = 5000           # 何ステップごとに検証するか
EARLY_STOP_PATIENCE = 5    # この回数連続で改善しなければ停止（5×EVAL_FREQ=25kステップ）
EARLY_STOP_MIN_EVALS = 5   # 最低この回数は評価してから停止判定

# ── パス ────────────────────────────────
MODEL_DIR = "."           # 既存どおりカレントに保存
TENSORBOARD_LOG = "./tb/"  # 学習曲線の記録（C-3）


def model_name(steps=None):
    """チェックポイント/最終モデルのファイル名プレフィックスを生成。"""
    base = f"nikkei_cp_{TRAIN_START}_{TRAIN_END}"
    if steps is None:
        return base
    return f"{base}_{steps}_steps"
