"""学習・評価で共有する設定（A-4 / C-4: ハイパーパラメータの一元管理）。

main.py / _eval_one.py / run_simulation.py はここを import して同じ値を使う。
学習と評価でコストやリスク制限がズレないようにするのが目的。

qrdqn_py12 からの主な変更（docs/improvements.md 対応）:
- G-1: アルゴリズムを RecurrentPPO（LSTM内蔵, sb3-contrib）に変更。
       window をネットワークに食わせる代わりに「1日1観測ベクトル」を LSTM に流す。
- B-3: 観測の MinMax 正規化を廃止し、リターン化＋ローリング・ロバスト z-score に変更。
- G-2: weight decay を optimizer に適用（過学習対策）。
- G-3: 複数OOS窓での評価（EVAL_WINDOWS）。
"""

# ── データ ──────────────────────────────
TICKER = "^N225"
TRAIN_START = "1997-01-01"
TRAIN_END = "2026-06-11"  # in-sample データの終端（≒最新）＝モデル名にも使用

# 学習/検証 分割（早期停止用）。学習はVAL_STARTまで、検証はVAL_START→TRAIN_END。
# 注意: 検証が最新期間まで及ぶため、純OOSのテスト期間は存在しない構成。
VAL_START = "2024-01-01"        # ここから先は学習に使わず検証に回す
VAL_WARMUP_START = "2022-06-01"  # 検証env の正規化warmup用にこの辺りからデータを渡す

# ── 環境 ────────────────────────────────
# WINDOW_SIZE は「観測ウィンドウ」ではなく warmup バー数になった（G-1）。
# 観測は1日1ベクトル（LSTMが時系列を記憶する）。トレード開始前に最低この本数の
# 履歴を確保し、評価時は同じ本数の観測を LSTM に流して隠れ状態を温める。
WINDOW_SIZE = 130
INITIAL_BALANCE = 1_000_000
TRANSACTION_COST = 0.001  # 学習・評価で統一（往復スプレッド+手数料想定, A-4）
RISK_LIMIT = 0.5          # 初期資産の RISK_LIMIT 未満で終了（学習・評価で統一）

# ── 観測の正規化（B-3）──────────────────────
# 旧: ウィンドウ内 MinMax（絶対水準を破壊し、ウィンドウ間で基準がズレる）。
# 新: ①リターン化/相対化で定常化（Open→対数差分, SMA/σバンド→Open比, ATR/MACD→Open比,
#     VIX→log など）→ ②ローリング・ロバスト z-score（中央値/MAD, 因果的=未来を見ない）
#     → ③±OBS_CLIP でクリップ。
Z_WINDOW = 252        # ロバストz-score のローリング窓（約1年）
Z_MIN_PERIODS = 60    # 統計を信用する最低サンプル数（それ以前は観測0埋め, warmup内）
OBS_CLIP = 5.0        # z-score のクリップ幅

# ── 報酬設計（G-3-1）──────────────────────
# "dsr"   : 差分シャープレシオ（Moody&Saffell）。リスク調整後の改善を即時報酬に。
#           生の対数リターンより学習が安定し、ドローダウンを抑えて一貫した勝ち方に。
# "ddr"   : 差分下方偏差レシオ（Differential Downside Deviation Ratio; Moody&Saffell 1998）。
#           DSRの分母（全分散）を下方偏差（負リターンのみ）に置換したSortino型（G-3-4）。
#           上昇のボラを罰さないので「上昇を取りに行く」方向に効き、DSRの様子見偏りを緩和。
# "excess": 超過リターン（対B&H）。reward = 戦略の対数リターン − 市場(Long固定)の対数リターン（G-3-5）。
# "excess_dsr": 差分インフォメーションレシオ（G-3-6）。DSRを超過リターン(戦略−市場)の系列に対して計算する。
# "logret": 1日分の対数リターン（従来）。比較用に残す。
#
# ── logret 骨格の派生（G-3-8）。logret の「学習が安定する」性質（スケール一定・分母なし）
#    を保ったまま、小さな修正項でリスク調整を注入する系統。
# "risk"   : ちょっといい。リスク感応 logret。reward = r − RISK_LAMBDA·r²/2（Ritter 2017 の平均分散効用）。
#            大きな変動を対称に罰する。λ=0 で logret に退化。
# "asym"   : 損失非対称 logret。reward = r (r>0) / ASYM_KAPPA·r (r≤0)。
#            Sortino の思想を分母なしで実現。下げ局面で Flat/Short へ逃げる誘因。
# "ddpen"  : ドローダウン・ペナルティ logret。reward = r − DD_LAMBDA·max(0, dd_t − dd_{t-1})。
#            新たにDDを掘った分だけ罰する。検証選抜で見る最大DDと直結。
#            dd はピーク依存で非マルコフ的だが LSTM が状態を補う前提（RecurrentPPO向き）。
# "volnorm": ボラ正規化 logret（DSR-lite）。reward = r / max(σ_ema, √DSR_VAR_FLOOR)。
#            DSR のリスク調整の核だけ残し不安定な微分項を捨てた版。分母が復活するので
#            warmup 中は報酬0・±DSR_CLIP でクリップ（DSR と同じ安定化を流用）。
REWARD_TYPE = "volnorm"
RISK_LAMBDA = 2.0    # risk: 2次ペナルティの強さ（1〜10目安）
ASYM_KAPPA = 2.0     # asym: 損失側の倍率（1.5〜3目安, 1で logret に退化）
DD_LAMBDA = 2.0      # ddpen: DD増分ペナルティの強さ（1〜5目安, 0で logret に退化）
# 部分ベンチマーク係数 β（excess / excess_dsr 用, G-3-7）:
#   reward = step_log_return − BENCHMARK_WEIGHT · market_log_return
BENCHMARK_WEIGHT = 0.2
DSR_ETA = 0.01            # 差分シャープ/下方偏差の適応率（リターン統計のEMA係数）
DSR_WARMUP = 100          # 最初の N ステップは報酬0（≒1/DSR_ETA。統計だけ温める）
DSR_VAR_FLOOR = 1e-4      # 分散/下方偏差の下限（日次対数リターン std≈0.01 → var≈1e-4 のスケール）
DSR_CLIP = 1.0            # 1ステップ報酬のクリップ幅 [-1,1]

# ── 学習アルゴリズム（G-1）────────────────────
# "recurrentppo": LSTM内蔵PPO（sb3-contrib）。部分観測の時系列に強く、window特徴抽出を
#                 内蔵LSTMに任せられる。オンポリシーで安定。
# "ppo"         : 通常のPPO（比較用）。同じ1日1ベクトル観測を使う（LSTMなし＝記憶なし）。
ALGO = "recurrentppo"
LSTM_HIDDEN = 128         # LSTM 隠れ状態の次元
LSTM_LAYERS = 1           # LSTM 層数
NET_ARCH = [128]          # LSTM後の policy/value ヘッドのMLP幅
WEIGHT_DECAY = 1e-4       # G-2: 過学習対策（AdamのL2正則化）

# ── 学習（PPO ハイパーパラメータ）──────────────
SEED = 42                 # 再現性（C-2）
TOTAL_TIMESTEPS = 300_000  # PPOはオフポリシーよりサンプル効率が低いので多め（早期停止前提）
LEARNING_RATE = 3e-4
# 線形減衰スケジュール（PPOの常套手段。終盤の方策の暴れ・破壊的更新を抑える）。
# "linear": progress_remaining(1→0) に比例して 初期値→0 へ線形減衰 / "constant": 固定。
# 減衰は TOTAL_TIMESTEPS 基準なので、早期停止した場合は途中の値で止まる
# （例: 500kのうち150kで停止 → LRは初期値の70%まで下がった状態）。
LR_SCHEDULE = "linear"
CLIP_RANGE_SCHEDULE = "linear"
N_STEPS = 2048            # 1回の rollout 長
BATCH_SIZE = 256          # ミニバッチ（N_STEPS を割り切る値）
N_EPOCHS = 10             # rollout ごとの再利用エポック数
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01           # エントロピーボーナス（様子見への早期収束を防ぐ）
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# ── 早期停止（過学習対策）──────────────────
# 検証スコア(EvalCallback)が頭打ちになったら学習を止め、検証ベスト版を保存する。
EVAL_FREQ = 10000         # 何ステップごとに検証するか
EARLY_STOP_PATIENCE = 6    # この回数連続で改善しなければ停止（6×EVAL_FREQ=60kステップ）
EARLY_STOP_MIN_EVALS = 5   # 最低この回数は評価してから停止判定

# ── 評価（G-3: 複数OOS窓）────────────────────
# 地合いの異なる窓ごとに B&H と比較してエッジの有無を判定する。
# TRAIN_END より前の窓は in-sample（参考値）。純OOSは TRAIN_END 以降の窓のみ。
# (開始日, 終了日 or None=最新, ラベル)
EVAL_WINDOWS = [
    ("2020-01-01", "2022-01-01", "コロナ暴落→回復"),
    ("2022-01-01", "2024-01-01", "軟調・もみ合い"),
    ("2024-01-01", None, "bull(純OOS)"),
]

# ── パス ────────────────────────────────
MODEL_DIR = "."           # 既存どおりカレントに保存
TENSORBOARD_LOG = None  # 学習曲線の記録（C-3）


def model_name(steps=None):
    """チェックポイント/最終モデルのファイル名プレフィックスを生成。"""
    base = f"nikkei_rppo_{TRAIN_START}_{TRAIN_END}"
    if steps is None:
        return base
    return f"{base}_{steps}_steps"
