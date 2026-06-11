# recurrentppo_py12 — 日経225トレーディング 強化学習エージェント（RecurrentPPO版）

日経平均株価（^N225）の時系列データから **買い / 待ち / 売り** を判断する強化学習エージェント。
`qrdqn_py12` をベースに、`docs/improvements.md` の改善を取り込んだ **RecurrentPPO（LSTM内蔵PPO）** 版。

> ローカルにGPUが無い前提で、**学習は Colab(GPU)・評価/推論はローカル(CPU)** の二段構えを想定（`notebooks/train_colab.ipynb`）。

---

## qrdqn_py12 からの主な変更（improvements.md 対応）

| 項目 | 変更内容 |
|---|---|
| **G-1: アルゴリズム** | QR-DQN → **RecurrentPPO**（`sb3-contrib`）。オンポリシーで安定。`config.ALGO="ppo"` で通常PPOにも切替可 |
| **G-1: 観測** | 「130日×29特徴のwindow」→ **1日1ベクトル（29特徴＋ポジションone-hot 3）**。時系列の記憶は内蔵LSTMに任せる。取引コストがあるため現在ポジションを観測に明示 |
| **B-3: 正規化** | ウィンドウ内MinMax（絶対水準破壊・基準ズレ）を廃止 → **リターン化/相対化＋ローリング・ロバスト z-score**（中央値/MAD・因果的・±5クリップ） |
| **G-2: 正則化** | optimizer に **weight decay**（`config.WEIGHT_DECAY`）を適用 |
| **G-3: 評価** | `_eval_one.py` が **複数OOS窓**（コロナ/軟調/bull, `config.EVAL_WINDOWS`）で **窓ごとに B&H と比較**し、「全窓でエッジが有るか」を判定 |
| **F-4: 当日データ** | `data.fetch_latest()` で最新OHLC・VIX・金利を自動取得。`server.py` に「📥 最新データ自動取得」ボタン。実OHLCで延長するので ATR/TR の歪み（D-2）も解消 |

学習・検証・テストは時系列で完全分離（リーク防止）:

- **学習**: `〜 VAL_START`（既定 2022-06-01）
- **検証**（早期停止・ベスト選択用）: `VAL_START 〜 TRAIN_END`（2024-01-01）
- **テスト（純OOS）**: `2024-01-01 〜`（学習・検証では一切触れない）

---

## セットアップ

Python 3.12 / [uv](https://docs.astral.sh/uv/) を使用。

```bash
uv sync          # pyproject.toml / uv.lock から仮想環境を構築
```

> ⚠️ ローカルは **必ず Python 3.12**（Colab 保存モデルの読込で segfault を避けるため）。

---

## 使い方

### 1. 学習

```bash
uv run python main.py          # config.SEED で学習
uv run python main.py 3        # シード3で学習（アンサンブル用に複数シード）
```

- `config.py` の設定（`TOTAL_TIMESTEPS` ほか）で学習し、検証スコアで早期停止。
  検証は `WarmupEvalCallback`（評価系と同じ LSTM warmup 手順）で行うため、checkpoint 選抜と本番評価の条件が一致する。
- チェックポイントを `nikkei_rppo_<TRAIN_START>_<TRAIN_END>_seed<N>_<steps>_steps.zip` として保存。
- 検証ベストは `best_seed<N>/best_model.zip` に保存され、最終的に正式名でコピーされる。

GPU で回したい場合は **`notebooks/train_colab.ipynb`**（clone → 依存導入 → 学習 → Drive 保存）を使う。

### 2. 評価 / バックテスト

評価系スクリプトは **カレントにある `nikkei_rppo_*.zip` を自動で拾って** 回す。
RecurrentPPO はトレード開始前に `WINDOW_SIZE` 本の観測を LSTM に流して隠れ状態を温めてから取引する（`main.rollout`）。

```bash
uv run python _eval_one.py        # 複数OOS窓（G-3）でバックテスト。窓ごとに B&H と比較
uv run python _eval_ensemble.py   # 複数モデルの多数決アンサンブルを評価（モデル別にLSTM状態を保持）
```

`_eval_one.py` は窓×モデルの **超過リターン（対B&H）/シャープのマトリクス**と「全窓edge」判定を出す。
`TRAIN_END` より前の窓は in-sample 参考値。

### 3. Web UI

```bash
uv run python visualize.py        # モデル性能の可視化（Gradio, http://localhost:7860）
uv run python server.py           # 単日シミュレーション（Gradio, http://localhost:8888）
```

`server.py` は「📥 最新データ自動取得」ボタンで yfinance/FRED から当日OHLC・VIX・金利を自動入力できる（F-4）。手打ちは上書き用途。

### 4. Docker

```bash
docker compose up --build         # visualize を起動 → http://localhost:13000
```

---

## プロジェクト構成

```
config.py            ハイパーパラメータ・期間設定の一元管理（全スクリプトが import）
data.py              yfinance/FRED からデータ取得＋テクニカル指標（generate_env_data / fetch_latest）
main.py              NikkeiEnv（1日1ベクトル観測＋ロバストz正規化）/ rollout / 学習エントリポイント
algo.py              config.ALGO に応じて RecurrentPPO / PPO を構築（build_model）
calc_performance.py  シャープ・年利・最大DD・勝率などの指標計算
_eval_one.py         複数OOS窓評価（G-3, zip をグロブ）
_eval_ensemble.py    多数決アンサンブル評価（モデル別LSTM状態）
visualize.py         Gradio 性能ビジュアライザ（ポート 7860）
server.py            Gradio 単日シミュレーション＋最新データ自動取得（ポート 8888）
run_simulation.py    server.py が呼ぶシミュレーション本体（実OHLCの manual_data 対応）
notebooks/
  train_colab.ipynb  Colab(GPU) 学習用ノートブック
docs/
  improvements.md    残課題・改善ロードマップ（本版での対応状況を冒頭に記載）
```

---

## 設定（`config.py` 抜粋）

| キー | 既定 | 説明 |
|---|---|---|
| `ALGO` | `recurrentppo` | `recurrentppo` / `ppo` |
| `REWARD_TYPE` | `dsr` | `dsr` / `ddr` / `excess` / `excess_dsr` / `logret` |
| `WINDOW_SIZE` | 130 | warmup バー数（観測ウィンドウではない。評価時のLSTM warmup長も兼ねる） |
| `Z_WINDOW` / `Z_MIN_PERIODS` | 252 / 60 | ロバストz-score のローリング窓 / 最低サンプル |
| `LSTM_HIDDEN` / `LSTM_LAYERS` | 128 / 1 | 内蔵LSTMのサイズ |
| `WEIGHT_DECAY` | 1e-4 | G-2: AdamのL2正則化 |
| `N_STEPS` / `BATCH_SIZE` / `N_EPOCHS` | 2048 / 256 / 10 | PPO rollout・ミニバッチ |
| `ENT_COEF` | 0.01 | エントロピーボーナス（様子見への早期収束を防ぐ） |
| `TOTAL_TIMESTEPS` | 500,000 | 学習ステップ数（早期停止前提） |
| `EVAL_FREQ` / `EARLY_STOP_PATIENCE` | 10000 / 6 | 検証頻度・早期停止 |
| `EVAL_WINDOWS` | 3窓 | G-3: 複数OOS評価窓（コロナ/軟調/bull） |

---

## メモ

- 差分シャープ報酬はエピソード序盤に数値が暴れやすいため、ウォームアップ・分散フロア・クリップで安定化済み（`config.DSR_*`）。
- 観測の正規化は因果的（ローリング統計のみ）なので学習/評価でリークしない。先頭 `Z_MIN_PERIODS` 行は0埋めだが warmup 区間内なので取引には使われない。
- 残っている既知の課題・今後の方針は `docs/improvements.md` を参照。
