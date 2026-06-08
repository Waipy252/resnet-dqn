# qrdqn_py12 — 日経225トレーディング 強化学習エージェント

日経平均株価（^N225）の時系列データから **買い / 待ち / 売り** を判断する強化学習エージェント。
分布型DQN（**QR-DQN**）＋ **1D ResNet** 特徴抽出器で学習し、リスク調整報酬（差分シャープレシオ）で安定化している。

> ローカルにGPUが無い前提で、**学習は Colab(GPU)・評価/推論はローカル(CPU)** の二段構えを想定（`notebooks/train_colab.ipynb`）。

---

## 主な構成

| 要素 | 内容 |
|---|---|
| アルゴリズム | QR-DQN（`sb3-contrib`）。`config.ALGO` で通常 DQN にも切替可 |
| 特徴抽出器 | 1D ResNet（`ResNetFeatures`）。GroupNorm 採用（DQN と相性の良い正規化） |
| 観測 | 直近 `WINDOW_SIZE=130` 日 × 29 特徴（移動平均/σバンド/RSI/MACD/RCI/ATR/金利/VIX 等）をウィンドウ内 MinMax 正規化 |
| 行動 | `0:LONG / 1:FLAT / 2:SHORT`（`Action` enum） |
| 報酬 | 差分シャープレシオ（Moody & Saffell）。`config.REWARD_TYPE` で対数リターンにも切替可 |
| 環境 | Gymnasium 準拠の `NikkeiEnv`（`terminated`=リスク失格 / `truncated`=データ終端） |

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
uv run python main.py
```

- `config.py` の設定（`TOTAL_TIMESTEPS` ほか）で学習し、検証スコアで早期停止。
- チェックポイントを `nikkei_cp_<TRAIN_START>_<TRAIN_END>_<steps>_steps.zip` として保存。
- 検証ベストは `best/best_model.zip` に保存され、最終的に正式名でコピーされる。

GPU で回したい場合は **`notebooks/train_colab.ipynb`**（clone → 依存導入 → 学習 → Drive 保存）を使う。

### 2. 評価 / バックテスト

評価系スクリプトは **カレントにある `nikkei_cp_*.zip` を自動で拾って** 回す（命名規約 `*_steps.zip` でなくリネーム済みモデルも対象）。

```bash
uv run python _eval_one.py        # 純OOS（2024-01-01〜）でバックテスト＋指標サマリ
uv run python _eval_ensemble.py   # 複数モデルの多数決アンサンブルを評価
uv run python eval.py             # 学習期間込みのバックテスト（プロット付き）
```

出力: 最終資産・総リターン・年利・シャープ・最大DD・勝率・PF などを表示し、バイ&ホールドと比較。

### 3. Web UI

```bash
uv run python visualize.py        # モデル性能の可視化（Gradio, http://localhost:7860）
uv run python server.py           # 単日シミュレーション（Gradio, http://localhost:8888）
```

`visualize.py` は全モデルの資産カーブ・指標比較・**アンサンブル（多数決）** をタブで表示する。

### 4. Docker

```bash
docker compose up --build         # visualize を起動 → http://localhost:13000
```

---

## プロジェクト構成

```
config.py            ハイパーパラメータ・期間設定の一元管理（全スクリプトが import）
data.py              yfinance/FRED からデータ取得＋テクニカル指標を生成（generate_env_data）
main.py              NikkeiEnv / ResNetFeatures / 学習エントリポイント・学習ユーティリティ
algo.py              config.ALGO に応じて QR-DQN / DQN を構築（build_model）
calc_performance.py  シャープ・年利・最大DD・勝率などの指標計算
eval.py              チェックポイントをバックテスト（matplotlib プロット）
_eval_one.py         純OOS 評価（zip をグロブ）
_eval_ensemble.py    多数決アンサンブル評価
visualize.py         Gradio 性能ビジュアライザ（ポート 7860）
server.py            Gradio 単日シミュレーション（ポート 8888）
run_simulation.py    server.py が呼ぶシミュレーション本体
notebooks/
  train_colab.ipynb  Colab(GPU) 学習用ノートブック
docs/
  improvements.md    残課題・改善ロードマップ
```

---

## 設定（`config.py` 抜粋）

| キー | 既定 | 説明 |
|---|---|---|
| `ALGO` | `qrdqn` | `qrdqn` / `dqn` |
| `REWARD_TYPE` | `dsr` | 差分シャープ `dsr` / 対数リターン `logret` |
| `WINDOW_SIZE` | 130 | 観測ウィンドウ日数 |
| `TRANSACTION_COST` | 0.001 | 取引コスト（学習・評価で統一） |
| `RISK_LIMIT` | 0.5 | 初期資産のこの割合を割ると終了 |
| `TOTAL_TIMESTEPS` | 180,000 | 学習ステップ数 |
| `BUFFER_SIZE` | 200,000 | リプレイバッファ |
| `EVAL_FREQ` / `EARLY_STOP_PATIENCE` | 5000 / 5 | 検証頻度・早期停止 |

---

## メモ

- 差分シャープ報酬はエピソード序盤に数値が暴れやすいため、ウォームアップ・分散フロア・クリップで安定化済み（`config.DSR_*`）。
- 残っている既知の課題・今後の方針は `docs/improvements.md` を参照。
