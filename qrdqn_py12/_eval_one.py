"""ダウンロード済みチェックポイントの簡易評価（確認後に削除可）。

純OOS評価: データは warmup 用に早めから取得しつつ、
実トレード開始を OOS_START（学習終了後）に固定して、純粋な未学習期間だけを測る。
"""
import glob
import os
import re
import warnings

import pandas as pd
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float32)

import config
from main import NikkeiEnv
from data import generate_env_data
from calc_performance import compute_sharpe_ratio, calculate_performance_metrics
from algo import get_algo_class

DATA_START = "2022-06-01"  # 正規化/特徴量の warmup 用（トレードはしない）
OOS_START = "2024-01-01"   # 学習終了日。ここからトレード開始＝純OOS
END = "2026-06-05"

print("テストデータ取得中...", DATA_START, "→", END)
test_data = generate_env_data(DATA_START, END, ticker=config.TICKER)

# env と同じ前処理(dropna)を再現して、OOS_START に対応する位置インデックスを求める
clean = test_data.dropna().reset_index()
date_col = clean.columns[0]
dates = pd.to_datetime(clean[date_col])
cutoff = pd.Timestamp(OOS_START)
start_idx = int((dates >= cutoff).to_numpy().argmax())
assert start_idx >= config.WINDOW_SIZE, (
    f"warmup不足: start_idx={start_idx} < window={config.WINDOW_SIZE}. DATA_STARTを早める"
)
trade_start_date = dates.iloc[start_idx].date()
trade_end_date = dates.iloc[-1].date()
print(f"全行(dropna後)={len(clean)} | トレード開始idx={start_idx} ({trade_start_date}) → {trade_end_date}")

AlgoClass = get_algo_class()
print("algo =", AlgoClass.__name__, "| reward =", config.REWARD_TYPE)

def _steps(p):
    """ファイル名から識別ラベル。seed があれば "sN:steps" 形式で区別する
    （複数シードで同じ step 番号が重複して見分けがつかなくなるのを防ぐ）。"""
    stem = os.path.splitext(os.path.basename(p))[0]
    m = re.search(r"seed(\d+)", stem)
    tag = f"s{m.group(1)}:" if m else ""
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[2] == "steps" and parts[1].isdigit():
        return f"{tag}{int(parts[1])}" if tag else int(parts[1])
    prefix = f"nikkei_cp_{config.TRAIN_START}_{config.TRAIN_END}_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


# `*_steps.zip` の命名に縛られず、リネーム済みのベストモデルも拾う
models = sorted(glob.glob("nikkei_cp_*.zip"), key=lambda p: str(_steps(p)))
print("対象モデル:", [_steps(p) for p in models])

summary = []
for path in models:
    env = NikkeiEnv(
        test_data,
        window_size=config.WINDOW_SIZE,
        transaction_cost=config.TRANSACTION_COST,
        risk_limit=config.RISK_LIMIT,
    )
    env.reset()
    # トレード開始を OOS_START に移動（warmup区間はスキップ）
    env.current_step = start_idx
    env.equity_curve = [env.balance]
    env.prev_action = 1  # flat から開始
    obs = env._get_observation()

    model = AlgoClass.load(path, device="cpu")

    done = False
    actions = []
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        a = int(a)
        actions.append(a)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc

    eq = env.get_equity_curve()
    sharpe = compute_sharpe_ratio(eq, yearly_risk_free_rate=0.01)
    m = calculate_performance_metrics(eq, actions)
    n_long, n_flat, n_short = actions.count(0), actions.count(1), actions.count(2)

    print("\n==============================")
    print(f"# {path}")
    print(f"  最終資産     : {int(eq[-1]):,} (初期 {config.INITIAL_BALANCE:,})")
    print(f"  総リターン   : {(eq[-1]/config.INITIAL_BALANCE - 1)*100:.1f}%")
    print(f"  年利         : {m['annual_return']:.2f}%")
    print(f"  シャープ     : {sharpe:.2f}")
    print(f"  最大DD       : {m['max_drawdown']:.2f}%")
    print(f"  勝率         : {m['win_rate']:.2f}%  (取引 {m['total_trades']})")
    print(f"  PF / 期待値  : {m['profit_factor']:.2f} / {m['expectancy']:.4f}%")
    print(f"  行動内訳     : 買{n_long} 待{n_flat} 売{n_short}")
    summary.append((_steps(path), (eq[-1]/config.INITIAL_BALANCE-1)*100, m["annual_return"], sharpe, m["max_drawdown"], m["win_rate"]))
    del model

# バイ&ホールド比較（OOS区間のみ）
o = clean["Open"].to_numpy()[start_idx:]
bh = (o[-1]/o[0]-1)*100
print("\n(参考) 同OOS区間バイ&ホールド:")
print(f"  {int(o[0]):,} → {int(o[-1]):,}  ({bh:.1f}%)")

# ── ランキング（OOSシャープ降順）──
print("\n================ OOS サマリ (シャープ降順) ================")
print(f"{'steps':>7} {'総ﾘﾀｰﾝ%':>8} {'年利%':>7} {'ｼｬｰﾌﾟ':>6} {'最大DD%':>7} {'勝率%':>6}")
for s in sorted(summary, key=lambda x: x[3], reverse=True):
    print(f"{s[0]:>7} {s[1]:>8.1f} {s[2]:>7.2f} {s[3]:>6.2f} {s[4]:>7.2f} {s[5]:>6.1f}")
print(f"{'B&H':>7} {bh:>8.1f}")
