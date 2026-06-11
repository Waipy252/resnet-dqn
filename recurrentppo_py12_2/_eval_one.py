"""チェックポイントの複数OOS窓評価（G-3）。

「2024-01〜の1窓だけ」では高分散の1回くじなので、地合いの異なる複数窓
（config.EVAL_WINDOWS: コロナ / 軟調 / bull）で各窓ごとに B&H と比較し、
「どの地合いでも B&H に対してプラスのエッジがあるか」を判定する。

注意: TRAIN_END(2024-01-01) より前の窓は学習データと重なる in-sample 参考値。
純OOSは TRAIN_END 以降の窓のみ。

RecurrentPPO: トレード開始前に window_size 本の観測を LSTM に流して
隠れ状態を温めてから取引を始める（main.rollout が面倒を見る）。
"""
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float32)

import config
from main import make_env, rollout
from data import generate_env_data
from calc_performance import compute_sharpe_ratio, calculate_performance_metrics
from algo import get_algo_class

# 最初の窓の warmup（z-score統計 + LSTM warmup）が確保できるよう余裕を持って取得
DATA_START = "2018-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")

print("テストデータ取得中...", DATA_START, "→", END)
test_data = generate_env_data(
    DATA_START, END, ticker=config.TICKER, save_csv="test_data_eval.csv"
)

AlgoClass = get_algo_class()
print("algo =", AlgoClass.__name__, "| reward =", config.REWARD_TYPE)


def _steps(p):
    """ファイル名から識別ラベル。seed があれば "sN:steps" 形式で区別する。"""
    stem = os.path.splitext(os.path.basename(p))[0]
    m = re.search(r"seed(\d+)", stem)
    tag = f"s{m.group(1)}:" if m else ""
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[2] == "steps" and parts[1].isdigit():
        return f"{tag}{int(parts[1])}" if tag else int(parts[1])
    prefix = f"{config.model_name()}_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


# `*_steps.zip` の命名に縛られず、リネーム済みのベストモデルも拾う
models = sorted(glob.glob(f"{config.model_name()}*.zip"), key=lambda p: str(_steps(p)))
print("対象モデル:", [_steps(p) for p in models])

# 窓ごとの B&H（環境と同じ前処理での Open 始点→終点）を先に計算
windows = []
for ws, we, label in config.EVAL_WINDOWS:
    env = make_env(test_data, trade_start_date=ws, trade_end_date=we)
    o = env.open_prices[env.trade_start: env.end_step + 1]
    bh = (o[-1] / o[0] - 1) * 100
    is_oos = pd.Timestamp(ws) >= pd.Timestamp(config.TRAIN_END)
    windows.append({
        "start": ws, "end": we, "label": label, "bh": bh, "oos": is_oos,
        "from": env.dates[env.trade_start].date(), "to": env.dates[env.end_step].date(),
    })
    print(f"窓 [{label}] {env.dates[env.trade_start].date()} → {env.dates[env.end_step].date()} "
          f"| B&H {bh:+.1f}% | {'純OOS' if is_oos else 'in-sample(参考)'}")

# results[model][window_label] = dict(指標)
results = {}
for path in models:
    model = AlgoClass.load(path, device="cpu")
    name = str(_steps(path))
    results[name] = {}
    for w in windows:
        env = make_env(test_data, trade_start_date=w["start"], trade_end_date=w["end"])
        actions, eq = rollout(model, env, deterministic=True, lstm_warmup=True)
        sharpe = compute_sharpe_ratio(eq, yearly_risk_free_rate=0.01)
        m = calculate_performance_metrics(eq, actions)
        ret = (eq[-1] / config.INITIAL_BALANCE - 1) * 100
        n_long, n_flat, n_short = actions.count(0), actions.count(1), actions.count(2)
        results[name][w["label"]] = {
            "ret": ret, "excess": ret - w["bh"], "sharpe": sharpe,
            "annual": m["annual_return"], "dd": m["max_drawdown"], "win": m["win_rate"],
        }
        print(f"\n# {name} | 窓 [{w['label']}] ({w['from']} → {w['to']})")
        print(f"  総リターン   : {ret:+.1f}%  (B&H {w['bh']:+.1f}% → 超過 {ret - w['bh']:+.1f}%)")
        print(f"  年利/シャープ: {m['annual_return']:.2f}% / {sharpe:.2f}")
        print(f"  最大DD/勝率  : {m['max_drawdown']:.2f}% / {m['win_rate']:.1f}% (取引 {m['total_trades']})")
        print(f"  行動内訳     : 買{n_long} 待{n_flat} 売{n_short}")
    del model

# ── 窓×モデルのサマリ（超過リターン=対B&H）──
print("\n================ 窓別サマリ: 超過リターン% (戦略 − B&H) ================")
header = f"{'model':>14} |" + "".join(f" {w['label'][:12]:>14}" for w in windows) + f" {'全窓edge':>8}"
print(header)
for name, r in results.items():
    cells = ""
    all_edge = True
    for w in windows:
        e = r[w["label"]]["excess"]
        all_edge &= e > 0
        cells += f" {e:>+14.1f}"
    print(f"{name:>14} |{cells} {'◎' if all_edge else '×':>8}")
print(f"{'B&H(絶対%)':>14} |" + "".join(f" {w['bh']:>+14.1f}" for w in windows))

print("\n================ 窓別サマリ: シャープレシオ ================")
print(f"{'model':>14} |" + "".join(f" {w['label'][:12]:>14}" for w in windows))
for name, r in results.items():
    print(f"{name:>14} |" + "".join(f" {r[w['label']]['sharpe']:>14.2f}" for w in windows))

print("\n判定基準: 「全窓edge=◎」＝どの地合いでも B&H 超え。1窓だけ良くても運の可能性が高い。")
print(f"※ {config.TRAIN_END} より前の窓は in-sample（参考値）。")
