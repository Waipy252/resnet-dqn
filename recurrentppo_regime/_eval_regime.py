"""レジーム別のパフォーマンス分解評価（R-1）。

_eval_one.py が「期間の窓」ごとに B&H と比較するのに対し、こちらは
HMM が検知したレジーム（低ボラ→高ボラ）ごとに日次リターンを集計し、
「どの地合いで勝てて、どの地合いで負けているか」を可視化する。

各日の帰属は RegimeMAP（filtered 事後確率の argmax）。行tのレジームは
寄付き時点の情報のみで決まるので、その日の行動・損益と因果的に対応する。
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
from algo import get_algo_class

assert config.USE_REGIME_OBS, "config.USE_REGIME_OBS=True にしてから実行してください"

# z-score統計 + LSTM warmup + レジームHMMの最低学習日数（REGIME_MIN_TRAIN）を確保
DATA_START = "2016-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
K = config.N_REGIMES
REGIME_LABELS = {0: "低ボラ(凪)", K - 1: "高ボラ(荒れ)"}  # 中間は「通常」


def regime_label(k):
    return REGIME_LABELS.get(k, f"通常{k}" if K > 3 else "通常")


print("テストデータ取得中...", DATA_START, "→", END)
test_data = generate_env_data(
    DATA_START, END, ticker=config.TICKER, save_csv="test_data_regime.csv"
)

AlgoClass = get_algo_class()
print("algo =", AlgoClass.__name__, "| reward =", config.REWARD_TYPE, "| K =", K)


def _steps(p):
    stem = os.path.splitext(os.path.basename(p))[0]
    m = re.search(r"seed(\d+)", stem)
    tag = f"s{m.group(1)}:" if m else ""
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[2] == "steps" and parts[1].isdigit():
        return f"{tag}{int(parts[1])}" if tag else int(parts[1])
    m = re.match(r"nikkei_\w+_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_?(.+)", stem)
    return f"{tag}{m.group(1)}" if m else stem


models = sorted(glob.glob("nikkei_*.zip"), key=lambda p: str(_steps(p)))
print("対象モデル:", [_steps(p) for p in models])


def per_regime(returns, regimes):
    """日次対数リターンをレジーム別に集計 → {k: (日数, 年率%, シャープ)}。"""
    out = {}
    for k in range(K):
        r = returns[regimes == k]
        if len(r) < 2:
            out[k] = (len(r), np.nan, np.nan)
            continue
        ann = (np.exp(r.mean() * 252) - 1) * 100
        sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(252)
        out[k] = (len(r), ann, sharpe)
    return out


def print_table(title, stats):
    print(f"\n  {title}")
    print(f"  {'レジーム':<12}{'日数':>6}{'年率%':>10}{'シャープ':>10}")
    for k in range(K):
        n, ann, sh = stats[k]
        a = f"{ann:>10.1f}" if np.isfinite(ann) else f"{'-':>10}"
        s = f"{sh:>10.2f}" if np.isfinite(sh) else f"{'-':>10}"
        print(f"  {regime_label(k):<12}{n:>6}{a}{s}")


for ws, we, label in config.EVAL_WINDOWS:
    env0 = make_env(test_data, trade_start_date=ws, trade_end_date=we)
    lo, hi = env0.trade_start, env0.end_step
    regimes = env0.df["RegimeMAP"].to_numpy()[lo: hi + 1]
    open_ = env0.open_prices
    # 行tの市場リターン = log(Open_{t+1}/Open_t)（envのステップ定義と同じ）
    nxt = np.minimum(np.arange(lo, hi + 1) + 1, env0.n - 1)
    mkt = np.log(open_[nxt] / open_[lo: hi + 1])
    is_oos = pd.Timestamp(ws) >= pd.Timestamp(config.TRAIN_END)

    print(f"\n========== 窓 [{label}] {env0.dates[lo].date()} → {env0.dates[hi].date()} "
          f"({'純OOS' if is_oos else 'in-sample(参考)'}) ==========")
    counts = " / ".join(f"{regime_label(k)} {int((regimes == k).sum())}日" for k in range(K))
    print(f"  レジーム内訳: {counts}")
    print_table("B&H（市場そのもの）", per_regime(mkt, regimes))

    for path in models:
        model = AlgoClass.load(path, device="cpu")
        env = make_env(test_data, trade_start_date=ws, trade_end_date=we)
        actions, eq = rollout(model, env, deterministic=True, lstm_warmup=True)
        eq = np.asarray(eq, dtype=np.float64)
        rets = np.log(eq[1:] / eq[:-1])  # 行tの戦略リターン（手数料込み）
        n = min(len(rets), len(regimes))
        print_table(f"モデル {_steps(path)}", per_regime(rets[:n], regimes[:n]))
        del model

cur = test_data["RegimeMAP"].iloc[-1]
cur_p = test_data[[f"Regime_{k}" for k in range(K)]].iloc[-1].to_numpy()
print(f"\n現在のレジーム: {regime_label(int(cur))}  P = {np.round(cur_p, 2)}")
print("見方: B&Hの段はその地合い自体の性質。モデルがB&Hよりシャープが高い地合いが「得意レジーム」。")
