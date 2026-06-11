"""アンサンブル（多数決）の純OOS評価。

複数モデルを読み込み、各ステップで全モデルの行動を多数決して1つの方策にする。
単体checkpointのばらつきを平均化できるか検証する。

RecurrentPPO 対応: 各モデルが自分の LSTM 状態を持つので、
warmup → トレードの間それぞれの状態を別々に保持して進める。

対象モデル: 既定で cwd の nikkei_rppo_*.zip と models/seed*/*.zip を両方拾う。
"""
import glob
import os
import sys
import warnings
from collections import Counter

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

DATA_START = "2022-06-01"
OOS_START = "2024-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")

# 対象モデル（引数でglob指定可: python _eval_ensemble.py "models/seed*/*.zip"）
patterns = sys.argv[1:] or [f"{config.model_name()}*.zip", "models/seed*/*.zip"]
paths = sorted({p for pat in patterns for p in glob.glob(pat)})
assert paths, f"モデルが見つからない: {patterns}"

print("テストデータ取得中...", DATA_START, "→", END)
test_data = generate_env_data(DATA_START, END, ticker=config.TICKER)

AlgoClass = get_algo_class()
models = [AlgoClass.load(p, device="cpu") for p in paths]

env0 = make_env(test_data, trade_start_date=OOS_START)
print(f"トレード開始 {env0.dates[env0.trade_start].date()} → {env0.dates[env0.end_step].date()} | メンバー数 {len(paths)}")


def summarize(eq, acts):
    m = calculate_performance_metrics(eq, acts)
    return {
        "ret": (eq[-1] / config.INITIAL_BALANCE - 1) * 100,
        "annual": m["annual_return"],
        "sharpe": compute_sharpe_ratio(eq, yearly_risk_free_rate=0.01),
        "dd": m["max_drawdown"],
        "win": m["win_rate"],
        "acts": acts,
    }


# ── 個別モデル（参考）──
print("\n--- 個別モデル（参考）---")
print(f"{'model':<36}{'総ﾘﾀｰﾝ%':>9}{'ｼｬｰﾌﾟ':>7}{'最大DD%':>8}")
indiv = []
for p, mdl in zip(paths, models):
    env = make_env(test_data, trade_start_date=OOS_START)
    acts, eq = rollout(mdl, env, deterministic=True, lstm_warmup=True)
    r = summarize(eq, acts)
    indiv.append(r["ret"])
    print(f"{os.path.basename(p):<36}{r['ret']:>9.1f}{r['sharpe']:>7.2f}{r['dd']:>8.2f}")

# ── アンサンブル（多数決）。各モデルのLSTM状態を個別に保持 ──
env = make_env(test_data, trade_start_date=OOS_START)
obs, _ = env.reset()

# warmup: トレード開始前 window_size 本の観測（FLAT）を全モデルに流す
states = [None] * len(models)
ep_start = np.ones(1, dtype=bool)
start = max(env.trade_start - env.window_size, 0)
for i in range(start, env.trade_start):
    w_obs = env.obs_at(i)
    for k, mdl in enumerate(models):
        _, states[k] = mdl.predict(w_obs, state=states[k], episode_start=ep_start, deterministic=True)
    ep_start = np.zeros(1, dtype=bool)

ens_acts = []
done = False
while not done:
    votes = []
    for k, mdl in enumerate(models):
        a, states[k] = mdl.predict(obs, state=states[k], episode_start=ep_start, deterministic=True)
        votes.append(int(a))
    ep_start = np.zeros(1, dtype=bool)
    c = Counter(votes).most_common()
    # 同数なら FLAT(1) を選ぶ（保守的）
    a = 1 if len(c) > 1 and c[0][1] == c[1][1] else c[0][0]
    ens_acts.append(a)
    obs, _, term, trunc, _ = env.step(a)
    done = term or trunc

ens = summarize(env.get_equity_curve(), ens_acts)

# ── バイ&ホールド ──
o = env.open_prices[env.trade_start: env.end_step + 1]
bh = (o[-1] / o[0] - 1) * 100

print("\n================ 比較 ================")
print(f"  個別 平均総ﾘﾀｰﾝ : {np.mean(indiv):>7.1f}%  (中央値 {np.median(indiv):.1f}%, 最小 {min(indiv):.1f}%, 最大 {max(indiv):.1f}%)")
print(f"  アンサンブル     : {ens['ret']:>7.1f}%  | 年利 {ens['annual']:.2f}% | シャープ {ens['sharpe']:.2f} | 最大DD {ens['dd']:.2f}% | 勝率 {ens['win']:.1f}%")
print(f"  バイ&ホールド    : {bh:>7.1f}%")
ca = Counter(ens["acts"])
print(f"  アンサンブル行動 : 買{ca.get(0,0)} 待{ca.get(1,0)} 売{ca.get(2,0)}")
