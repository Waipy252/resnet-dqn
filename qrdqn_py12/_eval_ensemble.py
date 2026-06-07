"""アンサンブル（多数決）の純OOS評価（確認後に削除可）。

複数モデルを読み込み、各ステップで全モデルの行動を多数決して1つの方策にする。
単体checkpointのばらつき（120k当たり/140k外れ）を平均化できるか検証する。

対象モデル: 既定で cwd の nikkei_cp_*_steps.zip と models/seed*/*.zip を両方拾う。
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
from main import NikkeiEnv
from data import generate_env_data
from calc_performance import compute_sharpe_ratio, calculate_performance_metrics
from algo import get_algo_class

DATA_START = "2022-06-01"
OOS_START = "2024-01-01"
END = "2026-06-05"

# 対象モデル（引数でglob指定可: python _eval_ensemble.py "models/seed*/*.zip"）
patterns = sys.argv[1:] or ["nikkei_cp_*_steps.zip", "models/seed*/*.zip"]
paths = sorted({p for pat in patterns for p in glob.glob(pat)})
assert paths, f"モデルが見つからない: {patterns}"

print("テストデータ取得中...", DATA_START, "→", END)
test_data = generate_env_data(DATA_START, END, ticker=config.TICKER)
clean = test_data.dropna().reset_index()
dates = pd.to_datetime(clean[clean.columns[0]])
start_idx = int((dates >= pd.Timestamp(OOS_START)).to_numpy().argmax())
print(f"トレード開始 {dates.iloc[start_idx].date()} → {dates.iloc[-1].date()} | メンバー数 {len(paths)}")

AlgoClass = get_algo_class()
models = [AlgoClass.load(p, device="cpu") for p in paths]


def fresh_env():
    env = NikkeiEnv(test_data, window_size=config.WINDOW_SIZE,
                    transaction_cost=config.TRANSACTION_COST, risk_limit=config.RISK_LIMIT)
    env.reset()
    env.current_step = start_idx
    env.equity_curve = [env.balance]
    env.prev_action = 1
    return env, env._get_observation()


def vote(obs):
    """全モデルの行動を多数決。同数なら FLAT(1) を選ぶ（保守的）。"""
    c = Counter(int(m.predict(obs, deterministic=True)[0]) for m in models)
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return 1  # 同数 → 待ち
    return top[0][0]


def run_policy(action_fn):
    env, obs = fresh_env()
    done, acts = False, []
    while not done:
        a = action_fn(obs)
        acts.append(a)
        obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
    eq = env.get_equity_curve()
    m = calculate_performance_metrics(eq, acts)
    return {
        "ret": (eq[-1]/config.INITIAL_BALANCE-1)*100,
        "annual": m["annual_return"],
        "sharpe": compute_sharpe_ratio(eq, yearly_risk_free_rate=0.01),
        "dd": m["max_drawdown"],
        "win": m["win_rate"],
        "acts": acts,
    }


# ── 個別モデル（参考）──
print("\n--- 個別モデル（参考）---")
print(f"{'model':<28}{'総ﾘﾀｰﾝ%':>9}{'ｼｬｰﾌﾟ':>7}{'最大DD%':>8}")
indiv = []
for p, mdl in zip(paths, models):
    _models_bak = models
    r = run_policy(lambda o, _m=mdl: int(_m.predict(o, deterministic=True)[0]))
    indiv.append(r["ret"])
    print(f"{os.path.basename(p):<28}{r['ret']:>9.1f}{r['sharpe']:>7.2f}{r['dd']:>8.2f}")

# ── アンサンブル（多数決）──
ens = run_policy(vote)

# ── バイ&ホールド ──
o = clean["Open"].to_numpy()[start_idx:]
bh = (o[-1]/o[0]-1)*100

print("\n================ 比較 ================")
print(f"  個別 平均総ﾘﾀｰﾝ : {np.mean(indiv):>7.1f}%  (中央値 {np.median(indiv):.1f}%, 最小 {min(indiv):.1f}%, 最大 {max(indiv):.1f}%)")
print(f"  アンサンブル     : {ens['ret']:>7.1f}%  | 年利 {ens['annual']:.2f}% | シャープ {ens['sharpe']:.2f} | 最大DD {ens['dd']:.2f}% | 勝率 {ens['win']:.1f}%")
print(f"  バイ&ホールド    : {bh:>7.1f}%")
ca = Counter(ens["acts"])
print(f"  アンサンブル行動 : 買{ca.get(0,0)} 待{ca.get(1,0)} 売{ca.get(2,0)}")
