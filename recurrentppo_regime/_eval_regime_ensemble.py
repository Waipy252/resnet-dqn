"""レジーム条件付きアンサンブル評価（R-2, MetaTrader-lite）。

「レジームごとに得意なモデルは違う」という前提で、
1) スコア窓で各モデルの **レジーム別シャープ** を測り、
2) トレード窓では現在のレジーム事後確率で加重したスコア
   score_m(t) = Σ_k P(regime_k | t) · sharpe_{m,k}
   が最大のモデルの行動に従う（毎日ゲートを引き直す）。

config.REGIME_GATE_FLAT_IF_NEG=True なら、どのモデルもスコアが負
（＝その地合いでは誰もエッジが無い）の日は FLAT に退避する。
これは前段の議論の「ロバストRL的な保守化をリスク制約として併用」に相当。

baseline として 個別モデル / 多数決アンサンブル / B&H も同条件で出す。

注意: スコア窓がモデルの学習期間と重なる場合、レジーム別スコア自体は
in-sample の情報になる（ゲートの選抜はトレード窓に対しては正直）。
純粋な検証にはスコア窓をモデルの学習期間外に取ること。

使い方:
    uv run python _eval_regime_ensemble.py                      # 既定glob
    uv run python _eval_regime_ensemble.py "models/seed*/*.zip" # glob指定
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
from main import make_env, rollout, Action
from data import generate_env_data
from calc_performance import compute_sharpe_ratio, calculate_performance_metrics
from algo import get_algo_class

assert config.USE_REGIME_OBS, "config.USE_REGIME_OBS=True にしてから実行してください"

K = config.N_REGIMES
DATA_START = "2018-01-01"          # SCORE_START まで z/HMM warmup を確保
SCORE_START, SCORE_END = "2022-01-01", "2024-01-01"  # レジーム別スコアの計測窓
TRADE_START = "2024-01-01"         # ゲート運用の評価窓（→最新）
END = pd.Timestamp.today().strftime("%Y-%m-%d")

patterns = sys.argv[1:] or [f"{config.model_name()}*.zip", "models/seed*/*.zip"]
paths = sorted({p for pat in patterns for p in glob.glob(pat)})
assert paths, f"モデルが見つからない: {patterns}"

print("テストデータ取得中...", DATA_START, "→", END)
test_data = generate_env_data(
    DATA_START, END, ticker=config.TICKER, save_csv="test_data_regime_ens.csv"
)

AlgoClass = get_algo_class()
models = [AlgoClass.load(p, device="cpu") for p in paths]
names = [os.path.basename(p) for p in paths]

# ── Phase 1: スコア窓でレジーム別シャープを計測 ──
print(f"\n--- Phase 1: レジーム別スコア計測 ({SCORE_START} → {SCORE_END}) ---")
score = np.zeros((len(models), K))  # sharpe_{m,k}
for m, (name, mdl) in enumerate(zip(names, models)):
    env = make_env(test_data, trade_start_date=SCORE_START, trade_end_date=SCORE_END)
    _, eq = rollout(mdl, env, deterministic=True, lstm_warmup=True)
    eq = np.asarray(eq, dtype=np.float64)
    rets = np.log(eq[1:] / eq[:-1])
    regimes = env.df["RegimeMAP"].to_numpy()[env.trade_start: env.trade_start + len(rets)]
    overall = rets.mean() / (rets.std() + 1e-12) * np.sqrt(252)
    for k in range(K):
        r = rets[regimes == k]
        # レジームkの観測が少なすぎる場合は全体シャープで代用
        score[m, k] = (
            r.mean() / (r.std() + 1e-12) * np.sqrt(252) if len(r) >= 20 else overall
        )
    print(f"  {name:<44} 別シャープ {np.round(score[m], 2)} (全体 {overall:.2f})")

# ── Phase 2: トレード窓でレジームゲート運用 ──
env = make_env(test_data, trade_start_date=TRADE_START)
print(f"\n--- Phase 2: ゲート運用 {env.dates[env.trade_start].date()} → "
      f"{env.dates[env.end_step].date()} | メンバー {len(models)} ---")
regime_cols = [f"Regime_{k}" for k in range(K)]
probs_all = env.df[regime_cols].to_numpy(dtype=np.float64)

obs, _ = env.reset()
states = [None] * len(models)
ep_start = np.ones(1, dtype=bool)
for i in range(max(env.trade_start - env.window_size, 0), env.trade_start):
    w_obs = env.obs_at(i)
    for k_, mdl in enumerate(models):
        _, states[k_] = mdl.predict(w_obs, state=states[k_], episode_start=ep_start, deterministic=True)
    ep_start = np.zeros(1, dtype=bool)

gate_acts, vote_acts_log, picked = [], [], []
flat_days = 0
done = False
while not done:
    p = probs_all[env.current_step]            # 当日のレジーム事後確率（寄付き時点）
    votes = []
    for k_, mdl in enumerate(models):
        a, states[k_] = mdl.predict(obs, state=states[k_], episode_start=ep_start, deterministic=True)
        votes.append(int(a))
    ep_start = np.zeros(1, dtype=bool)

    gate_scores = score @ p                     # score_m(t) = Σ_k P_k · sharpe_{m,k}
    best = int(np.argmax(gate_scores))
    if config.REGIME_GATE_FLAT_IF_NEG and gate_scores[best] <= 0:
        a = int(Action.FLAT)                    # 誰もエッジが無い地合いは退避
        flat_days += 1
    else:
        a = votes[best]
    picked.append(best)
    gate_acts.append(a)
    vote_acts_log.append(votes)

    obs, _, term, trunc, _ = env.step(a)
    done = term or trunc

gate_eq = env.get_equity_curve()

# ── baseline: 多数決アンサンブル（同じ予測ログから再構成はできないため再走） ──
env_v = make_env(test_data, trade_start_date=TRADE_START)
obs, _ = env_v.reset()
states = [None] * len(models)
ep_start = np.ones(1, dtype=bool)
for i in range(max(env_v.trade_start - env_v.window_size, 0), env_v.trade_start):
    w_obs = env_v.obs_at(i)
    for k_, mdl in enumerate(models):
        _, states[k_] = mdl.predict(w_obs, state=states[k_], episode_start=ep_start, deterministic=True)
    ep_start = np.zeros(1, dtype=bool)
vote_acts = []
done = False
while not done:
    votes = []
    for k_, mdl in enumerate(models):
        a, states[k_] = mdl.predict(obs, state=states[k_], episode_start=ep_start, deterministic=True)
        votes.append(int(a))
    ep_start = np.zeros(1, dtype=bool)
    c = Counter(votes).most_common()
    a = 1 if len(c) > 1 and c[0][1] == c[1][1] else c[0][0]
    vote_acts.append(a)
    obs, _, term, trunc, _ = env_v.step(a)
    done = term or trunc
vote_eq = env_v.get_equity_curve()


def summarize(eq, acts):
    m = calculate_performance_metrics(eq, acts)
    return {
        "ret": (eq[-1] / config.INITIAL_BALANCE - 1) * 100,
        "annual": m["annual_return"],
        "sharpe": compute_sharpe_ratio(eq, yearly_risk_free_rate=0.01),
        "dd": m["max_drawdown"],
        "win": m["win_rate"],
    }


# ── 個別モデル（参考） ──
indiv = []
for name, mdl in zip(names, models):
    e = make_env(test_data, trade_start_date=TRADE_START)
    acts, eq = rollout(mdl, e, deterministic=True, lstm_warmup=True)
    indiv.append(summarize(eq, acts)["ret"])

o = env.open_prices[env.trade_start: env.end_step + 1]
bh = (o[-1] / o[0] - 1) * 100
gate = summarize(gate_eq, gate_acts)
vote = summarize(vote_eq, vote_acts)

print("\n================ 比較 ================")
print(f"  個別 平均総ﾘﾀｰﾝ     : {np.mean(indiv):>7.1f}%  (最小 {min(indiv):.1f}% / 最大 {max(indiv):.1f}%)")
print(f"  多数決アンサンブル   : {vote['ret']:>7.1f}%  | シャープ {vote['sharpe']:.2f} | 最大DD {vote['dd']:.2f}%")
print(f"  レジームゲート(R-2)  : {gate['ret']:>7.1f}%  | シャープ {gate['sharpe']:.2f} | 最大DD {gate['dd']:.2f}% | 勝率 {gate['win']:.1f}%")
print(f"  バイ&ホールド        : {bh:>7.1f}%")
pc = Counter(picked)
print(f"  ゲートの選択内訳     : " + ", ".join(f"{names[m]}×{c}" for m, c in pc.most_common()))
print(f"  FLAT退避日数         : {flat_days} / {len(gate_acts)}")
ca = Counter(gate_acts)
print(f"  ゲート行動内訳       : 買{ca.get(0, 0)} 待{ca.get(1, 0)} 売{ca.get(2, 0)}")
if pd.Timestamp(SCORE_END) > pd.Timestamp(config.VAL_START):
    print(f"\n⚠ スコア窓 {SCORE_START}→{SCORE_END} がモデルの学習/検証期間と重なる場合、"
          "レジーム別スコアは in-sample 情報を含む（docstring 参照）。")
