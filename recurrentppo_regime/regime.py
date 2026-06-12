"""レジーム検知（R-1: ガウシアンHMMによる市場レジームの因果的推定）。

「刻々と変わる相場の地合い（レジーム）」を隠れマルコフモデル（HMM）で推定し、
各日のレジーム事後確率 P(regime_k | 当日寄付きまでの情報) を観測特徴として
エージェントに渡す。金融のレジームスイッチングモデルの伝統と、非定常RLの
文脈検知（MBCD 系）の発想を組み合わせた構成。

リーク防止（B-3 と同じ思想で因果性を厳守）:
- HMM入力は Open の対数差分とそのローリングσ。行tの値は寄付き時点で既知。
- パラメータ推定（EM）はウォークフォワード: 時点tで使うHMMは t より前の
  データだけで学習し、REGIME_REFIT_EVERY 日ごとに拡大窓で再学習する。
- 事後確率は前向きアルゴリズム（filtered probability）で計算する。
  hmmlearn の predict_proba は前向き後ろ向き（smoothed）で系列内の未来を
  参照するため使わず、forward filter を自前実装する。

ラベルの安定化:
- EM の状態番号は再学習のたびに入れ替わりうるので、毎回「ボラティリティの
  昇順」で並べ替える。Regime_0=低ボラ（凪・上昇相場側）〜 Regime_{K-1}=高ボラ
  （急落・危機側）という意味が全期間で一貫する。
"""

import logging

import numpy as np
import pandas as pd

import config

# EM終盤の微小な対数尤度の減少（数値誤差）で hmmlearn が出す
# "Model is not converging" を抑制する（結果には実害がない）
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def _hmm_input(open_series):
    """HMMの入力特徴を作る（因果的）。

    ret    : Open の対数差分（行tは寄付きで既知）
    logvol : ret のローリングσ（REGIME_VOL_WINDOW 日）の対数。
             リターン単体より状態分離が安定する（高ボラ/低ボラの持続性を拾う）。
    戻り値: DataFrame（先頭 REGIME_VOL_WINDOW 行は NaN）
    """
    open_ = pd.to_numeric(open_series, errors="coerce").astype(np.float64)
    ret = np.log(open_).diff()
    vol = ret.rolling(config.REGIME_VOL_WINDOW).std()
    return pd.DataFrame({"ret": ret, "logvol": np.log(vol.clip(lower=1e-8))})


def _sorted_params(hmm):
    """HMMパラメータを logvol 平均の昇順（低ボラ→高ボラ）に並べ替えて返す。"""
    order = np.argsort(hmm.means_[:, 1])
    means = hmm.means_[order]
    # covariance_type="diag" 前提。(K, D) に揃える
    covars = np.asarray(hmm.covars_)[order]
    if covars.ndim == 3:  # hmmlearn は diag でも (K, D, D) を返すことがある
        covars = np.array([np.diag(c) for c in covars])
    transmat = hmm.transmat_[np.ix_(order, order)]
    startprob = hmm.startprob_[order]
    return means, covars, transmat, startprob


def _log_gauss_diag(X, means, covars):
    """対角ガウスの対数尤度 (T, K)。X: (T, D), means/covars: (K, D)。"""
    covars = np.maximum(covars, 1e-10)
    diff = X[:, None, :] - means[None, :, :]  # (T, K, D)
    return -0.5 * (
        np.sum(np.log(2.0 * np.pi * covars), axis=1)[None, :]
        + np.sum(diff * diff / covars[None, :, :], axis=2)
    )


def _forward_filter(X, means, covars, transmat, startprob):
    """前向きアルゴリズムで filtered probability P(s_t | x_{1:t}) を返す (T, K)。

    smoothed（前向き後ろ向き）と違い未来を参照しないので、行tの値を
    そのまま当日の観測に使える。逐次正規化でアンダーフローを防ぐ。
    """
    log_lik = _log_gauss_diag(X, means, covars)
    lik = np.exp(log_lik - log_lik.max(axis=1, keepdims=True))  # (T, K) スケール不変
    T, K = lik.shape
    probs = np.empty((T, K))
    p = startprob * lik[0]
    probs[0] = p / max(p.sum(), 1e-300)
    for t in range(1, T):
        p = (probs[t - 1] @ transmat) * lik[t]
        probs[t] = p / max(p.sum(), 1e-300)
    return probs


def compute_regime_probs(open_series, verbose=True):
    """ウォークフォワードHMMで各日のレジーム事後確率を計算する。

    時点 t で使うパラメータは「t より前のデータで学習したHMM」
    （REGIME_REFIT_EVERY 日ごとに拡大窓で再学習）。事後確率は再学習のたびに
    系列先頭から forward filter を回し直し、当該セグメント分だけ採用する
    （filterは各行で過去のみ参照するので因果的）。

    学習データが REGIME_MIN_TRAIN 日に満たない先頭区間は一様分布 1/K で埋める
    （warmup 区間内に収まる想定。トレードには使われない）。
    戻り値: (len(open_series), N_REGIMES) の ndarray
    """
    from hmmlearn.hmm import GaussianHMM

    K = config.N_REGIMES
    feat = _hmm_input(open_series)
    valid = feat.dropna()
    X = valid.to_numpy()
    T = len(X)

    probs_valid = np.full((T, K), 1.0 / K)
    t = config.REGIME_MIN_TRAIN
    n_fit = 0
    while t < T:
        seg_end = min(t + config.REGIME_REFIT_EVERY, T)
        hmm = GaussianHMM(
            n_components=K,
            covariance_type="diag",
            n_iter=config.REGIME_HMM_ITER,
            random_state=config.REGIME_SEED,
        )
        hmm.fit(X[:t])  # t より前のみで学習（未来を見ない）
        params = _sorted_params(hmm)
        # 系列先頭から filter を回し直してセグメント分を採用
        # （パラメータは過去データ由来・filterも過去のみ参照 → 因果的）
        filtered = _forward_filter(X[:seg_end], *params)
        probs_valid[t:seg_end] = filtered[t:seg_end]
        n_fit += 1
        t = seg_end

    # 元のインデックス（NaN行含む）に戻す。NaN行（先頭のwarmup）は一様分布
    probs = np.full((len(feat), K), 1.0 / K)
    pos = feat.index.get_indexer(valid.index)
    probs[pos] = probs_valid
    if verbose and T > config.REGIME_MIN_TRAIN:
        last = probs_valid[-1]
        print(
            f"レジーム検知: HMM {n_fit}回再学習（拡大窓, {config.REGIME_REFIT_EVERY}日毎） "
            f"| 直近 P(low→high vol) = {np.round(last, 2)}"
        )
    return probs


def add_regime_features(df, verbose=True):
    """df に Regime_0..K-1（filtered事後確率）と RegimeMAP（argmax）を追加して返す。

    generate_env_data の最後に呼ばれる。確率は [0,1] のまま観測に入れる
    （z-score はかけない: 確率は既に正規化済みで、意味も解釈もそのまま保つ）。
    """
    probs = compute_regime_probs(df["Open"], verbose=verbose)
    for k in range(config.N_REGIMES):
        df[f"Regime_{k}"] = probs[:, k]
    df["RegimeMAP"] = probs.argmax(axis=1)
    return df
