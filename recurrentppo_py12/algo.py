"""学習アルゴリズムの切替（G-1）。

config.ALGO に応じて RecurrentPPO / PPO を返す。env・報酬は共通のまま、
アルゴリズムだけ差し替えて安定性を比較できるようにする。

qrdqn_py12 との違い:
- 観測は「1日1ベクトル」になったので特徴抽出器（ResNet等）は不要。
  RecurrentPPO は MlpLstmPolicy（内蔵LSTMが時系列を記憶）、PPO は MlpPolicy。
- G-2: optimizer_kwargs で weight decay を適用（過学習対策）。
"""

import torch

import config


def get_algo_class():
    """config.ALGO に対応するアルゴリズムクラスを返す。"""
    if config.ALGO == "recurrentppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO
    from stable_baselines3 import PPO

    return PPO


def build_model(env, device, tensorboard_log=config.TENSORBOARD_LOG, seed=None):
    """config の設定に従って学習器を構築する。

    seed を渡すと config.SEED の代わりにそれを使う（複数シード学習用, F-6）。
    """
    AlgoClass = get_algo_class()
    if seed is None:
        seed = config.SEED

    policy_kwargs = dict(
        net_arch=dict(pi=list(config.NET_ARCH), vf=list(config.NET_ARCH)),
        # G-2: weight decay（AdamのL2正則化）。dropoutに相当する正則化を簡便に効かせる。
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(weight_decay=config.WEIGHT_DECAY),
    )
    if config.ALGO == "recurrentppo":
        policy = "MlpLstmPolicy"
        policy_kwargs.update(
            lstm_hidden_size=config.LSTM_HIDDEN,
            n_lstm_layers=config.LSTM_LAYERS,
        )
    else:
        policy = "MlpPolicy"

    return AlgoClass(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device=device,
    )
