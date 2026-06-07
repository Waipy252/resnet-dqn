"""学習アルゴリズムの切替（G-1）。

config.ALGO に応じて QR-DQN / DQN を返す。env・特徴抽出器・報酬は共通のまま、
アルゴリズムだけ差し替えて安定性を比較できるようにする。
"""

import config


def get_algo_class():
    """config.ALGO に対応するアルゴリズムクラスを返す。"""
    if config.ALGO == "qrdqn":
        from sb3_contrib import QRDQN

        return QRDQN
    from stable_baselines3 import DQN

    return DQN


def build_model(env, device, features_extractor_class, tensorboard_log=config.TENSORBOARD_LOG):
    """config の設定に従って学習器を構築する。

    QR-DQN と DQN で共通のハイパーパラメータを使い、
    QR-DQN 固有の n_quantiles のみ条件付きで付与する。
    """
    AlgoClass = get_algo_class()

    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=dict(
            features_dim=config.FEATURES_DIM,
            num_blocks=config.NUM_BLOCKS,
        ),
    )
    if config.ALGO == "qrdqn":
        # n_quantiles は QRDQNPolicy のパラメータなので policy_kwargs に入れる
        policy_kwargs["n_quantiles"] = config.N_QUANTILES

    return AlgoClass(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=config.LEARNING_RATE,
        buffer_size=config.BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
        train_freq=config.TRAIN_FREQ,
        gradient_steps=config.GRADIENT_STEPS,
        exploration_fraction=config.EXPLORATION_FRACTION,
        exploration_final_eps=config.EXPLORATION_FINAL_EPS,
        seed=config.SEED,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device=device,
    )
