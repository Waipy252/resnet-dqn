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


def build_model(env, device, features_extractor_class, features_extractor_kwargs=None,
                tensorboard_log=config.TENSORBOARD_LOG,
                seed=None, buffer_size=None, optimize_memory_usage=False):
    """config の設定に従って学習器を構築する。

    QR-DQN と DQN で共通のハイパーパラメータを使い、
    QR-DQN 固有の n_quantiles のみ条件付きで付与する。
    seed を渡すと config.SEED の代わりにそれを使う（複数シード学習用, F-6）。
    buffer_size/optimize_memory_usage は並列学習時のRAM節約用に上書きできる。
    features_extractor_kwargs を省略すると ResNet 既定（features_dim/num_blocks）。
    アーキを切り替える場合は main.make_features_extractor() で (クラス, kwargs) を得て渡す（G-2）。
    """
    AlgoClass = get_algo_class()
    if seed is None:
        seed = config.SEED
    if buffer_size is None:
        buffer_size = config.BUFFER_SIZE
    if features_extractor_kwargs is None:
        features_extractor_kwargs = dict(
            features_dim=config.FEATURES_DIM,
            num_blocks=config.NUM_BLOCKS,
        )

    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=features_extractor_kwargs,
    )
    if config.ALGO == "qrdqn":
        # n_quantiles は QRDQNPolicy のパラメータなので policy_kwargs に入れる
        policy_kwargs["n_quantiles"] = config.N_QUANTILES

    # optimize_memory_usage=True は handle_timeout_termination=True と併用不可（SB3制約）
    replay_buffer_kwargs = {"handle_timeout_termination": False} if optimize_memory_usage else None

    return AlgoClass(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=config.LEARNING_RATE,
        buffer_size=buffer_size,
        optimize_memory_usage=optimize_memory_usage,
        replay_buffer_kwargs=replay_buffer_kwargs,
        batch_size=config.BATCH_SIZE,
        train_freq=config.TRAIN_FREQ,
        gradient_steps=config.GRADIENT_STEPS,
        exploration_fraction=config.EXPLORATION_FRACTION,
        exploration_final_eps=config.EXPLORATION_FINAL_EPS,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device=device,
    )
