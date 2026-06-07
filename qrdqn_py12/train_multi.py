"""複数シードを並列学習して seed アンサンブル用のモデル群を作る（F-6）。

GPUメモリが余っている（1run <1GB）ので、1つのGPUで複数runを同時に回す。
ただし律速はCPU側のenvステップなので、--jobs は Colab の vCPU 数程度（2〜4）が目安。

使い方:
  python train_multi.py --seeds 0 1 2 3 --jobs 2 --timesteps 200000
  # 各シードのモデルは models/seed{S}/ に保存される
出力命名: models/seed{S}/{model_name}_seed{S}_<steps>_steps.zip

仕組み: master が学習データを一度だけDLしてキャッシュ → 各シードを
        サブプロセス(worker)として最大 --jobs 並列で起動（CUDA+fork問題を回避）。
"""
import argparse
import os
import subprocess
import sys
import time

import config

CACHE_PATH = "_train_cache.pkl"


def ensure_data_cache():
    """学習データを一度だけDLして pickle キャッシュする（worker間で共有）。"""
    if os.path.exists(CACHE_PATH):
        print(f"[master] データキャッシュあり: {CACHE_PATH}")
        return
    from data import generate_env_data

    print("[master] 学習データDL中（初回のみ）...")
    df = generate_env_data(config.TRAIN_START, config.TRAIN_END, ticker=config.TICKER)
    df.to_pickle(CACHE_PATH)
    print(f"[master] キャッシュ保存: {CACHE_PATH} ({len(df)} rows)")


def run_worker(seed, timesteps, save_freq, buffer_size, optimize_memory):
    """1シード分の学習（サブプロセス内で実行される）。検証スコアで早期停止する。"""
    import pandas as pd
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed

    from main import ResNetFeatures, make_env, prepare_train_val_data, make_eval_callback, save_best_or_last
    from algo import build_model

    set_random_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = pd.read_pickle(CACHE_PATH)
    train_df, val_df = prepare_train_val_data(full)

    env = DummyVecEnv([lambda: make_env(train_df)])
    model = build_model(
        env, device, features_extractor_class=ResNetFeatures, tensorboard_log=None,
        seed=seed, buffer_size=buffer_size, optimize_memory_usage=optimize_memory,
    )

    out_dir = os.path.join("models", f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{config.model_name()}_seed{seed}"
    ckpt = CheckpointCallback(save_freq=save_freq, save_path=out_dir, name_prefix=prefix)
    best_dir = os.path.join(out_dir, "best")
    eval_cb = make_eval_callback(val_df, best_dir)  # 検証スコアで早期停止

    print(f"[worker seed={seed}] 学習開始 device={device} timesteps={timesteps}")
    model.learn(total_timesteps=timesteps, callback=[ckpt, eval_cb], progress_bar=False)
    # 検証ベストを seed フォルダ直下に正式名で保存
    save_best_or_last(model, best_dir, os.path.join(out_dir, prefix))
    print(f"[worker seed={seed}] 完了 → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--jobs", type=int, default=2, help="同時実行プロセス数（vCPU数程度）")
    ap.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    ap.add_argument("--save-freq", type=int, default=10000)
    # 並列時のRAM節約: バッファ縮小 / 省メモリモード（next_obsを別持ちしない）
    ap.add_argument("--buffer-size", type=int, default=50000,
                    help="1run あたりのリプレイバッファ。並列数に応じて小さく（既定5万≈1.5GB）")
    ap.add_argument("--optimize-memory", action="store_true",
                    help="optimize_memory_usage=True（バッファのRAMをほぼ半減）")
    # 内部用（worker起動フラグ）
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        run_worker(args.seed, args.timesteps, args.save_freq, args.buffer_size, args.optimize_memory)
        return

    # ── master ──
    ensure_data_cache()
    print(f"[master] seeds={args.seeds} jobs={args.jobs} timesteps={args.timesteps}")

    pending = list(args.seeds)
    running = {}  # proc -> seed
    while pending or running:
        # 空きがあれば起動
        while pending and len(running) < args.jobs:
            seed = pending.pop(0)
            cmd = [
                sys.executable, __file__, "--worker",
                "--seed", str(seed),
                "--timesteps", str(args.timesteps),
                "--save-freq", str(args.save_freq),
                "--buffer-size", str(args.buffer_size),
            ]
            if args.optimize_memory:
                cmd.append("--optimize-memory")
            log = open(f"train_seed{seed}.log", "w")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            running[proc] = (seed, log)
            print(f"[master] 起動 seed={seed} (pid={proc.pid}) ログ: train_seed{seed}.log")

        # 終了チェック
        done = [p for p in running if p.poll() is not None]
        for p in done:
            seed, log = running.pop(p)
            log.close()
            status = "OK" if p.returncode == 0 else f"FAIL(rc={p.returncode})"
            print(f"[master] 終了 seed={seed}: {status}")
        time.sleep(2)

    print("[master] 全シード完了。models/seed*/ を確認。")


if __name__ == "__main__":
    main()
