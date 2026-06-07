import torch
import warnings
from stable_baselines3 import DQN
import os

# CUDA完全無効化
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 警告を無視
warnings.filterwarnings("ignore")


def test_model_loading():
    """モデルロードのテスト"""
    try:
        print("テスト1: 環境なしでのモデルロード")
        model = DQN.load(
            "nikkei_cp_1997-01-01_2024-01-01_220000_steps.zip", device="cpu"
        )
        print("✓ 環境なしロード成功")
        del model
        return True
    except Exception as e:
        print(f"✗ 環境なしロードエラー: {e}")
        return False


def test_env_creation():
    """環境作成のテスト"""
    try:
        print("テスト2: 環境作成")
        from main import NikkeiEnv
        from data import generate_env_data

        # 小さなデータセットで試す
        test_data = generate_env_data("2024-01-01", "2024-06-01", ticker="^N225")
        test_env = NikkeiEnv(
            test_data,
            window_size=30,  # 小さなウィンドウサイズ
            transaction_cost=0.001,
            risk_limit=0.5,
        )
        obs, _ = test_env.reset()
        print(f"✓ 環境作成成功: obs shape = {obs.shape}")
        return test_env
    except Exception as e:
        print(f"✗ 環境作成エラー: {e}")
        return None


if __name__ == "__main__":
    print("=== デバッグテスト開始 ===")

    # テスト1: モデルロード
    model_ok = test_model_loading()

    # テスト2: 環境作成
    env = test_env_creation()

    if model_ok and env is not None:
        print("テスト3: 環境付きモデルロード")
        try:
            model = DQN.load(
                "nikkei_cp_1997-01-01_2024-01-01_220000_steps.zip",
                env=env,
                device="cpu",
            )
            print("✓ 環境付きロード成功")

            # テスト4: 予測
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            print(f"✓ 予測成功: action = {action}")

        except Exception as e:
            print(f"✗ 環境付きロードエラー: {e}")
