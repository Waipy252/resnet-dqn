import matplotlib.pyplot as plt
import pandas as pd
import warnings
from collections import Counter
from stable_baselines3 import DQN
from main import NikkeiEnv
from data import generate_env_data
from calc_performance import compute_sharpe_ratio, calculate_performance_metrics
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

warnings.filterwarnings(
    "ignore", category=UserWarning, module="stable_baselines3.common.buffers"
)

# PyTorchをCPU専用に設定（重要）
torch.set_default_tensor_type("torch.FloatTensor")

# 例: 手動でデータを追加する
manual_data = {
    "Date": ["2025-09-11", "2025-09-12"],
    "Open": [43930, 43930],
    "High": [43930, 43930],
    "Low": [43930, 43930],
    "Close": [43930, 43930],
    "Volume": [100000000, 100000000],
    "VIX": [14.77, 14.66],
    "Japan_10Y_Rate": [1.54, 1.54],
    "US_10Y_Rate": [4.33, 4.33],
}

# DataFrame に変換
manual_data = pd.DataFrame(manual_data)
manual_data.set_index("Date", inplace=True)


def load_model_safely(model_path, env):
    try:
        # まず env なしで CPU に強制ロード、BatchNorm 統計などは None に置換
        model = DQN.load(
            model_path,
            env=None,
            device="cpu",
            custom_objects={
                "lr_schedule": None,
                "exploration_schedule": None,
                "batch_norm_stats": None,  # 追加
                "batch_norm_stats_target": None,  # 追加
                "replay_buffer": None,  # もし含まれていても無視
            },
            # SB3 の新しめの版なら有効: print_system_info=True,
        )
        # 念のため CPU に固定し eval モードへ
        model.policy.to("cpu")
        model.policy.set_training_mode(False)
        # 後から環境を設定
        model.set_env(env)
        return model
    except Exception as e:
        print(f"モデルロードエラー: {e}")
        # ここに来るのは zip の破損やバージョン不整合の可能性大
        return DQN("MlpPolicy", env, verbose=0, device="cpu")


# ──────────────────────────────
# 5. バックテスト（テストデータ上で方策を実行）
start = "2023-01-01"
end = "2025-09-12"
test_data = generate_env_data(
    start, end, ticker="^N225", manual_data=manual_data
)  # 日経平均
window_size = 130

# バックテスト用（評価用）環境：通常の環境オブジェクトを利用
test_env = NikkeiEnv(
    test_data,
    window_size=window_size,
    transaction_cost=0.001,
    risk_limit=0.5,
    trade_penalty=0.00,
)

# 最後のアクションを保存するリスト
final_actions = []

try:
    for i in range(200000, 440001, 10000):
        print(f"## Step {i}")

        # 環境をリセット（重要）
        obs = test_env.reset()
        done = False
        action_history = []

        # モデルを安全にロード
        model_path = f"nikkei_cp_1997-01-01_2024-01-01_{i}_steps.zip"
        model = load_model_safely(model_path, test_env)

        step_count = 0
        max_steps = len(test_data) - window_size  # 無限ループ防止

        while not done and step_count < max_steps:
            try:
                # 決定論的に行動を選択
                action, _ = model.predict(obs, deterministic=True)
                action_history.append(int(action))
                obs, reward, done, info = test_env.step(action)
                step_count += 1

                if done:  # エピソード終了時のアクションを保存
                    final_actions.append(int(action))

            except Exception as e:
                print(f"ステップ実行エラー: {e}")
                break

        # テスト期間中のエクイティカーブを取得
        try:
            equity_curve = test_env.get_equity_curve()
            sharpe = compute_sharpe_ratio(equity_curve, yearly_risk_free_rate=0.01)

            # パフォーマンス指標の計算と表示
            metrics = calculate_performance_metrics(equity_curve, action_history)
            print("=== パフォーマンス指標 ===")
            print(f"年利: {metrics['annual_return']:.2f}%")
            print("年間シャープレシオ:", sharpe)
            print(f"最大ドローダウン: {metrics['max_drawdown']:.2f}%")
            print(f"最大ドローダウン期間: {metrics['max_drawdown_period']}")
            print(f"勝率: {metrics['win_rate']:.2f}%")
            print(f"平均勝ち%: {metrics['avg_win']:.4f}%")
            print(f"平均負け%: {metrics['avg_loss']:.4f}%")
            print(f"W/Lレシオ: {metrics['wl_ratio']:.2f}")
            print(f"期待値: {metrics['expectancy']:.4f}%")
            print(f"プロフィットファクター: {metrics['profit_factor']:.2f}")
            print(f"取引日数: {metrics['total_days']}")
            print(f"平均勝ち期間: {metrics['avg_win_holding_period']}")
            print(f"平均負け期間: {metrics['avg_loss_holding_period']}")

            # プロット処理
            buy_steps = [j for j, a in enumerate(action_history) if a == 0]
            buy_values = [equity_curve[j] for j in buy_steps if j < len(equity_curve)]
            steps = range(len(equity_curve))

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(steps, equity_curve, label="Equity Curve", color="blue")
            if buy_values:  # 買いシグナルがある場合のみプロット
                ax.scatter(
                    buy_steps[: len(buy_values)],
                    buy_values,
                    color="green",
                    marker="^",
                    s=100,
                    label="BUY Signal",
                )
            ax.set_xlabel("Step (Day)")
            ax.set_ylabel("Asset Balance")
            ax.set_title(f"Equity Curve with Buy Signals - Step {i}")
            ax.tick_params(axis="y", labelcolor="blue")

            # 株価データを追加（赤）
            stock_prices = test_data["Open"].values
            offset = len(stock_prices) - len(equity_curve)
            if offset >= 0:
                stock_prices = stock_prices[offset:]
                ax2 = ax.twinx()
                ax2.plot(
                    steps,
                    stock_prices[: len(equity_curve)],
                    label="Stock Price (N225)",
                    color="red",
                    linestyle="dashed",
                    alpha=0.7,
                )
                ax2.set_ylabel("Stock Price", color="red")
                ax2.tick_params(axis="y", labelcolor="red")
                ax2.legend(loc="upper right")

            fig.suptitle(f"Equity Curve & Stock Price - Step {i}")
            ax.legend(loc="upper left")
            plt.tight_layout()
            # plt.show()
            plt.close()  # メモリリーク防止

        except Exception as e:
            print(f"パフォーマンス計算エラー: {e}")

        # メモリをクリーンアップ
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

except Exception as e:
    print(f"メインループエラー: {e}")

finally:
    # 最後のアクションの出現回数をカウント
    if final_actions:
        final_action_counts = Counter(final_actions)
        count_0 = final_action_counts.get(0, 0)
        count_1 = final_action_counts.get(1, 0)
        count_2 = final_action_counts.get(2, 0)
        print(f"{end} 買い:{count_0}, 待ち:{count_1}, 売り:{count_2}")
    else:
        print("最終アクションが記録されませんでした")
