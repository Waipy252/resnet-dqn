import pandas as pd
from calc_performance import calculate_performance_metrics, compute_sharpe_ratio
import config
from algo import get_algo_class


def run_simulation_try18res(price, end_day, vix, jpy, usr):
    try:
        from main import NikkeiEnv
        from data import generate_env_data

        # 手動でデータを生成
        manual_data = {
            "Date": [end_day],
            "Open": [float(price)],
            "High": [float(price)],
            "Low": [float(price)],
            "Close": [float(price)],
            "Volume": [100000000],
            "VIX": [float(vix)],
            "Japan_10Y_Rate": [float(jpy)],
            "US_10Y_Rate": [float(usr)],
        }

        # DataFrame に変換
        manual_data = pd.DataFrame(manual_data)
        manual_data.set_index("Date", inplace=True)

        # バックテスト設定
        start = "2025-01-01"
        test_data = generate_env_data(
            start, end_day, ticker=config.TICKER, manual_data=manual_data
        )
        window_size = config.WINDOW_SIZE

        # バックテスト用環境作成（A-4: コスト・リスク制限を config に統一）
        test_env = NikkeiEnv(
            test_data,
            window_size=window_size,
            transaction_cost=config.TRANSACTION_COST,
            risk_limit=config.RISK_LIMIT,
        )

        # 結果を保存する変数
        final_actions = []
        results = []

        # 各モデルで評価
        for i in range(200000, 200001, 10000):
            obs, _ = test_env.reset()
            done = False
            action_history = []

            num_steps = i
            try:
                model = get_algo_class().load(
                    f"./{config.model_name(num_steps)}.zip",
                    env=test_env,
                )
            except FileNotFoundError:
                print(
                    f"モデルファイル {config.model_name(num_steps)}.zip が見つかりません。"
                )
                continue

            model_result = f"## Step {num_steps}\n"

            while not done:
                # 決定論的に行動を選択
                action, _ = model.predict(obs, deterministic=True)
                action_history.append(action)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                if done:  # エピソード終了時のアクションを保存
                    final_actions.append(int(action))

            # エクイティカーブとメトリクス計算
            equity_curve = test_env.get_equity_curve()
            sharpe = compute_sharpe_ratio(equity_curve, yearly_risk_free_rate=0.01)
            metrics = calculate_performance_metrics(equity_curve, action_history)

            # 結果を文字列に追加
            model_result += "=== パフォーマンス指標: try15res ===\n"
            model_result += f"アクション: {action}\n"
            model_result += f"年利: {metrics['annual_return']:.2f}%\n"
            model_result += f"年間シャープレシオ: {sharpe:.2f}\n"
            model_result += f"最大ドローダウン: {metrics['max_drawdown']:.2f}%\n"
            model_result += f"最大ドローダウン期間: {metrics['max_drawdown_period']}\n"
            model_result += f"勝率: {metrics['win_rate']:.2f}%\n"
            model_result += f"平均勝ち%: {metrics['avg_win']:.4f}%\n"
            model_result += f"平均負け%: {metrics['avg_loss']:.4f}%\n"
            model_result += f"W/Lレシオ: {metrics['wl_ratio']:.2f}\n"
            model_result += f"期待値: {metrics['expectancy']:.4f}%\n"
            model_result += f"プロフィットファクター: {metrics['profit_factor']:.2f}\n"
            model_result += f"取引日数: {metrics['total_days']}\n"
            model_result += f"平均勝ち期間: {metrics['avg_win_holding_period']}\n"
            model_result += f"平均負け期間: {metrics['avg_loss_holding_period']}\n\n"

            results.append(model_result)
        return results, final_actions, test_data
    except Exception as e:
        print(f"run_simulation_try15res でエラーが発生: {e}")
        return [], [], test_data
