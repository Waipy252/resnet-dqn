import glob
import os

import pandas as pd

from calc_performance import calculate_performance_metrics, compute_sharpe_ratio
import config
from algo import get_algo_class


def run_simulation_rppo(open_, high, low, close, volume, end_day, vix, jpy, usr):
    """当日データ1行を付け足してバックテストし、各モデルの「明日のアクション」を得る。

    F-4: 実OHLC を受け取る（High=Low=Open の合成行をやめ、ATR/TR の歪み（D-2）を解消）。
    値は server.py の「最新データ自動取得」（data.fetch_latest）で自動入力できる。
    """
    test_data = None
    try:
        from main import make_env, rollout
        from data import generate_env_data

        # 当日の実OHLC行（D-2: High/Low/Close も実値）
        manual_data = pd.DataFrame(
            {
                "Date": [end_day],
                "Open": [float(open_)],
                "High": [float(high)],
                "Low": [float(low)],
                "Close": [float(close)],
                "Volume": [float(volume)],
                "VIX": [float(vix)],
                "Japan_10Y_Rate": [float(jpy)],
                "US_10Y_Rate": [float(usr)],
            }
        ).set_index("Date")

        # バックテスト設定（z-score warmup の分も含めて余裕を持って取得）
        start = "2025-01-01"
        test_data = generate_env_data(
            start, end_day, ticker=config.TICKER, manual_data=manual_data
        )

        # 結果を保存する変数
        final_actions = []
        results = []

        def _steps_from_path(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            parts = stem.rsplit("_", 2)
            if len(parts) == 3 and parts[2] == "steps" and parts[1].isdigit():
                return int(parts[1])
            return stem

        # カレントにある学習済みzipを全部拾う（_eval_one.py と同じ流儀）
        model_paths = sorted(
            glob.glob(f"{config.model_name()}*.zip"),
            key=lambda p: str(_steps_from_path(p)),
        )
        if not model_paths:
            print(f"モデルzipが見つかりません（{config.model_name()}*.zip）")

        # 各モデルで評価
        for model_path in model_paths:
            num_steps = _steps_from_path(model_path)
            try:
                model = get_algo_class().load(model_path, device="cpu")
            except FileNotFoundError:
                print(f"モデルファイル {model_path} が見つかりません。")
                continue

            test_env = make_env(test_data)
            # RecurrentPPO: rollout が LSTM warmup と状態管理をやってくれる
            action_history, equity_curve = rollout(
                model, test_env, deterministic=True, lstm_warmup=True
            )
            final_actions.append(int(action_history[-1]))

            sharpe = compute_sharpe_ratio(equity_curve, yearly_risk_free_rate=0.01)
            metrics = calculate_performance_metrics(equity_curve, action_history)

            model_result = f"## Step {num_steps}\n"
            model_result += "=== パフォーマンス指標: recurrentppo ===\n"
            model_result += f"アクション: {action_history[-1]}\n"
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
        print(f"run_simulation_rppo でエラーが発生: {e}")
        return [], [], test_data
