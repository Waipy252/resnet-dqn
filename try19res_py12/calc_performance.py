import numpy as np


def calculate_performance_metrics(equity_curve, action_history):
    """
    トレーディングモデルのパフォーマンス指標を計算

    Parameters:
    - equity_curve (list): 資産残高の推移
    - action_history (list): 各ステップでのアクション（0:ロング, 1:フラット, 2:ショート）

    Returns:
    dict: 各種パフォーマンス指標
    """
    daily_returns = []
    for i in range(1, len(equity_curve)):
        r = (equity_curve[i] / equity_curve[i - 1]) - 1
        daily_returns.append(r)

    # ───────────────────────────────
    # トレード単位に分割するためのロジック
    trades = []
    position = None
    entry_step = None
    entry_value = None

    for i, action in enumerate(action_history):
        if i >= len(equity_curve):
            break
        if position is None:
            if action in [0, 2]:  # ロング or ショート開始
                position = action
                entry_step = i
                entry_value = equity_curve[i]
        else:
            # ポジション変更 or 最後まで保有
            if action != position or i == len(action_history) - 1:
                exit_step = i
                exit_value = equity_curve[min(i, len(equity_curve) - 1)]
                pnl = (
                    (exit_value / entry_value) - 1
                    if position == 0
                    else (entry_value / exit_value) - 1
                )
                trades.append({"pnl": pnl, "holding_period": exit_step - entry_step})
                # 次のポジションを開始
                if action in [0, 2] and i < len(equity_curve):
                    position = action
                    entry_step = i
                    entry_value = equity_curve[i]
                else:
                    position = None
                    entry_step = None
                    entry_value = None

    # 勝ちトレードと負けトレードを分ける
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    # ───────────────────────────────
    # 各種指標の計算
    total_days = len(daily_returns)
    cumulative_return = (equity_curve[-1] / equity_curve[0]) - 1
    annual_return = (
        (1 + cumulative_return) ** (250 / total_days) - 1 if total_days > 0 else 0
    )

    # 最大ドローダウン
    peak = equity_curve[0]
    max_drawdown = 0
    max_dd_start = 0
    max_dd_end = 0
    current_dd_start = 0
    for i, value in enumerate(equity_curve):
        if value > peak:
            peak = value
            current_dd_start = i
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_dd_start = current_dd_start
            max_dd_end = i

    # 勝率、平均勝ち・負け%
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    profit_factor = (
        sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))
        if losses
        else float("inf")
    )

    # 追加: 勝ち/負けトレードの平均保持期間
    avg_win_holding = np.mean([t["holding_period"] for t in wins]) if wins else 0
    avg_loss_holding = np.mean([t["holding_period"] for t in losses]) if losses else 0

    # 最大勝ち期間・最大負け期間
    max_win_holding = max([t["holding_period"] for t in wins]) if wins else 0
    max_loss_holding = max([t["holding_period"] for t in losses]) if losses else 0

    return {
        "annual_return": annual_return * 100,
        "max_drawdown": max_drawdown * 100,
        "max_drawdown_period": f"{max_dd_start}日目から{max_dd_end}日目",
        "win_rate": win_rate * 100,
        "avg_win": avg_win * 100,
        "avg_loss": avg_loss * 100,
        "wl_ratio": wl_ratio,
        "expectancy": expectancy * 100,
        "profit_factor": profit_factor,
        "total_days": total_days,
        "avg_win_holding_period": avg_win_holding,
        "avg_loss_holding_period": avg_loss_holding,
        "max_win_holding_period": max_win_holding,
        "max_loss_holding_period": max_loss_holding,
        "total_trades": len(trades),
    }


def compute_sharpe_ratio(equity_curve, yearly_risk_free_rate=0.0, periods_per_year=252):
    """
    equity_curve            : 資産残高のリストまたはnumpy配列（各時点の口座残高）
    yearly_risk_free_rate   : 年率の無リスク利率（例: 0.01 → 1%）
    periods_per_year        : 年間取引日数（例: 252）
    """
    # equity_curve をnumpy配列に変換
    equity_curve = np.array(equity_curve)

    # 負の値や0の値をチェック
    if np.any(equity_curve <= 0):
        return 0.0

    # 各日の対数リターンの計算
    daily_log_returns = np.diff(np.log(equity_curve))

    # 年率の無リスク利率を日次に変換
    risk_free_daily = (1 + yearly_risk_free_rate) ** (1 / periods_per_year) - 1

    # 超過リターン
    excess_returns = daily_log_returns - risk_free_daily

    # 日次シャープレシオ ※分母が0にならないよう注意
    sharpe_daily = (
        np.mean(excess_returns) / np.std(excess_returns)
        if np.std(excess_returns) != 0
        else 0
    )

    # 年率シャープレシオに換算
    sharpe_annualized = sharpe_daily * np.sqrt(periods_per_year)

    return sharpe_annualized
