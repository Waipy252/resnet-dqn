import os
import glob
import re
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from main import NikkeiEnv, rollout
from data import generate_env_data
from calc_performance import compute_sharpe_ratio, calculate_performance_metrics
from collections import Counter
import config
from algo import get_algo_class


def clean_data_for_plot(data):
    """プロット用データからNaN、無限値、複素数を除去"""
    if isinstance(data, (list, tuple)):
        cleaned = []
        for item in data:
            if isinstance(item, (int, float, np.number)):
                if np.isfinite(item) and np.isreal(item):
                    cleaned.append(float(np.real(item)))
                else:
                    cleaned.append(0.0)
            else:
                cleaned.append(item)
        return cleaned
    elif isinstance(data, np.ndarray):
        # 複素数を実数部に変換し、NaN/Infを0で置換
        real_data = np.real(data)
        return np.where(np.isfinite(real_data), real_data, 0.0).tolist()
    return data


def _steps_from_path(path):
    """ファイル名から識別ラベルを取り出す。

    命名規約 nikkei_cp_..._<steps>_steps.zip なら steps(int) を、
    それ以外（リネーム済みベストモデル等）は拡張子なしのファイル名を返す。
    seed があれば "sN:steps" 形式にして複数シードを区別する。
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"seed(\d+)", stem)
    tag = f"s{m.group(1)}:" if m else ""
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[2] == "steps" and parts[1].isdigit():
        return f"{tag}{int(parts[1])}" if tag else int(parts[1])
    # リネーム済みモデルは共通プレフィックスを剥がして短く（_eval_one.py と同じ流儀）
    m = re.match(r"nikkei_\w+_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}_?(.+)", stem)
    return f"{tag}{m.group(1)}" if m else stem


def discover_models(pattern=None):
    """カレントにある学習済みzipを全部拾う（_eval_one.py と同じ流儀）。

    `*_steps.zip` の命名に縛られず、TRAIN_END が違う旧モデルや
    リネーム済みのベストモデルも対象にする。
    """
    if pattern is None:
        # R-1: レジーム版（nikkei_regime_*）も旧名（nikkei_rppo_*）も拾う。
        # 観測次元が違う旧モデルはロード時に弾かれるので、置かないのが前提。
        pattern = "nikkei_*.zip"
    return sorted(glob.glob(pattern), key=lambda p: str(_steps_from_path(p)))


def load_model_safely(model_path):
    AlgoClass = get_algo_class()  # G-1: config.ALGO に応じて RecurrentPPO / PPO
    try:
        model = AlgoClass.load(model_path, device="cpu")
        model.policy.to("cpu")
        model.policy.set_training_mode(False)
        return model
    except Exception as e:
        print(f"モデルロードエラー: {e}")
        return None


def evaluate_all_models(ticker="^N225", start="2000-01-01", end="2010-01-01"):
    """全モデルの性能を評価して結果を返す"""
    test_data = generate_env_data(start, end, ticker=ticker)
    window_size = 130

    test_env = NikkeiEnv(
        test_data,
        window_size=window_size,
        transaction_cost=config.TRANSACTION_COST,
        risk_limit=config.RISK_LIMIT,
    )

    results = []

    model_paths = discover_models()
    if not model_paths:
        print("モデルzipが見つかりません（nikkei_*.zip）")
    for model_path in model_paths:
        i = _steps_from_path(model_path)
        try:
            print(f"Evaluating Step {i} ({model_path})...")

            model = load_model_safely(model_path)
            if model is None:
                continue

            # RecurrentPPO: rollout が LSTM warmup と状態管理をやってくれる
            action_history, equity_curve = rollout(
                model, test_env, deterministic=True, lstm_warmup=True
            )
            sharpe = compute_sharpe_ratio(equity_curve, yearly_risk_free_rate=0.01)

            metrics = calculate_performance_metrics(equity_curve, action_history)

            # 最終アクションの統計
            action_counter = Counter(action_history)

            result = {
                "steps": i,
                "annual_return": metrics["annual_return"],
                "sharpe_ratio": sharpe,
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
                "avg_win": metrics["avg_win"],
                "avg_loss": metrics["avg_loss"],
                "wl_ratio": metrics["wl_ratio"],
                "expectancy": metrics["expectancy"],
                "profit_factor": metrics["profit_factor"],
                "total_trades": metrics["total_trades"],
                "avg_win_holding_period": metrics["avg_win_holding_period"],
                "avg_loss_holding_period": metrics["avg_loss_holding_period"],
                "max_win_holding_period": metrics["max_win_holding_period"],
                "max_loss_holding_period": metrics["max_loss_holding_period"],
                "final_balance": equity_curve[-1],
                "long_actions": action_counter.get(0, 0),
                "flat_actions": action_counter.get(1, 0),
                "short_actions": action_counter.get(2, 0),
                "last_action": action_history[-1] if action_history else None,
                "equity_curve": equity_curve,
                "action_history": action_history,
            }

            results.append(result)
            del model

        except Exception as e:
            print(f"Error evaluating model {i}: {e}")
            continue

    return results


def create_performance_comparison(results):
    """モデル性能比較グラフを作成"""

    # gr.Plot に文字列を返すと postprocess で落ちるため、空のときは None
    if not results:
        return None

    df = pd.DataFrame(results)

    # 4つのメトリクスを表示するサブプロット
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "年利 (%)",
            "シャープレシオ",
            "最大ドローダウン (%)",
            "勝率 (%)",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    # 年利
    fig.add_trace(
        go.Scatter(
            x=df["steps"],
            y=df["annual_return"],
            mode="lines+markers",
            name="年利",
            line=dict(color="green", width=2),
            marker=dict(size=8),
        ),
        row=1,
        col=1,
    )

    # シャープレシオ
    fig.add_trace(
        go.Scatter(
            x=df["steps"],
            y=df["sharpe_ratio"],
            mode="lines+markers",
            name="シャープレシオ",
            line=dict(color="blue", width=2),
            marker=dict(size=8),
        ),
        row=1,
        col=2,
    )

    # 最大ドローダウン
    fig.add_trace(
        go.Scatter(
            x=df["steps"],
            y=df["max_drawdown"],
            mode="lines+markers",
            name="最大ドローダウン",
            line=dict(color="red", width=2),
            marker=dict(size=8),
        ),
        row=2,
        col=1,
    )

    # 勝率
    fig.add_trace(
        go.Scatter(
            x=df["steps"],
            y=df["win_rate"],
            mode="lines+markers",
            name="勝率",
            line=dict(color="purple", width=2),
            marker=dict(size=8),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title={
            "text": "トレーニングステップごとのモデルパフォーマンス比較",
            "x": 0.5,
            "font": {"size": 20},
        },
        height=700,
        showlegend=False,
        template="plotly_white",
        margin=dict(b=80),
    )

    # X軸ラベル（モデルが少数なのでカテゴリ軸・水平ラベルで見やすく）
    fig.update_xaxes(type="category", tickangle=0)
    fig.update_xaxes(title_text="モデル", row=2, col=1)
    fig.update_xaxes(title_text="モデル", row=2, col=2)

    return fig


def create_summary_stats(results):
    """サマリー統計テーブルを作成"""

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # 統計サマリーを計算
    metrics = [
        "annual_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "avg_win_holding_period",
        "avg_loss_holding_period",
        "max_win_holding_period",
        "max_loss_holding_period",
    ]
    summary_data = []

    metric_names = {
        "annual_return": "年利 (%)",
        "sharpe_ratio": "シャープレシオ",
        "max_drawdown": "最大ドローダウン (%)",
        "win_rate": "勝率 (%)",
        "profit_factor": "プロフィットファクター",
        "avg_win_holding_period": "平均勝ち期間 (日)",
        "avg_loss_holding_period": "平均負け期間 (日)",
        "max_win_holding_period": "最大勝ち期間 (日)",
        "max_loss_holding_period": "最大負け期間 (日)",
    }

    for metric in metrics:
        values = df[metric]
        summary_data.append(
            {
                "メトリクス名": metric_names.get(
                    metric, metric.replace("_", " ").title()
                ),
                "最小": f"{values.min():.2f}",
                "最大": f"{values.max():.2f}",
                "平均": f"{values.mean():.2f}",
                "標準偏差": f"{values.std():.2f}",
                "範囲": f"{values.max() - values.min():.2f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def create_action_distribution(results):
    """アクション分布の可視化"""

    # gr.Plot に文字列を返すと postprocess で落ちるため、空のときは None
    if not results:
        return None

    df = pd.DataFrame(results)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["steps"],
            y=df["long_actions"],
            name="買いアクション",
            marker_color="green",
        )
    )

    fig.add_trace(
        go.Bar(
            x=df["steps"],
            y=df["flat_actions"],
            name="待ちアクション",
            marker_color="gray",
        )
    )

    fig.add_trace(
        go.Bar(
            x=df["steps"],
            y=df["short_actions"],
            name="売りアクション",
            marker_color="red",
        )
    )

    fig.update_layout(
        title="Action Distribution Across Models",
        xaxis_title="トレーニングステップ",
        yaxis_title="アクション数",
        barmode="stack",
        template="plotly_white",
        height=500,
    )

    return fig


def create_ensemble_result(
    results, ticker="^N225", start="2023-01-01", end="2025-08-23"
):
    """多数決によるアンサンブルの結果を作成（エクイティカーブと性能指標）"""
    if not results:
        return [], {}

    # 既存の結果からアクション履歴を取得（効率化）
    all_actions = {}
    max_length = 0

    for result in results:
        steps = result["steps"]
        action_history = result["action_history"]
        all_actions[steps] = action_history
        max_length = max(max_length, len(action_history))

    # 各ステップでの多数決を計算
    ensemble_actions = []
    for i in range(max_length):
        votes = []
        for steps, actions in all_actions.items():
            if i < len(actions):
                votes.append(actions[i])

        if votes:
            # 多数決（最も多い票のアクション）
            vote_counts = Counter(votes)
            ensemble_action = vote_counts.most_common(1)[0][0]
            ensemble_actions.append(ensemble_action)

    print(f"Ensemble actions length: {len(ensemble_actions)}")

    # アンサンブルのエクイティカーブを計算
    test_data = generate_env_data(start, end, ticker=ticker)
    window_size = 130

    ensemble_env = NikkeiEnv(
        test_data,
        window_size=window_size,
        transaction_cost=config.TRANSACTION_COST,
        risk_limit=config.RISK_LIMIT,
    )

    obs, _ = ensemble_env.reset()
    done = False
    step_count = 0

    for action in ensemble_actions:
        if done or step_count >= len(ensemble_actions):
            break
        try:
            obs, reward, terminated, truncated, info = ensemble_env.step(action)
            done = terminated or truncated
            step_count += 1
        except Exception:
            break

    equity_curve = ensemble_env.get_equity_curve()

    # アンサンブルの性能指標を計算
    sharpe = compute_sharpe_ratio(equity_curve, yearly_risk_free_rate=0.01)
    metrics = calculate_performance_metrics(equity_curve, ensemble_actions)
    action_counter = Counter(ensemble_actions)

    ensemble_metrics = {
        "annual_return": metrics["annual_return"],
        "sharpe_ratio": sharpe,
        "max_drawdown": metrics["max_drawdown"],
        "win_rate": metrics["win_rate"],
        "avg_win": metrics["avg_win"],
        "avg_loss": metrics["avg_loss"],
        "wl_ratio": metrics["wl_ratio"],
        "expectancy": metrics["expectancy"],
        "profit_factor": metrics["profit_factor"],
        "total_trades": metrics["total_trades"],
        "avg_win_holding_period": metrics["avg_win_holding_period"],
        "avg_loss_holding_period": metrics["avg_loss_holding_period"],
        "max_win_holding_period": metrics["max_win_holding_period"],
        "max_loss_holding_period": metrics["max_loss_holding_period"],
        "final_balance": equity_curve[-1],
        "long_actions": action_counter.get(0, 0),
        "flat_actions": action_counter.get(1, 0),
        "short_actions": action_counter.get(2, 0),
        "last_action": ensemble_actions[-1] if ensemble_actions else None,
    }

    return equity_curve, ensemble_metrics


def create_test_data_table(ticker="^N225", start="2000-01-01", end="2010-01-01"):
    """テストデータをDataFrame形式で表示（日時降順）"""
    test_data = generate_env_data(start, end, ticker=ticker)

    # DataFrameとして準備（インデックスを列に変換）
    df = test_data.reset_index()
    df.rename(columns={"index": "Date"}, inplace=True)

    # 日付を降順でソート
    df = df.sort_values("Date", ascending=False)

    # 数値を適切にフォーマット
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].round(2)

    # 日付を文字列に変換（表示用）
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # 最初の50行だけを返す（表示用）
    return df.head(50)


def create_action_log_table(
    results, ticker="^N225", start="2000-01-01", end="2010-01-01"
):
    """日付ごとに各モデル・アンサンブル（多数決）のアクションを一覧にする（日時降順）。

    アクションは「その日のOpenで建てて翌日のOpenまで持つ」判断なので、
    翌日Open比（ロングがその日に得るリターン）も併記する。
    """
    if not results:
        return pd.DataFrame()

    test_data = generate_env_data(start, end, ticker=ticker)
    env = NikkeiEnv(
        test_data,
        window_size=130,
        transaction_cost=config.TRANSACTION_COST,
        risk_limit=config.RISK_LIMIT,
    )

    action_label = {0: "🟢 買", 1: "⚪ 待", 2: "🔴 売"}
    max_len = max(len(r["action_history"]) for r in results)

    # 多数決（create_ensemble_result と同じロジック）
    ensemble_actions = []
    for i in range(max_len):
        votes = [
            r["action_history"][i] for r in results if i < len(r["action_history"])
        ]
        ensemble_actions.append(Counter(votes).most_common(1)[0][0])

    # 多数決の資産カーブ（test_data を使い回して再ダウンロードを避ける）
    ens_env = NikkeiEnv(
        test_data,
        window_size=130,
        transaction_cost=config.TRANSACTION_COST,
        risk_limit=config.RISK_LIMIT,
    )
    ens_env.reset()
    for a in ensemble_actions:
        _, _, terminated, truncated, _ = ens_env.step(a)
        if terminated or truncated:
            break
    ens_equity = ens_env.get_equity_curve()  # [i+1] が i 日目のアクション後の資産

    rows = []
    for i in range(max_len):
        step = env.trade_start + i
        if step >= env.n:
            break
        open_today = float(env.open_prices[step])
        open_next = (
            float(env.open_prices[step + 1]) if step + 1 < env.n else float("nan")
        )
        row = {
            "日付": env.dates[step].strftime("%Y-%m-%d"),
            "Open": round(open_today, 1),
            "翌日Open比 (%)": (
                round((open_next / open_today - 1) * 100, 2)
                if np.isfinite(open_next)
                else None
            ),
        }
        votes = []
        for r in results:
            a = (
                r["action_history"][i]
                if i < len(r["action_history"])
                else None
            )
            row[f"モデル {r['steps']}"] = action_label.get(a, "—")
            if a is not None:
                votes.append(a)
        row["多数決"] = action_label.get(ensemble_actions[i], "—")
        row["全員一致"] = "✓" if len(set(votes)) == 1 else ""
        if i + 1 < len(ens_equity):
            row["多数決 資産 (円)"] = round(ens_equity[i + 1])
            row["多数決 日次損益 (%)"] = round(
                (ens_equity[i + 1] / ens_equity[i] - 1) * 100, 2
            )
        else:
            row["多数決 資産 (円)"] = None
            row["多数決 日次損益 (%)"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("日付", ascending=False)


def create_equity_curves_with_ensemble(
    results, ticker="^N225", start="2000-01-01", end="2010-01-01"
):
    """全モデルのエクイティカーブとアンサンブルを表示し、アンサンブル性能も返す"""

    # 1要素目は gr.Plot 行きなので文字列ではなく None（モデル未発見の旨は2要素目に）
    if not results:
        return None, "モデルzipが見つかりません（nikkei_*.zip を置いてください）", pd.DataFrame()

    # 銘柄の価格データを取得
    test_data = generate_env_data(start, end, ticker=ticker)
    window_size = 130

    # 実際の価格データを取得（window_size以降のデータ）
    # 'Close'列を使用（存在しない場合は'Open'を使用）
    if "Close" in test_data.columns:
        price_column = "Close"
    else:
        price_column = "Open"

    # 最長のequity_curveの長さを取得して、銘柄価格データも同じ長さに調整
    max_equity_length = (
        max(len(result["equity_curve"]) for result in results) if results else 0
    )

    # 銘柄価格データを資産カーブと同じ長さに調整（最新のデータを残して古いデータを削る）
    actual_prices = (
        test_data[price_column].iloc[-max_equity_length:]
        if max_equity_length > 0
        else test_data[price_column]
    )
    # 初期価格で正規化（初期資産1,000,000円に対応）
    if len(actual_prices) > 0:
        initial_price = actual_prices.iloc[0]
        normalized_prices = (actual_prices / initial_price) * 1000000
    else:
        normalized_prices = pd.Series([])

    # サブプロットを作成（2つのY軸を持つ）
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    # 個別モデルのエクイティカーブ
    for result in results:
        steps = result["steps"]
        equity_curve = result["equity_curve"]

        fig.add_trace(
            go.Scatter(
                x=list(range(len(equity_curve))),
                y=clean_data_for_plot(equity_curve),
                mode="lines",
                name=f"Step {steps}",
                line=dict(width=1.5),
                opacity=0.6,
            ),
            secondary_y=False,
        )

    # アンサンブル（多数決）のエクイティカーブと性能指標
    ensemble_curve, ensemble_metrics = create_ensemble_result(
        results, ticker, start, end
    )

    if ensemble_curve:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(ensemble_curve))),
                y=clean_data_for_plot(ensemble_curve),
                mode="lines",
                name="Ensemble (Majority Vote)",
                line=dict(width=4, color="black"),
                opacity=1.0,
            ),
            secondary_y=False,
        )

    # 銘柄の値動きを追加（第2Y軸）
    fig.add_trace(
        go.Scatter(
            x=list(range(len(normalized_prices))),
            y=normalized_prices.tolist(),
            mode="lines",
            name=f"{ticker} 価格（正規化）",
            line=dict(width=2, color="orange", dash="dash"),
            opacity=0.8,
        ),
        secondary_y=True,
    )

    # レイアウトの更新
    fig.update_xaxes(title_text="日数")
    fig.update_yaxes(title_text="資産 (円)", secondary_y=False)
    # 第2軸はグリッドを消す（主軸のグリッドと重なって読みにくくなるため）
    fig.update_yaxes(title_text="銘柄価格（正規化）", secondary_y=True, showgrid=False)

    fig.update_layout(
        title="資産カーブの比較 + アンサンブル（多数決）+ 銘柄価格",
        template="plotly_white",
        height=600,
        hovermode="x unified",
        # カーブは右肩上がりなので左上の空き領域に凡例を重ねる
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )

    # アンサンブルの性能指標をテキスト形式で表示
    ensemble_text = ""
    if ensemble_metrics:
        action_map = {0: "🟢 買い (Long)", 1: "⚪ 待ち (Flat)", 2: "🔴 売り (Short)"}
        last_action_label = action_map.get(ensemble_metrics.get("last_action"), "不明")
        ensemble_text = f"""
## 📌 最後のアクション: **{last_action_label}**

---

## 🎯 アンサンブル（多数決）性能指標

- **年利**: {ensemble_metrics['annual_return']:.2f}%
- **シャープレシオ**: {ensemble_metrics['sharpe_ratio']:.3f}
- **最大ドローダウン**: {ensemble_metrics['max_drawdown']:.2f}%
- **勝率**: {ensemble_metrics['win_rate']:.2f}%
- **平均勝ち**: {ensemble_metrics['avg_win']:.4f}%
- **平均負け**: {ensemble_metrics['avg_loss']:.4f}%
- **W/Lレシオ**: {ensemble_metrics['wl_ratio']:.2f}
- **期待値**: {ensemble_metrics['expectancy']:.4f}%
- **プロフィットファクター**: {ensemble_metrics['profit_factor']:.2f}
- **総取引数**: {ensemble_metrics['total_trades']}
- **最終残高**: ¥{ensemble_metrics['final_balance']:,.0f}

### 保持期間統計
- **平均勝ち期間**: {ensemble_metrics['avg_win_holding_period']:.1f} 日
- **平均負け期間**: {ensemble_metrics['avg_loss_holding_period']:.1f} 日
- **最大勝ち期間**: {ensemble_metrics['max_win_holding_period']:.0f} 日
- **最大負け期間**: {ensemble_metrics['max_loss_holding_period']:.0f} 日

### アクション分布
- **ロング**: {ensemble_metrics['long_actions']} 回
- **フラット**: {ensemble_metrics['flat_actions']} 回
- **ショート**: {ensemble_metrics['short_actions']} 回
"""

    # テストデータのDataFrameも返す
    test_data_df = create_test_data_table(ticker, start, end)

    return fig, ensemble_text, test_data_df


# Gradioインターフェース
with gr.Blocks(
    title="RecurrentPPO Model Performance Visualizer"
) as demo:
    gr.Markdown(
        """
    # 🚀 RecurrentPPO Trading Model Performance Analyzer

    このツールは、異なる訓練ステップで保存されたRecurrentPPOモデルの性能を比較・可視化します。
    各モデルの年利、シャープレシオ、最大ドローダウン、勝率などの指標の幅を確認できます。
    """
    )

    with gr.Row():
        with gr.Column(scale=2):
            ticker_input = gr.Textbox(
                value="^N225",
                label="📈 ティッカーシンボルを入力",
                info="有効なティッカーシンボルを入力(e.g., ^N225, ^GSPC, AAPL, MSFT)",
                placeholder="^N225",
            )
        with gr.Column(scale=2):
            start_date_input = gr.Textbox(
                value="2023-01-01",
                label="📅 開始日",
                info="開始日を入力 (YYYY-MM-DD形式)",
                placeholder="2023-01-01",
            )
        with gr.Column(scale=2):
            end_date_input = gr.Textbox(
                value=pd.Timestamp.today().strftime("%Y-%m-%d"),
                label="📅 終了日",
                info="終了日を入力 (YYYY-MM-DD形式)。当日を指定すると当日バーも取得を試みる",
                placeholder="YYYY-MM-DD",
            )
        with gr.Column(scale=1):
            analyze_btn = gr.Button(
                "📊 全てのモデルで分析する", variant="primary", size="lg"
            )

    with gr.Tab("📈 パフォーマンス比較"):
        performance_plot = gr.Plot()

    with gr.Tab("📋 サマリー統計"):
        summary_table = gr.DataFrame()

    with gr.Tab("アクション分布"):
        action_plot = gr.Plot()

    with gr.Tab("💰 資産カーブ"):
        with gr.Row():
            with gr.Column(scale=3):
                equity_plot = gr.Plot()
            with gr.Column(scale=1):
                ensemble_metrics = gr.Markdown()

    with gr.Tab("📅 日次アクション履歴"):
        action_log_table = gr.Dataframe(
            label="日付ごとの各モデル・多数決のアクション（日時降順）",
            wrap=True,
            interactive=False,
        )

    with gr.Tab("📊 テストデータ (CSV)"):
        test_data_table = gr.Dataframe(
            label="テストデータ（日時降順）", wrap=True, interactive=False
        )

    # ボタンクリック時の動作
    def analyze_all_models(ticker, start, end):
        """全モデルを一度だけ評価して、結果を各関数に渡す"""
        # evaluate_all_modelsを一度だけ実行
        results = evaluate_all_models(ticker, start, end)

        # 各関数に結果を渡す
        return [
            create_performance_comparison(results),
            create_summary_stats(results),
            create_action_distribution(results),
            create_action_log_table(results, ticker, start, end),
            *create_equity_curves_with_ensemble(
                results, ticker, start, end
            ),  # グラフ、メトリクス、テストデータを返す
        ]

    analyze_btn.click(
        fn=analyze_all_models,
        inputs=[ticker_input, start_date_input, end_date_input],
        outputs=[
            performance_plot,
            summary_table,
            action_plot,
            action_log_table,
            equity_plot,
            ensemble_metrics,
            test_data_table,
        ],
    )

    gr.Markdown(
        """
    ### 📖 使い方
    1. ティッカーシンボルを入力（任意の銘柄が可能）
    2. **全てのモデルで分析する**ボタンをクリックして全モデルを評価
    3. 各タブで異なる観点からの分析結果を確認
    4. **パフォーマンス比較**: 主要指標の推移
    5. **サマリー統計**: 指標の統計サマリー
    6. **アクション分布**: アクション選択の分布
    7. **資産曲線**: 資産曲線の比較 + **アンサンブル（多数決）結果**
    8. **テストデータ (CSV)**: 使用されたテストデータを日時降順で表示

    ### 📈 ティッカーシンボル例
    **指数:**
    - **^N225**: 日経225（日本） | **^GSPC**: S&P 500（米国） | **^IXIC**: NASDAQ（米国）
    - **^DJI**: ダウジョーンズ（米国） | **^RUT**: ラッセル2000（米国） | **^FTSE**: FTSE100（英国）
    
    **個別株:**
    - **AAPL**: Apple | **MSFT**: Microsoft | **GOOGL**: Google | **AMZN**: Amazon
    - **7203.T**: トヨタ | **6758.T**: ソニー | **9984.T**: ソフトバンク
    
    ### 🎯 アンサンブル機能
    - 全モデルの各ステップでのアクションを多数決で決定
    - 黒い太線で「Ensemble (Majority Vote)」として表示
    - 個別モデルの性能のばらつきを平滑化した安定的な戦略
    
    ### 📊 指標説明
    - **Annual Return**: 年間収益率 (%)
    - **Sharpe Ratio**: シャープレシオ (リスク調整済みリターン)
    - **Max Drawdown**: 最大ドローダウン (%)
    - **Win Rate**: 勝率 (%)
    - **Profit Factor**: プロフィットファクター (総利益/総損失)
    """
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, show_error=True, theme="soft")
