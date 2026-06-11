import gradio as gr
from collections import Counter

from data import fetch_latest
from run_simulation import run_simulation_rppo


def run_simulation(open_, high, low, close, volume, end_day, vix, jpy, usr):
    # 結果を保存する変数
    final_actions = []
    results = []
    results1, final_actions1, test_data = run_simulation_rppo(
        open_, high, low, close, volume, end_day, vix, jpy, usr
    )

    # 結合する
    final_actions.extend(final_actions1)
    results.extend(results1)

    # 最後のアクションの出現回数をカウント
    final_action_counts = Counter(final_actions)
    count_0 = final_action_counts.get(0, 0)
    count_1 = final_action_counts.get(1, 0)
    count_2 = final_action_counts.get(2, 0)

    action_summary = f"\n{end_day} 買い:{count_0}, 待ち:{count_1}, 売り:{count_2}"

    if test_data is None:
        return "データ生成に失敗しました" + action_summary, None

    # CSVファイルとして保存（オプション）
    test_data.to_csv("test_data.csv")

    # インデックスをカラムに変換して返す
    display_data = test_data.sort_index(ascending=False).reset_index()

    # 結果を返す：シミュレーション結果文字列とデータフレーム
    return "".join(results) + action_summary, display_data


def autofill_latest(current_jpy):
    """F-4: yfinance / FRED から最新OHLC・VIX・金利を取得してフォームを埋める。

    日本10年金利（FRED月次）が取れなかった場合のみ、現在の入力値を残す。
    """
    try:
        latest = fetch_latest()
    except Exception as e:
        gr.Warning(f"最新データの取得に失敗しました: {e}")
        return [gr.skip()] * 9

    jpy = latest["jp_10y"] if latest["jp_10y"] is not None else current_jpy
    return (
        latest["open"],
        latest["high"],
        latest["low"],
        latest["close"],
        latest["volume"],
        latest["date"],
        latest["vix"],
        jpy,
        latest["us_10y"],
    )


# Gradioインターフェースの作成
with gr.Blocks(title="日経平均株価シミュレーション") as demo:
    gr.Markdown("# 日経平均株価 運用シミュレーション（RecurrentPPO）")

    with gr.Row():
        with gr.Column(scale=1):
            fetch_btn = gr.Button("📥 最新データ自動取得", variant="secondary")
            open_input = gr.Number(label="始値 (Open)", value=63000)
            high_input = gr.Number(label="高値 (High)", value=63300)
            low_input = gr.Number(label="安値 (Low)", value=62700)
            close_input = gr.Number(label="終値 (Close)", value=63000)
            volume_input = gr.Number(label="出来高", value=130000000)
            end_day_input = gr.Textbox(label="日付 (YYYY-MM-DD)", value="2026-06-06")
            vix_input = gr.Number(label="VIX指数", value=15.21)
            jpy_input = gr.Number(label="日本10年債利回り", value=2.6)
            usr_input = gr.Number(label="米国10年債利回り", value=4.53)
            submit_btn = gr.Button("シミュレーション実行", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(label="シミュレーション結果", lines=25)

    # テストデータ表示用のDataframeコンポーネント
    gr.Markdown("## 生成されたテストデータ")
    data_display = gr.Dataframe(label="テストデータ", wrap=True)

    inputs = [
        open_input,
        high_input,
        low_input,
        close_input,
        volume_input,
        end_day_input,
        vix_input,
        jpy_input,
        usr_input,
    ]

    # F-4: 最新データ自動取得 → フォームを上書き（手打ちは上書き用途に限定）
    fetch_btn.click(fn=autofill_latest, inputs=[jpy_input], outputs=inputs)

    # ボタンクリック時の動作
    submit_btn.click(fn=run_simulation, inputs=inputs, outputs=[output, data_display])

# アプリの起動
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8888, share=False)
