import gradio as gr
from collections import Counter
from run_simulation import run_simulation_try18res


def run_simulation(price, end_day, vix, jpy, usr, volume):
    # 結果を保存する変数
    final_actions = []
    results = []
    results1, final_actions1, test_data = run_simulation_try18res(
        price, end_day, vix, jpy, usr
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

    # CSVファイルとして保存（オプション）
    test_data.to_csv("test_data.csv")

    # インデックスをカラムに変換して返す
    display_data = test_data.sort_index(ascending=False).reset_index()

    # 結果を返す：シミュレーション結果文字列とデータフレーム
    return "".join(results) + action_summary, display_data


# Gradioインターフェースの作成
with gr.Blocks(title="日経平均株価シミュレーション") as demo:
    gr.Markdown("# 日経平均株価 運用シミュレーション")

    with gr.Row():
        with gr.Column(scale=1):
            price_input = gr.Number(label="株価", value=44000)
            volume_input = gr.Number(label="出来高", value=130000000)
            end_day_input = gr.Textbox(label="日付 (YYYY-MM-DD)", value="2025-10-01")
            vix_input = gr.Number(label="VIX指数", value=15.21)
            jpy_input = gr.Number(label="日本10年債利回り", value=1.6)
            usr_input = gr.Number(label="米国10年債利回り", value=4.33)
            submit_btn = gr.Button("シミュレーション実行", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(label="シミュレーション結果", lines=25)

    # テストデータ表示用のDataframeコンポーネント
    gr.Markdown("## 生成されたテストデータ")
    data_display = gr.Dataframe(label="テストデータ", wrap=True)

    # ボタンクリック時の動作
    submit_btn.click(
        fn=run_simulation,
        inputs=[
            price_input,
            end_day_input,
            vix_input,
            jpy_input,
            usr_input,
            volume_input,
        ],
        outputs=[output, data_display],
    )

# アプリの起動
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8888, share=False)
