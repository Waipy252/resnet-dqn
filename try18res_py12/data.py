import pandas_datareader.data as web
import yfinance as yf
import pandas as pd
import numpy as np


def generate_env_data(start, end, ticker="JPY=X", manual_data=None):
    # ^N225の取得
    test_data = yf.download(ticker, start=start, end=end)
    # test_data = yf.download("^GSPC", start=start, end=end)#S&P500
    # ダウンロード直後にカラムをフラット化する
    test_data.columns = test_data.columns.get_level_values(0)

    # -------------------------
    # 金利データ(FRED)の取得
    # -------------------------
    us_rate = web.DataReader("FEDFUNDS", "fred", start, end)
    us_rate.rename(columns={"FEDFUNDS": "US_10Y_Rate"}, inplace=True)
    us_rate.index.name = "Date"
    date_range = pd.date_range(start=start, end=end, freq="D")
    us_rate = us_rate.reindex(date_range).ffill()

    jp_rate = web.DataReader("IRLTLT01JPM156N", "fred", start, end)
    jp_rate.rename(columns={"IRLTLT01JPM156N": "Japan_10Y_Rate"}, inplace=True)
    jp_rate.index.name = "Date"
    date_range = pd.date_range(start=start, end=end, freq="D")
    jp_rate = jp_rate.reindex(date_range).ffill()

    # 3. アメリカの恐怖指数 VIX のデータを取得（終値を使用）
    vix_data = yf.download("^VIX", start=start, end=end)[["Close"]]
    vix_data.rename(columns={"Close": "VIX"}, inplace=True)
    vix_data.columns = vix_data.columns.get_level_values(0)
    # 文字列の"null"をNaNに変換してから前日データで埋める
    vix_data["VIX"] = vix_data["VIX"].replace("null", np.nan).ffill()
    test_data = test_data.join(vix_data, how="left")
    # 結合後も文字列の"null"をNaNに変換してから前日データで埋める
    test_data["VIX"] = test_data["VIX"].replace("null", np.nan).ffill()

    rate_data = pd.merge(
        jp_rate, us_rate, left_index=True, right_index=True, how="left"
    )
    test_data = pd.merge(
        test_data, rate_data, left_index=True, right_index=True, how="left"
    )
    # 手動データがある場合は追加
    if manual_data is not None:
        manual_data = pd.DataFrame(manual_data)
        manual_data.index = pd.to_datetime(manual_data.index)  # 日付データを適切に変換
        test_data = pd.concat([test_data, manual_data])

    # テクニカル指標の計算（例ではOpenを使用）
    test_data["SMA_5"] = test_data["Open"].rolling(window=5).mean()
    test_data["SMA_25"] = test_data["Open"].rolling(window=25).mean()
    test_data["SMA_75"] = test_data["Open"].rolling(window=75).mean()
    test_data["STD_25"] = test_data["Open"].rolling(window=25).std()
    test_data["Upper_3σ"] = test_data["SMA_25"] + 3 * test_data["STD_25"]
    test_data["Lower_3σ"] = test_data["SMA_25"] - 3 * test_data["STD_25"]
    test_data["Upper_2σ"] = test_data["SMA_25"] + 2 * test_data["STD_25"]
    test_data["Lower_2σ"] = test_data["SMA_25"] - 2 * test_data["STD_25"]
    test_data["Upper_1σ"] = test_data["SMA_25"] + 1 * test_data["STD_25"]
    test_data["Lower_1σ"] = test_data["SMA_25"] - 1 * test_data["STD_25"]
    test_data["偏差値25"] = 50 + 10 * (
        (test_data["Open"] - test_data["SMA_25"]) / test_data["STD_25"]
    )

    test_data["STD_75"] = test_data["Open"].rolling(window=75).std()
    test_data["Upper2_3σ"] = test_data["SMA_75"] + 3 * test_data["STD_75"]
    test_data["Lower2_3σ"] = test_data["SMA_75"] - 3 * test_data["STD_75"]
    test_data["Upper2_2σ"] = test_data["SMA_75"] + 2 * test_data["STD_75"]
    test_data["Lower2_2σ"] = test_data["SMA_75"] - 2 * test_data["STD_75"]
    test_data["Upper2_1σ"] = test_data["SMA_75"] + 1 * test_data["STD_75"]
    test_data["Lower2_1σ"] = test_data["SMA_75"] - 1 * test_data["STD_75"]
    test_data["偏差値75"] = 50 + 10 * (
        (test_data["Open"] - test_data["SMA_75"]) / test_data["STD_75"]
    )

    # -------------------------
    # RSIの計算
    # -------------------------
    def calc_rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    test_data["RSI_14"] = calc_rsi(test_data["Open"], 14)
    test_data["RSI_22"] = calc_rsi(test_data["Open"], 22)

    # -------------------------
    # MACDの計算
    # -------------------------
    test_data["EMA12"] = test_data["Open"].ewm(span=12, adjust=False).mean()
    test_data["EMA26"] = test_data["Open"].ewm(span=26, adjust=False).mean()
    test_data["MACD"] = test_data["EMA12"] - test_data["EMA26"]
    test_data["MACD_signal"] = test_data["MACD"].ewm(span=9, adjust=False).mean()

    # -------------------------
    # RCIの計算 (9日と26日)
    # -------------------------
    def calc_rci(series, period):
        def rci_calc(arr):
            N = len(arr)
            order = np.arange(1, N + 1)
            rank_ = pd.Series(arr).rank(method="first").values
            d = order - rank_
            return (1 - 6 * np.sum(d**2) / (N * (N**2 - 1))) * 100

        return series.rolling(window=period).apply(rci_calc, raw=True)

    test_data["RCI_9"] = calc_rci(test_data["Open"], 9)
    test_data["RCI_26"] = calc_rci(test_data["Open"], 26)

    # ATR（Average True Range）の計算 (5日と25日)
    # 前日のOpenを取得
    test_data["Previous_Open"] = test_data["Open"].shift(1)
    # True Range (TR) の各構成要素を計算
    tr1 = test_data["High"] - test_data["Low"]
    tr2 = (test_data["High"] - test_data["Previous_Open"]).abs()
    tr3 = (test_data["Low"] - test_data["Previous_Open"]).abs()
    # 各日のTRは3要素の中で最大の値
    test_data["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # ATRはTRの単純移動平均値
    test_data["ATR_5"] = test_data["TR"].rolling(window=5).mean()
    test_data["ATR_25"] = test_data["TR"].rolling(window=25).mean()
    # 途中計算用のカラム（例: Previous_Open）は削除
    test_data.drop(columns=["Previous_Open"], inplace=True)

    print(test_data)
    test_data.to_csv("test_data.csv")
    return test_data
