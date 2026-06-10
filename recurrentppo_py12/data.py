import pandas_datareader.data as web
import yfinance as yf
import pandas as pd
import numpy as np


def generate_env_data(start, end, ticker="JPY=X", manual_data=None, save_csv=False):
    # ^N225の取得
    test_data = yf.download(ticker, start=start, end=end)
    # test_data = yf.download("^GSPC", start=start, end=end)#S&P500
    # ダウンロード直後にカラムをフラット化する
    test_data.columns = test_data.columns.get_level_values(0)
    if getattr(test_data.index, "tz", None) is not None:
        test_data.index = test_data.index.tz_localize(None)

    # yf.download は end が排他的なうえ、当日の進行中バーを返さないことが多い。
    # Ticker.history は当日のライブバーを含むので、範囲内の欠け（≒当日分）を補完する。
    # VIX・金利の join より前に行うこと（後だと当日行のそれらが NaN のままになる）。
    try:
        recent = yf.Ticker(ticker).history(period="5d")
        if not recent.empty:
            recent.index = recent.index.tz_localize(None).normalize()
            last_have = test_data.index.max() if len(test_data) else pd.Timestamp(start)
            new_rows = recent.loc[
                (recent.index > last_have) & (recent.index <= pd.Timestamp(end)),
                ["Open", "High", "Low", "Close", "Volume"],
            ]
            if not new_rows.empty:
                test_data = pd.concat([test_data, new_rows]).sort_index()
                # auto_adjust されていない列（Adj Close 等）は Close で補完
                for col in test_data.columns.difference(new_rows.columns):
                    test_data.loc[new_rows.index, col] = test_data.loc[
                        new_rows.index, col
                    ].fillna(test_data.loc[new_rows.index, "Close"])
                print(
                    "当日バーを補完:",
                    list(new_rows.index.strftime("%Y-%m-%d")),
                )
    except Exception as e:
        print(f"当日データの補完に失敗（取得済みデータのみで継続）: {e}")

    date_range = pd.date_range(start=start, end=end, freq="D")

    # -------------------------
    # 米国10年債利回り（A-2 / F-3）
    # FEDFUNDS（FF金利・月次）ではなく ^TNX（CBOE 10年債利回り・日次）を使用。
    # 名前と中身を一致させ、かつ日次で更新されるようにする。
    # -------------------------
    us_10y = yf.download("^TNX", start=start, end=end)[["Close"]]
    us_10y.columns = us_10y.columns.get_level_values(0)
    us_10y.rename(columns={"Close": "US_10Y_Rate"}, inplace=True)
    us_10y.index.name = "Date"
    # 日次だが取引所休場日を埋めるため reindex + ffill
    us_rate = us_10y.reindex(date_range).ffill()

    # -------------------------
    # 日本10年債利回り（F-3）
    # FRED IRLTLT01JPM156N は月次。日次 reindex 後、階段状を避けるため線形補間。
    # -------------------------
    jp_rate = web.DataReader("IRLTLT01JPM156N", "fred", start, end)
    jp_rate.rename(columns={"IRLTLT01JPM156N": "Japan_10Y_Rate"}, inplace=True)
    jp_rate.index.name = "Date"
    jp_rate = jp_rate.reindex(date_range).interpolate(method="linear").ffill().bfill()

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
    # 手動データがある場合は追加（F-4: fetch_latest の実OHLC行を想定。
    # High=Low=Open の合成行ではなく実OHLCで延長すれば ATR/TR の歪み（D-2）も出ない）
    if manual_data is not None:
        manual_data = pd.DataFrame(manual_data)
        manual_data.index = pd.to_datetime(manual_data.index)  # 日付データを適切に変換
        # 既にyfinanceで取得済みの日付と重複したら手動データ側を優先
        test_data = test_data[~test_data.index.isin(manual_data.index)]
        test_data = pd.concat([test_data, manual_data]).sort_index()

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

    # 副作用は呼び出し側の判断に委ねる（D-1）。デフォルトでは保存・全件printしない。
    if save_csv:
        test_data.to_csv("test_data.csv")
    return test_data


def fetch_latest(ticker="^N225"):
    """当日（直近営業日）のOHLC・VIX・金利を自動取得する（F-4）。

    run_simulation の manual_data を手打ちする代わりに使う。
    High=Low=Open の合成行ではなく実OHLCを返すので、ATR/TR の歪み（D-2）も起きない。

    戻り値 dict:
        date(YYYY-MM-DD), open, high, low, close, volume, vix, us_10y, jp_10y
        jp_10y は FRED が月次のため直近値（取得失敗時は None → 手入力で補完）。
    """
    px = yf.Ticker(ticker).history(period="5d")
    if px.empty:
        raise RuntimeError(f"{ticker} の直近データを取得できませんでした")
    last = px.iloc[-1]
    date = px.index[-1].strftime("%Y-%m-%d")

    def _last_close(symbol):
        h = yf.Ticker(symbol).history(period="5d")
        return float(h["Close"].iloc[-1]) if not h.empty else None

    vix = _last_close("^VIX")
    us_10y = _last_close("^TNX")

    # 日本10年金利は FRED 月次系列の直近値（低頻度なので近似でよい）
    try:
        today = pd.Timestamp.today()
        jp = web.DataReader("IRLTLT01JPM156N", "fred", today - pd.Timedelta(days=180), today)
        jp_10y = float(jp.dropna().iloc[-1, 0])
    except Exception as e:
        print(f"日本10年金利の取得に失敗（手入力で補完してください）: {e}")
        jp_10y = None

    return {
        "date": date,
        "open": float(last["Open"]),
        "high": float(last["High"]),
        "low": float(last["Low"]),
        "close": float(last["Close"]),
        "volume": float(last["Volume"]),
        "vix": vix,
        "us_10y": us_10y,
        "jp_10y": jp_10y,
    }
