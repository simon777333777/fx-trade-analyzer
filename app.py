# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキーの指定 ---
API_KEY = st.secrets["API_KEY"]

# --- ユーザーインターフェース ---
st.title("FXトレード分析ツール")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"])
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"])

if st.button("実行"):

    # --- トレードスタイルに応じた時間足定義 ---
    tf_map = {
        "スキャルピング": ["5min", "15min", "1h"],
        "デイトレード": ["15min", "1h", "4h"],
        "スイング": ["1h", "4h", "1day"]
    }
    timeframes = tf_map[style]

    # --- データ取得関数 ---
    def fetch_data(symbol, interval):
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
        r = requests.get(url)
        data = r.json()
        if "values" not in data:
            st.error(f"データ取得失敗: {symbol} - {interval}")
            return None
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df = df.sort_index()
        df["close"] = df["close"].astype(float)
        return df

    # --- テクニカル指標（MACD・SMA・BB・RCI） ---
    def calc_indicators(df):
        df = df.copy()
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["SMA_5"] = df["close"].rolling(window=5).mean()
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["Signal"] = df["MACD"].ewm(span=9).mean()
        df["Upper"] = df["SMA_20"] + 2 * df["close"].rolling(window=20).std()
        df["Lower"] = df["SMA_20"] - 2 * df["close"].rolling(window=20).std()
        df["RCI"] = df["close"].rolling(9).apply(lambda x: pd.Series(x).rank().corr(pd.Series(range(1, len(x)+1))), raw=False)
        return df

    # --- シグナル抽出 ---
    def extract_signal(row):
        guide = []
        signal = "待ち"
        if row["MACD"] > row["Signal"] and row["SMA_5"] > row["SMA_20"] and row["close"] > row["Lower"]:
            signal = "買い"
            guide.append("MACDがゴールデンクロス")
            guide.append("SMA短期 > 長期")
            guide.append("BB下限反発")
        elif row["MACD"] < row["Signal"] and row["SMA_5"] < row["SMA_20"] and row["close"] < row["Upper"]:
            signal = "売り"
            guide.append("MACDがデッドクロス")
            guide.append("SMA短期 < 長期")
            guide.append("BB上限反発")
        else:
            guide.append("全条件未達")
        return signal, ", ".join(guide)

    # --- バックテスト ---
    def run_backtest(df):
        df = df.copy()
        df = calc_indicators(df)
        results = []
        for i in range(20, len(df)-5):
            row = df.iloc[i]
            signal, guide = extract_signal(row)
            if signal == "待ち":
                continue
            entry = row["close"]
            future = df.iloc[i+1:i+6]  # 次の5本で判断
            atr = df["close"].rolling(window=14).std().iloc[i]
            tp = entry + atr * 1.6 if signal == "買い" else entry - atr * 1.6
            sl = entry - atr * 1.0 if signal == "買い" else entry + atr * 1.0
            result = "保留"
            for j, frow in future.iterrows():
                if signal == "買い":
                    if frow["close"] >= tp:
                        result = "勝ち"
                        break
                    elif frow["close"] <= sl:
                        result = "負け"
                        break
                else:
                    if frow["close"] <= tp:
                        result = "勝ち"
                        break
                    elif frow["close"] >= sl:
                        result = "負け"
                        break
            pips = (tp - entry)*100 if result == "勝ち" else (sl - entry)*100 if result == "負け" else 0
            results.append({"日時": row.name, "判定": signal, "エントリー": entry, "結果": result, "損益(pips)": round(pips, 1)})
        df_result = pd.DataFrame(results)
        return df_result

    # --- 分析・表示 ---
    st.subheader(f"通貨ペア：{symbol} | スタイル：{style}")
    df_all = fetch_data(symbol, timeframes[1])
    if df_all is not None:
        df_all = calc_indicators(df_all)
        latest_row = df_all.iloc[-1]
        decision, reason = extract_signal(latest_row)

        st.markdown("### 🧭 エントリーガイド")
        st.write(f"現時点での判定：**{decision}**")
        st.write(reason)

        if decision != "待ち":
            atr = df_all["close"].rolling(window=14).std().iloc[-1]
            entry = latest_row["close"]
            tp = entry + atr * 1.6 if decision == "買い" else entry - atr * 1.6
            sl = entry - atr * 1.0 if decision == "買い" else entry + atr * 1.0
            rr = abs((tp - entry) / (sl - entry))
            st.markdown("### 🎯 トレードプラン（想定）")
            st.write(f"エントリーレート：{entry:.2f}")
            st.write(f"指値（利確）：{tp:.2f}（+{abs(tp-entry)*100:.0f} pips）")
            st.write(f"逆指値（損切）：{sl:.2f}（-{abs(sl-entry)*100:.0f} pips）")
            st.write(f"リスクリワード比：{rr:.2f}")

        # バックテスト実施
        bt_df = run_backtest(df_all)
        win_rate = bt_df[bt_df["結果"]=="勝ち"].shape[0] / bt_df.shape[0] * 100 if bt_df.shape[0]>0 else 0
        total_pips = bt_df["損益(pips)"].sum()

        with st.expander("📊 バックテスト結果（詳細表示）"):
            st.dataframe(bt_df)
            st.write(f"勝率：{win_rate:.1f}%")
            st.write(f"合計損益：{total_pips:.1f} pips")
    else:
        st.warning("データ取得に失敗しました。")
