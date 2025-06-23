# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキーの指定 ---
API_KEY = st.secrets["API_KEY"]

# --- ユーザーインターフェース ---
st.title("FXトレード分析ツール（マルチタイム＆戦略対応）")

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

    def calc_indicators(df):
        df = df.copy()
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["SMA_5"] = df["close"].rolling(window=5).mean()
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["Signal"] = df["MACD"].ewm(span=9).mean()
        df["Upper"] = df["close"].rolling(window=20).mean() + 2 * df["close"].rolling(window=20).std()
        df["Lower"] = df["close"].rolling(window=20).mean() - 2 * df["close"].rolling(window=20).std()
        return df

    def extract_signal(df):
        guide = []
        last = df.iloc[-1]
        if last["MACD"] > last["Signal"] and last["SMA_5"] > last["SMA_20"] and last["close"] > last["Lower"]:
            signal = "買い"
            guide.append("MACDがゴールデンクロス")
            guide.append("SMA短期 > 長期")
            guide.append("BB下限反発")
        elif last["MACD"] < last["Signal"] and last["SMA_5"] < last["SMA_20"] and last["close"] < last["Upper"]:
            signal = "売り"
            guide.append("MACDがデッドクロス")
            guide.append("SMA短期 < 長期")
            guide.append("BB上限反発")
        else:
            signal = "待ち"
            guide.append("全条件未達")
        return signal, guide

    def suggest_trade_plan(df, direction):
        price = df["close"].iloc[-1]
        atr = df["close"].rolling(window=14).std().iloc[-1]
        if direction == "買い":
            sl = price - atr * 1.0
            tp = price + atr * 1.6
        elif direction == "売り":
            sl = price + atr * 1.0
            tp = price - atr * 1.6
        else:
            return price, None, None, 0
        rr = abs((tp - price) / (sl - price))
        return price, tp, sl, rr

    def real_backtest(df):
        signals = []
        for i in range(30, len(df) - 10):
            window = df.iloc[i - 30:i + 1]
            latest = df.iloc[i]
            if latest["MACD"] > latest["Signal"] and latest["SMA_5"] > latest["SMA_20"] and latest["close"] > latest["Lower"]:
                entry = latest["close"]
                future = df.iloc[i + 1:i + 11]
                tp = entry + (entry * 0.004)
                sl = entry - (entry * 0.003)
                result = "保留"
                for j, row in future.iterrows():
                    if row["close"] >= tp:
                        result = "勝ち"
                        break
                    elif row["close"] <= sl:
                        result = "負け"
                        break
                signals.append({"日時": latest.name, "エントリーレート": entry, "結果": result})
        df_bt = pd.DataFrame(signals).tail(100)
        win_rate = round((df_bt["結果"] == "勝ち").sum() / len(df_bt), 3) if not df_bt.empty else 0.0
        return win_rate, df_bt

    st.subheader(f"通貨ペア：{symbol} | スタイル：{style}")
    final_signal = []
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            continue
        df = calc_indicators(df)
        sig, guide = extract_signal(df)
        final_signal.append(sig)
        st.markdown(f"### ⏱ {tf} 判定：{sig}")
        for g in guide:
            st.write("-", g)

    decision = "待ち"
    if final_signal.count("買い") >= 2:
        decision = "買い"
    elif final_signal.count("売り") >= 2:
        decision = "売り"

    df_all = fetch_data(symbol, timeframes[1])
    df_all = calc_indicators(df_all)
    entry, tp, sl, rr = suggest_trade_plan(df_all, decision)
    win_rate, bt_log = real_backtest(df_all)

    if decision != "待ち":
        st.subheader("\n🎯 トレードプラン（想定）")
        st.write(f"エントリーレート：{entry:.2f}")
        st.write(f"指値（利確）：{tp:.2f}（+{abs(tp-entry)*100:.0f} pips）")
        st.write(f"逆指値（損切）：{sl:.2f}（-{abs(sl-entry)*100:.0f} pips）")
        st.write(f"リスクリワード比：{rr:.2f}")
        st.write(f"想定勝率：{win_rate*100:.1f}%")
    else:
        st.subheader("現在はエントリー待ちです。")

    with st.expander("バックテスト（直近100件）"):
        st.dataframe(bt_log)
        st.write(f"勝率：{win_rate*100:.1f}%")
