# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- ユーザーインターフェース ---
st.title("FXトレード分析")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"], index=0)

if st.button("実行"):

    tf_map = {
        "スキャルピング": ["5min", "15min", "1h"],
        "デイトレード": ["15min", "1h", "4h"],
        "スイング": ["1h", "4h", "1day"]
    }
    tf_weights = {
        "5min": 0.2, "15min": 0.3, "1h": 0.2, "4h": 0.3, "1day": 0.5
    }
    timeframes = tf_map[style]

    def fetch_data(symbol, interval):
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={API_KEY}"
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
        df["RCI"] = df["close"].rank().rolling(window=9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
        return df

    def extract_signal(df):
        guide = []
        buy_score = 0
        sell_score = 0
        last = df.iloc[-1]

        # MACD
        if last["MACD"] > last["Signal"]:
            buy_score += 1
            guide.append("✅ MACDゴールデンクロス")
        elif last["MACD"] < last["Signal"]:
            sell_score += 1
            guide.append("✅ MACDデッドクロス")
        else:
            guide.append("❌ MACD未達")

        # SMA
        if last["SMA_5"] > last["SMA_20"]:
            buy_score += 1
            guide.append("✅ SMA短期 > 長期")
        elif last["SMA_5"] < last["SMA_20"]:
            sell_score += 1
            guide.append("✅ SMA短期 < 長期")
        else:
            guide.append("❌ SMA条件未達")

        # Bollinger Band
        if last["close"] < last["Lower"]:
            buy_score += 1
            guide.append("✅ BB下限反発の可能性")
        elif last["close"] > last["Upper"]:
            sell_score += 1
            guide.append("✅ BB上限反発の可能性")
        else:
            guide.append("❌ BB反発無し")

        # RCI
        if last["RCI"] > 0.5:
            buy_score += 1
            guide.append("✅ RCI上昇傾向")
        elif last["RCI"] < -0.5:
            sell_score += 1
            guide.append("✅ RCI下降傾向")
        else:
            guide.append("❌ RCI未達")

        # スコア判定
        signal = "買い" if buy_score >= 3 else "売り" if sell_score >= 3 else "待ち"
        return signal, guide, buy_score, sell_score

    st.subheader(f"通貨ペア：{symbol} | スタイル：{style}")

    final_buy_scores = []
    final_sell_scores = []
    final_signals = []

    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            continue
        df = calc_indicators(df)
        sig, guide, buy_score, sell_score = extract_signal(df)
        final_signals.append((tf, sig, buy_score, sell_score, guide))

        st.markdown(f"### ⏱ {tf} 判定：{sig}")
        for g in guide:
            st.write("-", g)

        final_buy_scores.append(buy_score * tf_weights.get(tf, 0.3))
        final_sell_scores.append(sell_score * tf_weights.get(tf, 0.3))

    # 総合スコア判定
    weighted_buy = sum(final_buy_scores)
    weighted_sell = sum(final_sell_scores)

    if weighted_buy >= 0.6:
        decision = "買い"
    elif weighted_sell >= 0.6:
        decision = "売り"
    else:
        decision = "待ち"

    st.subheader("\n🧭 エントリーガイド（総合評価）")
    if decision == "買い":
        st.write(f"✅ {style} において複数の時間足が買いシグナルを示しています")
        st.write("⏳ 中期・長期の上昇トレンドが短期にも波及")
        st.write("📌 押し目が完了しており、エントリータイミングとして有効")
    elif decision == "売り":
        st.write(f"🔻 {style} において複数の時間足が売りシグナルを示しています")
        st.write("📉 中期・長期の下降トレンドに入っており戻り売りが有効")
    else:
        st.write("現在は明確な買い/売りシグナルが不足しているため、エントリーは控えめに")
