import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# --- APIキー設定 ---
API_KEY = st.secrets["API_KEY"]

# --- UI設定 ---
st.title("FXトレード分析")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"], index=0)

# --- 時間足と重み設定 ---
tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}

# --- データ取得 ---
def fetch_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df = df.astype(float)
    return df

# --- 指標計算 ---
def calc_indicators(df):
    df = df.copy()
    df["SMA_5"] = df["close"].rolling(5).mean()
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Upper"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
    df["Lower"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()
    df["STD"] = df["close"].rolling(20).std()
    df["RCI"] = df["close"].rank().rolling(9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
    df.dropna(inplace=True)
    return df

# --- 相場構造判定 ---
def detect_market_structure(df):
    last = df.iloc[-1]
    trend_votes = 0
    range_votes = 0
    
    # 疑似ADX判定
    adx_proxy = abs(last["SMA_5"] - last["SMA_20"]) / last["close"]
    if adx_proxy > 0.015:
        trend_votes += 1
    else:
        range_votes += 1

    # SMA乖離率
    sma_ratio = abs(last["SMA_5"] - last["SMA_20"]) / last["close"]
    if sma_ratio > 0.015:
        trend_votes += 1
    else:
        range_votes += 1

    # 標準偏差
    if last["STD"] > last["close"] * 0.005:
        trend_votes += 1
    else:
        range_votes += 1

    return "トレンド" if trend_votes >= 2 else "レンジ"

# --- シグナル抽出（売買別スコア） ---
def extract_signal(df):
    last = df.iloc[-1]
    guide = []
    buy_score = 0
    sell_score = 0

    if last["MACD"] > last["Signal"]:
        buy_score += 1
        guide.append("✅ MACDゴールデンクロス")
    else:
        sell_score += 1
        guide.append("✅ MACDデッドクロス")

    if last["SMA_5"] > last["SMA_20"]:
        buy_score += 1
        guide.append("✅ SMA短期 > 長期")
    else:
        sell_score += 1
        guide.append("❌ SMA条件未達")

    if last["close"] < last["Lower"]:
        buy_score += 1
        guide.append("✅ BB下限反発")
    elif last["close"] > last["Upper"]:
        sell_score += 1
        guide.append("✅ BB上限反発")
    else:
        guide.append("❌ BB反発無し")

    if last["RCI"] > 0.5:
        buy_score += 1
        guide.append("✅ RCI上昇傾向")
    elif last["RCI"] < -0.5:
        sell_score += 1
        guide.append("✅ RCI下降傾向")
    else:
        guide.append("❌ RCI未達")

    if buy_score >= 3:
        return "買い", guide, buy_score
    elif sell_score >= 3:
        return "売り", guide, sell_score
    else:
        return "待ち", guide, max(buy_score, sell_score)

# --- 実行ボタン付きメイン処理 ---
if st.button("実行"):
    timeframes = tf_map[style]
    final_scores = []
    st.subheader(f"\n💱 通貨ペア：{symbol} | スタイル：{style}\n\n⸻")
    st.markdown("### ⏱ 各時間足シグナル詳細")

    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            st.error(f"データ取得失敗：{tf}")
            continue

        df = calc_indicators(df)
        market_type = detect_market_structure(df)
        signal, guide, score = extract_signal(df)
        final_scores.append(score * tf_weights.get(tf, 0.3))

        st.markdown(f"⏱ {tf} 判定：{signal}（スコア：{score:.1f}）")
        st.markdown(f"• 市場判定：{market_type}")
        for g in guide:
            st.markdown(f"\t•\t{g}")

    avg_score = sum(final_scores)
    st.markdown("\n⸻")
    st.markdown("### 🧭 エントリーガイド（総合評価）")
    if avg_score >= 2.4:
        st.write("✅ 複数の時間足が買いシグナルを示しています")
    elif avg_score <= 1.2:
        st.write("✅ 複数の時間足が売りシグナルを示しています")
    else:
        st.write("現在は明確な買い/売りシグナルが不足しています")
