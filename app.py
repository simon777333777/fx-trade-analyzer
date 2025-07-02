import streamlit as st
import pandas as pd
import numpy as np
import requests

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- UI ---
st.title("FXトレード分析ツール（シグナル判定特化）")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"])
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=2)

# --- 時間足と重み ---
tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}

# --- データ取得 ---
@st.cache_data(ttl=600)
def fetch_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        raise ValueError(f"APIデータ取得エラー: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    df = df.astype(float)
    return df

# --- インジケーター ---
def calc_indicators(df):
    df["SMA_5"] = df["close"].rolling(5).mean()
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Upper"] = df["SMA_20"] + 2 * df["close"].rolling(20).std()
    df["Lower"] = df["SMA_20"] - 2 * df["close"].rolling(20).std()
    df["RCI"] = df["close"].rank().rolling(9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
    df["ADX"] = abs(df["MACD"] - df["Signal"]).rolling(14).mean()
    df["STD"] = df["close"].rolling(20).std()
    return df

# --- 市場構造判定 ---
def detect_market_structure(last):
    score = 0
    if last["ADX"] > 25: score += 1
    elif last["ADX"] < 20: score -= 1
    if abs(last["SMA_5"] - last["SMA_20"]) / last["close"] > 0.015: score += 1
    else: score -= 1
    if last["STD"] > last["close"] * 0.005: score += 1
    else: score -= 1
    return "トレンド" if score >= 2 else "レンジ"

# --- シグナル抽出 ---
def extract_signal(df):
    last = df.iloc[-1]
    logs = [f"• 市場判定：{detect_market_structure(last)}"]
    buy = sell = 0

    if last["MACD"] > last["Signal"]:
        buy += 1; logs.append("🟢 MACDゴールデンクロス")
    else:
        sell += 1; logs.append("🔴 MACDデッドクロス")

    if last["SMA_5"] > last["SMA_20"]:
        buy += 1; logs.append("🟢 SMA短期 > 長期")
    else:
        sell += 1; logs.append("🔴 SMA短期 < 長期")

    if last["close"] < last["Lower"]:
        buy += 1; logs.append("🟢 BB下限反発の可能性")
    elif last["close"] > last["Upper"]:
        sell += 1; logs.append("🔴 BB上限反発の可能性")
    else:
        logs.append("⚪ BB反発無し")

    if last["RCI"] > 0.5:
        buy += 1; logs.append("🟢 RCI上昇傾向")
    elif last["RCI"] < -0.5:
        sell += 1; logs.append("🔴 RCI下降傾向")
    else:
        logs.append("⚪ RCI未達")

    decision = (
        "買い" if buy >= 3 and buy > sell else
        "売り" if sell >= 3 and sell > buy else
        "待ち"
    )
    return decision, logs, buy, sell

# --- 実行ボタン ---
if st.button("実行"):
    timeframes = tf_map[style]
    total_buy = total_sell = 0
    score_log = []

    st.subheader(f"📊 通貨ペア：{symbol} | スタイル：{style}")
    st.markdown("### ⏱ 各時間足シグナル詳細\n\n凡例：🟢=買い、🔴=売り、⚪=未達")

    for tf in timeframes:
        with st.spinner(f"{tf} データ取得中..."):
            df = fetch_data(symbol.replace("/", ""), tf)
            df = calc_indicators(df)
            decision, logs, buy, sell = extract_signal(df)
            weight = tf_weights[tf]
            total_buy += buy * weight
            total_sell += sell * weight
            score_log.append((tf, buy, sell, weight))

            st.markdown(f"⏱ {tf} 判定：{decision}（スコア：{max(buy,sell):.1f}）")
            for log in logs:
                st.markdown(log)

    # --- エントリーガイド ---
    st.markdown("⸻\n### 🧭 エントリーガイド（総合評価）")
    st.markdown(f"総合スコア：{total_buy:.2f}（買） / {total_sell:.2f}（売）")
    for tf, b, s, w in score_log:
        st.markdown(f"• {tf}：買 {b} × {w} = {b*w:.2f} / 売 {s} × {w} = {s*w:.2f}")

    if total_buy >= 2.4 and total_buy > total_sell:
        st.success("✅ 買いシグナル")
    elif total_sell >= 2.4 and total_sell > total_buy:
        st.warning("✅ 売りシグナル")
    elif abs(total_buy - total_sell) >= 1.0:
        if total_buy > total_sell:
            st.success("✅ 買いシグナル（補足判定）")
        else:
            st.warning("✅ 売りシグナル（補足判定）")
    else:
        st.info("⏸ エントリー見送り")
