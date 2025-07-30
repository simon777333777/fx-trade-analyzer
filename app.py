import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="RCI主軸FXトレード分析", layout="centered")
API_KEY = st.secrets["API_KEY"]

st.title("📈 RCI主軸FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=0)
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=1)
use_dummy = st.checkbox("📦 ダミーデータで実行", value=False)

tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}

def get_dummy_data():
    idx = pd.date_range(end=pd.Timestamp.now(), periods=150, freq="H")
    price = np.cumsum(np.random.randn(len(idx))) + 150
    return pd.DataFrame({
        "datetime": idx,
        "open": price + np.random.randn(len(idx)),
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": 1000
    }).set_index("datetime")

@st.cache_data(ttl=300)
def fetch_data(symbol, interval, use_dummy):
    if use_dummy:
        return get_dummy_data()
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        raise ValueError(f"APIエラー: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def calc_indicators(df):
    for period in [9, 26, 52]:
        df[f"RCI_{period}"] = df["close"].rolling(period).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if x.notna().all() else np.nan,
            raw=False
        )
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["SMA_9"] = df["close"].rolling(9).mean()
    df["SMA_26"] = df["close"].rolling(26).mean()
    df["STD"] = df["close"].rolling(20).std()
    return df

def rci_based_signal(df):
    last = df.iloc[-1]
    score = 0
    logs = []

    # 短期RCI
    if last["RCI_9"] >= 0.8:
        logs.append("• 短期RCI（9）：+80以上 → 強い上昇トレンド")
        score += 2
    else:
        logs.append("• 短期RCI（9）：未達")

    # 中期RCI
    if df["RCI_26"].iloc[-1] > df["RCI_26"].iloc[-2]:
        logs.append("• 中期RCI（26）：上昇中 → 支持")
        score += 1
    else:
        logs.append("• 中期RCI（26）：下降傾向")

    # 長期RCI
    if last["RCI_52"] >= 0.5:
        logs.append("• 長期RCI（52）：+50超 → 中長期も上昇傾向")
        score += 1
    else:
        logs.append("• 長期RCI（52）：未達")

    # MACD
    if last["MACD"] > last["Signal"] and df["MACD"].diff().iloc[-1] > 0:
        logs.append("• MACD：ゴールデンクロス直後（買い支持）")
        score += 1
    else:
        logs.append("• MACD：判定弱")

    # SMA位置
    if last["close"] > last["SMA_9"] and last["close"] > last["SMA_26"]:
        logs.append("• SMA：ローソク足が短期・中期SMAより上（順行）")
        score += 1
    else:
        logs.append("• SMA：順行でない")

    # ボラティリティ
    if 0 < last["STD"] < df["STD"].mean() * 1.5:
        logs.append("• ボラティリティ：安定上昇（過熱感なし）")
        score += 1
    else:
        logs.append("• ボラティリティ：高騰 or 低迷")

    return score, logs

if st.button("実行"):
    for tf in tf_map[style]:
        st.subheader(f"⏱ 時間足：{tf}")
        df = fetch_data(symbol, tf, use_dummy)
        df = calc_indicators(df)
        score, logs = rci_based_signal(df)
        decision = "🟢 エントリー判定：買い" if score >= 6 else "⚪ 判定保留"

        st.markdown(f"**{decision}**")
        for log in logs:
            st.markdown(log)
        st.markdown(f"**信頼度スコア：{score} / 7点**")
