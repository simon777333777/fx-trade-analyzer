# --- ライブラリ読み込み ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキーの取得（st.secrets使用） ---
API_KEY = st.secrets["API_KEY"]

# --- 通貨ペアとスタイルの選択 ---
st.title("\ud83d\udcb1 FX\u30c8\u30ec\u30fc\u30c9\u5206\u6790")
symbol = st.selectbox("\u901a\u8ca8\u30da\u30a2\u3092\u9078\u629e", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("\u30c8\u30ec\u30fc\u30c9\u30b9\u30bf\u30a4\u30eb\u3092\u9078\u629e", ["\u30b9\u30a4\u30f3\u30b0", "\u30c7\u30a4\u30c8\u30ec\u30fc\u30c9", "\u30b9\u30ad\u30e3\u30eb\u30d4\u30f3\u30b0"], index=1)

# --- スタイルに応じた時間足と重み ---
tf_map = {
    "\u30b9\u30ad\u30e3\u30eb\u30d4\u30f3\u30b0": ["5min", "15min", "1h"],
    "\u30c7\u30a4\u30c8\u30ec\u30fc\u30c9": ["15min", "1h", "4h"],
    "\u30b9\u30a4\u30f3\u30b0": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.4, "1day": 0.5}
timeframes = tf_map[style]

# --- データ取得関数 ---
def fetch_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()

# --- インジケータ計算 ---
def calc_indicators(df):
    df = df.copy()
    df["SMA_5"] = df["close"].rolling(5).mean()
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Upper"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
    df["Lower"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()
    df["RCI"] = df["close"].rank().rolling(9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
    return df.dropna()

# --- スコア独立判定 ---
def extract_signal(df):
    last = df.iloc[-1]
    guide = []
    buy_score = 0
    sell_score = 0

    # MACD
    if last["MACD"] > last["Signal"]:
        buy_score += 1
        guide.append("\u2705 MACD\u30b4\u30fc\u30eb\u30c7\u30f3\u30af\u30ed\u30b9")
    else:
        sell_score += 1
        guide.append("\u274c MACD\u30c7\u30c3\u30c9\u30af\u30ed\u30b9")

    # SMA
    if last["SMA_5"] > last["SMA_20"]:
        buy_score += 1
        guide.append("\u2705 SMA\u77ed\u671f > \u9577\u671f")
    else:
        sell_score += 0
        guide.append("\u274c SMA\u6761\u4ef6\u672a\u9054")

    # BB
    if last["close"] < last["Lower"]:
        buy_score += 1
        guide.append("\u2705 BB\u4e0b\u9650\u53cd\u767a\u306e\u53ef\u80fd\u6027")
    elif last["close"] > last["Upper"]:
        sell_score += 1
        guide.append("\u2705 BB\u4e0a\u9650\u53cd\u8ee2\u306e\u53ef\u80fd\u6027")
    else:
        guide.append("\u274c BB\u53cd\u767a\u7121\u3057")

    # RCI
    if last["RCI"] > 0.5:
        buy_score += 1
        guide.append("\u2705 RCI\u4e0a\u6607\u50be\u5411")
    elif last["RCI"] < -0.5:
        sell_score += 1
        guide.append("\u2705 RCI\u4e0b\u964d\u50be\u5411")
    else:
        guide.append("\u274c RCI\u672a\u9054")

    if buy_score >= 3:
        signal = "\u8cb7\u3044"
        final_score = buy_score
    elif sell_score >= 3:
        signal = "\u58f2\u308a"
        final_score = sell_score
    else:
        signal = "\u5f85\u3061"
        final_score = max(buy_score, sell_score)

    return signal, guide, final_score

# --- 実行ボタン ---
if st.button("\u5b9f\u884c"):
    st.markdown(f"## \ud83d\udcb1 \u901a\u8ca8\u30da\u30a2\uff1a{symbol} | \u30b9\u30bf\u30a4\u30eb\uff1a{style}")
    st.markdown("\u23b3")
    st.markdown("### \u23f1 \u5404\u6642\u9593\u8db3\u30b7\u30b0\u30ca\u30eb\u8a73\u7d30")

    final_scores = []
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            st.warning(f"{tf} \u306e\u30c7\u30fc\u30bf\u306a\u3057")
            continue
        df = calc_indicators(df)
        sig, guide, score = extract_signal(df)
        final_scores.append(score * tf_weights.get(tf, 0.3))

        st.markdown(f"**\u23f1 {tf} \u5224\u5b9a\uff1a{sig}（\u30b9\u30b3\u30a2\uff1a{score}.0）**")
        for g in guide:
            st.markdown(f"\t\u2022 {g}")

    st.markdown("\n\u23b3")

    # 総合評価（旧形式出力コメント含む）はこの後に続けてください（略）
