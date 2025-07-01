import streamlit as st
import pandas as pd
import numpy as np
import requests

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- UI ---
st.title("FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=2)

# --- 時間足と重み ---
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
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    return df

# --- インジケータ計算 ---
def calc_indicators(df):
    df["SMA_5"] = df["close"].rolling(5).mean()
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Upper"] = df["SMA_20"] + 2 * df["close"].rolling(20).std()
    df["Lower"] = df["SMA_20"] - 2 * df["close"].rolling(20).std()
    df["RCI"] = df["close"].rank().rolling(9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0,1])
    df["ADX"] = abs(df["MACD"] - df["Signal"]).rolling(14).mean()
    df["STD"] = df["close"].rolling(20).std()
    return df

# --- 市場構造判定 ---
def detect_market_structure(last):
    trend = 0
    if last["ADX"] > 25: trend += 1
    elif last["ADX"] < 20: trend -= 1
    if abs(last["SMA_5"] - last["SMA_20"]) / last["close"] > 0.015:
        trend += 1
    else:
        trend -= 1
    if last["STD"] > last["close"] * 0.005:
        trend += 1
    else:
        trend -= 1
    return "トレンド" if trend >= 2 else "レンジ"

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
    return "買い" if buy >= 3 and buy > sell else "売り" if sell >= 3 and sell > buy else "待ち", logs, buy, sell

# --- 高値/安値取得 ---
def get_recent_high_low(df, direction):
    high = df["high"].rolling(20).max().iloc[-2]
    low = df["low"].rolling(20).min().iloc[-2]
    return (high, low) if direction == "買い" else (low, high)
