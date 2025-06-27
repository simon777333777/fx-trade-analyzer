import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# --- APIキー設定 ---
API_KEY = st.secrets["API_KEY"]

# --- ユーザーインターフェース ---
st.title("FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"], index=1)

# --- 時間足設定 ---
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

# --- インジケータ計算 ---
def calc_indicators(df):
    df["SMA_5"] = df["close"].rolling(window=5).mean()
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Upper"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
    df["Lower"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()
    df["STD"] = df["close"].rolling(20).std()
    df["RCI"] = df["close"].rank().rolling(window=9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0,1])
    df["TR"] = np.maximum.reduce([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    ])
    df["ATR"] = df["TR"].rolling(window=14).mean()
    df["+DM"] = np.where((df["high"] - df["high"].shift()) > (df["low"].shift() - df["low"]),
                          np.maximum(df["high"] - df["high"].shift(), 0), 0)
    df["-DM"] = np.where((df["low"].shift() - df["low"]) > (df["high"] - df["high"].shift()),
                          np.maximum(df["low"].shift() - df["low"], 0), 0)
    df["+DI"] = 100 * (df["+DM"].ewm(span=14).mean() / df["ATR"])
    df["-DI"] = 100 * (df["-DM"].ewm(span=14).mean() / df["ATR"])
    df["ADX"] = 100 * abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])
    return df

# --- トレンド / レンジ判定 ---
def detect_market_structure(df):
    last = df.iloc[-1]
    trend_votes = 0
    range_votes = 0

    # 1. ADX
    if last["ADX"] > 25:
        trend_votes += 1
    elif last["ADX"] < 20:
        range_votes += 1

    # 2. SMA乖離
    sma_diff_ratio = abs(last["SMA_5"] - last["SMA_20"]) / last["close"]
    if sma_diff_ratio > 0.015:
        trend_votes += 1
    else:
        range_votes += 1

    # 3. 標準偏差
    if last["STD"] > (last["close"] * 0.005):
        trend_votes += 1
    else:
        range_votes += 1

    return "トレンド" if trend_votes >= 2 else "レンジ"

# --- 売買個別スコア判定 ---
def extract_signal(df):
    last = df.iloc[-1]
    logs = []
    buy_score = 0
    sell_score = 0

    if last["MACD"] > last["Signal"]:
        buy_score += 1
        logs.append("✅ MACDゴールデンクロス")
    else:
        sell_score += 1
        logs.append("✅ MACDデッドクロス")

    if last["SMA_5"] > last["SMA_20"]:
        buy_score += 1
        logs.append("✅ SMA短期 > 長期")
    else:
        sell_score += 1
        logs.append("✅ SMA短期 < 長期")

    if last["close"] < last["Lower"]:
        buy_score += 1
        logs.append("✅ BB下限反発")
    elif last["close"] > last["Upper"]:
        sell_score += 1
        logs.append("✅ BB上限反発")
    else:
        logs.append("❌ BB反発無し")

    if last["RCI"] > 0.5:
        buy_score += 1
        logs.append("✅ RCI上昇傾向")
    elif last["RCI"] < -0.5:
        sell_score += 1
        logs.append("✅ RCI下降傾向")
    else:
        logs.append("❌ RCI未達")

    if buy_score >= 3:
        decision = "買い"
        score = buy_score
    elif sell_score >= 3:
        decision = "売り"
        score = sell_score
    else:
        decision = "待ち"
        score = max(buy_score, sell_score)

    return decision, logs, score
