import streamlit as st
import pandas as pd
import numpy as np
import requests

API_KEY = st.secrets["API_KEY"]

st.title("FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=0)

tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}

def fetch_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={API_KEY}&format=JSON"
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

def calc_indicators(df):
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["SMA_5"] = df["close"].rolling(window=5).mean()
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Upper"] = df["SMA_20"] + 2 * df["close"].rolling(window=20).std()
    df["Lower"] = df["SMA_20"] - 2 * df["close"].rolling(window=20).std()
    df["RCI"] = df["close"].rank().rolling(window=9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
    df["ADX"] = abs(df["MACD"] - df["Signal"]).rolling(window=14).mean()
    df["STD"] = df["close"].rolling(window=20).std()
    return df

def detect_market_structure(last):
    trend_votes = 0
    range_votes = 0
    if last["ADX"] > 25:
        trend_votes += 1
    elif last["ADX"] < 20:
        range_votes += 1
    sma_diff_ratio = abs(last["SMA_5"] - last["SMA_20"]) / last["close"]
    if sma_diff_ratio > 0.015:
        trend_votes += 1
    else:
        range_votes += 1
    if last["STD"] > last["close"] * 0.005:
        trend_votes += 1
    else:
        range_votes += 1
    return "トレンド" if trend_votes >= 2 else "レンジ"

def extract_signal(df):
    last = df.iloc[-1]
    structure = detect_market_structure(last)
    logs = [f"• 市場判定：{structure}"]
    buy_score = sell_score = 0

    if last["MACD"] > last["Signal"]:
        buy_score += 1
        logs.append("🟢 MACDゴールデンクロス")
    else:
        sell_score += 1
        logs.append("🔴 MACDデッドクロス")

    if last["SMA_5"] > last["SMA_20"]:
        buy_score += 1
        logs.append("🟢 SMA短期 > 長期")
    else:
        sell_score += 1
        logs.append("🔴 SMA短期 < 長期")

    if last["close"] < last["Lower"]:
        buy_score += 1
        logs.append("🟢 BB下限反発の可能性")
    elif last["close"] > last["Upper"]:
        sell_score += 1
        logs.append("🔴 BB上限反発の可能性")
    else:
        logs.append("⚪ BB反発無し")

    if last["RCI"] > 0.5:
        buy_score += 1
        logs.append("🟢 RCI上昇傾向")
    elif last["RCI"] < -0.5:
        sell_score += 1
        logs.append("🔴 RCI下降傾向")
    else:
        logs.append("⚪ RCI未達")

    return buy_score, sell_score, logs

def suggest_trade_plan(price, atr, direction):
    if direction == "買い":
        tp = price + atr * 1.6
        sl = price - atr * 1.0
    elif direction == "売り":
        tp = price - atr * 1.6
        sl = price + atr * 1.0
    else:
        return price, None, None, 0, 0, 0
    rr = abs((tp - price) / (sl - price))
    pips_tp = abs(tp - price) * (100 if "JPY" in symbol else 10000)
    pips_sl = abs(sl - price) * (100 if "JPY" in symbol else 10000)
    return price, tp, sl, rr, pips_tp, pips_sl

def backtest(df):
    log = []
    win = loss = 0
    for i in range(20, len(df)-1):
        sample = df.iloc[:i+1]
        buy_score, sell_score, _ = extract_signal(sample)
        if buy_score < 3 and sell_score < 3:
            continue
        signal = "買い" if buy_score > sell_score else "売り"
        price = sample["close"].iloc[-1]
        atr = sample["close"].rolling(window=14).std().iloc[-1]
        if np.isnan(atr): continue
        entry, tp, sl, rr, ptp, psl = suggest_trade_plan(price, atr, signal)
        next_candle = df.iloc[i+1]
        high = next_candle["high"]
        low = next_candle["low"]
        if signal == "買い":
            if low <= sl and high >= tp:
                result = "損切" if low - sl < high - tp else "利確"
            elif high >= tp:
                result = "利確"
            elif low <= sl:
                result = "損切"
            else:
                result = "-"
        elif signal == "売り":
            if high >= sl and low <= tp:
                result = "損切" if high - sl < tp - low else "利確"
            elif low <= tp:
                result = "利確"
            elif high >= sl:
                result = "損切"
            else:
                result = "-"
        if result == "利確": win += 1
        if result == "損切": loss += 1
        pips = ptp if result == "利確" else (-psl if result == "損切" else 0)
        log.append({
            "No": len(log)+1,
            "日時": sample.index[-1].strftime("%Y-%m-%d %H:%M"),
            "判定": signal,
            "エントリー価格": round(entry, 2),
            "TP価格": round(tp, 2),
            "SL価格": round(sl, 2),
            "結果": result,
            "損益(pips)": int(pips),
        })
    total = win + loss
    win_rate = (win / total) * 100 if total > 0 else 0
    total_pips = sum([l["損益(pips)"] for l in log])
    return win_rate, total_pips, pd.DataFrame(log)
