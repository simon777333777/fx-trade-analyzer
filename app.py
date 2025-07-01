import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta

# --- API設定 ---
API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://api.twelvedata.com/time_series"

# --- 時間足と重み ---
TIMEFRAMES = {
    "デイトレード": [("15min", 0.3), ("1h", 0.3), ("4h", 0.3)],
    "スイング": [("1h", 0.3), ("4h", 0.3), ("1day", 0.4)]
}

# --- 補助関数：データ取得 ---
@st.cache_data(show_spinner=False)
def fetch_data(symbol: str, interval: str):
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": 200
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df = df.astype(float)
    return df

# --- インジケーター判定 ---
def judge_indicators(df):
    result = {"判定": "待ち", "スコア": 0, "詳細": [], "市場": "レンジ"}
    
    # インジケーター追加
    macd = ta.trend.MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    
    df["sma_fast"] = ta.trend.sma_indicator(df["close"], window=7)
    df["sma_slow"] = ta.trend.sma_indicator(df["close"], window=25)
    
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_low"] = bb.bollinger_lband()
    df["bb_high"] = bb.bollinger_hband()
    
    rci = df["close"].rolling(9).apply(lambda x: pd.Series(x).rank().corr(pd.Series(range(len(x)))))
    df["rci"] = rci

    # 最終行のみ使用
    last = df.iloc[-1]
    score = 0
    details = []
    
    # トレンド判定（ADX + SMA 乖離）
    sma_trend = abs(df["sma_fast"] - df["sma_slow"]).iloc[-1]
    std = df["close"].rolling(14).std().iloc[-1]
    trend = "トレンド" if sma_trend > std * 0.5 else "レンジ"
    result["市場"] = trend

    # MACD
    if last["macd_diff"] > 0:
        score += 1
        details.append("🟢 MACDゴールデンクロス")
    else:
        score += 1
        details.append("🔴 MACDデッドクロス")
        
    # SMA
    if last["sma_fast"] > last["sma_slow"]:
        score += 1
        details.append("🟢 SMA短期 > 長期")
    else:
        score += 1
        details.append("🔴 SMA短期 < 長期")

    # BB反発（レンジ用）
    if last["close"] < last["bb_low"]:
        details.append("🟢 BB下限反発の可能性")
    elif last["close"] > last["bb_high"]:
        details.append("🔴 BB上限反発の可能性")
    else:
        details.append("⚪ BB反発無し")

    # RCI
    if last["rci"] > 0.5:
        score += 1
        details.append("🟢 RCI上昇傾向")
    elif last["rci"] < -0.5:
        score += 1
        details.append("🔴 RCI下降傾向")
    else:
        details.append("⚪ RCI未達")

    # 判定
    result["スコア"] = score
    if score >= 3:
        result["判定"] = "買い" if last["macd_diff"] > 0 else "売り"

    result["詳細"] = details
    return result

# --- 総合シグナル評価 ---
def synthesize_signals(signals, style):
    buy_score, sell_score = 0, 0
    log = []
    for tf, weight in TIMEFRAMES[style]:
        sig = signals[tf]
        b = 0
        s = 0
        for item in sig["詳細"]:
            if "🟢" in item:
                b += 1
            elif "🔴" in item:
                s += 1
        buy_score += b * weight
        sell_score += s * weight
        log.append(f"• {tf}: 買 {b} × {weight} = {b * weight:.2f} / 売 {s} × {weight} = {s * weight:.2f}")
    return buy_score, sell_score, log

# --- Streamlit UI ---
st.title("📊 FX シグナル判定ツール")
symbol = st.selectbox("通貨ペアを選択", ["EUR/USD", "USD/JPY", "GBP/JPY", "AUD/JPY"])
style = st.selectbox("トレードスタイル", list(TIMEFRAMES.keys()))

# --- 各時間足のシグナル判定 ---
signals = {}
for tf, _ in TIMEFRAMES[style]:
    df = fetch_data(symbol.replace("/", ""), tf)
    sig = judge_indicators(df)
    signals[tf] = sig

# --- 表示 ---
st.markdown(f"### 通貨ペア：{symbol} | スタイル：{style}")
st.subheader("⏱ 各時間足シグナル詳細")
st.caption("凡例：🟢=買い、🔴=売り、⚪=未達")

for tf, sig in signals.items():
    st.write(f"⏱ {tf} 判定：{sig['判定']}（スコア：{sig['スコア']}）")
    st.write(f"• 市場判定：{sig['市場']}")
    for item in sig["詳細"]:
        st.write(item)

# --- 総合評価 ---
st.subheader("🧭 エントリーガイド（総合評価）")
buy_score, sell_score, logs = synthesize_signals(signals, style)
st.write(f"総合スコア：{buy_score:.2f}（買） / {sell_score:.2f}（売）")
for log in logs:
    st.write(log)

# --- 判定出力 ---
if buy_score >= 2.0 and buy_score > sell_score:
    st.success("✅ 買いシグナル")
elif sell_score >= 2.0 and sell_score > buy_score:
    st.warning("✅ 売りシグナル")
else:
    st.info("⏸ エントリー見送り")
