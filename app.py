import streamlit as st
import pandas as pd
import requests
import ta
from datetime import datetime

# --- APIキー（安全に読み込み） ---
API_KEY = st.secrets["API_KEY"]

# --- データ取得関数（キャッシュ付き） ---
@st.cache_data
def fetch_data(symbol, interval, limit=200):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize={limit}&order=desc"
    response = requests.get(url)
    data = response.json()
    if "values" not in data:
        raise ValueError(f"APIデータ取得エラー: {data}")
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.astype({
        "open": "float",
        "high": "float",
        "low": "float",
        "close": "float",
        "volume": "float"
    })
    df = df.sort_values("time").reset_index(drop=True)
    return df

# --- インジケーター計算 ---
def add_indicators(df):
    df["MACD"] = ta.trend.macd_diff(df["close"])
    df["SMA_fast"] = ta.trend.sma_indicator(df["close"], window=5)
    df["SMA_slow"] = ta.trend.sma_indicator(df["close"], window=20)
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["RCI"] = df["close"].rolling(9).apply(lambda s: pd.Series(s).rank().corr(pd.Series(range(len(s)))))
    return df

# --- 市場構造（トレンド or レンジ）判定 ---
def detect_market_structure(df):
    sma_fast = ta.trend.sma_indicator(df["close"], window=5)
    sma_slow = ta.trend.sma_indicator(df["close"], window=20)
    adx = ta.trend.adx(df["high"], df["low"], df["close"])
    std = df["close"].rolling(window=20).std()
    recent_adx = adx.iloc[-1]
    recent_std = std.iloc[-1]
    if recent_adx > 25 and abs(sma_fast.iloc[-1] - sma_slow.iloc[-1]) > recent_std * 0.5:
        return "トレンド"
    else:
        return "レンジ"

# --- シグナル判定ロジック（トレンドフォロー、逆張り、ローソク、ダウ理論簡易対応） ---
def judge_signal(df, market_type):
    latest = df.iloc[-1]
    result = {"score_buy": 0, "score_sell": 0, "log": [], "structure": market_type}

    # MACDクロス
    if df["MACD"].iloc[-1] > 0:
        result["score_buy"] += 1
        result["log"].append("🟢 MACDゴールデンクロス")
    else:
        result["score_sell"] += 1
        result["log"].append("🔴 MACDデッドクロス")

    # SMA順序
    if df["SMA_fast"].iloc[-1] > df["SMA_slow"].iloc[-1]:
        result["score_buy"] += 1
        result["log"].append("🟢 SMA短期 > 長期")
    else:
        result["score_sell"] += 1
        result["log"].append("🔴 SMA短期 < 長期")

    # BB反発（逆張り）
    close = df["close"].iloc[-1]
    bb_upper = df["BB_upper"].iloc[-1]
    bb_lower = df["BB_lower"].iloc[-1]
    if close < bb_lower * 1.01:
        result["score_buy"] += 1
        result["log"].append("🟢 BB下限反発の可能性")
    elif close > bb_upper * 0.99:
        result["score_sell"] += 1
        result["log"].append("🔴 BB上限反発の可能性")
    else:
        result["log"].append("⚪ BB反発無し")

    # RCI
    rci = df["RCI"].iloc[-1]
    if rci > 0.3:
        result["score_buy"] += 1
        result["log"].append("🟢 RCI上昇傾向")
    elif rci < -0.3:
        result["score_sell"] += 1
        result["log"].append("🔴 RCI下降傾向")
    else:
        result["log"].append("⚪ RCI未達")

    return result

# --- Streamlit画面構成 ---
st.title("📊 FXシグナル判定ツール（軽量・精度重視）")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD", "EUR/JPY"])
style = st.selectbox("トレードスタイル", ["デイトレード", "スイング"])
symbol_api = symbol.replace("/", "")

# 時間足構成（スタイル別）
timeframes = {
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}[style]

weights = {
    "デイトレード": [0.3, 0.3, 0.3],
    "スイング": [0.3, 0.3, 0.4]
}[style]

df_dict, signals = {}, []

# 各時間足でデータ取得と判定
for tf in timeframes:
    df = fetch_data(symbol_api, tf)
    df = add_indicators(df)
    market_type = detect_market_structure(df)
    sig = judge_signal(df, market_type)
    df_dict[tf] = df
    signals.append(sig)

# --- 出力表示 ---
st.markdown(f"### 📊 通貨ペア：{symbol} | スタイル：{style}")
st.markdown("### ⏱ 各時間足シグナル詳細\n凡例：🟢=買い、🔴=売り、⚪=未達")

buy_total = 0
sell_total = 0

for i, tf in enumerate(timeframes):
    sig = signals[i]
    buy_score = sig["score_buy"]
    sell_score = sig["score_sell"]
    buy_total += buy_score * weights[i]
    sell_total += sell_score * weights[i]
    st.markdown(f"⏱ {tf} 判定：{'買い' if buy_score > sell_score else '売り' if sell_score > buy_score else '待ち'}（スコア：{max(buy_score, sell_score)}）")
    st.markdown(f"• 市場判定：{sig['structure']}")
    for line in sig["log"]:
        st.markdown(line)

# --- 総合評価 ---
st.markdown("### 🧭 エントリーガイド（総合評価）")
st.markdown(f"総合スコア：{round(buy_total, 2)}（買） / {round(sell_total, 2)}（売）")
for i, tf in enumerate(timeframes):
    st.markdown(f"• {tf}：買 {signals[i]['score_buy']} × {weights[i]} = {round(signals[i]['score_buy']*weights[i], 2)} / 売 {signals[i]['score_sell']} × {weights[i]} = {round(signals[i]['score_sell']*weights[i], 2)}")

# 判定表示
if buy_total > sell_total and buy_total >= 1.5:
    st.success("✅ 買いシグナル")
elif sell_total > buy_total and sell_total >= 1.5:
    st.warning("✅ 売りシグナル")
else:
    st.info("⏸ エントリー見送り")
