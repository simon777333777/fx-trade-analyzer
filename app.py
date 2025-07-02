import streamlit as st
import pandas as pd
import requests
import ta
from datetime import datetime

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- 通貨ペアと時間足選択 ---
st.title("📊 FXシグナル分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY"])
style = st.selectbox("トレードスタイル", ["デイトレード", "スイング"])
timeframes = {
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}[style]

# --- データ取得関数 ---
@st.cache_data(show_spinner=False)
def fetch_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}&format=JSON"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.astype(float)
    df = df.sort_index()
    return df

# --- テクニカル指標計算 ---
def calculate_indicators(df):
    df["sma_fast"] = ta.trend.sma_indicator(df["close"], window=10)
    df["sma_slow"] = ta.trend.sma_indicator(df["close"], window=20)
    df["macd"] = ta.trend.macd_diff(df["close"])
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["rci"] = df["close"].rolling(window=9).apply(lambda x: pd.Series(x).rank().corr(pd.Series(range(len(x)))), raw=False)
    return df

# --- シグナル判定 ---
def analyze_signals(df, tf_name):
    df = calculate_indicators(df)
    latest = df.iloc[-1]

    market_structure = "レンジ"  # 現時点はレンジ固定（後でADX等で改善可）
    score_buy, score_sell = 0, 0
    signals = []

    # --- トレンドフォロー（MACD・SMA） ---
    if latest["macd"] > 0:
        score_buy += 1
        signals.append("🟢 MACDゴールデンクロス")
    else:
        score_sell += 1
        signals.append("🔴 MACDデッドクロス")

    if latest["sma_fast"] > latest["sma_slow"]:
        score_buy += 1
        signals.append("🟢 SMA短期 > 長期")
    else:
        score_sell += 1
        signals.append("🔴 SMA短期 < 長期")

    # --- レンジ逆張り型（BB反発） ---
    if latest["close"] < latest["bb_low"] * 1.005:
        score_buy += 1
        signals.append("🟢 BB下限反発の可能性")
    elif latest["close"] > latest["bb_high"] * 0.995:
        score_sell += 1
        signals.append("🔴 BB上限反発の可能性")
    else:
        signals.append("⚪ BB反発無し")

    # --- オシレーター系（RCI） ---
    if latest["rci"] > 0.5:
        score_buy += 1
        signals.append("🟢 RCI上昇傾向")
    elif latest["rci"] < -0.5:
        score_sell += 1
        signals.append("🔴 RCI下降傾向")
    else:
        signals.append("⚪ RCI未達")

    # 判定
    if score_buy >= 3 and score_buy > score_sell:
        decision = "買い"
    elif score_sell >= 3 and score_sell > score_buy:
        decision = "売り"
    else:
        decision = "待ち"

    return {
        "timeframe": tf_name,
        "market": market_structure,
        "decision": decision,
        "score_buy": score_buy,
        "score_sell": score_sell,
        "signals": signals
    }

# --- 総合スコア評価 ---
def summarize_signals(results):
    weight = {"15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}
    total_buy = total_sell = 0
    log = []
    for res in results:
        w = weight[res["timeframe"]]
        buy_score = res["score_buy"] * w
        sell_score = res["score_sell"] * w
        total_buy += buy_score
        total_sell += sell_score
        log.append(f"• {res['timeframe']}: 買 {res['score_buy']} × {w} = {buy_score:.2f} / 売 {res['score_sell']} × {w} = {sell_score:.2f}")
    return total_buy, total_sell, log

# --- 表示実行 ---
results = []
for tf in timeframes:
    df = fetch_data(symbol.replace("/", ""), tf)
    res = analyze_signals(df, tf)
    results.append(res)

# --- 各時間足シグナル詳細表示 ---
st.subheader("⏱ 各時間足シグナル詳細")
for res in results:
    st.markdown(f"**⏱ {res['timeframe']} 判定：{res['decision']}（スコア：{res['score_buy'] if res['decision']=='買い' else res['score_sell']}）**")
    st.markdown(f"• 市場判定：{res['market']}")
    for sig in res["signals"]:
        st.write(sig)

# --- 総合評価 ---
st.markdown("---")
st.subheader("🧭 エントリーガイド（総合評価）")
total_buy, total_sell, logs = summarize_signals(results)
for log in logs:
    st.write(log)
st.markdown(f"**総合スコア：{total_buy:.2f}（買） / {total_sell:.2f}（売）**")

if total_buy >= 2.5 and total_buy > total_sell:
    st.success("✅ 買いシグナル")
elif total_sell >= 2.5 and total_sell > total_buy:
    st.warning("✅ 売りシグナル")
else:
    st.info("⏸ エントリー見送り")
