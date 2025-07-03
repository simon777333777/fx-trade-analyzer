import streamlit as st
import pandas as pd
import numpy as np
import requests

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- UI ---
st.title("FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], key="symbol_box")
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], key="style_box")
use_dummy = st.checkbox("ダミーデータを使用する（API制限回避）", key="dummy_checkbox")

# --- ログ表示 ---
st.write(f"✅ 選択通貨ペア: {symbol}")
st.write(f"✅ ダミーモード: {'ON' if use_dummy else 'OFF'}")

# --- 時間足と重み ---
tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}

# --- ダミーデータ生成 ---
def get_dummy_data():
    date_rng = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='H')
    df = pd.DataFrame(date_rng, columns=['datetime'])
    df["open"] = np.random.uniform(100, 110, size=(100,))
    df["high"] = df["open"] + np.random.uniform(0, 1, size=(100,))
    df["low"] = df["open"] - np.random.uniform(0, 1, size=(100,))
    df["close"] = df["open"] + np.random.uniform(-0.5, 0.5, size=(100,))
    df["volume"] = np.random.randint(100, 1000, size=(100,))
    df.set_index("datetime", inplace=True)
    return df

# --- APIからデータ取得 ---
@st.cache_data(show_spinner=False)
def fetch_data(symbol, interval):
    if use_dummy:
        return get_dummy_data()
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        st.error(f"{interval} のデータ取得に失敗: APIデータ取得エラー: {data}")
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    df = df.astype(float)
    return df

# --- インジケータ計算 ---
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
    trend = 0
    if last["ADX"] > 25: trend += 1
    elif last["ADX"] < 20: trend -= 1
    if abs(last["SMA_5"] - last["SMA_20"]) / last["close"] > 0.015: trend += 1
    else: trend -= 1
    if last["STD"] > last["close"] * 0.005: trend += 1
    else: trend -= 1
    return "トレンド" if trend >= 2 else "レンジ"

# --- ダウ理論判定 ---
def detect_dow_theory(df):
    highs = df["high"].rolling(3).apply(lambda x: x[1] > x[0] and x[1] > x[2])
    lows = df["low"].rolling(3).apply(lambda x: x[1] < x[0] and x[1] < x[2])
    return highs.iloc[-5:].sum(), lows.iloc[-5:].sum()

# --- プライスアクション判定 ---
def detect_price_action(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["close"] > prev["open"]:
        return "bullish_engulfing"
    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["close"] < prev["open"]:
        return "bearish_engulfing"
    return None

# --- シグナル抽出 ---
def extract_signal(df):
    last = df.iloc[-1]
    logs = [f"• 市場判定：{detect_market_structure(last)}"]
    buy = sell = 0
    if last["MACD"] > last["Signal"]: buy += 1; logs.append("🟢 MACDゴールデンクロス")
    else: sell += 1; logs.append("🔴 MACDデッドクロス")
    if last["SMA_5"] > last["SMA_20"]: buy += 1; logs.append("🟢 SMA短期 > 長期")
    else: sell += 1; logs.append("🔴 SMA短期 < 長期")
    if last["close"] < last["Lower"]: buy += 1; logs.append("🟢 BB下限反発の可能性")
    elif last["close"] > last["Upper"]: sell += 1; logs.append("🔴 BB上限反発の可能性")
    else: logs.append("⚪ BB反発無し")
    if last["RCI"] > 0.5: buy += 1; logs.append("🟢 RCI上昇傾向")
    elif last["RCI"] < -0.5: sell += 1; logs.append("🔴 RCI下降傾向")
    else: logs.append("⚪ RCI未達")
    # ダウ理論
    hi_sig, lo_sig = detect_dow_theory(df)
    if hi_sig >= 2: buy += 1; logs.append("🟢 高値切り上げ")
    elif lo_sig >= 2: sell += 1; logs.append("🔴 安値切り下げ")
    else: logs.append("⚪ ダウ理論未達")
    # プライスアクション
    pa = detect_price_action(df)
    if pa == "bullish_engulfing": buy += 1; logs.append("🟢 陽線包み足")
    elif pa == "bearish_engulfing": sell += 1; logs.append("🔴 陰線包み足")
    else: logs.append("⚪ プライスアクション未達")
    return ("買い" if buy >= 4 and buy > sell else
            "売り" if sell >= 4 and sell > buy else
            "待ち"), logs, buy, sell

# --- 実行 ---
if st.button("実行"):
    timeframes = tf_map[style]
    total_buy_score = total_sell_score = 0
    score_log = []
    st.subheader(f"📊 通貨ペア：{symbol} | スタイル：{style}")
    st.markdown("### ⏱ 各時間足シグナル詳細\n\n凡例：🟢=買い、🔴=売り、⚪=未達")
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            continue
        df = calc_indicators(df)
        sig, logs, b, s = extract_signal(df)
        weight = tf_weights[tf]
        total_buy_score += b * weight
        total_sell_score += s * weight
        score_log.append((tf, b, s, weight))
        st.markdown(f"⏱ {tf} 判定：{sig}（スコア：{max(b,s):.1f}）")
        for log in logs:
            st.markdown(log)
    st.markdown("⸻\n### 🧭 エントリーガイド（総合評価）")
    st.markdown(f"総合スコア：{total_buy_score:.2f}（買） / {total_sell_score:.2f}（売）")
    for tf, b, s, w in score_log:
        st.markdown(f"　• {tf}：買 {b} × {w} = {b*w:.2f} / 売 {s} × {w} = {s*w:.2f}")
    if total_buy_score >= 2.4 and total_buy_score > total_sell_score:
        st.success("✅ 買いシグナル")
    elif total_sell_score >= 2.4 and total_sell_score > total_buy_score:
        st.warning("✅ 売りシグナル")
    else:
        st.info("⏸ エントリー見送り")
