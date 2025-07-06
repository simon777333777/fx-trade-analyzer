import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="FXトレード分析", layout="centered")

API_KEY = st.secrets["API_KEY"]

# --- UI ---
st.title("FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=2)
use_dummy = st.checkbox("📦 ダミーデータで実行", value=False)

# --- 時間足と重み ---
tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}

# --- ダミーデータ生成 ---
def get_dummy_data():
    idx = pd.date_range(end=pd.Timestamp.now(), periods=150, freq="H")
    np.random.seed(42)
    price = np.cumsum(np.random.randn(len(idx))) + 180
    return pd.DataFrame({
        "datetime": idx,
        "open": price + np.random.randn(len(idx)),
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": 1000
    }).set_index("datetime")

# --- API取得 ---
@st.cache_data(ttl=300)
def fetch_data(symbol, interval):
    if use_dummy:
        return get_dummy_data()
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        st.error(f"❌ APIエラー発生：{data.get('message', '不明なエラー')}")
        raise ValueError(f"APIデータ取得エラー: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df.set_index("datetime", inplace=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df

# --- インジケータ算出 ---
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
    if abs(last["SMA_5"] - last["SMA_20"]) / last["close"] > 0.015: trend += 1
    if last["STD"] > last["close"] * 0.005: trend += 1
    return "トレンド" if trend >= 2 else "レンジ"

# --- シグナル抽出 ---
def extract_signal(df):
    last = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    market = detect_market_structure(last)
    logs = [f"• 市場判定：{market}"]
    buy = sell = 0

    trend_w = 2 if market == "トレンド" else 1
    range_w = 2 if market == "レンジ" else 1

    # MACD
    if prev1["MACD"] < prev1["Signal"] and last["MACD"] > last["Signal"]:
        buy += trend_w
        logs.append("🟢 MACDゴールデンクロス")
    elif prev1["MACD"] > prev1["Signal"] and last["MACD"] < last["Signal"]:
        sell += trend_w
        logs.append("🔴 MACDデッドクロス")
    else:
        logs.append("⚪ MACD未達")

    # SMA
    if last["SMA_5"] > last["SMA_20"]:
        buy += trend_w
        logs.append("🟢 SMA短期 > 長期")
    else:
        sell += trend_w
        logs.append("🔴 SMA短期 < 長期")

    # BB
    if last["close"] < last["Lower"]:
        buy += range_w
        logs.append("🟢 BB下限反発の可能性")
    elif last["close"] > last["Upper"]:
        sell += range_w
        logs.append("🔴 BB上限反発の可能性")
    else:
        logs.append("⚪ BB反発無し")

    # RCI
    if last["RCI"] > 0.5:
        buy += range_w
        logs.append("🟢 RCI上昇傾向")
    elif last["RCI"] < -0.5:
        sell += range_w
        logs.append("🔴 RCI下降傾向")
    else:
        logs.append("⚪ RCI未達")

    # プライスアクション：包み足
    if last["close"] < last["open"] and last["high"] > prev1["high"] and last["low"] < prev1["low"]:
        sell += 1
        logs.append("🔴 陰線包み足")
    elif last["close"] > last["open"] and last["high"] > prev1["high"] and last["low"] < prev1["low"]:
        buy += 1
        logs.append("🟢 陽線包み足")
    else:
        logs.append("⚪ プライスアクション未達")

    # ダウ理論
    if last["high"] > prev1["high"] > prev2["high"]:
        buy += 1
        logs.append("🟢 高値切り上げ")
    elif last["low"] < prev1["low"] < prev2["low"]:
        sell += 1
        logs.append("🔴 安値切り下げ")
    else:
        logs.append("⚪ ダウ理論未達")

    return ("買い" if buy >= 4 and buy > sell else
            "売り" if sell >= 4 and sell > buy else
            "待ち"), logs, buy, sell

# --- トレードプラン ---
def suggest_trade_plan(price, atr, decision, tf, df):
    hi = df["high"].rolling(20).max().iloc[-2]
    lo = df["low"].rolling(20).min().iloc[-2]
    if decision == "買い":
        tp = hi * 0.997
        sl = lo * 1.003
    else:
        tp = lo * 0.997
        sl = hi * 1.003
    rr = abs((tp - price) / (sl - price)) if sl != price else 0
    factor = 100 if "JPY" in symbol else 10000
    pips_tp = abs(tp - price) * factor
    pips_sl = abs(sl - price) * factor
    return price, tp, sl, rr, pips_tp, pips_sl

# --- 実行処理 ---
if st.button("実行"):
    timeframes = tf_map[style]
    total_buy_score = total_sell_score = 0
    score_log = []
    main_df = None
    main_tf = ""

    st.subheader(f"📊 通貨ペア：{symbol} | スタイル：{style}")
    st.markdown("### ⏱ 各時間足シグナル詳細\n\n凡例：🟢=買い、🔴=売り、⚪=未達")

    for tf in timeframes:
        try:
            df = fetch_data(symbol, tf)
            df = calc_indicators(df)
        except Exception as e:
            st.error(f"{tf} のデータ取得に失敗: {e}")
            continue

        sig, logs, b, s = extract_signal(df)
        weight = tf_weights[tf]
        total_buy_score += b * weight
        total_sell_score += s * weight
        score_log.append((tf, b, s, weight))
        st.markdown(f"⏱ {tf} 判定：{sig}（スコア：{max(b,s):.1f}）")
        for log in logs:
            st.markdown(log)
        main_df = df
        main_tf = tf

    st.markdown("⸻\n### 🧭 エントリーガイド（総合評価）")
    if total_buy_score >= 2.4 and total_buy_score > total_sell_score:
        decision = "買い"
    elif total_sell_score >= 2.4 and total_sell_score > total_buy_score:
        decision = "売り"
    elif abs(total_buy_score - total_sell_score) >= 1.0:
        decision = "買い" if total_buy_score > total_sell_score else "売り"
    else:
        decision = "待ち"

    st.markdown(f"総合スコア：{total_buy_score:.2f}（買） / {total_sell_score:.2f}（売）")
    for tf, b, s, w in score_log:
        st.markdown(f"• {tf}：買 {b} × {w} = {b*w:.2f} / 売 {s} × {w} = {s*w:.2f}")

    if decision == "買い":
        st.success("✅ 買いシグナル")
    elif decision == "売り":
        st.warning("✅ 売りシグナル")
    else:
        st.info("⏸ エントリー見送り")

    st.markdown("⸻\n### 🎯 トレードプラン（想定）")
    if decision != "待ち":
        price = main_df["close"].iloc[-1]
        atr = main_df["close"].rolling(14).std().iloc[-1]
        entry, tp, sl, rr, ptp, psl = suggest_trade_plan(price, atr, decision, main_tf, main_df)
        st.markdown(f"• エントリー価格：{entry:.5f}")
        st.markdown(f"• TP：{tp:.5f}（+{ptp:.0f}pips）")
        st.markdown(f"• SL：{sl:.5f}（−{psl:.0f}pips）")
        st.markdown(f"• リスクリワード：{rr:.2f}（高値/安値）")
    else:
        st.markdown("現在はエントリー見送りです。")
