import streamlit as st
import pandas as pd
import numpy as np
import requests

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- UI ---
st.title("FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=1)

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
    if abs(last["SMA_5"] - last["SMA_20"]) / last["close"] > 0.015: trend += 1
    else: trend -= 1
    if last["STD"] > last["close"] * 0.005: trend += 1
    else: trend -= 1
    return "トレンド" if trend >= 2 else "レンジ"

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
    return "買い" if buy >= 3 and buy > sell else "売り" if sell >= 3 and sell > buy else "待ち", logs, buy, sell

# --- 高値/安値取得 ---
def get_recent_high_low(df, direction):
    hi = df["high"].rolling(20).max().iloc[-2]
    lo = df["low"].rolling(20).min().iloc[-2]
    return (hi, lo) if direction == "買い" else (lo, hi)

# --- トレードプラン（改善版） ---
def suggest_trade_plan(price, atr, decision, tf, df):
    rr_comment = "（ATR）"
    atr_tp = price + atr*1.6 if decision=="買い" else price - atr*1.6
    atr_sl = price - atr*1.0 if decision=="買い" else price + atr*1.0

    use_high_low = False
    if style == "デイトレード" and tf == "4h":
        use_high_low = True
    elif style == "スイング" and tf == "1day":
        use_high_low = True

    if use_high_low:
        hi, lo = get_recent_high_low(df, decision)
        tp = hi*0.997 if decision=="買い" else hi*1.003
        sl = lo*1.003 if decision=="買い" else lo*0.997
        min_sl_distance = 0.001 * price
        if abs(price - sl) < min_sl_distance:
            tp = atr_tp
            sl = atr_sl
            rr_comment = "（ATR自動切替）"
        else:
            rr_comment = "（高値/安値）"
    else:
        tp = atr_tp
        sl = atr_sl

    rr = abs((tp - price) / (sl - price)) if sl != price else 0
    rr = min(rr, 10.0)
    pips_tp = abs(tp - price) * (100 if "JPY" in symbol else 10000)
    pips_sl = abs(sl - price) * (100 if "JPY" in symbol else 10000)
    return price, tp, sl, rr, pips_tp, pips_sl, rr_comment

# --- 実行 ---
if st.button("実行"):
    st.subheader(f"\n📊 通貨ペア：{symbol} | スタイル：{style}")
    timeframes = tf_map[style]
    total_buy = total_sell = 0
    score_log = []
    main_df = None
    main_tf = ""

    st.markdown("### ⏱ 各時間足シグナル詳細\n\n凡例：🟢=買い、🔴=売り、⚪=未達")
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            st.error(f"{tf}のデータ取得に失敗")
            continue
        df = calc_indicators(df)
        sig, logs, b, s = extract_signal(df)
        total_buy += b * tf_weights[tf]
        total_sell += s * tf_weights[tf]
        score_log.append((tf, b, s, tf_weights[tf]))
        st.markdown(f"⏱ {tf} 判定：{sig}（スコア：{max(b,s):.1f}）")
        for log in logs:
            st.markdown(log)
        if tf == timeframes[-1]:
            main_df = df.copy()
            main_tf = tf

    st.markdown("⸻\n### 🧭 エントリーガイド（総合評価）")
    if total_buy >= 2.4 and total_buy > total_sell:
        decision = "買い"
    elif total_sell >= 2.4 and total_sell > total_buy:
        decision = "売り"
    elif abs(total_buy - total_sell) >= 1.0:
        decision = "買い" if total_buy > total_sell else "売り"
    else:
        decision = "待ち"
    st.markdown(f"総合スコア：{total_buy:.2f}（買） / {total_sell:.2f}（売）")
    for tf, b, s, w in score_log:
        st.markdown(f"　• {tf}：買 {b} × {w} = {b*w:.2f} / 売 {s} × {w} = {s*w:.2f}")
    if decision == "買い":
    st.success("✅ 買いシグナル")
elif decision == "売り":
    st.warning("✅ 売りシグナル")
else:
    st.info("⏸ エントリー見送り")

    st.markdown("⸻\n### 🎯 トレードプラン（改善版）")
    price = main_df["close"].iloc[-1]
    atr = main_df["close"].rolling(14).std().iloc[-1]
    entry, tp, sl, rr, ptp, psl, rr_comment = suggest_trade_plan(price, atr, decision, main_tf, main_df)
    if decision != "待ち":
        st.markdown(f"• エントリー価格：{entry:.5f}")
        st.markdown(f"• TP：{tp:.5f}（+{ptp:.0f}pips）")
        st.markdown(f"• SL：{sl:.5f}（−{psl:.0f}pips）")
        st.markdown(f"• リスクリワード：{rr:.2f} {rr_comment}")
    else:
        st.markdown("現在はエントリー待ちです。")
