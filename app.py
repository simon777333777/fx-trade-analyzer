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

# --- トレードプラン ---
def suggest_trade_plan(price, atr, decision, tf, df):
    rr_comment = "（ATR）"
    if style == "スキャルピング":
        tp = price + atr*1.6 if decision=="買い" else price - atr*1.6
        sl = price - atr*1.0 if decision=="買い" else price + atr*1.0
    elif style == "デイトレード" and tf == "4h":
        hi, lo = get_recent_high_low(df, decision)
        tp = hi*0.997 if decision=="買い" else hi*1.003
        sl = lo*1.003 if decision=="買い" else lo*0.997
        rr_comment = "（高値/安値）"
    elif style == "スイング" and tf == "1day":
        hi, lo = get_recent_high_low(df, decision)
        tp = hi*0.997 if decision=="買い" else hi*1.003
        sl = lo*1.003 if decision=="買い" else lo*0.997
        rr_comment = "（高値/安値）"
    else:
        tp = price + atr*1.6 if decision=="買い" else price - atr*1.6
        sl = price - atr*1.0 if decision=="買い" else price + atr*1.0
    rr = abs((tp-price)/(sl-price)) if sl != price else 0
    pips_tp = abs(tp - price) * (100 if "JPY" in symbol else 10000)
    pips_sl = abs(sl - price) * (100 if "JPY" in symbol else 10000)
    return price, tp, sl, rr, pips_tp, pips_sl, rr_comment

# --- バックテスト（順序考慮）---
def backtest(df, tf, df_tf):
    logs = []
    win = loss = 0
    for i in range(20, len(df)-1):
        sample = df.iloc[:i+1]
        signal, _, b, s = extract_signal(sample)
        price = sample["close"].iloc[-1]
        atr = sample["close"].rolling(14).std().iloc[-1]
        if np.isnan(atr): continue
        entry, tp, sl, rr, ptp, psl, _ = suggest_trade_plan(price, atr, signal, tf, df_tf)
        hi = df["high"].iloc[i+1]
        lo = df["low"].iloc[i+1]
        result = "-"
        if signal == "買い":
            if hi >= tp and lo <= sl:
                result = "利確" if tp-price < price-sl else "損切"
            elif hi >= tp:
                result = "利確"
            elif lo <= sl:
                result = "損切"
        elif signal == "売り":
            if lo <= tp and hi >= sl:
                result = "利確" if price-tp < sl-price else "損切"
            elif lo <= tp:
                result = "利確"
            elif hi >= sl:
                result = "損切"
        if result == "利確": win += 1
        if result == "損切": loss += 1
        pips = ptp if result == "利確" else -psl if result == "損切" else 0
        logs.append({
            "No": len(logs)+1,
            "日時": sample.index[-1].strftime("%Y-%m-%d %H:%M"),
            "判定": signal,
            "エントリー価格": round(entry,2) if signal!="待ち" else "-",
            "TP価格": round(tp,2) if signal!="待ち" else "-",
            "SL価格": round(sl,2) if signal!="待ち" else "-",
            "結果": result,
            "損益(pips)": int(pips) if signal!="待ち" else "-"
        })
    total = win + loss
    win_rate = (win / total * 100) if total else 0
    total_pips = sum([l["損益(pips)"] for l in logs if isinstance(l["損益(pips)"], int)])
    return win_rate, total_pips, pd.DataFrame(logs)

# --- 実行ボタン ---
if st.button("実行"):
    st.subheader(f"\n📊 通貨ペア：{symbol} | スタイル：{style}")
    timeframes = tf_map[style]
    total_buy_score = total_sell_score = 0
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
        weight = tf_weights[tf]
        total_buy_score += b * weight
        total_sell_score += s * weight
        score_log.append((tf, b, s, weight))
        st.markdown(f"⏱ {tf} 判定：{sig}（スコア：{max(b,s):.1f}）")
        for log in logs:
            st.markdown(log)
        if tf == timeframes[-1]:
            main_df = df.copy()
            main_tf = tf

    st.markdown("⸻")
    st.markdown("### 🧭 エントリーガイド（総合評価）")
    # 判定 + 補完
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
        st.markdown(f"　• {tf}：買 {b} × {w} = {b*w:.2f} / 売 {s} × {w} = {s*w:.2f}")
    if decision == "買い":
        st.success("✅ 買いシグナルが優勢です")
    elif decision == "売り":
        st.warning("✅ 売りシグナルが優勢です")
    else:
        st.info("⏸ 現在はエントリー待機中です")

    st.markdown("⸻\n### 🎯 トレードプラン（想定）")
    price = main_df["close"].iloc[-1]
    atr = main_df["close"].rolling(14).std().iloc[-1]
    entry, tp, sl, rr, ptp, psl, comment = suggest_trade_plan(price, atr, decision, main_tf, main_df)
    if decision != "待ち":
        st.markdown(f"• エントリー価格：{entry:.2f}")
        st.markdown(f"• TP：{tp:.2f}（+{ptp:.0f}pips）")
        st.markdown(f"• SL：{sl:.2f}（−{psl:.0f}pips）")
        st.markdown(f"• リスクリワード：{rr:.2f} {comment}")
    else:
        st.markdown("現在はエントリー待ちです。")

    st.markdown("⸻\n### 📈 バックテスト結果")
    win_rate, total_pips, df_bt = backtest(main_df, main_tf, main_df)
    if not df_bt.empty:
        st.write(f"勝率：{win_rate:.1f}%（{int(win_rate)}勝 / {len(df_bt)}件）")
        st.write(f"合計損益：{total_pips:+.0f} pips")
        st.dataframe(df_bt)
    else:
        st.warning("バックテスト結果が0件です")
