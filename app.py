import streamlit as st
import pandas as pd
import numpy as np
import requests

API_KEY = st.secrets["API_KEY"]
st.title("FXトレード分析ツール")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイル", ["スキャルピング", "デイトレード", "スイング"], index=1)

tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}
tf_lower = {"5min": "1min", "15min": "5min", "1h": "15min", "4h": "1h", "1day": "4h"}

def fetch_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df

def calc_indicators(df):
    df["SMA_5"] = df["close"].rolling(5).mean()
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["MACD"] = df["close"].ewm(12).mean() - df["close"].ewm(26).mean()
    df["Signal"] = df["MACD"].ewm(9).mean()
    df["Upper"] = df["SMA_20"] + 2 * df["close"].rolling(20).std()
    df["Lower"] = df["SMA_20"] - 2 * df["close"].rolling(20).std()
    df["RCI"] = df["close"].rank().rolling(9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
    df["ADX"] = abs(df["MACD"] - df["Signal"]).rolling(14).mean()
    df["STD"] = df["close"].rolling(20).std()
    return df

def detect_market_structure(last):
    trend = 0
    if last["ADX"] > 25: trend += 1
    elif last["ADX"] < 20: trend -= 1
    if abs(last["SMA_5"] - last["SMA_20"]) / last["close"] > 0.015: trend += 1
    else: trend -= 1
    if last["STD"] > last["close"] * 0.005: trend += 1
    else: trend -= 1
    return "トレンド" if trend >= 2 else "レンジ"

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

def get_recent_high_low(df, direction):
    high = df["high"].rolling(20).max().iloc[-2]
    low = df["low"].rolling(20).min().iloc[-2]
    return (high, low) if direction == "買い" else (low, high)

def suggest_trade_plan(price, atr, decision, tf, df):
    if (style == "デイトレード" and tf == "4h") or (style == "スイング" and tf == "1day"):
        hi, lo = get_recent_high_low(df, decision)
        tp = hi * 0.997 if decision == "買い" else hi * 1.003
        sl = lo * 1.003 if decision == "買い" else lo * 0.997
        rr_comment = "（高値/安値）"
    else:
        tp = price + atr * 1.6 if decision == "買い" else price - atr * 1.6
        sl = price - atr * 1.0 if decision == "買い" else price + atr * 1.0
        rr_comment = "（ATR）"
    rr = abs((tp - price) / (sl - price)) if sl != price else 0
    pips_tp = abs(tp - price) * (100 if "JPY" in symbol else 10000)
    pips_sl = abs(sl - price) * (100 if "JPY" in symbol else 10000)
    return price, tp, sl, rr, pips_tp, pips_sl, rr_comment

def backtest(df_main, tf_main, df_ref):
    df_sub = fetch_data(symbol, tf_lower[tf_main])
    if df_sub is None:
        return 0.0, 0.0, pd.DataFrame()
    df_sub = calc_indicators(df_sub)
    logs = []
    win = loss = 0
    for i in range(20, len(df_main) - 1):
        sample = df_main.iloc[:i + 1]
        signal, _, b, s = extract_signal(sample)
        price = sample["close"].iloc[-1]
        atr = sample["close"].rolling(14).std().iloc[-1]
        if np.isnan(atr): continue
        entry, tp, sl, rr, ptp, psl, _ = suggest_trade_plan(price, atr, signal, tf_main, df_ref)
        entry_time = sample.index[-1]
        subset = df_sub[df_sub.index > entry_time]
        result = "-"
        for _, row in subset.iterrows():
            if signal == "買い":
                if row["low"] <= sl: result = "損切"; break
                if row["high"] >= tp: result = "利確"; break
            elif signal == "売り":
                if row["high"] >= sl: result = "損切"; break
                if row["low"] <= tp: result = "利確"; break
        if result == "利確": win += 1
        if result == "損切": loss += 1
        pips = ptp if result == "利確" else -psl if result == "損切" else 0
        logs.append({
            "No": len(logs) + 1,
            "日時": entry_time.strftime("%Y-%m-%d %H:%M"),
            "判定": signal,
            "エントリー価格": round(entry, 5),
            "TP価格": round(tp, 5),
            "SL価格": round(sl, 5),
            "結果": result,
            "損益(pips)": int(pips)
        })
    total = win + loss
    win_rate = (win / total * 100) if total else 0
    total_pips = sum([l["損益(pips)"] for l in logs])
    return win_rate, total_pips, pd.DataFrame(logs)

if st.button("実行"):
    st.subheader(f"📊 通貨ペア：{symbol} | スタイル：{style}")
    timeframes = tf_map[style]
    total_buy = total_sell = 0
    score_log = []
    main_df = main_tf = None

    st.markdown("### ⏱ 各時間足シグナル詳細\n\n凡例：🟢=買い、🔴=売り、⚪=未達")
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            st.error(f"{tf} データ取得失敗")
            continue
        df = calc_indicators(df)
        sig, logs, b, s = extract_signal(df)
        weight = tf_weights[tf]
        total_buy += b * weight
        total_sell += s * weight
        score_log.append((tf, b, s, weight))
        st.markdown(f"⏱ {tf} 判定：{sig}（スコア：{max(b,s):.1f}）")
        for log in logs: st.markdown(log)
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

    st.markdown("⸻\n### 🎯 トレードプラン（想定）")
    price = main_df["close"].iloc[-1]
    atr = main_df["close"].rolling(14).std().iloc[-1]
    entry, tp, sl, rr, ptp, psl, comment = suggest_trade_plan(price, atr, decision, main_tf, main_df)
    if decision != "待ち":
        st.markdown(f"• エントリー価格：{entry:.5f}")
        st.markdown(f"• TP：{tp:.5f}（+{ptp:.0f}pips）")
        st.markdown(f"• SL：{sl:.5f}（−{psl:.0f}pips）")
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
