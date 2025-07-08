import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="FXトレード分析", layout="centered")
API_KEY = st.secrets["API_KEY"]

if st.button("🧹 キャッシュをクリア"):
    st.cache_data.clear()
    st.success("キャッシュをクリアしました。")

st.title("FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=2)
use_dummy = st.checkbox("📦 ダミーデータで実行", value=False)

tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}

def get_dummy_data():
    idx = pd.date_range(end=pd.Timestamp.now(), periods=150, freq="H")
    np.random.seed(0)
    price = np.cumsum(np.random.randn(len(idx))) + 150
    return pd.DataFrame({
        "datetime": idx,
        "open": price + np.random.randn(len(idx)),
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": 1000
    }).set_index("datetime")

@st.cache_data(ttl=300)
def fetch_data(symbol, interval, use_dummy):
    if use_dummy:
        return get_dummy_data()
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
    r = requests.get(url); data = r.json()
    if "values" not in data:
        raise ValueError(f"APIエラー: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    if "volume" not in df.columns:
        df["volume"] = 1000
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

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

def get_hi_lo(df, style):
    cnt = {"スキャルピング":12,"デイトレード":20,"スイング":10}.get(style,20)
    return df["high"].iloc[-cnt:].max(), df["low"].iloc[-cnt:].min()

def suggest_trade_plan(price, atr, decision, df, style):
    hi, lo = get_hi_lo(df, style)
    atr_mult = 1.5
    is_break = False
    if decision == "買い":
        if price > hi:
            tp = price + atr * atr_mult
            sl = price - atr * atr_mult
            is_break = True
        else:
            tp = hi * 0.997
            sl = price - abs(tp - price)/1.7
    elif decision == "売り":
        if price < lo:
            tp = price - atr * atr_mult
            sl = price + atr * atr_mult
            is_break = True
        else:
            tp = lo * 0.997
            sl = price + abs(tp - price)/1.7
    else:
        tp = sl = rr = pips_tp = pips_sl = 0
        # プラン出力なし
        return 0,0,0,0,0,0

    rr = abs((tp - price) / (sl - price)) if sl != price else 0
    pips_tp = abs(tp - price) * (100 if "JPY" in symbol else 10000)
    pips_sl = abs(sl - price) * (100 if "JPY" in symbol else 10000)

    st.markdown("#### 🔍 最新トレードプラン")
    st.markdown(f"• TP：{tp:.5f}（+{pips_tp:.0f}pips）")
    st.markdown(f"• SL：{sl:.5f}（−{pips_sl:.0f}pips）")
    st.markdown(f"• RR比：{rr:.2f}")
    return price, tp, sl, rr, pips_tp, pips_sl

def detect_market_structure(last):
    trend = 0
    if last["ADX"] > 25: trend += 1
    if abs(last["SMA_5"] - last["SMA_20"]) / last["close"] > 0.015: trend += 1
    if last["STD"] > last["close"] * 0.005: trend += 1
    return "トレンド" if trend >= 2 else "レンジ"

def detect_dow(df):
    highs = df["high"].iloc[-3:]
    lows = df["low"].iloc[-3:]
    if highs[2] > highs[1] > highs[0] and lows[2] > lows[1] > lows[0]:
        return "上昇", "🟢 高値切り上げ"
    elif highs[2] < highs[1] < highs[0] and lows[2] < lows[1] < lows[0]:
        return "下降", "🔴 安値切り下げ"
    else:
        return "保ち合い", "⚪ ダウ理論未達"

def detect_price_action(df):
    last2, last1 = df.iloc[-2], df.iloc[-1]
    if last2["close"] < last2["open"] and last1["close"] > last1["open"] and last1["close"] > last2["open"] and last1["open"] < last2["close"]:
        return "🟢 陽線包み足"
    elif last2["close"] > last2["open"] and last1["close"] < last1["open"] and last1["close"] < last2["open"] and last1["open"] > last2["close"]:
        return "🔴 陰線包み足"
    else:
        return "⚪ プライスアクション未達"

def extract_signal(df):
    last = df.iloc[-1]
    market = detect_market_structure(last)
    logs = [f"• 市場判定：{market}"]
    buy = sell = 0
    trend_weight = 2 if market == "トレンド" else 1
    range_weight = 2 if market == "レンジ" else 1

    macd, signal = df["MACD"].iloc[-3:], df["Signal"].iloc[-3:]
    if macd.iloc[-1] > signal.iloc[-1] and macd.is_monotonic_increasing:
        buy += trend_weight; logs.append("🟢 MACDゴールデンクロス + 上昇圧")
    elif macd.iloc[-1] < signal.iloc[-1] and macd.is_monotonic_decreasing:
        sell += trend_weight; logs.append("🔴 MACDデッドクロス + 下降圧")
    else:
        logs.append("⚪ MACD判定微妙")

    sma5, sma20 = df["SMA_5"].iloc[-3:], df["SMA_20"].iloc[-3:]
    if sma5.iloc[-1] > sma20.iloc[-1] and sma5.is_monotonic_increasing:
        buy += trend_weight; logs.append("🟢 SMA短期>長期 + 上昇傾向")
    elif sma5.iloc[-1] < sma20.iloc[-1] and sma5.is_monotonic_decreasing:
        sell += trend_weight; logs.append("🔴 SMA短期<長期 + 下降傾向")
    else:
        logs.append("⚪ SMA判定微妙")

    if last["close"] < last["Lower"]:
        buy += range_weight; logs.append("🟢 BB下限反発")
    elif last["close"] > last["Upper"]:
        sell += range_weight; logs.append("🔴 BB上限反発")
    else:
        logs.append("⚪ BB反発無し")

    if last["RCI"] > 0.5:
        buy += range_weight; logs.append("🟢 RCI上昇")
    elif last["RCI"] < -0.5:
        sell += range_weight; logs.append("🔴 RCI下降")
    else:
        logs.append("⚪ RCI未達")

    _, log_dow = detect_dow(df)
    if "高値" in log_dow:
        buy += 1
    elif "安値" in log_dow:
        sell += 1
    logs.append(log_dow)

    pa = detect_price_action(df)
    if "陽線" in pa:
        buy += 1
    elif "陰線" in pa:
        sell += 1
    logs.append(pa)

    if buy >= 4 and buy > sell:
        return "買い", logs, buy, sell
    elif sell >= 4 and sell > buy:
        return "売り", logs, buy, sell
    else:
        return "待ち", logs, buy, sell

def run_backtest(df, style):
    results = []
    for i in range(100, len(df) - 5):
        sub = df.iloc[i - 50:i + 1].copy()
        sig, logs, _, _ = extract_signal(sub)
        price = sub["close"].iloc[-1]
        atr = sub["close"].rolling(14).std().iloc[-1]
        entry, tp, sl, rr, ptp, psl = suggest_trade_plan(price, atr, sig, sub, style)
        future_high = df["high"].iloc[i + 1:i + 5].max()
        future_low = df["low"].iloc[i + 1:i + 5].min()
        hit = None
        if sig == "買い":
            if future_high >= tp: hit = "win"
            elif future_low <= sl: hit = "lose"
        elif sig == "売り":
            if future_low <= tp: hit = "win"
            elif future_high >= sl: hit = "lose"
        if hit:
            results.append({
                "index": i,
                "date": df.index[i].strftime("%Y-%m-%d %H:%M:%S"),
                "signal": sig,
                "result": hit,
                "tp": tp,
                "sl": sl,
                "entry": price,
                "logs": logs
            })
    if results:
        wins = sum(1 for r in results if r["result"] == "win")
        total = len(results)
        win_rate = wins / total * 100
        avg_pips = np.mean([abs(r["tp"] - r["entry"]) * (100 if "JPY" in symbol else 10000) for r in results])

        st.markdown("### 📈 バックテスト総合結果")
        st.markdown(f"- 勝率：{win_rate:.1f}%（{wins}勝 / {total}回）")
        st.markdown(f"- 平均獲得pips（期待値）：{avg_pips:.1f}")

        with st.expander("📋 バックテスト判定ログ（クリックで展開）", expanded=False):
            st.markdown("| 本数 | 日付 | シグナル | 結果 | TP | SL | エントリー価格 | 判定ログ |")
            st.markdown("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for r in results:
                log_text = ", ".join([l.replace("• ", "") for l in r["logs"]])
                st.markdown(f"| {r['index']} | {r['date']} | {r['signal']} | {r['result']} | {r['tp']:.3f} | {r['sl']:.3f} | {r['entry']:.3f} | {log_text} |")

if st.button("実行"):
    main_df = None
    st.subheader(f"📌 通貨: {symbol} ｜ スタイル: {style}")
    st.markdown("### ⏱ 各TFシグナル")
    total_buy = total_sell = 0
    for tf in tf_map[style]:
        df = fetch_data(symbol, tf, use_dummy)
        df = calc_indicators(df)
        sig, logs, b, s = extract_signal(df)
        wt = tf_weights[tf]
        total_buy += b * wt
        total_sell += s * wt
        st.markdown(f"- {tf}: {sig}（スコア {max(b, s):.1f}）")
        for log in logs:
            st.markdown(f"　{log}")
        if tf == tf_map[style][-1]:  # 最後のTFを代表として保持
            main_df = df

    st.markdown("### 🧭 エントリーガイド")
    diff = total_buy - total_sell
    if total_buy >= 2.4 and total_buy > total_sell:
        decision = "買い"
    elif total_sell >= 2.4 and total_sell > total_buy:
        decision = "売り"
    elif abs(diff) >= 1.0:
        decision = "買い" if diff > 0 else "売り"
    else:
        decision = "待ち"
    st.write(f"総合買:{total_buy:.2f} vs 売:{total_sell:.2f} → **{decision}**")
    if decision != "待ち":
        price = main_df["close"].iloc[-1]
        atr = main_df["close"].rolling(14).std().iloc[-1]
        suggest_trade_plan(price, atr, decision, main_df, style)
    else:
        st.info("エントリー見送り")

    run_backtest(main_df, style)
