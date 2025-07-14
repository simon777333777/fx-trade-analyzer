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
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        raise ValueError(f"APIエラー: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 1000
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
    df["HH"] = df["high"].rolling(20).max()
    df["LL"] = df["low"].rolling(20).min()
    return df

def detect_dow(df):
    highs = df["high"].iloc[-3:]
    lows = df["low"].iloc[-3:]
    is_hh = highs[2] > highs[1] > highs[0]
    is_ll = lows[2] < lows[1] < lows[0]
    if is_hh and is_ll:
        return "保ち合い", "⚪ ダウ理論：保ち合い"
    elif is_hh:
        return "上昇", "🟢 高値切り上げ"
    elif is_ll:
        return "下降", "🔴 安値切り下げ"
    else:
        return "不明", "⚪ ダウ理論未達"

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
    market = detect_market_structure(df)
    logs = [f"• 市場判定：{market}"]

    buy = sell = 0
    tw = 2 if market == "トレンド" else 1
    rw = 2 if market == "レンジ" else 1

    # --- ADXとSTD（勢い） ---
    adx_score = 0
    if last["ADX"] > 20:
        adx_score += 1
    else:
        logs.append("⚪ ADX<20（勢い不足）")

    if last["STD"] > df["close"].mean() * 0.0015:
        adx_score += 1
    else:
        logs.append("⚪ STD低（ボラ不足）")

    if adx_score == 0:
        logs.append("⚠ 勢い・ボラともに不足 → 信頼度低")

    # --- MACD ---
    macd = df["MACD"].iloc[-3:]
    sig = df["Signal"].iloc[-3:]
    if macd.iloc[-1] > sig.iloc[-1] and any(macd.diff().iloc[1:] > 0):
        buy += tw
        logs.append("🟢 MACDゴールデンクロス傾向")
    elif macd.iloc[-1] < sig.iloc[-1] and any(macd.diff().iloc[1:] < 0):
        sell += tw
        logs.append("🔴 MACDデッドクロス傾向")
    else:
        logs.append("⚪ MACD判定微妙")

    # --- SMA ---
    sma5 = df["SMA_5"].iloc[-3:]
    sma20 = df["SMA_20"].iloc[-3:]
    if sma5.iloc[-1] > sma20.iloc[-1] and any(sma5.diff().iloc[1:] > 0):
        buy += tw
        logs.append("🟢 SMA短期>長期")
    elif sma5.iloc[-1] < sma20.iloc[-1] and any(sma5.diff().iloc[1:] < 0):
        sell += tw
        logs.append("🔴 SMA短期<長期")
    else:
        logs.append("⚪ SMA判定微妙")

    # --- BB反発 ---
    if last["close"] < last["Lower"]:
        buy += rw
        logs.append("🟢 BB下限反発")
    elif last["close"] > last["Upper"]:
        sell += rw
        logs.append("🔴 BB上限反発")
    else:
        logs.append("⚪ BB反発無し")

    # --- RCI ---
    if last["RCI"] > 0.4:
        buy += rw
        logs.append("🟢 RCI上昇")
    elif last["RCI"] < -0.4:
        sell += rw
        logs.append("🔴 RCI下降")
    else:
        logs.append("⚪ RCI未達")

    # --- ダウ理論 + プライスアクション ---
    _, log_dow = detect_dow(df)
    pa = detect_price_action(df)
    if "高値" in log_dow or "陽線" in pa:
        buy += 1
    if "安値" in log_dow or "陰線" in pa:
        sell += 1
    logs.append(log_dow)
    logs.append(pa)

    # --- RR・ボラ補正（減点処理） ---
    rr_penalty = 0
    if last["STD"] < df["close"].mean() * 0.0015:
        rr_penalty += 1
    if last["ADX"] < 20:
        rr_penalty += 1

    buy = max(buy - rr_penalty * 0.5, 0)
    sell = max(sell - rr_penalty * 0.5, 0)

    # --- 非対称エントリー基準 ---
    decision = "待ち"
    if buy >= 4 and buy > sell:
        decision = "買い"
    elif sell >= 5 and sell > buy:
        decision = "売り"

    score = max(buy, sell)
    logs.append(f"🧠 信頼度スコア: {score:.1f}")

    return decision, logs, buy, sell

def detect_market_structure(df):
    last = df.iloc[-1]
    trend_score = 0
    if last["ADX"] > 25:
        trend_score += 1
    if abs(last["SMA_5"] - last["SMA_20"]) / last["close"] > 0.015:
        trend_score += 1
    if last["close"] > df["HH"].iloc[-1] * 0.995:
        trend_score += 1
    if last["close"] < df["LL"].iloc[-1] * 1.005:
        trend_score += 1
    return "トレンド" if trend_score >= 2 else "レンジ"

def suggest_trade_plan(price, atr, decision, df, style, show_detail=True):
    hi = df["high"].iloc[-20:].max()
    lo = df["low"].iloc[-20:].min()
    std = df["STD"].iloc[-1]
    tp = sl = rr = pips_tp = pips_sl = 0
    is_break = False

    if decision == "買い":
        if price > hi:
            tp = price + std * 2
            sl = price - std * 1.2
            is_break = True
        else:
            tp = hi * 0.997
            sl = price - abs(tp - price) / 1.7

    elif decision == "売り":
        if price < lo:
            tp = price - std * 2
            sl = price + std * 1.2
            is_break = True
        else:
            tp = lo * 0.997
            sl = price + abs(tp - price) / 1.7

    else:
        return price, 0, 0, 0, 0, 0

    # --- このチェックを緩和 or 削除 ---
    if not (sl < price < tp):
        st.warning("⚠ エントリー価格がTP/SLの範囲に収まっていませんが、参考としてトレードプランを表示します。")

    rr = abs((tp - price) / (sl - price)) if sl != price else 0
    pips_tp = abs(tp - price) * (100 if "JPY" in symbol else 10000)
    pips_sl = abs(sl - price) * (100 if "JPY" in symbol else 10000)

    if show_detail:
        st.markdown("### 🔍 トレードプラン")
        st.markdown(f"• TP: `{tp:.3f}` (+{pips_tp:.0f}pips), SL: `{sl:.3f}` (-{pips_sl:.0f}pips)")
        st.markdown(f"• RR比: `{rr:.2f}`")
        if rr < 1.0:
            st.warning("⚠ RR（リスクリワード比）が1.0未満のため、リスクに対してリターンが見合っていません。非推奨トレードです。")

    return price, tp, sl, rr, pips_tp, pips_sl

def run_backtest(df, style):
    results = []
    for i in range(100, len(df) - 5):
        sub = df.iloc[i - 50:i + 1].copy()
        sig, logs, _, _ = extract_signal(sub)
        price = sub["close"].iloc[-1]
        atr = sub["close"].rolling(14).std().iloc[-1]
        entry, tp, sl, rr, ptp, psl = suggest_trade_plan(price, atr, sig, sub, style, show_detail=False)
        if tp == 0 or sl == 0:
            continue

        future_high = df["high"].iloc[i + 1:i + 5].max()
        future_low = df["low"].iloc[i + 1:i + 5].min()
        hit = None
        if sig == "買い":
            if future_high >= tp:
                hit = "win"
            elif future_low <= sl:
                hit = "lose"
        elif sig == "売り":
            if future_low <= tp:
                hit = "win"
            elif future_high >= sl:
                hit = "lose"

        if sig != "待ち":
            results.append({
                "No": i,
                "日付": df.index[i].strftime("%Y-%m-%d %H:%M:%S"),
                "シグナル": sig,
                "結果": hit if hit else "-",
                "TP": round(tp, 3),
                "SL": round(sl, 3),
                "エントリー価格": round(price, 3),
                "判定ログ": ", ".join(logs),
                "損益pips": ptp if hit == "win" else (-psl if hit == "lose" else 0)
            })

    if results:
        df_result = pd.DataFrame(results).sort_values("No", ascending=False)
        wins = df_result["結果"].value_counts().get("win", 0)
        total = df_result["結果"].isin(["win", "lose"]).sum()
        win_rate = (wins / total * 100) if total > 0 else 0
        avg_pips = df_result[df_result["結果"].isin(["win", "lose"])]["損益pips"].mean()

        st.markdown("### 📊 バックテスト結果")
        st.markdown(f"• 勝率：{win_rate:.1f}%（{wins}勝 / {total}回）")
        st.markdown(f"• 平均pips：{avg_pips:.1f}")

        with st.expander("📋 詳細ログ"):
            st.dataframe(df_result)

# ----------------- Streamlit 実行処理 ------------------

if st.button("実行"):
    st.subheader(f"📌 通貨: {symbol} ｜ スタイル: {style}")
    st.markdown("### ⏱ 各時間足シグナル")

    total_buy = total_sell = 0
    decisions = []
    main_df = None

    for tf in tf_map[style]:
        df = fetch_data(symbol, tf, use_dummy)
        df = calc_indicators(df)
        sig, logs, b, s = extract_signal(df)
        weight = tf_weights[tf]
        total_buy += b * weight
        total_sell += s * weight
        decisions.append(sig)

        st.markdown(f"#### 🕒 {tf}足: **{sig}**（スコア: {max(b, s):.1f}）")
        for log in logs:
            st.markdown(f"・{log}")

        if tf == tf_map[style][1]:
            main_df = df.copy()

    if any(d in ["買い", "売り"] for d in decisions):
        st.markdown("### 🧭 総合判断")
        diff = total_buy - total_sell
        if total_buy >= 2.4 and total_buy > total_sell:
            decision = "買い"
        elif total_sell >= 2.4 and total_sell > total_buy:
            decision = "売り"
        elif abs(diff) >= 1.0:
            decision = "買い" if diff > 0 else "売り"
        else:
            decision = "待ち"

        st.markdown(f"• 買いスコア: `{total_buy:.2f}`, 売りスコア: `{total_sell:.2f}`")
        st.success(f"✅ エントリー判定：**{decision}**")

        if main_df is not None:
            price = main_df["close"].iloc[-1]
            atr = main_df["close"].rolling(14).std().iloc[-1]
            suggest_trade_plan(price, atr, decision, main_df, style)
    else:
        st.info("📭 シグナルが不明確なため、トレードプランは表示されません。")

    if main_df is not None:
        run_backtest(main_df, style)



