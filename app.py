import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="FXトレード分析", layout="centered")
API_KEY = st.secrets["API_KEY"]

# --- キャッシュクリアボタン ---
if st.button("🧹 キャッシュをクリア"):
    st.cache_data.clear()
    st.success("キャッシュをクリアしました。")

# --- UI ---
st.title("FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=2)
use_dummy = st.checkbox("📦 ダミーデータで実行", value=False)

# --- 時間足マッピングと重み付け ---
tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}

# --- ダミーデータ生成 ---
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

# --- APIデータ取得 ---
@st.cache_data(ttl=300)
def fetch_data(symbol, interval, use_dummy):
    if use_dummy:
        return get_dummy_data()
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        raise ValueError(f"APIデータ取得エラー: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    if "volume" not in df.columns:
        df["volume"] = 1000
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# --- インジケーター計算 ---
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

# --- 価格/構造/ダウ理論/PA判定 ---
def detect_market_structure(last):
    trend = 0
    if last["ADX"] > 25: trend += 1
    if abs(last["SMA_5"] - last["SMA_20"]) / last["close"] > 0.015: trend += 1
    if last["STD"] > last["close"] * 0.005: trend += 1
    return "トレンド" if trend >= 2 else "レンジ"

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
    market = detect_market_structure(last)
    logs = [f"• 市場判定：{market}"]
    buy = sell = 0
    tw = 2 if market == "トレンド" else 1
    rw = 2 if market == "レンジ" else 1

    macd = df["MACD"].iloc[-3:]
    sig = df["Signal"].iloc[-3:]
    if macd.iloc[-1] > sig.iloc[-1] and macd.is_monotonic_increasing:
        buy += tw; logs.append("🟢 MACDゴールデンクロス + 上昇傾向")
    elif macd.iloc[-1] < sig.iloc[-1] and macd.is_monotonic_decreasing:
        sell += tw; logs.append("🔴 MACDデッドクロス + 下降傾向")
    else:
        logs.append("⚪ MACD判定微妙")

    sma5 = df["SMA_5"].iloc[-3:]; sma20 = df["SMA_20"].iloc[-3:]
    if sma5.iloc[-1] > sma20.iloc[-1] and sma5.is_monotonic_increasing:
        buy += tw; logs.append("🟢 SMA短期>長期 + 上昇傾向")
    elif sma5.iloc[-1] < sma20.iloc[-1] and sma5.is_monotonic_decreasing:
        sell += tw; logs.append("🔴 SMA短期<長期 + 下降傾向")
    else:
        logs.append("⚪ SMA判定微妙")

    if last["close"] < last["Lower"]:
        buy += rw; logs.append("🟢 BB下限反発")
    elif last["close"] > last["Upper"]:
        sell += rw; logs.append("🔴 BB上限反発")
    else:
        logs.append("⚪ BB反発無し")

    if last["RCI"] > 0.5:
        buy += rw; logs.append("🟢 RCI上昇")
    elif last["RCI"] < -0.5:
        sell += rw; logs.append("🔴 RCI下降")
    else:
        logs.append("⚪ RCI未達")

    _, log_dow = detect_dow(df)
    buy += "高値" in log_dow
    sell += "安値" in log_dow
    logs.append(log_dow)

    pa = detect_price_action(df)
    buy += "陽線" in pa
    sell += "陰線" in pa
    logs.append(pa)

    if buy >= 4 and buy > sell:
        return "買い", logs, buy, sell
    elif sell >= 4 and sell > buy:
        return "売り", logs, buy, sell
    else:
        return "待ち", logs, buy, sell

def suggest_trade_plan(price, atr, decision, df, style, show_detail=True):
    hi = df["high"].iloc[-20:].max()
    lo = df["low"].iloc[-20:].min()
    atr_mult = 1.5
    is_break = False
    if decision == "買い":
        if price > hi:
            tp = price + atr * atr_mult
            sl = price - atr * atr_mult
            is_break = True
        else:
            tp = hi * 0.997
            sl = price - abs(tp - price) / 1.7
    elif decision == "売り":
        if price < lo:
            tp = price - atr * atr_mult
            sl = price + atr * atr_mult
            is_break = True
        else:
            tp = lo * 0.997
            sl = price + abs(tp - price) / 1.7
    else:
        return price, 0, 0, 0, 0, 0
    rr = abs((tp - price) / (sl - price)) if sl != price else 0
    pips_tp = abs(tp - price) * (100 if "JPY" in symbol else 10000)
    pips_sl = abs(sl - price) * (100 if "JPY" in symbol else 10000)
    if show_detail:
        st.markdown("#### 🔍 トレードプラン詳細")
        st.markdown(f"• ATR: `{atr:.5f}`, 倍率: `{atr_mult}`, ブレイク: `{is_break}`")
        st.markdown(f"• TP: `{tp:.5f}` (+{pips_tp:.0f}pips), SL: `{sl:.5f}` (-{pips_sl:.0f}pips)")
        st.markdown(f"• RR比: `{rr:.2f}`")
    return price, tp, sl, rr, pips_tp, pips_sl

def run_backtest(df, style):
    results = []
    for i in range(100, len(df) - 5):
        sub = df.iloc[i - 50:i + 1].copy()
        sig, logs, _, _ = extract_signal(sub)
        price = sub["close"].iloc[-1]
        atr = sub["close"].rolling(14).std().iloc[-1]
        entry, tp, sl, rr, ptp, psl = suggest_trade_plan(price, atr, sig, sub, style, show_detail=False)
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

        st.markdown("### 📈 バックテスト総合結果")
        st.markdown(f"• 勝率：{win_rate:.1f}%（{wins}勝 / {total}回）")
        st.markdown(f"• 平均獲得pips（期待値）：{avg_pips:.1f}")

        with st.expander("📋 バックテスト判定ログ（クリックで展開）", expanded=False):
            st.dataframe(df_result)

# --- 実行 ---
if st.button("実行"):
    st.subheader(f"📌 通貨: {symbol} ｜ スタイル: {style}")
    st.markdown("### ⏱ 各時間足シグナル")
    total_buy = total_sell = 0
    main_df = None
    for tf in tf_map[style]:
        df = fetch_data(symbol, tf, use_dummy)
        df = calc_indicators(df)
        sig, logs, b, s = extract_signal(df)
        weight = tf_weights[tf]
        total_buy += b * weight
        total_sell += s * weight
        st.markdown(f"#### 🕒 {tf} 足: **{sig}**（スコア: {max(b, s):.1f}）")
        for log in logs:
            st.markdown(f"・{log}")
        if tf == tf_map[style][1]:
            main_df = df.copy()
    st.markdown("### 🧭 総合エントリー判断")
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
    if decision != "待ち" and main_df is not None:
        price = main_df["close"].iloc[-1]
        atr = main_df["close"].rolling(14).std().iloc[-1]
        suggest_trade_plan(price, atr, decision, main_df, style)
    else:
        st.info("📭 明確なエントリーシグナルがないため、トレードプランは表示しません。")
    if main_df is not None:
        run_backtest(main_df, style)
