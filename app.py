import streamlit as st
import pandas as pd
import numpy as np
import requests

# --- APIキー設定 ---
API_KEY = st.secrets["API_KEY"]

# --- ユーザーインターフェース ---
st.title("FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=0)
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=0)

# --- 時間足と重み設定 ---
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
    df = df.astype(float)
    return df

# --- インジケーター計算 ---
def calc_indicators(df):
    df = df.copy()
    df["SMA_5"] = df["close"].rolling(window=5).mean()
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Upper"] = df["close"].rolling(window=20).mean() + 2 * df["close"].rolling(window=20).std()
    df["Lower"] = df["close"].rolling(window=20).mean() - 2 * df["close"].rolling(window=20).std()
    df["STD"] = df["close"].rolling(window=20).std()
    df["ADX"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()  # 簡易的ADX
    df["RCI"] = df["close"].rank().rolling(window=9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
    return df

# --- 相場構造判定 ---
def detect_market_structure(df):
    last = df.iloc[-1]
    trend_votes = 0
    range_votes = 0

    if last["ADX"] > 25:
        trend_votes += 1
    elif last["ADX"] < 20:
        range_votes += 1

    sma_diff_ratio = abs(last["SMA_5"] - last["SMA_20"]) / last["close"]
    if sma_diff_ratio > 0.015:
        trend_votes += 1
    else:
        range_votes += 1

    if last["STD"] > (last["close"] * 0.005):
        trend_votes += 1
    else:
        range_votes += 1

    return "トレンド" if trend_votes >= 2 else "レンジ"

# --- シグナル判定 ---
def extract_signal(df):
    last = df.iloc[-1]
    logs = []
    score_buy = score_sell = 0

    # MACD
    if last["MACD"] > last["Signal"]:
        score_buy += 1
        logs.append("🟢 MACDゴールデンクロス")
    else:
        score_sell += 1
        logs.append("🔴 MACDデッドクロス")

    # SMA
    if last["SMA_5"] > last["SMA_20"]:
        score_buy += 1
        logs.append("🟢 SMA短期 > 長期")
    else:
        score_sell += 1
        logs.append("🔴 SMA短期 < 長期")

    # BB
    if last["close"] < last["Lower"]:
        score_buy += 1
        logs.append("🟢 BB下限反発")
    elif last["close"] > last["Upper"]:
        score_sell += 1
        logs.append("🔴 BB上限反発")
    else:
        logs.append("⚪ BB反発無し")

    # RCI
    if last["RCI"] > 0.5:
        score_buy += 1
        logs.append("🟢 RCI上昇傾向")
    elif last["RCI"] < -0.5:
        score_sell += 1
        logs.append("🔴 RCI下降傾向")
    else:
        logs.append("⚪ RCI未達")

    final = "買い" if score_buy >= 3 else "売り" if score_sell >= 3 else "待ち"
    return final, logs, max(score_buy, score_sell), score_buy, score_sell

# --- トレードプラン生成 ---
def suggest_trade_plan(price, atr, direction):
    if direction == "買い":
        tp = price + atr * 1.6
        sl = price - atr * 1.0
    elif direction == "売り":
        tp = price - atr * 1.6
        sl = price + atr * 1.0
    else:
        return price, None, None, 0, 0, 0
    rr = abs((tp - price) / (sl - price))
    pips_tp = abs(tp - price) * (100 if "JPY" in symbol else 10000)
    pips_sl = abs(sl - price) * (100 if "JPY" in symbol else 10000)
    return price, tp, sl, rr, pips_tp, pips_sl

# --- バックテスト ---
def backtest(df):
    log = []
    win = 0
    loss = 0
    for i in range(20, len(df)-1):
        sample = df.iloc[:i+1]
        signal, _, score, _, _ = extract_signal(sample)
        price = sample["close"].iloc[-1]
        atr = sample["close"].rolling(window=14).std().iloc[-1]
        if np.isnan(atr):
            continue
        entry, tp, sl, rr, ptp, psl = suggest_trade_plan(price, atr, signal)
        next_price = df["close"].iloc[i+1]
        if signal == "買い":
            result = "利確" if next_price >= tp else ("損切" if next_price <= sl else "-")
        elif signal == "売り":
            result = "利確" if next_price <= tp else ("損切" if next_price >= sl else "-")
        else:
            result = "-"
        if result == "利確":
            win += 1
        elif result == "損切":
            loss += 1
        pips = ptp if result == "利確" else (-psl if result == "損切" else 0)
        log.append({
            "No": len(log)+1,
            "日時": sample.index[-1].strftime("%Y-%m-%d %H:%M"),
            "判定": signal,
            "エントリー価格": round(entry, 2) if signal != "待ち" else "-",
            "TP価格": round(tp, 2) if signal != "待ち" else "-",
            "SL価格": round(sl, 2) if signal != "待ち" else "-",
            "結果": result if signal != "待ち" else "-",
            "損益(pips)": int(pips) if signal != "待ち" else "-",
        })
    total = win + loss
    win_rate = (win / total) * 100 if total > 0 else 0
    total_pips = sum([l["損益(pips)"] for l in log if isinstance(l["損益(pips)"], int)])
    return win_rate, total_pips, pd.DataFrame(log)

# --- 実行 ---
if st.button("実行"):
    st.markdown(f"### \n💱 通貨ペア：{symbol} | スタイル：{style}\n\n⸻")
    timeframes = tf_map[style]
    final_score = []
    score_details = []
    df_all = None

    st.markdown("### ⏱ 各時間足シグナル詳細\n\n凡例：🟢=買い条件達成、🔴=売り条件達成、⚪=未達")
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            st.error(f"データ取得失敗：{tf}")
            continue
        df = calc_indicators(df)
        structure = detect_market_structure(df)
        signal, logs, score, buy_score, sell_score = extract_signal(df)
        final_score.append((buy_score - sell_score) * tf_weights.get(tf, 0.3))
        st.markdown(f"\n⏱ {tf} 判定：{signal}（スコア：{score:.1f}）")
        st.markdown(f"• • 市場判定：{structure}")
        for g in logs:
            st.markdown(f"• {g}")
        if tf == timeframes[1]:
            df_all = df.copy()

    st.markdown("\n⸻")
    avg_score = sum(final_score)
    decision = "買い" if avg_score >= 1.2 else ("売り" if avg_score <= -1.2 else "待ち")

    st.markdown("### 🧭 エントリーガイド（総合評価）")
    if decision == "買い":
        st.write("✅ 複数の時間足で買い優勢。押し目完了後の上昇トレンド入りに注目")
    elif decision == "売り":
        st.write("✅ 売り優勢。戻り売りの好機、下落トレンド継続の可能性")
    else:
        st.write("現在は明確な買い/売りシグナルが不足しているため、エントリーは控えめに")

    st.markdown("\n⸻")
    price = df_all["close"].iloc[-1]
    atr = df_all["close"].rolling(window=14).std().iloc[-1]
    entry, tp, sl, rr, ptp, psl = suggest_trade_plan(price, atr, decision)
    win_rate, total_pips, bt_df = backtest(df_all)

    st.markdown("### 🎯 トレードプラン（想定）")
    if decision != "待ち":
        st.write(f"\t•\tエントリーレート：{entry:.2f}")
        st.write(f"\t•\t指値（利確）：{tp:.2f}（+{int(ptp)} pips）")
        st.write(f"\t•\t逆指値（損切）：{sl:.2f}（−{int(psl)} pips）")
        st.write(f"\t•\tリスクリワード比：{rr:.2f}")
        st.write(f"\t•\t想定勝率：{win_rate:.1f}%")
    else:
        st.write("現在はエントリー待ちです。")

    st.markdown("\n### 📈 バックテスト結果（最大100件）")
    if len(bt_df) > 0:
        st.write(f"勝率：{win_rate:.1f}%（{int(win_rate)}勝 / {len(bt_df)}件）")
        st.write(f"合計損益：{total_pips:+.0f} pips")
        st.dataframe(bt_df)
    else:
        st.write("⚠ バックテスト結果が0件です。ATRが計算できないか、TP/SLが未達成かもしれません")
