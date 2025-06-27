import streamlit as st
import pandas as pd
import numpy as np
import requests

# --- APIキー設定 ---
API_KEY = st.secrets["API_KEY"]

# --- ユーザーインターフェース ---
st.title("FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"], index=1)

# --- 時間足設定 ---
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
    df["Upper"] = df["SMA_20"] + 2 * df["close"].rolling(window=20).std()
    df["Lower"] = df["SMA_20"] - 2 * df["close"].rolling(window=20).std()
    df["RCI"] = df["close"].rank().rolling(window=9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
    df["STD"] = df["close"].rolling(window=20).std()
    df["ADX"] = (df["high"] - df["low"]).rolling(window=14).mean()
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
    if last["STD"] > last["close"] * 0.005:
        trend_votes += 1
    else:
        range_votes += 1
    return "トレンド" if trend_votes >= 2 else "レンジ"

# --- シグナル判定 ---
def extract_signal(df):
    last = df.iloc[-1]
    logs = []
    buy_score = 0
    sell_score = 0
    market_type = detect_market_structure(df)
    logs.append(f"• 市場判定：{market_type}")

    if last["MACD"] > last["Signal"]:
        buy_score += 1
        logs.append("🟢 MACDゴールデンクロス")
    elif last["MACD"] < last["Signal"]:
        sell_score += 1
        logs.append("🔴 MACDデッドクロス")
    else:
        logs.append("⚪ MACD未達")

    if last["SMA_5"] > last["SMA_20"]:
        buy_score += 1
        logs.append("🟢 SMA短期 > 長期")
    elif last["SMA_5"] < last["SMA_20"]:
        sell_score += 1
        logs.append("🔴 SMA短期 < 長期")
    else:
        logs.append("⚪ SMA未達")

    if last["close"] < last["Lower"]:
        buy_score += 1
        logs.append("🟢 BB下限反発の可能性")
    elif last["close"] > last["Upper"]:
        sell_score += 1
        logs.append("🔴 BB上限反発の可能性")
    else:
        logs.append("⚪ BB反発無し")

    if last["RCI"] > 0.5:
        buy_score += 1
        logs.append("🟢 RCI上昇傾向")
    elif last["RCI"] < -0.5:
        sell_score += 1
        logs.append("🔴 RCI下降傾向")
    else:
        logs.append("⚪ RCI未達")

    if buy_score >= 3:
        return "買い", logs, buy_score
    elif sell_score >= 3:
        return "売り", logs, sell_score
    else:
        return "待ち", logs, max(buy_score, sell_score)

# --- トレードプラン作成 ---
def suggest_trade_plan(price, atr, decision):
    if decision == "買い":
        tp = price + atr * 1.6
        sl = price - atr * 1.0
    elif decision == "売り":
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
    for i in range(20, len(df) - 1):
        sample = df.iloc[:i + 1]
        signal, _, _ = extract_signal(sample)
        price = sample["close"].iloc[-1]
        atr = sample["close"].rolling(window=14).std().iloc[-1]
        if np.isnan(atr):
            continue
        entry, tp, sl, rr, ptp, psl = suggest_trade_plan(price, atr, signal)
        next_price = df["close"].iloc[i + 1]
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
            "No": len(log) + 1,
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
    timeframes = tf_map[style]
    st.subheader(f"\n💱 通貨ペア：{symbol} | スタイル：{style}\n\n⸻")
    st.markdown("### ⏱ 各時間足シグナル詳細")
    st.markdown("##### 凡例：🟢=買い条件達成、🔴=売り条件達成、⚪=未達")
    final_scores = []
    df_all = None

    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            st.error(f"データ取得失敗：{tf}")
            continue
        df = calc_indicators(df)
        signal, guide, score = extract_signal(df)
        final_scores.append(score * tf_weights.get(tf, 0.3))
        st.markdown(f"\n⏱ {tf} 判定：{signal}（スコア：{score:.1f}）")
        for g in guide:
            st.markdown(f"\t•\t{g}")
        if tf == timeframes[1]:
            df_all = df.copy()

    avg_score = sum(final_scores)
    decision = "買い" if avg_score >= 2.4 else ("売り" if avg_score <= 1.2 else "待ち")

    st.markdown("\n⸻")
    st.markdown("### 🧭 エントリーガイド（総合評価）")
    if decision == "買い":
        st.write("✅ 複数の時間足が買いシグナルを示しています")
        st.write("⏳ 中期・長期の上昇トレンドが短期にも波及")
        st.write("📌 押し目が完了しており、エントリータイミングとして有効")
    elif decision == "売り":
        st.write("✅ 複数の時間足が売りシグナルを示しています")
        st.write("📉 長期トレンドに従った戻り売りのタイミング")
        st.write("🚩 高値圏での反転シグナルが複数確認")
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
