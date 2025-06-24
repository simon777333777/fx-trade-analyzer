# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキーの指定 ---
API_KEY = st.secrets["API_KEY"]

# --- ユーザーインターフェース ---
st.title("FXトレード分析ツール（改良バックテスト対応）")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"], index=0)

if st.button("実行"):

    tf_map = {
        "スキャルピング": ["5min", "15min", "1h"],
        "デイトレード": ["15min", "1h", "4h"],
        "スイング": ["1h", "4h", "1day"]
    }
    tf_weights = {
        "5min": 0.2, "15min": 0.3, "1h": 0.2, "4h": 0.3, "1day": 0.5
    }
    timeframes = tf_map[style]

    def fetch_data(symbol, interval):
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=200&apikey={API_KEY}"
        r = requests.get(url)
        data = r.json()
        if "values" not in data:
            st.error(f"データ取得失敗: {symbol} - {interval}")
            return None
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df = df.sort_index()
        df["close"] = df["close"].astype(float)
        return df

    def calc_indicators(df):
        df = df.copy()
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["SMA_5"] = df["close"].rolling(window=5).mean()
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["Signal"] = df["MACD"].ewm(span=9).mean()
        df["Upper"] = df["close"].rolling(window=20).mean() + 2 * df["close"].rolling(window=20).std()
        df["Lower"] = df["close"].rolling(window=20).mean() - 2 * df["close"].rolling(window=20).std()
        df["RCI"] = df["close"].rank().rolling(window=9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
        return df

    def extract_signal(df):
        guide = []
        score = 0
        last = df.iloc[-1]
        if last["MACD"] > last["Signal"]:
            score += 1
            guide.append("✅ MACDゴールデンクロス")
        else:
            guide.append("❌ MACD未達")

        if last["SMA_5"] > last["SMA_20"]:
            score += 1
            guide.append("✅ SMA短期 > 長期")
        else:
            guide.append("❌ SMA条件未達")

        if last["close"] < last["Lower"]:
            score += 1
            guide.append("✅ BB下限反発の可能性")
        else:
            guide.append("❌ BB反発無し")

        if last["RCI"] > 0.5:
            score += 1
            guide.append("✅ RCI上昇傾向")
        else:
            guide.append("❌ RCI未達")

        if score >= 3:
            signal = "買い"
        elif score <= 1:
            signal = "売り"
        else:
            signal = "待ち"
        return signal, guide, score / 4

    def suggest_trade_plan(df, direction):
        price = df["close"].iloc[-1]
        pips_factor = 100 if "/JPY" in symbol else 10000
        sl = tp = None
        if direction == "買い":
            sl = price * 0.9975
            tp = price * 1.004
        elif direction == "売り":
            sl = price * 1.0025
            tp = price * 0.996
        rr = abs((tp - price) / (sl - price)) if sl and tp else 0
        pips_tp = abs(tp - price) * pips_factor if tp else 0
        pips_sl = abs(sl - price) * pips_factor if sl else 0
        return price, tp, sl, rr, (pips_tp, pips_sl)

    def backtest(df, direction):
        log = []
        pips_factor = 100 if "/JPY" in symbol else 10000
        for i in range(30, len(df) - 30):
            segment = df.iloc[:i+1]
            segment = calc_indicators(segment)
            sig, guide, _ = extract_signal(segment)
            now = df.iloc[i]
            if sig != direction:
                log.append({"日時": df.index[i], "レート": now["close"], "判定": sig, "結果": "無視", "備考": ", ".join(guide)})
                continue

            price = now["close"]
            if direction == "買い":
                sl = price * 0.9975
                tp = price * 1.004
            else:
                sl = price * 1.0025
                tp = price * 0.996

            future = df["close"].iloc[i+1:i+31]
            result = "ノートレード"
            for f in future:
                if direction == "買い" and f <= sl:
                    result = "損切"
                    break
                elif direction == "買い" and f >= tp:
                    result = "利確"
                    break
                elif direction == "売り" and f >= sl:
                    result = "損切"
                    break
                elif direction == "売り" and f <= tp:
                    result = "利確"
                    break

            log.append({
                "日時": df.index[i],
                "レート": price,
                "判定": direction,
                "SL": round(sl, 4),
                "TP": round(tp, 4),
                "結果": result,
                "備考": ", ".join(guide)
            })
        return log

    st.subheader(f"通貨ペア：{symbol} | スタイル：{style}")

    final_scores = []
    final_signals = []
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            continue
        df = calc_indicators(df)
        sig, guide, score = extract_signal(df)
        final_signals.append((tf, sig, score, guide))
        st.markdown(f"### ⏱ {tf} 判定：{sig}")
        for g in guide:
            st.write("-", g)
        final_scores.append(score * tf_weights.get(tf, 0.3))

    weighted_avg_score = sum(final_scores)
    decision = "買い" if weighted_avg_score >= 0.6 else "売り" if weighted_avg_score <= 0.3 else "待ち"

    df_all = fetch_data(symbol, timeframes[1])
    df_all = calc_indicators(df_all)
    entry, tp, sl, rr, (pips_tp, pips_sl) = suggest_trade_plan(df_all, decision)
    bt_log = backtest(df_all, decision)
    win_count = sum(1 for r in bt_log if r.get("結果") == "利確")
    total_count = sum(1 for r in bt_log if r.get("結果") in ["利確", "損切"])
    win_rate = win_count / total_count if total_count else 0

    st.subheader("🧭 エントリーガイド（総合評価）")
    if decision == "買い":
        st.write("✅ 現在は買いシグナルが優勢です")
    elif decision == "売り":
        st.write("✅ 現在は売りシグナルが優勢です")
    else:
        st.write("現時点では明確な方向感がありません")

    if decision != "待ち":
        st.subheader("🎯 トレードプラン（想定）")
        st.write(f"エントリーレート：{entry:.4f}")
        st.write(f"指値（利確）：{tp:.4f}（+{pips_tp:.0f} pips）")
        st.write(f"逆指値（損切）：{sl:.4f}（-{pips_sl:.0f} pips）")
        st.write(f"リスクリワード比：{rr:.2f}")
        st.write(f"想定勝率：{win_rate*100:.1f}%")
    else:
        st.subheader("現在はエントリー待ちです。")

    st.subheader("📊 バックテスト結果（最大100件）")
    if bt_log:
        df_bt = pd.DataFrame(bt_log).tail(100)
        st.dataframe(df_bt)
        st.write(f"勝率：{win_rate*100:.1f}%（{win_count}勝 / {total_count}件）")
    else:
        st.write("⚠ バックテスト結果が0件です。条件未達かヒット無しの可能性があります。")
