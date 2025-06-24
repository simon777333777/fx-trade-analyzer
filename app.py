# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキーの指定 ---
API_KEY = st.secrets["API_KEY"]

# --- ユーザーインターフェース ---
st.title("FXトレード分析ツール（改良版）")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"], index=0)

if st.button("実行"):

    tf_map = {
        "スキャルピング": ["5min", "15min", "1h"],
        "デイトレード": ["15min", "1h", "4h"],
        "スイング": ["1h", "4h", "1day"]
    }
    tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.2, "4h": 0.3, "1day": 0.5}
    timeframes = tf_map[style]

    def fetch_data(symbol, interval):
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={API_KEY}"
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
        df = df.dropna()
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

        signal = "買い" if score >= 3 else "待ち"
        return signal, guide, score / 4

    def suggest_trade_plan(df, direction):
        price = df["close"].iloc[-1]
        min_gap = 0.0005 if "/" in symbol and "JPY" not in symbol else 0.05
        sl, tp = price, price
        if direction == "買い":
            sl = price - max(price * 0.002, min_gap)
            tp = price + max(price * 0.0032, min_gap)
        elif direction == "売り":
            sl = price + max(price * 0.002, min_gap)
            tp = price - max(price * 0.0032, min_gap)
        rr = abs((tp - price) / (sl - price)) if sl != price else 0
        pips_multiplier = 10000 if "/" in symbol and "JPY" not in symbol else 100
        return price, tp, sl, rr, ((tp - price) * pips_multiplier, (sl - price) * pips_multiplier)

    def backtest(df, direction):
        results = []
        price_col = df["close"]
        for i in range(len(df) - 30):
            entry = price_col.iloc[i]
            min_gap = 0.0005 if "/" in symbol and "JPY" not in symbol else 0.05
            sl, tp = entry, entry
            if direction == "買い":
                sl = entry - max(entry * 0.002, min_gap)
                tp = entry + max(entry * 0.0032, min_gap)
                future = price_col.iloc[i+1:i+30]
                if any(f <= sl for f in future):
                    results.append((df.index[i], entry, sl, tp, "損切"))
                elif any(f >= tp for f in future):
                    results.append((df.index[i], entry, sl, tp, "利確"))
            elif direction == "売り":
                sl = entry + max(entry * 0.002, min_gap)
                tp = entry - max(entry * 0.0032, min_gap)
                future = price_col.iloc[i+1:i+30]
                if any(f >= sl for f in future):
                    results.append((df.index[i], entry, sl, tp, "損切"))
                elif any(f <= tp for f in future):
                    results.append((df.index[i], entry, sl, tp, "利確"))
        return results

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
    decision = "買い" if weighted_avg_score >= 0.6 else "待ち"

    df_all = fetch_data(symbol, timeframes[1])
    if df_all is not None:
        df_all = calc_indicators(df_all)
        entry, tp, sl, rr, (pips_tp, pips_sl) = suggest_trade_plan(df_all, decision)
        bt_results = backtest(df_all, decision)
        wins = sum(1 for r in bt_results if r[-1] == "利確")
        total = len(bt_results)
        win_rate = wins / total if total > 0 else 0

        st.subheader("\n🧭 エントリーガイド（総合評価）")
        if decision == "買い":
            st.write(f"✅ {style} において複数の時間足が買いシグナルを示しています")
            st.write("⏳ 中期・長期の上昇トレンドが短期にも波及")
            st.write("📌 押し目が完了しており、エントリータイミングとして有効")
        else:
            st.write("現在は明確な買い/売りシグナルが不足しているため、エントリーは控えめに")

        if decision != "待ち":
            st.subheader("\n🎯 トレードプラン（想定）")
            st.write(f"エントリーレート：{entry:.4f}")
            st.write(f"指値（利確）：{tp:.4f}（+{pips_tp:.0f} pips）")
            st.write(f"逆指値（損切）：{sl:.4f}（{pips_sl:.0f} pips）")
            st.write(f"リスクリワード比：{rr:.2f}")
            st.write(f"想定勝率：{win_rate*100:.1f}%")
        else:
            st.subheader("現在はエントリー待ちです。")

        with st.expander("バックテスト結果（最大100件）"):
            if total > 0:
                df_bt = pd.DataFrame(bt_results, columns=["日時", "エントリー", "損切", "利確", "結果"])
                st.dataframe(df_bt)
                st.write(f"勝率：{win_rate*100:.1f}%  | 件数：{total}")
            else:
                st.write("⚠ バックテスト結果が0件です。TP/SL条件が未来足でヒットしない可能性があります。")
