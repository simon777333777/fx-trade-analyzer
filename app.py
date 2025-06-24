# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキーの指定 ---
API_KEY = st.secrets["API_KEY"]  # secrets.tomlから取得

# --- ユーザーインターフェース ---
st.title("FXトレード分析ツール（改良版・判定ロジック＆バックテスト拡張）")

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
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
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
        buy_score = 0
        sell_score = 0
        last = df.iloc[-1]

        # MACD 判定
        if last["MACD"] > last["Signal"]:
            buy_score += 1
            guide.append("✅ MACDゴールデンクロス")
        elif last["MACD"] < last["Signal"]:
            sell_score += 1
            guide.append("✅ MACDデッドクロス")
        else:
            guide.append("❌ MACD未達")

        # SMA 判定
        if last["SMA_5"] > last["SMA_20"]:
            buy_score += 1
            guide.append("✅ SMA短期 > 長期")
        elif last["SMA_5"] < last["SMA_20"]:
            sell_score += 1
            guide.append("✅ SMA短期 < 長期")
        else:
            guide.append("❌ SMA条件未達")

        # BB 判定
        if last["close"] < last["Lower"]:
            buy_score += 1
            guide.append("✅ BB下限反発の可能性")
        elif last["close"] > last["Upper"]:
            sell_score += 1
            guide.append("✅ BB上限反発の可能性")
        else:
            guide.append("❌ BB反発無し")

        # RCI 判定
        if last["RCI"] > 0.5:
            buy_score += 1
            guide.append("✅ RCI上昇傾向")
        elif last["RCI"] < -0.5:
            sell_score += 1
            guide.append("✅ RCI下降傾向")
        else:
            guide.append("❌ RCI未達")

        if buy_score >= 3:
            signal = "買い"
        elif sell_score >= 3:
            signal = "売り"
        else:
            signal = "待ち"

        return signal, guide, buy_score, sell_score

    def suggest_trade_plan(df, direction):
        price = df["close"].iloc[-1]
        pip_unit = 0.01 if "JPY" in symbol else 0.0001
        buffer = 50 * pip_unit
        if direction == "買い":
            sl = price - buffer
            tp = price + buffer * 1.6
        elif direction == "売り":
            sl = price + buffer
            tp = price - buffer * 1.6
        else:
            return price, None, None, 0, (0, 0)
        rr = abs((tp - price) / (sl - price))
        pips_tp = int((tp - price) / pip_unit)
        pips_sl = int((price - sl) / pip_unit)
        return price, tp, sl, rr, (pips_tp, pips_sl)

    def backtest(df):
        results = []
        pip_unit = 0.01 if "JPY" in symbol else 0.0001
        buffer = 50 * pip_unit
        for i in range(len(df) - 15):
            row = df.iloc[i]
            price = row["close"]
            signal, _, buy_score, sell_score = extract_signal(df.iloc[:i+1])
            entry_time = df.index[i]
            tp = sl = None
            outcome = "対象外"

            if signal == "買い":
                tp = price + buffer * 1.6
                sl = price - buffer
                future = df["close"].iloc[i+1:i+15]
                if any(f <= sl for f in future):
                    outcome = "損切"
                elif any(f >= tp for f in future):
                    outcome = "利確"
            elif signal == "売り":
                tp = price - buffer * 1.6
                sl = price + buffer
                future = df["close"].iloc[i+1:i+15]
                if any(f >= sl for f in future):
                    outcome = "損切"
                elif any(f <= tp for f in future):
                    outcome = "利確"
            results.append({
                "日時": entry_time,
                "終値": price,
                "判定": signal,
                "買いスコア": buy_score,
                "売りスコア": sell_score,
                "TP": tp,
                "SL": sl,
                "結果": outcome
            })
        return results

    st.subheader(f"通貨ペア：{symbol} | スタイル：{style}")
    final_scores = []
    final_signals = []
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            continue
        df = calc_indicators(df)
        sig, guide, buy_score, sell_score = extract_signal(df)
        final_signals.append((tf, sig, buy_score, sell_score))

        st.markdown(f"### ⏱ {tf} 判定：{sig}")
        for g in guide:
            st.write("-", g)

        score = buy_score if sig == "買い" else sell_score if sig == "売り" else 0
        final_scores.append(score * tf_weights.get(tf, 0.3))

    weighted_avg_score = sum(final_scores)
    if final_scores.count(0) == len(final_scores):
        decision = "待ち"
    elif final_scores[-1] >= 3:
        decision = final_signals[-1][1]
    else:
        decision = "待ち"

    df_all = fetch_data(symbol, timeframes[1])
    if df_all is not None:
        df_all = calc_indicators(df_all)
        entry, tp, sl, rr, (pips_tp, pips_sl) = suggest_trade_plan(df_all, decision)
        bt_log = backtest(df_all)
        wins = sum(1 for r in bt_log if r["結果"] == "利確")
        total = sum(1 for r in bt_log if r["結果"] in ["利確", "損切"])
        win_rate = wins / total if total > 0 else 0

        st.subheader("🧭 エントリーガイド（総合評価）")
        if decision == "買い":
            st.write("✅ 複数の時間足が買いシグナルを示しています")
        elif decision == "売り":
            st.write("✅ 複数の時間足が売りシグナルを示しています")
        else:
            st.write("現在は明確な買い/売りシグナルが不足しているため、エントリーは控えめに")

        if decision != "待ち":
            st.subheader("🎯 トレードプラン（想定）")
            st.write(f"エントリーレート：{entry:.4f}")
            st.write(f"指値（利確）：{tp:.4f}（+{pips_tp} pips）")
            st.write(f"逆指値（損切）：{sl:.4f}（-{pips_sl} pips）")
            st.write(f"リスクリワード比：{rr:.2f}")
            st.write(f"想定勝率：{win_rate*100:.1f}%")

        st.subheader("📊 バックテスト結果（最大100件）")
        df_bt = pd.DataFrame(bt_log)
        st.dataframe(df_bt)
        st.write(f"勝率：{win_rate*100:.1f}%  | 判定回数：{total}件")
