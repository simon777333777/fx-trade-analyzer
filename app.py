# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- ユーザーインターフェース ---
st.title("FXトレード分析")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"], index=0)

if st.button("実行"):

    # --- 時間足と重み定義 ---
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
        df["ATR"] = df["close"].rolling(window=14).std()
        return df.dropna()

    def extract_signal(row):
        buy_score = 0
        sell_score = 0
        guide = []

        # MACD
        if row["MACD"] > row["Signal"]:
            buy_score += 1
            guide.append("✅ MACDゴールデンクロス")
        else:
            sell_score += 1
            guide.append("❌ MACDデッドクロス")

        # SMA
        if row["SMA_5"] > row["SMA_20"]:
            buy_score += 1
            guide.append("✅ SMA短期 > 長期")
        else:
            sell_score += 1
            guide.append("❌ SMA短期 < 長期")

        # BB
        if row["close"] < row["Lower"]:
            buy_score += 1
            guide.append("✅ BB下限反発の可能性")
        elif row["close"] > row["Upper"]:
            sell_score += 1
            guide.append("✅ BB上限反落の可能性")
        else:
            guide.append("❌ BB反発無し")

        # RCI
        if row["RCI"] > 0.5:
            buy_score += 1
            guide.append("✅ RCI上昇傾向")
        elif row["RCI"] < -0.5:
            sell_score += 1
            guide.append("✅ RCI下降傾向")
        else:
            guide.append("❌ RCI未達")

        if buy_score >= 3:
            return "買い", guide, buy_score / 4
        elif sell_score >= 3:
            return "売り", guide, sell_score / 4
        else:
            return "待ち", guide, 0

    def suggest_trade_plan(row, signal):
        price = row["close"]
        atr = row["ATR"]
        if pd.isna(atr) or atr == 0:
            return price, price, price, 0, (0, 0)

        pip_unit = 100 if "/JPY" in symbol else 10000

        if signal == "買い":
            tp = price + atr * 1.6
            sl = price - atr * 1.0
        elif signal == "売り":
            tp = price - atr * 1.6
            sl = price + atr * 1.0
        else:
            return price, price, price, 0, (0, 0)

        rr = abs((tp - price) / (sl - price))
        return price, tp, sl, rr, (abs(tp - price) * pip_unit, abs(sl - price) * pip_unit)

    def backtest(df):
        logs = []
        for i in range(-100, 0):
            row = df.iloc[i]
            signal, guide, score = extract_signal(row)
            entry, tp, sl, rr, (pips_tp, pips_sl) = suggest_trade_plan(row, signal)

            logs.append({
                "日時": row.name.strftime("%Y-%m-%d %H:%M"),
                "終値": round(row["close"], 5),
                "判定": signal,
                "スコア": round(score, 2),
                "利確(pips)": round(pips_tp),
                "損切(pips)": round(pips_sl),
                "RR比": round(rr, 2),
                "根拠": " / ".join(guide)
            })
        return pd.DataFrame(logs)

    st.subheader(f"通貨ペア：{symbol} | スタイル：{style}")

    final_scores = []
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            continue
        df = calc_indicators(df)
        latest = df.iloc[-1]
        signal, guide, score = extract_signal(latest)

        st.markdown(f"### ⏱ {tf} 判定：{signal}")
        for g in guide:
            st.write("-", g)
        final_scores.append(score * tf_weights.get(tf, 0.3))

    weighted_avg = sum(final_scores)
    decision = "買い" if weighted_avg >= 0.6 else "売り" if weighted_avg <= 0.2 else "待ち"

    df_all = fetch_data(symbol, timeframes[1])
    df_all = calc_indicators(df_all)
    latest = df_all.iloc[-1]
    entry, tp, sl, rr, (pips_tp, pips_sl) = suggest_trade_plan(latest, decision)

    st.subheader("🧭 エントリーガイド（総合評価）")
    if decision in ["買い", "売り"]:
        st.write(f"✅ 複数の時間足が{decision}シグナルを示しています")
        st.subheader("🎯 トレードプラン")
        st.write(f"エントリーレート：{entry:.3f}")
        st.write(f"指値（利確）：{tp:.3f}（+{pips_tp:.0f} pips）")
        st.write(f"逆指値（損切）：{sl:.3f}（-{pips_sl:.0f} pips）")
        st.write(f"リスクリワード比：{rr:.2f}")
    else:
        st.write("現在は明確な買い/売りシグナルが不足しているため、エントリーは控えめに")

    st.subheader("📊 バックテスト結果（直近100本）")
    bt_log = backtest(df_all)
    st.dataframe(bt_log)
    st.write(f"対象件数：{len(bt_log)}")
