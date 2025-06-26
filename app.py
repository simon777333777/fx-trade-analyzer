# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- タイトル ---
st.title("💱 FXトレード分析ツール")

# --- ユーザー入力 ---
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=1)

# --- スタイルに応じた時間足とSMA設定 ---
tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.5}
sma_periods = {"スキャルピング": (5, 20), "デイトレード": (10, 40), "スイング": (20, 80)}
timeframes = tf_map[style]
sma_short, sma_long = sma_periods[style]

# --- ATRの取得関数 ---
def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

# --- シグナル判定 ---
def fetch_data(symbol, interval):
    apikey = st.secrets["API_KEY"]
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=200&apikey={apikey}&format=JSON"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df

def analyze_signals(df):
    df = df.copy()
    df["SMA_S"] = df["close"].rolling(window=sma_short).mean()
    df["SMA_L"] = df["close"].rolling(window=sma_long).mean()
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Upper"] = df["close"].rolling(window=20).mean() + 2 * df["close"].rolling(window=20).std()
    df["Lower"] = df["close"].rolling(window=20).mean() - 2 * df["close"].rolling(window=20).std()
    df["RCI"] = df["close"].rank().rolling(window=9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
    df["ATR"] = calculate_atr(df)
    
    def judge(row):
        guide = []
        buy_score = 0
        sell_score = 0

        # MACD
        if row["MACD"] > row["Signal"]:
            buy_score += 1
            guide.append("✅ MACDゴールデンクロス")
        elif row["MACD"] < row["Signal"]:
            sell_score += 1
            guide.append("✅ MACDデッドクロス")
        else:
            guide.append("❌ MACD未達")

        # SMA
        if row["SMA_S"] > row["SMA_L"]:
            buy_score += 1
            guide.append("✅ SMA短期 > 長期")
        elif row["SMA_S"] < row["SMA_L"]:
            sell_score += 1
            guide.append("✅ SMA短期 < 長期")
        else:
            guide.append("❌ SMA条件未達")

        # BB反発
        if row["close"] < row["Lower"]:
            buy_score += 1
            guide.append("✅ BB下限反発の可能性")
        elif row["close"] > row["Upper"]:
            sell_score += 1
            guide.append("✅ BB上限反発の可能性")
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
            decision = "買い"
        elif sell_score >= 3:
            decision = "売り"
        else:
            decision = "待ち"
        
        return decision, guide, buy_score, sell_score

    latest = df.iloc[-1]
    decision, guide, bscore, sscore = judge(latest)
    return decision, guide, bscore, sscore, df

# --- 実行 ---
if st.button("実行"):
    st.markdown(f"## 💱 通貨ペア：{symbol} | スタイル：{style}\n\n")
    final_scores = []
    total_guide = []
    all_decisions = []
    
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            st.error(f"{tf} のデータ取得に失敗")
            continue
        decision, guide, b, s, df = analyze_signals(df)
        score = b if decision == "買い" else s if decision == "売り" else 0
        weighted_score = score * tf_weights.get(tf, 0.3)
        final_scores.append(weighted_score)
        all_decisions.append(decision)

        st.markdown(f"### ⏱ {tf} 判定：{decision}（スコア：{score:.1f}）")
        for g in guide:
            st.write("-", g)

    avg_score = sum(final_scores)
    buy_count = all_decisions.count("買い")
    sell_count = all_decisions.count("売り")
    
    if buy_count >= 2:
        final_decision = "買い"
    elif sell_count >= 2:
        final_decision = "売り"
    else:
        final_decision = "待ち"

    st.markdown("\n---\n")
    st.subheader("🧭 エントリーガイド（総合評価）")
    if final_decision == "買い":
        st.write(f"✅ {style} において複数の時間足が買いシグナルを示しています")
        st.write("⏳ 中期・長期の上昇トレンドが短期にも波及")
        st.write("📌 押し目が完了しており、エントリータイミングとして有効")
    elif final_decision == "売り":
        st.write(f"✅ {style} において複数の時間足が売りシグナルを示しています")
        st.write("⏳ 中期・長期の下降トレンドが短期にも波及")
        st.write("📌 戻りの終盤でエントリーの好機")
    else:
        st.write("現在は明確な買い/売りシグナルが不足しているため、エントリーは控えめに")
