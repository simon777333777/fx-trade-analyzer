# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- ユーザーインターフェース ---
st.title("FXトレード分析ツール")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"], index=0)

# --- 通貨ペアのPIPS単位定義 ---
def get_pip_unit(symbol):
    return 0.01 if "JPY" in symbol else 0.0001

# --- トレードスタイルに応じた時間足と重み ---
tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.2, "4h": 0.3, "1day": 0.5}

# --- データ取得関数 ---
def fetch_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    for col in ["close", "high", "low"]:
        df[col] = df[col].astype(float)
    return df

# --- インジケーター計算 ---
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

# --- シグナル抽出 ---
def extract_signal(row):
    buy_score, sell_score = 0, 0
    guide = []

    # MACD
    if row["MACD"] > row["Signal"]:
        buy_score += 1
        guide.append("✅ MACDゴールデンクロス")
    elif row["MACD"] < row["Signal"]:
        sell_score += 1
        guide.append("✅ MACDデッドクロス")
    else:
        guide.append("❌ MACD横ばい")

    # SMA
    if row["SMA_5"] > row["SMA_20"]:
        buy_score += 1
        guide.append("✅ SMA短期 > 長期")
    elif row["SMA_5"] < row["SMA_20"]:
        sell_score += 1
        guide.append("✅ SMA短期 < 長期")
    else:
        guide.append("❌ SMA横ばい")

    # ボリンジャーバンド
    if row["close"] < row["Lower"]:
        buy_score += 1
        guide.append("✅ BB下限反発の可能性")
    elif row["close"] > row["Upper"]:
        sell_score += 1
        guide.append("✅ BB上限反落の可能性")
    else:
        guide.append("❌ BB反応無し")

    # RCI
    if row["RCI"] > 0.5:
        buy_score += 1
        guide.append("✅ RCI上昇傾向")
    elif row["RCI"] < -0.5:
        sell_score += 1
        guide.append("✅ RCI下降傾向")
    else:
        guide.append("❌ RCI中立")

    if buy_score >= 3:
        return "買い", guide, buy_score / 4
    elif sell_score >= 3:
        return "売り", guide, sell_score / 4
    else:
        return "待ち", guide, 0.0

# --- トレードプラン ---
def suggest_trade_plan(entry_price, direction, pip_unit):
    tp, sl = None, None
    atr = 20 * pip_unit  # ATR代用
    if direction == "買い":
        tp = entry_price + atr * 1.6
        sl = entry_price - atr * 1.0
    elif direction == "売り":
        tp = entry_price - atr * 1.6
        sl = entry_price + atr * 1.0
    rr = abs((tp - entry_price) / (sl - entry_price)) if tp and sl else 0
    return tp, sl, rr

# --- バックテスト ---
def backtest(df, pip_unit):
    logs = []
    wins = 0
    count = 0

    for i in range(30, len(df) - 10):
        row = df.iloc[i]
        signal, guide, score = extract_signal(row)
        entry_price = row["close"]
        tp, sl, _ = suggest_trade_plan(entry_price, signal, pip_unit)
        outcome = "スキップ"

        if signal in ["買い", "売り"]:
            future = df.iloc[i+1:i+10]
            for _, frow in future.iterrows():
                if signal == "買い":
                    if frow["high"] >= tp:
                        outcome = "利確"
                        wins += 1
                        break
                    elif frow["low"] <= sl:
                        outcome = "損切"
                        break
                elif signal == "売り":
                    if frow["low"] <= tp:
                        outcome = "利確"
                        wins += 1
                        break
                    elif frow["high"] >= sl:
                        outcome = "損切"
                        break
            count += 1

        logs.append({
            "日時": row.name.strftime("%Y-%m-%d %H:%M"),
            "シグナル": signal,
            "価格": entry_price,
            "TP": tp,
            "SL": sl,
            "判定": outcome
        })

    return wins, count, pd.DataFrame(logs)

# --- 実行 ---
if st.button("実行"):
    timeframes = tf_map[style]
    pip_unit = get_pip_unit(symbol)
    final_scores = []
    signals = []

    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            st.error(f"データ取得失敗: {symbol} - {tf}")
            continue
        df = calc_indicators(df)
        row = df.iloc[-1]
        signal, guide, score = extract_signal(row)
        signals.append((tf, signal, guide))
        final_scores.append(score * tf_weights.get(tf, 0.3))

        st.markdown(f"### ⏱ {tf} 判定：{signal}")
        for g in guide:
            st.write("-", g)

    # --- 総合判断 ---
    avg_score = sum(final_scores)
    decision = "買い" if avg_score >= 0.6 else "売り" if avg_score <= 0.3 else "待ち"

    st.subheader("\n🧭 エントリーガイド（総合評価）")
    st.write(f"スコア合計：{avg_score:.2f} → 判定：{decision}")

    # --- トレードプラン ---
    df_latest = fetch_data(symbol, timeframes[1])
    df_latest = calc_indicators(df_latest)
    entry = df_latest["close"].iloc[-1]
    tp, sl, rr = suggest_trade_plan(entry, decision, pip_unit)

    if decision != "待ち":
        st.subheader("🎯 トレードプラン")
        st.write(f"エントリー価格：{entry:.3f}")
        st.write(f"利確(TP)：{tp:.3f}（+{abs(tp - entry)/pip_unit:.0f} pips）")
        st.write(f"損切(SL)：{sl:.3f}（-{abs(sl - entry)/pip_unit:.0f} pips）")
        st.write(f"リスクリワード：{rr:.2f}")

    # --- バックテスト実施 ---
    df_bt = fetch_data(symbol, timeframes[1])
    df_bt = calc_indicators(df_bt)
    wins, total, log = backtest(df_bt, pip_unit)

    st.subheader("📊 バックテスト結果")
    if total == 0:
        st.write("⚠ バックテスト対象なし（買い/売りシグナルが少ない）")
    else:
        st.write(f"勝率：{(wins/total)*100:.1f}% ({wins}/{total})")
        with st.expander("ログ詳細を表示"):
            st.dataframe(log)
