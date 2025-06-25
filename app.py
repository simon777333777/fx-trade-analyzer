# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキーの指定 ---
API_KEY = st.secrets["API_KEY"]  # セキュアに管理

# --- ユーザーインターフェース ---
st.title("FXトレード分析")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"], index=0)

# --- ユーティリティ ---
def get_pip_unit(symbol):
    return 0.01 if "JPY" in symbol else 0.0001

def fetch_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=150&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        st.error(f"データ取得失敗: {symbol} - {interval}")
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    df = df.astype(float)
    return df

def calc_indicators(df):
    df = df.copy()
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["SMA_5"] = df["close"].rolling(window=5).mean()
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Upper"] = df["close"].rolling(window=20).mean() + 2 * df["close"].rolling(window=20).std()
    df["Lower"] = df["close"].rolling(window=20).mean() - 2 * df["close"].rolling(window=20).std()
    df["RCI"] = df["close"].rank().rolling(window=9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0,1], raw=True)
    return df

def extract_signal(df):
    last = df.iloc[-1]
    score = 0
    if last["MACD"] > last["Signal"]:
        score += 1
    if last["SMA_5"] > last["SMA_20"]:
        score += 1
    if last["close"] > last["Upper"]:
        score += 1
    if last["RCI"] > 0.5:
        score += 1
    buy_score = score

    score = 0
    if last["MACD"] < last["Signal"]:
        score += 1
    if last["SMA_5"] < last["SMA_20"]:
        score += 1
    if last["close"] < last["Lower"]:
        score += 1
    if last["RCI"] < -0.5:
        score += 1
    sell_score = score

    if buy_score >= 3:
        return "買い", buy_score
    elif sell_score >= 3:
        return "売り", sell_score
    else:
        return "待ち", max(buy_score, sell_score)

def suggest_trade_plan(price, direction, pip_unit):
    atr = 50 * pip_unit
    if direction == "買い":
        sl = price - atr
        tp = price + atr * 1.6
    elif direction == "売り":
        sl = price + atr
        tp = price - atr * 1.6
    else:
        return price, price, price, 0
    rr = abs((tp - price) / (sl - price))
    return price, tp, sl, rr

def backtest(df, pip_unit):
    logs = []
    wins = 0
    total = 0
    for i in range(len(df)):
        row = df.iloc[i]
        dt = row.name.strftime("%Y-%m-%d %H:%M")
        signal, score = extract_signal(df.iloc[:i+1])
        price = row["close"]
        entry, tp, sl, rr = suggest_trade_plan(price, signal, pip_unit)
        result = "-"
        pips = 0

        if signal == "買い":
            if i+1 < len(df):
                next_close = df.iloc[i+1]["close"]
                if next_close >= tp:
                    result = "利確"
                    pips = (tp - price) / pip_unit
                    wins += 1
                elif next_close <= sl:
                    result = "損切"
                    pips = (sl - price) / pip_unit
                total += 1
        elif signal == "売り":
            if i+1 < len(df):
                next_close = df.iloc[i+1]["close"]
                if next_close <= tp:
                    result = "利確"
                    pips = (price - tp) / pip_unit
                    wins += 1
                elif next_close >= sl:
                    result = "損切"
                    pips = (price - sl) / pip_unit
                total += 1

        logs.append({"日時": dt, "終値": round(price, 3), "判定": signal, "スコア": score,
                      "指値": round(tp, 3), "逆指値": round(sl, 3), "結果": result, "損益(pips)": int(pips)})

    df_bt = pd.DataFrame(logs)
    win_rate = wins / total * 100 if total > 0 else 0
    total_pips = df_bt["損益(pips)"].sum()
    return df_bt, win_rate, total_pips

# --- メインロジック ---
if st.button("実行"):
    tf_map = {
        "スキャルピング": ["5min", "15min", "1h"],
        "デイトレード": ["15min", "1h", "4h"],
        "スイング": ["1h", "4h", "1day"]
    }
    tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}
    timeframes = tf_map[style]
    pip_unit = get_pip_unit(symbol)

    final_scores = []
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None: continue
        df = calc_indicators(df)
        signal, score = extract_signal(df)
        st.markdown(f"### ⏱ {tf} 判定：{signal}（スコア：{score}）")
        final_scores.append(score * tf_weights.get(tf, 0.3))

    avg_score = sum(final_scores)
    decision = "買い" if avg_score >= 2 else "売り" if avg_score <= 1 else "待ち"

    st.subheader("\n🧭 エントリーガイド（総合評価）")
    if decision == "買い":
        st.write(f"✅ {style} において複数の時間足が買いシグナルを示しています")
        st.write("⏳ 中期・長期の上昇トレンドが短期にも波及")
        st.write("📌 押し目が完了しており、エントリータイミングとして有効")
    elif decision == "売り":
        st.write(f"✅ {style} において複数の時間足が売りシグナルを示しています")
        st.write("⏳ 中期・長期の下降トレンドが短期にも波及")
        st.write("📌 戻り売りのチャンスが近づいています")
    else:
        st.write("現在は明確な買い/売りシグナルが不足しているため、エントリーは控えめに")

    df_all = fetch_data(symbol, timeframes[1])
    df_all = calc_indicators(df_all)
    entry = df_all.iloc[-1]["close"]
    entry, tp, sl, rr = suggest_trade_plan(entry, decision, pip_unit)

    bt_log, win_rate, total_pips = backtest(df_all, pip_unit)

    if decision != "待ち":
        st.subheader("\n🎯 トレードプラン（想定）")
        st.write(f"エントリーレート：{entry:.3f}")
        st.write(f"指値（利確）：{tp:.3f}（+{abs(tp-entry)/pip_unit:.0f} pips）")
        st.write(f"逆指値（損切）：{sl:.3f}（-{abs(sl-entry)/pip_unit:.0f} pips）")
        st.write(f"リスクリワード比：{rr:.2f}")
        st.write(f"想定勝率：{win_rate:.1f}%")
    else:
        st.subheader("現在はエントリー待ちです。")

    st.subheader("\n📊 バックテスト結果（最大100件）")
    st.dataframe(bt_log.tail(100), use_container_width=True)
    st.write(f"合計損益：{total_pips:.0f} pips　勝率：{win_rate:.1f}%（買い/売りシグナルのみ対象）")
