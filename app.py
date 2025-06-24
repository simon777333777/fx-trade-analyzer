# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキーの指定 ---
API_KEY = st.secrets["API_KEY"]  # secrets.tomlから取得

# --- ユーザーインターフェース ---
st.title("FXトレード分析")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"], index=0)

# 通貨ペアによってpips計算単位を設定
def get_pips_unit(symbol):
    # 円絡みは0.01、ドル絡みは0.0001に設定（単純化）
    if "JPY" in symbol:
        return 0.01
    else:
        return 0.0001

if st.button("実行"):

    # --- トレードスタイルに応じた時間足定義 ---
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
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
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

        # ATR計算 (14期間の単純移動平均版)
        high_low = df["high"] - df["low"]
        high_close_prev = np.abs(df["high"] - df["close"].shift(1))
        low_close_prev = np.abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(window=14).mean()
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
        return signal, guide, score / 4  # 正規化スコア

    def suggest_trade_plan(df, direction, pips_unit):
        price = df["close"].iloc[-1]
        atr = df["ATR"].iloc[-1]  # ATRを使用
        if np.isnan(atr) or atr == 0:
            return price, None, None, 0, (0, 0)
        if direction == "買い":
            sl = price - atr * 1.0
            tp = price + atr * 1.6
        elif direction == "売り":
            sl = price + atr * 1.0
            tp = price - atr * 1.6
        else:
            return price, None, None, 0, (0, 0)
        rr = abs((tp - price) / (sl - price))
        pips_tp = abs(tp - price) / pips_unit
        pips_sl = abs(sl - price) / pips_unit
        return price, tp, sl, rr, (pips_tp, pips_sl)

    def backtest(df, direction, pips_unit):
        results = []
        start_idx = df["ATR"].first_valid_index()
        if start_idx is None:
            st.warning("ATRの有効なデータがありません。")
            return results
        start_pos = df.index.get_loc(start_idx)
        for i in range(start_pos, len(df) - 15):
            price = df["close"].iloc[i]
            atr = df["ATR"].iloc[i]
            if np.isnan(atr) or atr == 0:
                st.write(f"スキップ：インデックス {i}, ATR={atr}")
                continue
            if direction == "買い":
                sl = price - atr * 1.0
                tp = price + atr * 1.6
                future = df["close"].iloc[i + 1:i + 15]
                if any(f <= sl for f in future):
                    results.append((df.index[i], price, sl, tp, "損切"))
                elif any(f >= tp for f in future):
                    results.append((df.index[i], price, sl, tp, "利確"))
            elif direction == "売り":
                sl = price + atr * 1.0
                tp = price - atr * 1.6
                future = df["close"].iloc[i + 1:i + 15]
                if any(f >= sl for f in future):
                    results.append((df.index[i], price, sl, tp, "損切"))
                elif any(f <= tp for f in future):
                    results.append((df.index[i], price, sl, tp, "利確"))
        return results

    st.subheader(f"通貨ペア：{symbol} | スタイル：{style}")

    pips_unit = get_pips_unit(symbol)

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
    if df_all is None:
        st.error("詳細分析用のデータが取得できませんでした。")
        st.stop()

    df_all = calc_indicators(df_all)
    entry, tp, sl, rr, (pips_tp, pips_sl) = suggest_trade_plan(df_all, decision, pips_unit)
    bt_results = backtest(df_all, decision, p
