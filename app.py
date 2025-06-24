# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- UI ---
st.title("FXトレード分析（改良版）")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"], index=0)

# バックテスト期間を選択できるように
backtest_period = st.slider("バックテスト期間（本数）", min_value=15, max_value=50, value=15, step=5)

# TP/SL倍率（ATRに対する倍率）を設定可能に
sl_multiplier = st.slider("損切り幅倍率（ATR比）", min_value=0.3, max_value=2.0, value=1.0, step=0.1)
tp_multiplier = st.slider("利確幅倍率（ATR比）", min_value=0.5, max_value=3.0, value=1.6, step=0.1)

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
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=300&apikey={API_KEY}"
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
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        return df

    def calc_indicators(df):
        df = df.copy()
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["SMA_5"] = df["close"].rolling(window=5).mean()
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["Signal"] = df["MACD"].ewm(span=9).mean()
        df["Upper"] = df["close"].rolling(window=20).mean() + 2 * df["close"].rolling(window=20).std()
        df["Lower"] = df["close"].rolling(window=20).mean() - 2 * df["close"].rolling(window=20).std()
        df["RCI"] = df["close"].rank().rolling(window=9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1], raw=True)
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

    def suggest_trade_plan(df, direction):
        price = df["close"].iloc[-1]
        atr = df["high"].rolling(window=14).max() - df["low"].rolling(window=14).min()  # 14期間の高低差で代用
        atr_val = atr.iloc[-1]
        if np.isnan(atr_val) or atr_val == 0:
            atr_val = 0.001  # 非常に小さい値で回避
        if direction == "買い":
            sl = price - atr_val * sl_multiplier
            tp = price + atr_val * tp_multiplier
        elif direction == "売り":
            sl = price + atr_val * sl_multiplier
            tp = price - atr_val * tp_multiplier
        else:
            return price, None, None, 0, 0, 0
        rr = abs((tp - price) / (sl - price)) if sl != price else 0
        # pips計算（円通貨は100倍、小数通貨は10000倍）
        pips_tp = abs(tp - price) * (100 if any(c in symbol for c in ["JPY"]) else 10000)
        pips_sl = abs(sl - price) * (100 if any(c in symbol for c in ["JPY"]) else 10000)
        return price, tp, sl, rr, pips_tp, pips_sl

    def backtest(df, direction):
        results = []
        n = backtest_period
        for i in range(len(df) - n):
            atr = df["high"].iloc[i:i+14].max() - df["low"].iloc[i:i+14].min()
            if np.isnan(atr) or atr == 0:
                st.write(f"スキップ：インデックス {i}, ATR={atr}")
                continue
            price = df["close"].iloc[i]
            if direction == "買い":
                sl = price - atr * sl_multiplier
                tp = price + atr * tp_multiplier
                future_highs = df["high"].iloc[i+1:i+1+n]
                future_lows = df["low"].iloc[i+1:i+1+n]
                if any(future_lows <= sl):
                    results.append((df.index[i], price, sl, tp, "損切"))
                elif any(future_highs >= tp):
                    results.append((df.index[i], price, sl, tp, "利確"))
            elif direction == "売り":
                sl = price + atr * sl_multiplier
                tp = price - atr * tp_multiplier
                future_highs = df["high"].iloc[i+1:i+1+n]
                future_lows = df["low"].iloc[i+1:i+1+n]
                if any(future_highs >= sl):
                    results.append((df.index[i], price, sl, tp, "損切"))
                elif any(future_lows <= tp):
                    results.append((df.index[i], price, sl, tp, "利確"))
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
    if df_all is None:
        st.error("中期時間足のデータ取得に失敗しました。")
        st.stop()

    df_all = calc_indicators(df_all)
    entry, tp, sl, rr, pips_tp, pips_sl = suggest_trade_plan(df_all, decision)

    bt_results = backtest(df_all, decision)
    total_bt = len(bt_results)
    wins = sum(1 for r in bt_results if r[-1] == "利確")
    win_rate = wins / total_bt if total_bt > 0 else 0

    st.subheader("\n🧭 エントリーガイド（総合評価）")
    if decision == "買い":
        st.write(f"✅ {style} において複数の時間足が買いシグナルを示しています")
        st.write("⏳ 中期・長期の上昇トレンドが短期にも波及")
        st.write("📌 押し目が完了しており、エントリータイミングとして有効")
    else:
        st.write("現在は明確な買い/売りシグナルが不足しているため、エントリーは控えめに")

    if decision != "待ち":
        st.subheader("\n🎯 トレードプラン（想定）")
        st.write(f"エントリーレート：{entry:.5f}")
        st.write(f"指値（利確）：{tp:.5f}（+{pips_tp:.1f} pips）")
        st.write(f"逆指値（損切）：{sl:.5f}（-{pips_sl:.1f} pips）")
        st.write(f"リスクリワード比：{rr:.2f}")
        st.write(f"想定勝率：{win_rate*100:.1f}%")
    else:
        st.subheader("現在はエントリー待ちです。")

    with st.expander(f"バックテスト結果（最大{backtest_period}件）"):
        if total_bt == 0:
            st.warning("⚠ バックテスト結果が0件です。ATRが0か、TP/SLがヒットしない可能性があります。")
        else:
            df_bt = pd.DataFrame(bt_results, columns=["日時", "エントリー", "損切", "利確", "結果"])
            st.dataframe(df_bt)
            st.write(f"勝率：{win_rate*100:.1f}%")
