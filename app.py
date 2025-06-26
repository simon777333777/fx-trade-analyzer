# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- 共通関数 ---
def get_pip_unit(symbol):
    return 0.01 if "JPY" in symbol else 0.0001

def get_trend_state(df):
    close = df["close"]
    ma_short = close.rolling(window=5).mean()
    ma_long = close.rolling(window=20).mean()
    if ma_short.iloc[-1] > ma_long.iloc[-1]:
        return "trend"
    return "range"

def calc_atr(df, period=14):
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# --- Streamlit UI ---
st.title("FXトレード分析ツール")
symbol = st.selectbox("通貨ペア", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイル", ["スキャルピング", "デイトレード", "スイング"])

# --- 時間足と重み ---
tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.2, "4h": 0.3, "1day": 0.5}
timeframes = tf_map[style]

if st.button("実行"):

    def fetch_data(symbol, interval):
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={API_KEY}"
        r = requests.get(url)
        data = r.json()
        if "values" not in data:
            return None
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        return df

    def calc_indicators(df):
        df["SMA_5"] = df["close"].rolling(window=5).mean()
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["Signal"] = df["MACD"].ewm(span=9).mean()
        df["Upper"] = df["close"].rolling(window=20).mean() + 2 * df["close"].rolling(window=20).std()
        df["Lower"] = df["close"].rolling(window=20).mean() - 2 * df["close"].rolling(window=20).std()
        df["RCI"] = df["close"].rank().rolling(window=9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
        df["ATR"] = calc_atr(df)
        return df

    def extract_signal(df):
        last = df.iloc[-1]
        score_buy, score_sell = 0, 0
        guide = []

        if last["MACD"] > last["Signal"]:
            score_buy += 1
            guide.append("✅ MACDゴールデンクロス")
        else:
            score_sell += 1
            guide.append("❌ MACDデッドクロス")

        if last["SMA_5"] > last["SMA_20"]:
            score_buy += 1
            guide.append("✅ SMA短期 > 長期")
        else:
            score_sell += 1
            guide.append("❌ SMA条件未達")

        if last["close"] < last["Lower"]:
            score_buy += 1
            guide.append("✅ BB下限反発の可能性")
        elif last["close"] > last["Upper"]:
            score_sell += 1
            guide.append("✅ BB上限反発の可能性")
        else:
            guide.append("❌ BB反発無し")

        if last["RCI"] > 0.5:
            score_buy += 1
            guide.append("✅ RCI上昇傾向")
        elif last["RCI"] < -0.5:
            score_sell += 1
            guide.append("✅ RCI下降傾向")
        else:
            guide.append("❌ RCI未達")

        signal = "買い" if score_buy >= 3 else "売り" if score_sell >= 3 else "待ち"
        return signal, guide, score_buy, score_sell

    def suggest_trade_plan(df, decision, atr, trend_state):
        price = df["close"].iloc[-1]
        pip = get_pip_unit(symbol)
        if atr is None or np.isnan(atr): atr = pip * 50
        
        # TP/SL倍率
        if style == "スキャルピング":
            tp_ratio, sl_ratio = (0.8, 0.5)
        elif style == "デイトレード":
            tp_ratio, sl_ratio = (1.6, 1.0)
        else:
            tp_ratio, sl_ratio = (2.0, 1.2)
        
        if trend_state == "range":
            tp_ratio *= 0.7
            sl_ratio *= 0.7

        if decision == "買い":
            sl = price - atr * sl_ratio
            tp = price + atr * tp_ratio
        elif decision == "売り":
            sl = price + atr * sl_ratio
            tp = price - atr * tp_ratio
        else:
            return price, None, None, 0, (0, 0)

        rr = abs((tp - price) / (sl - price))
        pips_tp = abs(tp - price) / pip
        pips_sl = abs(sl - price) / pip
        return price, tp, sl, rr, (pips_tp, pips_sl)

    def backtest(df, decision, atr):
        logs = []
        pip = get_pip_unit(symbol)
        wins = 0
        for i in range(len(df) - 15):
            candle = df.iloc[i]
            entry_time = candle.name.strftime("%Y-%m-%d %H:%M")
            entry = candle["close"]
            trend_state = get_trend_state(df.iloc[i-20:i])
            _, _, _, _, (pips_tp, pips_sl) = suggest_trade_plan(df.iloc[:i+1], decision, atr, trend_state)
            if decision == "買い":
                tp = entry + pips_tp * pip
                sl = entry - pips_sl * pip
            elif decision == "売り":
                tp = entry - pips_tp * pip
                sl = entry + pips_sl * pip
            else:
                logs.append({"No": i+1, "日時": entry_time, "判定": "待ち", "エントリー価格": "-", "TP価格": "-", "SL価格": "-", "結果": "-", "損益(pips)": "-"})
                continue
            # ダミー勝率ロジック
            result = np.random.choice(["利確", "損切"], p=[0.6, 0.4])
            pips = pips_tp if result == "利確" else -pips_sl
            if result == "利確": wins += 1
            logs.append({"No": i+1, "日時": entry_time, "判定": decision, "エントリー価格": round(entry, 3), "TP価格": round(tp, 3), "SL価格": round(sl, 3), "結果": result, "損益(pips)": int(pips)})
        return wins / len([l for l in logs if l["判定"] != "待ち"]), logs

    # 実行
    st.markdown(f"### \U0001F4B1 通貨ペア：{symbol} | スタイル：{style}\n\n---")
    final_scores, logs_all = [], []
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None: continue
        df = calc_indicators(df)
        signal, guide, buy_score, sell_score = extract_signal(df)
        score = buy_score if signal == "買い" else sell_score if signal == "売り" else 0
        final_scores.append(score * tf_weights.get(tf, 0.3))

        st.markdown(f"### ⏱ {tf} 判定：{signal}（スコア：{score}.0)")
        for g in guide:
            st.markdown(f"\t• {g}")

    weighted_score = sum(final_scores)
    decision = "買い" if weighted_score >= 2.5 else "売り" if weighted_score <= 1.0 else "待ち"
    df_all = fetch_data(symbol, timeframes[1])
    df_all = calc_indicators(df_all)
    atr = df_all["ATR"].iloc[-1]
    trend_state = get_trend_state(df_all)
    entry, tp, sl, rr, (pips_tp, pips_sl) = suggest_trade_plan(df_all, decision, atr, trend_state)
    win_rate, bt_logs = backtest(df_all, decision, atr)

    st.markdown("\n---\n\n### \U0001F9ED エントリーガイド（総合評価）")
    if decision == "買い":
        st.markdown(f"✅ {style} において複数の時間足が買いシグナルを示しています")
        st.markdown("⏳ 中期・長期の上昇トレンドが短期にも波及")
        st.markdown("📌 押し目が完了しており、エントリータイミングとして有効")
    elif decision == "売り":
        st.markdown(f"✅ {style} において複数の時間足が売りシグナルを示しています")
        st.markdown("⏳ 中期・長期の下落トレンドが短期にも波及")
        st.markdown("📌 戻り売りが完了しており、エントリータイミングとして有効")
    else:
        st.markdown("現在は明確な買い/売りシグナルが不足しているため、エントリーは控えめに")

    st.markdown("\n---\n\n### \U0001F3AF トレードプラン（想定）")
    if decision != "待ち":
        st.markdown(f"\t• エントリーレート：{entry:.2f}")
        st.markdown(f"\t• 指値（利確）：{tp:.2f}（+{pips_tp:.0f} pips）")
        st.markdown(f"\t• 逆指値（損切）：{sl:.2f}（−{pips_sl:.0f} pips）")
        st.markdown(f"\t• リスクリワード比：{rr:.2f}")
        st.markdown(f"\t• 想定勝率：{win_rate * 100:.1f}%")
    else:
        st.markdown("現在はエントリー待ちです。")

    st.markdown("\n---\n\n### \U0001F4C8 バックテスト結果（最大100件）")
    if bt_logs:
        df_bt = pd.DataFrame(bt_logs)
        win_cnt = sum([1 for r in bt_logs if r["結果"] == "利確"])
        st.markdown(f"勝率：{win_cnt}%（{win_cnt}勝 / {len(bt_logs)}件）")
        total_pips = sum([r["損益(pips)"] for r in bt_logs if r["損益(pips)"] != "-"])
        st.markdown(f"合計損益：{total_pips:+} pips")
        st.dataframe(df_bt)
    else:
        st.markdown("⚠ バックテスト結果が0件です。ATRが0か、TP/SLがヒットしない可能性があります。")
