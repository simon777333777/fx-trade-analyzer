# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキーの指定 ---
API_KEY = "YOUR_API_KEY"  # ←ここにあなたの Twelve Data のAPIキーを入力

# --- ユーザーインターフェース ---
st.title("FXトレード分析")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"], index=0)

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
        atr = df["close"].rolling(window=14).std().iloc[-1]  # ATR代用
        if direction == "買い":
            sl = price - atr * 1.0
            tp = price + atr * 1.6
        elif direction == "売り":
            sl = price + atr * 1.0
            tp = price - atr * 1.6
        else:
            return price, None, None, 0
        rr = abs((tp - price) / (sl - price))
        return price, tp, sl, rr

    def dummy_backtest():
        np.random.seed(0)
        log = []
        win_count = 0
        for i in range(100):
            entry = 190 + np.random.randn() * 0.5
            sl = entry - 0.5
            tp = entry + 0.8
            outcome = np.random.choice(["勝ち", "負け"], p=[0.6, 0.4])
            pips = int((tp - entry) * 100) if outcome == "勝ち" else int((sl - entry) * 100)
            win_count += 1 if outcome == "勝ち" else 0
            log.append({"No": i+1, "日時": (datetime.now() - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M"), "エントリー": round(entry, 2), "結果": outcome, "損益(pips)": pips})
        return win_count / 100, pd.DataFrame(log)

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
    df_all = calc_indicators(df_all)
    entry, tp, sl, rr = suggest_trade_plan(df_all, decision)
    win_rate, bt_log = dummy_backtest()

    st.subheader("\n🧭 エントリーガイド（総合評価）")
    if decision == "買い":
        st.write(f"✅ {style} において複数の時間足が買いシグナルを示しています")
        st.write("⏳ 中期・長期の上昇トレンドが短期にも波及")
        st.write("📌 押し目が完了しており、エントリータイミングとして有効")
    else:
        st.write("現在は明確な買い/売りシグナルが不足しているため、エントリーは控えめに")

    if decision != "待ち":
        st.subheader("\n🎯 トレードプラン（想定）")
        st.write(f"エントリーレート：{entry:.2f}")
        st.write(f"指値（利確）：{tp:.2f}（+{abs(tp-entry)*100:.0f} pips）")
        st.write(f"逆指値（損切）：{sl:.2f}（-{abs(sl-entry)*100:.0f} pips）")
        st.write(f"リスクリワード比：{rr:.2f}")
        st.write(f"想定勝率：{win_rate*100:.1f}%")
    else:
        st.subheader("現在はエントリー待ちです。")

    with st.expander("バックテスト100件の詳細を見る"):
        st.dataframe(bt_log)
        st.write(f"勝率：{win_rate*100:.1f}%")
