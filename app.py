# --- ライブラリ読み込み ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- タイトル ---
st.title("FXトレード分析ツール")

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- ユーザーインターフェース ---
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=1)

if st.button("分析実行"):

    # --- トレードスタイル別時間足 ---
    tf_map = {
        "スキャルピング": ["5min", "15min", "1h"],
        "デイトレード": ["15min", "1h", "4h"],
        "スイング": ["1h", "4h", "1day"]
    }
    tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}
    timeframes = tf_map[style]

    # --- 為替ペアでpips単位を決定 ---
    pip_unit = 100.0 if "JPY" in symbol else 10000.0

    # --- データ取得 ---
    def fetch_data(symbol, interval):
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=200&apikey={API_KEY}"
        r = requests.get(url)
        data = r.json()
        if "values" not in data:
            st.error(f"データ取得失敗: {symbol} - {interval}")
            return None
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df = df.astype(float)
        return df

    # --- インジケータ計算 ---
    def calc_indicators(df):
        df = df.copy()
        df["SMA_5"] = df["close"].rolling(window=5).mean()
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["EMA_12"] = df["close"].ewm(span=12).mean()
        df["EMA_26"] = df["close"].ewm(span=26).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["Signal"] = df["MACD"].ewm(span=9).mean()
        df["Upper"] = df["close"].rolling(window=20).mean() + 2 * df["close"].rolling(window=20).std()
        df["Lower"] = df["close"].rolling(window=20).mean() - 2 * df["close"].rolling(window=20).std()
        df["RCI"] = df["close"].rank().rolling(window=9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
        return df

    # --- シグナル判定（買い・売り別スコア） ---
    def extract_signals(df):
        last = df.iloc[-1]
        buy_score, sell_score = 0, 0
        guide = []

        # MACD
        if last["MACD"] > last["Signal"]:
            buy_score += 1
            guide.append("✅ MACDゴールデンクロス")
        else:
            sell_score += 1
            guide.append("❌ MACDデッドクロス")

        # SMA
        if last["SMA_5"] > last["SMA_20"]:
            buy_score += 1
            guide.append("✅ SMA短期 > 長期")
        else:
            sell_score += 1
            guide.append("❌ SMA条件未達")

        # BB
        if last["close"] < last["Lower"]:
            buy_score += 1
            guide.append("✅ BB下限反発の可能性")
        elif last["close"] > last["Upper"]:
            sell_score += 1
            guide.append("✅ BB上限反発の可能性")
        else:
            guide.append("❌ BB反発無し")

        # RCI
        if last["RCI"] > 0.5:
            buy_score += 1
            guide.append("✅ RCI上昇傾向")
        elif last["RCI"] < -0.5:
            sell_score += 1
            guide.append("✅ RCI下降傾向")
        else:
            guide.append("❌ RCI未達")

        # 最終判定
        if buy_score >= 3 and sell_score <= 1:
            signal = "買い"
        elif sell_score >= 3 and buy_score <= 1:
            signal = "売り"
        else:
            signal = "待ち"

        return signal, guide, buy_score, sell_score

    # --- トレードプラン作成 ---
    def trade_plan(df, signal):
        price = df["close"].iloc[-1]
        atr = df["close"].rolling(window=14).std().iloc[-1]
        if np.isnan(atr): atr = 0.3  # デフォルト安全値
        if signal == "買い":
            tp = price + atr * 1.6
            sl = price - atr * 1.0
        elif signal == "売り":
            tp = price - atr * 1.6
            sl = price + atr * 1.0
        else:
            return price, None, None, 0
        rr = abs((tp - price) / (sl - price))
        return price, tp, sl, rr

    # --- 総合判定 ---
    all_scores = []
    buy_scores, sell_scores = [], []
    signal_results = []

    st.markdown(f"""
        ### 💱 通貨ペア：{symbol} | スタイル：{style}
        ⸻
        ⏱ 各時間足シグナル詳細
    """)

    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None: continue
        df = calc_indicators(df)
        sig, guide, b, s = extract_signals(df)
        signal_results.append(sig)
        buy_scores.append(b * tf_weights.get(tf, 0.3))
        sell_scores.append(s * tf_weights.get(tf, 0.3))

        st.markdown(f"**⏱ {tf} 判定：{sig}（スコア：{b if sig=='買い' else s}.0)**")
        for g in guide:
            st.write("•", g)

    st.markdown("⸻")

    # --- エントリーガイド ---
    avg_buy = sum(buy_scores)
    avg_sell = sum(sell_scores)

    if avg_buy >= 2.5 and avg_sell <= 1.0:
        decision = "買い"
        comment = [
            "✅ 複数時間足で買いシグナルが確認されました",
            "⏳ 中期・長期の上昇トレンドが短期にも波及",
            "📌 押し目が完了しており、エントリータイミングとして有効"
        ]
    elif avg_sell >= 2.5 and avg_buy <= 1.0:
        decision = "売り"
        comment = [
            "✅ 複数時間足で売りシグナルが確認されました",
            "⏳ 中期・長期の下降トレンドが短期にも波及",
            "📌 戻りの終盤でエントリーの好機"
        ]
    else:
        decision = "待ち"
        comment = ["現在は明確な買い/売りシグナルが不足しているため、エントリーは控えめに"]

    st.markdown("### 🧭 エントリーガイド（総合評価）")
    for c in comment:
        st.write(c)

    st.markdown("⸻")

    # --- トレードプラン表示 ---
    df_final = fetch_data(symbol, timeframes[1])
    df_final = calc_indicators(df_final)
    entry, tp, sl, rr = trade_plan(df_final, decision)

    if decision != "待ち":
        pips_tp = round((tp - entry) * pip_unit)
        pips_sl = round((entry - sl) * pip_unit)
        win_rate = 1 / (1 + rr)

        st.markdown("### 🎯 トレードプラン（想定）")
        st.write(f"• エントリーレート：{entry:.2f}")
        st.write(f"• 指値（利確）：{tp:.2f}（+{pips_tp} pips）")
        st.write(f"• 逆指値（損切）：{sl:.2f}（−{pips_sl} pips）")
        st.write(f"• リスクリワード比：{rr:.2f}")
        st.write(f"• 想定勝率：{win_rate*100:.1f}%")
    else:
        st.write("現在はトレードプランは提示されません。")
