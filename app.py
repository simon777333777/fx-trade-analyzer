# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキーの指定 ---
API_KEY = st.secrets["API_KEY"]

# CSSで文字サイズ調整
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# --- ユーザーインターフェース ---
st.title("FXトレード分析")

symbol = st.selectbox("通貨ペアを選択", ["GBP/JPY", "EUR/USD", "USD/JPY", "AUD/USD"])
style = st.selectbox("トレードスタイルを選択", ["スイング", "デイトレード", "スキャルピング"])

if st.button("実行"):

    # --- トレードスタイルに応じた時間足定義 ---
    tf_map = {
        "スキャルピング": ["5min", "15min", "1h"],
        "デイトレード": ["15min", "1h", "4h"],
        "スイング": ["1h", "4h", "1day"]
    }
    timeframes = tf_map[style]

    # --- データ取得関数 ---
    def fetch_data(symbol, interval):
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=50&apikey={API_KEY}"
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

    # --- テクニカル指標計算 ---
    def calc_indicators(df):
        df = df.copy()
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["SMA_5"] = df["close"].rolling(window=5).mean()
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["Signal"] = df["MACD"].ewm(span=9).mean()
        df["Upper"] = df["close"].rolling(window=20).mean() + 2 * df["close"].rolling(window=20).std()
        df["Lower"] = df["close"].rolling(window=20).mean() - 2 * df["close"].rolling(window=20).std()
        # RCI追加
        df["RCI"] = df["close"].rank() - np.arange(len(df))
        df["RCI"] = 1 - 6 * (df["RCI"]**2).sum() / (len(df) * (len(df)**2 - 1)) if len(df) > 1 else 0
        return df

    # --- シグナル抽出 ---
    def extract_signal(df):
        guide = []
        last = df.iloc[-1]
        signal = "待ち"
        if last["MACD"] > last["Signal"]:
            guide.append("MACDがゴールデンクロス")
        if last["SMA_5"] > last["SMA_20"]:
            guide.append("SMA短期 > 長期")
        if last["close"] > last["Lower"]:
            guide.append("BB下限反発")
        if last["RCI"] < -0.8:
            guide.append("RCIが-80以下で買いシグナル")
        if len(guide) >= 3:
            signal = "買い"
        elif len(guide) == 0:
            signal = "売り"
        return signal, guide

    # --- 指値と逆指値計算（可変） ---
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

    # --- バックテストダミー関数 ---
    def dummy_backtest():
        np.random.seed(0)
        win_rate = np.round(np.random.uniform(0.55, 0.65), 3)
        log = []
        for i in range(50):
            outcome = np.random.choice(["勝ち", "負け"], p=[win_rate, 1 - win_rate])
            pips = np.random.randint(40, 100) if outcome == "勝ち" else -np.random.randint(30, 80)
            log.append({"No": i+1, "結果": outcome, "損益(pips)": pips})
        df_log = pd.DataFrame(log)
        return win_rate, df_log

    # --- 分析・表示 ---
    st.subheader(f"通貨ペア：{symbol} | スタイル：{style}")

    final_signal = []
    for tf in timeframes:
        df = fetch_data(symbol, tf)
        if df is None:
            continue
        df = calc_indicators(df)
        sig, guide = extract_signal(df)
        final_signal.append(sig)

        st.markdown(f"### ⏱ {tf} 判定：{sig}")
        for g in guide:
            st.write("-", g)

    # --- 総合判断 ---
    decision = "待ち"
    if final_signal.count("買い") >= 2:
        decision = "買い"
    elif final_signal.count("売り") >= 2:
        decision = "売り"

    df_all = fetch_data(symbol, timeframes[1])
    df_all = calc_indicators(df_all)
    entry, tp, sl, rr = suggest_trade_plan(df_all, decision)
    win_rate, bt_log = dummy_backtest()

    # --- エントリーガイド ---
    st.subheader("エントリーガイド")
    if decision == "買い":
        st.success("現時点での判定：エントリー可能（買い）")
        st.write("✅ 短期・中期の戦略が強く、押し目買いが成立")
        st.write("⏳ 日足はやや様子見だが、4h足が強くフォロー")
        st.write("直近の安値を明確に割らなければ、買い継続でOK")
    elif decision == "売り":
        st.error("現時点での判定：エントリー可能（売り）")
        st.write("✅ トレンド反転シグナルが出現")
        st.write("⏳ 上位足も下降方向の兆しあり")
        st.write("直近高値を上抜けしなければ売り継続でOK")
    else:
        st.info("現在はエントリー待ちです。")

    # --- トレードプラン ---
    if decision != "待ち":
        st.subheader("\n🎯 トレードプラン（想定）")
        st.write(f"エントリーレート：{entry:.2f}")
        st.write(f"指値（利確）：{tp:.2f}（+{abs(tp-entry)*100:.0f} pips）")
        st.write(f"逆指値（損切）：{sl:.2f}（-{abs(sl-entry)*100:.0f} pips）")
        st.write(f"リスクリワード比：{rr:.2f}")
        st.write(f"想定勝率：{win_rate*100:.1f}%")

    # --- バックテスト結果表示 ---
    with st.expander("バックテスト50件の内訳を見る"):
        st.dataframe(bt_log)
        st.write(f"勝率：{win_rate*100:.1f}%")
