# --- ライブラリ ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキーの指定 ---
API_KEY = st.secrets["API_KEY"]  # secrets.tomlから取得

# --- ユーザーインターフェース ---
st.title("FXトレード分析ツール（マルチタイム＆戦略対応）")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"])
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"])

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

    # --- テクニカル指標 ---
    def calc_indicators(df):
        df = df.copy()
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["SMA_5"] = df["close"].rolling(window=5).mean()
        df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["Signal"] = df["MACD"].ewm(span=9).mean()
        df["Upper"] = df["close"].rolling(window=20).mean() + 2 * df["close"].rolling(window=20).std()
        df["Lower"] = df["close"].rolling(window=20).mean() - 2 * df["close"].rolling(window=20).std()
        df["RCI"] = df["close"].rank(pct=True).rolling(window=9).mean() * 100 - 50
        return df

    # --- シグナル抽出 ---
    def extract_signal(df):
        guide = []
        last = df.iloc[-1]
        if last["MACD"] > last["Signal"] and last["SMA_5"] > last["SMA_20"] and last["close"] > last["Lower"]:
            signal = "買い"
            guide.append("MACDがゴールデンクロス")
            guide.append("SMA短期 > 長期")
            guide.append("BB下限反発")
        elif last["MACD"] < last["Signal"] and last["SMA_5"] < last["SMA_20"] and last["close"] < last["Upper"]:
            signal = "売り"
            guide.append("MACDがデッドクロス")
            guide.append("SMA短期 < 長期")
            guide.append("BB上限反発")
        else:
            signal = "待ち"
            guide.append("全条件未達")
        return signal, guide

    # --- 指値と逆指値計算 ---
    def suggest_trade_plan(df, direction):
        price = df["close"].iloc[-1]
        atr = df["close"].rolling(window=14).std().iloc[-1]
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

    # --- 実バックテスト関数 ---
    def backtest(df, direction):
        results = []
        for i in range(len(df)-15):
            price = df["close"].iloc[i]
            atr = df["close"].rolling(window=14).std().iloc[i]
            if atr == 0 or np.isnan(atr):
                continue
            if direction == "買い":
                sl = price - atr * 1.0
                tp = price + atr * 1.6
                future = df["close"].iloc[i+1:i+15]
                if any(f <= sl for f in future):
                    results.append((df.index[i], price, sl, tp, "損切"))
                elif any(f >= tp for f in future):
                    results.append((df.index[i], price, sl, tp, "利確"))
            elif direction == "売り":
                sl = price + atr * 1.0
                tp = price - atr * 1.6
                future = df["close"].iloc[i+1:i+15]
                if any(f >= sl for f in future):
                    results.append((df.index[i], price, sl, tp, "損切"))
                elif any(f <= tp for f in future):
                    results.append((df.index[i], price, sl, tp, "利確"))
        return results

    # --- 分析開始 ---
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
    bt_results = backtest(df_all, decision)
    wins = sum(1 for r in bt_results if r[-1] == "利確")
    total = len(bt_results)
    win_rate = wins / total if total > 0 else 0

    # --- エントリーガイド ---
    st.subheader("🧭 エントリーガイド（総合評価）")
    if decision == "買い":
        st.write("現時点での判定：エントリー可能（買い）")
        if final_signal[0] == "買い" and final_signal[1] == "買い":
            st.write("✅ 短期・中期の戦略が強く、押し目買いが成立")
        if final_signal[2] == "待ち":
            st.write("⏳ 日足はやや様子見だが、4h足が強くフォロー")
        st.write(f"直近の安値 {sl:.2f} を明確に割らなければ、買い継続でOK")
    elif decision == "売り":
        st.write("現時点での判定：エントリー可能（売り）")
        if final_signal[0] == "売り" and final_signal[1] == "売り":
            st.write("✅ 短期・中期の戦略が強く、戻り売りが成立")
        if final_signal[2] == "待ち":
            st.write("⏳ 日足はやや様子見だが、4h足が弱く牽引中")
        st.write(f"直近の高値 {sl:.2f} を明確に超えなければ、売り継続でOK")
    else:
        st.write("現時点では明確なシグナルは出ていません。")
        st.write("👀 さらなる動き待ち。次の押し目・戻りに備えましょう。")

    # --- トレードプラン ---
    if decision != "待ち":
        st.subheader("🎯 トレードプラン（想定）")
        st.write(f"エントリーレート：{entry:.2f}")
        st.write(f"指値（利確）：{tp:.2f}（+{abs(tp-entry)*100:.0f} pips）")
        st.write(f"逆指値（損切）：{sl:.2f}（-{abs(sl-entry)*100:.0f} pips）")
        st.write(f"リスクリワード比：{rr:.2f}")
        st.write(f"想定勝率：{win_rate*100:.1f}%")
    else:
        st.subheader("現在はエントリー待ちです。")

    # --- バックテスト結果詳細 ---
    with st.expander("バックテスト結果（100件）"):
        df_bt = pd.DataFrame(bt_results, columns=["日時", "エントリー", "損切", "利確", "結果"])
        st.dataframe(df_bt)
        st.write(f"勝率：{win_rate*100:.1f}%  | 件数：{total}")
