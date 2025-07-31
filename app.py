import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="RCI主軸FXトレード分析", layout="centered")
API_KEY = st.secrets["API_KEY"]

st.title("📈 RCI主軸FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=0)
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=1)
use_dummy = st.checkbox("📦 ダミーデータで実行", value=False)

tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}

def get_dummy_data():
    idx = pd.date_range(end=pd.Timestamp.now(), periods=150, freq="H")
    price = np.cumsum(np.random.randn(len(idx))) + 150
    return pd.DataFrame({
        "datetime": idx,
        "open": price + np.random.randn(len(idx)),
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": 1000
    }).set_index("datetime")

@st.cache_data(ttl=300)
def fetch_data(symbol, interval, use_dummy):
    if use_dummy:
        return get_dummy_data()
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        raise ValueError(f"APIエラー: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def calc_indicators(df):
    for period in [9, 26, 52]:
        df[f"RCI_{period}"] = df["close"].rolling(period).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if x.notna().all() else np.nan,
            raw=False
        )
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["SMA_9"] = df["close"].rolling(9).mean()
    df["SMA_26"] = df["close"].rolling(26).mean()
    df["STD"] = df["close"].rolling(20).std()
    return df

def rci_based_signal(df):
    last = df.iloc[-1]
    logs = []

    rci_short = last["RCI_9"]
    rci_mid = last["RCI_26"]
    rci_long = last["RCI_52"]
    macd = last["MACD"]
    signal = last["Signal"]
    close = last["close"]
    sma9 = last["SMA_9"]
    sma26 = last["SMA_26"]
    std = last["STD"]

    signal_flag = False

    # 総合判定ロジック
    if rci_short >= 0.8 and rci_mid > df["RCI_26"].iloc[-2] and rci_long >= 0.5:
        if macd > signal and df["MACD"].diff().iloc[-1] > 0:
            if close > sma9 and close > sma26:
                if 0 < std < df["STD"].mean() * 1.5:
                    signal_flag = True

    if signal_flag:
        logs.append("✅ 総合判定：RCI・MACD・SMA・ボラティリティすべてが買い傾向を支持")
        score = 7
    else:
        score = 0
        logs.append("⚠️ 総合判定：買いシグナルの条件を満たしていません")

    # 詳細ログ
    logs.append(f"• 短期RCI（9）: {round(rci_short, 2)}")
    logs.append(f"• 中期RCI（26）: {'上昇中' if rci_mid > df['RCI_26'].iloc[-2] else '下降中'}")
    logs.append(f"• 長期RCI（52）: {round(rci_long, 2)}")
    logs.append(f"• MACD判定: {'GC直後' if macd > signal and df['MACD'].diff().iloc[-1] > 0 else '弱め'}")
    logs.append(f"• SMA判定: {'順行' if close > sma9 and close > sma26 else '逆行'}")
    logs.append(f"• ボラティリティ: {'通常範囲' if 0 < std < df['STD'].mean() * 1.5 else '高騰/低迷'}")

    return score, logs

def generate_trade_plan(df):
    entry = df["close"].iloc[-1]
    recent_high = df["high"].rolling(window=20).max().iloc[-1]
    recent_low = df["low"].rolling(window=20).min().iloc[-1]

    tp = recent_high if recent_high > entry else entry + df["STD"].iloc[-1] * 1.5
    sl = recent_low if recent_low < entry else entry - df["STD"].iloc[-1] * 1.0

    rr = round((tp - entry) / (entry - sl), 2) if (entry - sl) != 0 else 0

    comment = ""
    if rr < 1.0:
        comment = "⚠️ リスクリワード比が1.0未満のため注意が必要です。"
    elif rr < 1.5:
        comment = "🟡 リスクリワードは平均的ですが、トレンド確認を。"
    else:
        comment = "🟢 十分なRRで、シグナルとの整合性も高い可能性あり。"

    return {
        "エントリー価格": round(entry, 3),
        "利確（TP）": round(tp, 3),
        "損切り（SL）": round(sl, 3),
        "リスクリワード比（RR）": rr,
        "コメント": comment
    }

if st.button("実行"):
    for tf in tf_map[style]:
        st.subheader(f"⏱ 時間足：{tf}")
        df = fetch_data(symbol, tf, use_dummy)
        df = calc_indicators(df)
        score, logs = rci_based_signal(df)
        decision = "🟢 エントリー判定：買い" if score == 7 else "⚪ 判定保留"

        st.markdown(f"**{decision}**")
        for log in logs:
            st.markdown(log)
        st.markdown(f"**信頼度スコア：{score} / 7点**")

        if score == 7:
            plan = generate_trade_plan(df)
            st.subheader("🧮 トレードプラン（RCI主軸型）")
            for k, v in plan.items():
                st.write(f"{k}: {v}")
