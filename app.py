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
    prev = df.iloc[-2]
    logs = []

    # 状態取得
    rci_short = last["RCI_9"]
    rci_mid = last["RCI_26"]
    rci_long = last["RCI_52"]

    # 短期RCIクロス判定
    short_cross_neg80 = prev["RCI_9"] < -0.8 and rci_short >= -0.8
    short_cross_zero = prev["RCI_9"] < 0 and rci_short >= 0

    # 中長期RCIの上昇・プラス圏判定
    mid_up = df["RCI_26"].iloc[-1] > df["RCI_26"].iloc[-2]
    long_up = df["RCI_52"].iloc[-1] > df["RCI_52"].iloc[-2]

    mid_pos = rci_mid > 0
    long_pos = rci_long > 0

    # MACD補助
    macd_bullish = last["MACD"] > last["Signal"] and df["MACD"].diff().iloc[-1] > 0
    sma_bullish = last["close"] > last["SMA_9"] and last["close"] > last["SMA_26"]

    # ロジック判定
    if short_cross_neg80 and mid_up and long_up and mid_pos and long_pos:
        logs.append("✅ パターン①：短期RCIが-80上抜け＋中長期クロス上昇＋＋圏 → 強い買い")
        decision = "🟢 エントリー判定：買い"
    elif short_cross_neg80 and mid_pos and long_pos:
        logs.append("✅ パターン②：短期RCIが-80上抜け＋中長期＋圏維持 → 買い")
        decision = "🟢 エントリー判定：買い"
    elif short_cross_zero and mid_pos and long_pos:
        logs.append("✅ パターン③：短期RCIが0上抜け＋中長期＋圏維持 → 買い")
        decision = "🟢 エントリー判定：買い"
    elif rci_short < -0.8 and not (mid_pos and long_pos):
        logs.append("⌛ 短期RCIが底で推移中＋中長期弱気 → 待ち")
        decision = "🟡 待機"
    else:
        logs.append("❌ 条件不一致 → エントリー見送り")
        decision = "⚪ 判定保留"

    # 補足情報（MACD・SMA・ボラティリティ）
    if macd_bullish:
        logs.append("• MACD：GC直後 → モメンタム良好")
    else:
        logs.append("• MACD：弱含み")

    if sma_bullish:
        logs.append("• SMA：順行")
    else:
        logs.append("• SMA：逆行 or 接触中")

    if 0 < last["STD"] < df["STD"].mean() * 1.5:
        logs.append("• ボラティリティ：安定範囲")
    else:
        logs.append("• ボラティリティ：高騰 or 低迷")

    return decision, logs

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
        decision, logs = rci_based_signal(df)
st.markdown(f"**{decision}**")
for log in logs:
    st.markdown(log)

if "買い" in decision:
    plan = generate_trade_plan(df)
    st.subheader("🧮 トレードプラン（RCI主軸型）")
    for k, v in plan.items():
        st.write(f"{k}: {v}")
