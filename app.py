import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

# -----------------------------
# APIキー設定（ベースコード準拠）
# -----------------------------
API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")  # 環境変数から取得
BASE_URL = "https://www.alphavantage.co/query"

# -----------------------------
# データ取得（15分足）
# -----------------------------
@st.cache_data(ttl=600)
def get_fx_data(symbol: str) -> pd.DataFrame:
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": symbol[:3],
        "to_symbol": symbol[3:],
        "interval": "15min",
        "apikey": API_KEY,
        "outputsize": "compact"
    }
    res = requests.get(BASE_URL, params=params)
    data = res.json()

    if "Time Series FX (15min)" not in data:
        return None

    df = pd.DataFrame.from_dict(data["Time Series FX (15min)"], orient="index", dtype=float)
    df.columns = ["open", "high", "low", "close"]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

# -----------------------------
# RCI計算
# -----------------------------
def calculate_rci(series: pd.Series, period: int) -> pd.Series:
    def _calc(window):
        if window.isnull().any():
            return np.nan
        rank_price = window.rank(ascending=False)
        rank_time = pd.Series(range(1, len(window)+1), index=window.index)
        d = (rank_price - rank_time) ** 2
        rci = 1 - (6 * d.sum()) / (period * (period**2 - 1))
        return rci * 100
    return series.rolling(period).apply(_calc, raw=False)

# -----------------------------
# シグナル判定ロジック（パターン②）
# -----------------------------
def evaluate(df: pd.DataFrame, setting: dict) -> dict:
    df["rci_s"] = calculate_rci(df["close"], setting["short"])
    df["rci_m"] = calculate_rci(df["close"], setting["middle"])
    df["rci_l"] = calculate_rci(df["close"], setting["long"])
    latest = df.iloc[-1]

    if pd.isna(latest[["rci_s", "rci_m", "rci_l"]]).any():
        return {"signal": "データ不足", "logs": ["RCI計算に必要なデータが不足しています"]}

    score = 0
    logs = []

    if latest["rci_s"] > 60:
        score += 2
        logs.append("短期RCIが上昇圏（+2）")
    elif latest["rci_s"] < -60:
        score -= 2
        logs.append("短期RCIが下降圏（-2）")

    if latest["rci_m"] > 60:
        score += 2
        logs.append("中期RCIが上昇圏（+2）")
    elif latest["rci_m"] < -60:
        score -= 2
        logs.append("中期RCIが下降圏（-2）")

    if latest["rci_l"] > 60:
        score += 1
        logs.append("長期RCIが上昇圏（+1）")
    elif latest["rci_l"] < -60:
        score -= 1
        logs.append("長期RCIが下降圏（-1）")

    if score >= 4:
        signal = "買い（強）"
    elif score >= 2:
        signal = "買い（中）"
    elif score <= -4:
        signal = "売り（強）"
    elif score <= -2:
        signal = "売り（中）"
    else:
        signal = "見送り"

    entry = round(latest["close"], 3)
    pip = setting["tp_sl_pips"] * 0.01
    tp = round(entry + pip, 3)
    sl = round(entry - pip, 3)

    return {
        "signal": signal,
        "score": score,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "rr": 1.0,
        "logs": logs
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 FXシグナル分析ツール（RCIロジック / TP・SL付き）")

symbol = st.selectbox("通貨ペアを選択", ["USDJPY", "EURUSD", "GBPJPY", "AUDJPY"])
style = st.radio("トレードスタイルを選択", ["スキャル", "デイトレ", "スイング"])

style_map = {
    "スキャル": 20,
    "デイトレ": 50,
    "スイング": 100
}
tp_sl_pips = style_map[style]

setting = {
    "short": 9,
    "middle": 26,
    "long": 52,
    "tp_sl_pips": tp_sl_pips
}

if st.button("🔍 シグナルチェック"):
    with st.spinner("データ取得中..."):
        df = get_fx_data(symbol)

    if df is None:
        st.error("データ取得に失敗しました。APIキーや通貨ペアを確認してください。")
    else:
        result = evaluate(df, setting)

        st.subheader("✅ シグナル結果")
        st.write(f"### シグナル：{result['signal']}")
        st.write(f"- スコア：{result['score']}")
        if "entry" in result:
            st.write(f"- エントリー価格：{result['entry']}")
            st.write(f"- 利確（TP）：{result['tp']}")
            st.write(f"- 損切（SL）：{result['sl']}")
            st.write(f"- リスクリワード：{result['rr']}")

        st.subheader("📌 根拠ログ")
        for log in result["logs"]:
            st.write(f"- {log}")
