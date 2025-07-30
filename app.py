import streamlit as st
import pandas as pd
import numpy as np
import requests

# ----------------------------------------
# API設定・トレードスタイル選択
# ----------------------------------------
st.title("RCI主軸FX分析ツール")

api_key = st.text_input("APIキーを入力してください", type="password")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY"])
style = st.radio("トレードスタイル", ["スキャル", "デイトレ", "スイング"])

# ----------------------------------------
# RCIパラメータ設定（スタイル別）
# ----------------------------------------
if style == "スキャル":
    rci_params = [9, 26]
elif style == "デイトレ":
    rci_params = [12, 52]
else:  # スイング
    rci_params = [26, 104]

# ----------------------------------------
# データ取得関数（仮）
# ----------------------------------------
def fetch_data(symbol: str, api_key: str, interval: str = "1h", length: int = 300) -> pd.DataFrame:
    """APIからOHLCVデータ取得（仮のローカルサンプルで代用）"""
    # 本来はAPIリクエスト。ここではダミーデータ生成。
    dates = pd.date_range(end=pd.Timestamp.now(), periods=length, freq="1h")
    prices = np.cumsum(np.random.randn(length)) + 150
    highs = prices + np.random.rand(length)
    lows = prices - np.random.rand(length)
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices + np.random.randn(length)*0.2,
        "high": highs,
        "low": lows,
        "close": prices,
    })
    df.set_index("datetime", inplace=True)
    return df

# ----------------------------------------
# RCI算出
# ----------------------------------------
def calculate_rci(df: pd.DataFrame, period: int) -> pd.Series:
    rci = []
    for i in range(len(df)):
        if i < period:
            rci.append(np.nan)
            continue
        close = df['close'].iloc[i - period + 1:i + 1]
        time_rank = np.arange(period, 0, -1)
        price_rank = close.rank(ascending=False).values
        diff_sq = (time_rank - price_rank) ** 2
        rci_value = (1 - (6 * np.sum(diff_sq)) / (period * (period ** 2 - 1))) * 100
        rci.append(rci_value)
    return pd.Series(rci, index=df.index)

# ----------------------------------------
# シグナル判定
# ----------------------------------------
def detect_signal(df: pd.DataFrame, rci_short: pd.Series, rci_long: pd.Series):
    signal = []
    for i in range(len(df)):
        if np.isnan(rci_short[i]) or np.isnan(rci_long[i]):
            signal.append("")
            continue
        if rci_short[i] > 80 and rci_long[i] > 50:
            signal.append("売り")
        elif rci_short[i] < -80 and rci_long[i] < -50:
            signal.append("買い")
        else:
            signal.append("")
    return pd.Series(signal, index=df.index)

# ----------------------------------------
# スイング高値/安値の取得（TP/SL用）
# ----------------------------------------
def get_swing_points(df: pd.DataFrame, index: int, window: int = 10):
    """直近の高値・安値をエントリー基準に取得"""
    high = df["high"].iloc[index-window:index].max()
    low = df["low"].iloc[index-window:index].min()
    return high, low

# ----------------------------------------
# トレードプラン生成
# ----------------------------------------
def generate_trade_plan(df: pd.DataFrame, signal: pd.Series):
    plans = []
    for i in range(len(signal)):
        if signal[i] not in ["買い", "売り"]:
            plans.append(None)
            continue
        entry_price = df["close"].iloc[i]
        swing_high, swing_low = get_swing_points(df, i)

        if signal[i] == "買い":
            sl = swing_low
            tp = entry_price + (entry_price - sl)
        else:  # 売り
            sl = swing_high
            tp = entry_price - (sl - entry_price)

        rr = abs(tp - entry_price) / abs(entry_price - sl) if sl != entry_price else 0
        plan = {
            "エントリー価格": round(entry_price, 3),
            "TP": round(tp, 3),
            "SL": round(sl, 3),
            "RR": round(rr, 2),
            "注意": "※リスクリワード比が1.0未満です。" if rr < 1.0 else ""
        }
        plans.append(plan)
    return plans

# ----------------------------------------
# メイン処理
# ----------------------------------------
if st.button("分析開始"):

    if not api_key:
        st.warning("APIキーを入力してください")
        st.stop()

    df = fetch_data(symbol, api_key)
    rci_short = calculate_rci(df, rci_params[0])
    rci_long = calculate_rci(df, rci_params[1])
    df["RCI短期"] = rci_short
    df["RCI長期"] = rci_long
    df["シグナル"] = detect_signal(df, rci_short, rci_long)
    df["トレードプラン"] = generate_trade_plan(df, df["シグナル"])

    latest_signal = df["シグナル"].iloc[-1]
    st.subheader(f"最新シグナル: {latest_signal or 'なし'}")

    if latest_signal in ["買い", "売り"]:
        plan = df["トレードプラン"].iloc[-1]
        if plan:
            st.subheader("📊 トレードプラン")
            st.write(f"エントリー価格: {plan['エントリー価格']}")
            st.write(f"TP: {plan['TP']}")
            st.write(f"SL: {plan['SL']}")
            st.write(f"リスクリワード比: {plan['RR']}")
            if plan["RR"] < 1.0:
                st.warning(plan["注意"])
        else:
            st.info("有効なトレードプランが生成できませんでした。")
    else:
        st.info("現在、エントリーシグナルは出ていません。")

    # チャート確認用
    st.line_chart(df[["close", "RCI短期", "RCI長期"]].dropna())
