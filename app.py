import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- UI構成 ---
st.title("FXトレード分析ツール")

symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=0, key="symbol")
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=2, key="style")
use_dummy = st.checkbox("ダミーデータを使用する", key="use_dummy")

# --- 時間足と重み設定 ---
tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}

# --- ダミーデータ生成 ---
def generate_dummy_data(interval):
    now = datetime.now()
    dates = pd.date_range(end=now, periods=200, freq="1H" if "day" in interval else "15min")
    base = 150 if "JPY" in symbol else 1.1
    prices = base + np.cumsum(np.random.normal(0, 0.05, len(dates)))
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices + np.random.normal(0, 0.02, len(dates)),
        "high": prices + np.random.normal(0.05, 0.02, len(dates)),
        "low": prices - np.random.normal(0.05, 0.02, len(dates)),
        "close": prices,
        "volume": np.random.randint(100, 1000, len(dates))
    })
    df.set_index("datetime", inplace=True)
    return df

# --- データ取得関数 ---
@st.cache_data(ttl=3600)
def fetch_data(symbol, interval, use_dummy=False):
    if use_dummy:
        return generate_dummy_data(interval)

    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        st.error(f"❌ APIエラー発生：{data.get('message')}")
        raise ValueError(f"APIデータ取得エラー: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.astype(float).sort_index()
    return df

# --- インジケータ計算 ---
def calc_indicators(df):
    df["SMA_5"] = df["close"].rolling(5).mean()
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Upper"] = df["SMA_20"] + 2 * df["close"].rolling(20).std()
    df["Lower"] = df["SMA_20"] - 2 * df["close"].rolling(20).std()
    df["RCI"] = df["close"].rank().rolling(9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1])
    df["ADX"] = abs(df["MACD"] - df["Signal"]).rolling(14).mean()
    df["STD"] = df["close"].rolling(20).std()
    # --- ダウ理論（高値・安値切り下げ/上げ）仮ロジック
    df["HH"] = df["high"].rolling(3).apply(lambda x: x[-1] > x[-2] > x[-3])
    df["LL"] = df["low"].rolling(3).apply(lambda x: x[-1] < x[-2] < x[-3])
    return df

# --- 市場構造判定 ---
def detect_market_structure(last):
    trend = 0
    if last["ADX"] > 25: trend += 1
    elif last["ADX"] < 20: trend -= 1
    if abs(last["SMA_5"] - last["SMA_20"]) / last["close"] > 0.015: trend += 1
    else: trend -= 1
    if last["STD"] > last["close"] * 0.005: trend += 1
    else: trend -= 1
    return "トレンド" if trend >= 2 else "レンジ"

# --- シグナル抽出（拡張版） ---
def extract_signal(df):
    last = df.iloc[-1]
    logs = [f"• 市場判定：{detect_market_structure(last)}"]
    buy = sell = 0
    if last["MACD"] > last["Signal"]: buy += 1; logs.append("🟢 MACDゴールデンクロス")
    else: sell += 1; logs.append("🔴 MACDデッドクロス")
    if last["SMA_5"] > last["SMA_20"]: buy += 1; logs.append("🟢 SMA短期 > 長期")
    else: sell += 1; logs.append("🔴 SMA短期 < 長期")
    if last["close"] < last["Lower"]: buy += 1; logs.append("🟢 BB下限反発の可能性")
    elif last["close"] > last["Upper"]: sell += 1; logs.append("🔴 BB上限反発の可能性")
    else: logs.append("⚪ BB反発無し")
    if last["RCI"] > 0.5: buy += 1; logs.append("🟢 RCI上昇傾向")
    elif last["RCI"] < -0.5: sell += 1; logs.append("🔴 RCI下降傾向")
    else: logs.append("⚪ RCI未達")
    # プライスアクション風味（大陽線/大陰線判定）
    if df["close"].iloc[-1] > df["open"].iloc[-1] * 1.005: buy += 1; logs.append("🟢 陽線ブレイク")
    elif df["close"].iloc[-1] < df["open"].iloc[-1] * 0.995: sell += 1; logs.append("🔴 陰線ブレイク")
    else: logs.append("⚪ ローソク足判断無し")
    # ダウ理論風（切り上げ/下げ）判定
    if df["HH"].iloc[-1]: buy += 1; logs.append("🟢 高値切り上げ")
    if df["LL"].iloc[-1]: sell += 1; logs.append("🔴 安値切り下げ")
    return ("買い" if buy >= 3 and buy > sell else
            "売り" if sell >= 3 and sell > buy else
            "待ち"), logs, buy, sell

# --- トレードプラン（ATR or 高値/安値） ---
def get_recent_high_low(df, direction):
    hi = df["high"].rolling(20).max().iloc[-2]
    lo = df["low"].rolling(20).min().iloc[-2]
    return (hi, lo) if direction == "買い" else (lo, hi)

def suggest_trade_plan(price, atr, decision, tf, df):
    rr_comment = "（ATR）"
    if style == "スイング" and tf == "1day":
        hi, lo = get_recent_high_low(df, decision)
        tp = hi * 0.997 if decision == "買い" else lo * 0.997
        sl = lo * 1.003 if decision == "買い" else hi * 1.003
        rr_comment = "（高値/安値）"
    elif style == "デイトレード" and tf == "4h":
        hi, lo = get_recent_high_low(df, decision)
        tp = hi * 0.997 if decision == "買い" else lo * 0.997
        sl = lo * 1.003 if decision == "買い" else hi * 1.003
        rr_comment = "（高値/安値）"
    else:
        tp = price + atr * 1.6 if decision == "買い" else price - atr * 1.6
        sl = price - atr * 1.0 if decision == "買い" else price + atr * 1.0
    rr = abs((tp - price) / (sl - price)) if sl != price else 0
    pips_tp = abs(tp - price) * (100 if "JPY" in symbol else 10000)
    pips_sl = abs(sl - price) * (100 if "JPY" in symbol else 10000)
    return price, tp, sl, rr, pips_tp, pips_sl, rr_comment

# --- 実行ボタン ---
if st.button("実行", key="run_button"):
    timeframes = tf_map[style]
    total_buy_score = total_sell_score = 0
    score_log = []
    main_df = None
    main_tf = ""

    st.subheader(f"📊 通貨ペア：{symbol} | スタイル：{style}")
    st.markdown("### ⏱ 各時間足シグナル詳細\n\n凡例：🟢=買い、🔴=売り、⚪=未達")

    df_dict = {}
    for tf in timeframes:
        try:
            df = fetch_data(symbol.replace("/", ""), tf, use_dummy)
            if df is None:
                st.error(f"{tf} のデータ取得に失敗")
                continue
        except Exception as e:
            st.error(f"{tf} のデータ取得に失敗: {e}")
            continue
        df = calc_indicators(df)
        sig, logs, b, s = extract_signal(df)
        weight = tf_weights[tf]
        total_buy_score += b * weight
        total_sell_score += s * weight
        score_log.append((tf, b, s, weight))
        df_dict[tf] = df
        st.markdown(f"⏱ {tf} 判定：{sig}（スコア：{max(b,s):.1f}）")
        for log in logs:
            st.markdown(log)
        main_df = df
        main_tf = tf

    st.markdown("⸻\n### 🧭 エントリーガイド（総合評価）")
    if total_buy_score >= 2.4 and total_buy_score > total_sell_score:
        decision = "買い"
    elif total_sell_score >= 2.4 and total_sell_score > total_buy_score:
        decision = "売り"
    elif abs(total_buy_score - total_sell_score) >= 1.0:
        decision = "買い" if total_buy_score > total_sell_score else "売り"
    else:
        decision = "待ち"

    st.markdown(f"総合スコア：{total_buy_score:.2f}（買） / {total_sell_score:.2f}（売）")
    for tf, b, s, w in score_log:
        st.markdown(f"　• {tf}：買 {b} × {w} = {b*w:.2f} / 売 {s} × {w} = {s*w:.2f}")
    if decision == "買い":
        st.success("✅ 買いシグナル")
    elif decision == "売り":
        st.warning("✅ 売りシグナル")
    else:
        st.info("⏸ エントリー見送り")

    st.markdown("⸻\n### 🎯 トレードプラン（想定）")
    if main_df is not None and decision != "待ち":
        price = main_df["close"].iloc[-1]
        atr = main_df["close"].rolling(14).std().iloc[-1]
        entry, tp, sl, rr, ptp, psl, comment = suggest_trade_plan(price, atr, decision, main_tf, main_df)
        st.markdown(f"• エントリー価格：{entry:.5f}")
        st.markdown(f"• TP：{tp:.5f}（+{ptp:.0f}pips）")
        st.markdown(f"• SL：{sl:.5f}（−{psl:.0f}pips）")
        st.markdown(f"• リスクリワード：{rr:.2f} {comment}")
    else:
        st.markdown("現在はエントリー待機中です。")
