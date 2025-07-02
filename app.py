import streamlit as st
import pandas as pd
import numpy as np
import requests

# --- APIキー ---
API_KEY = st.secrets["API_KEY"]

# --- UI設定 ---
st.title("FXトレード分析ツール")

# 通貨ペア選択
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=0)

# トレードスタイル選択
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=2)

# ダミーデータ切り替えチェックボックス
use_dummy = st.checkbox("ダミーデータで実行する", value=False)

# --- 時間足と重み設定 ---
tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}

# --- APIからのデータ取得関数 ---
@st.cache_data(ttl=60*60)
def fetch_data(symbol, interval, use_dummy=False):
    if use_dummy:
        # ダミーデータ生成（簡易）
        date_rng = pd.date_range(end=pd.Timestamp.now(), periods=500, freq=interval.upper())
        np.random.seed(0)
        df = pd.DataFrame({
            "datetime": date_rng,
            "open": np.random.rand(len(date_rng)) + 100,
            "high": np.random.rand(len(date_rng)) + 101,
            "low": np.random.rand(len(date_rng)) + 99,
            "close": np.random.rand(len(date_rng)) + 100,
            "volume": np.random.randint(100, 1000, size=len(date_rng)),
        })
        df = df.set_index("datetime")
        return df

    # APIデータ取得
    url = f"https://api.twelvedata.com/time_series?symbol={symbol.replace('/', '')}&interval={interval}&outputsize=500&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        st.error(f"APIエラー: {data.get('message', '不明なエラー')}")
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df

# --- インジケーター計算 ---
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
    return df

# --- 市場構造の判定 ---
def detect_market_structure(last):
    trend = 0
    if last["ADX"] > 25: trend += 1
    elif last["ADX"] < 20: trend -= 1
    if abs(last["SMA_5"] - last["SMA_20"]) / last["close"] > 0.015: trend += 1
    else: trend -= 1
    if last["STD"] > last["close"] * 0.005: trend += 1
    else: trend -= 1
    return "トレンド" if trend >= 2 else "レンジ"

# --- プライスアクション・ダウ理論判定 ---
def detect_price_action(df):
    recent = df["close"].iloc[-4:]
    if all(recent[i] < recent[i + 1] for i in range(3)):
        return "上昇3連", 1
    elif all(recent[i] > recent[i + 1] for i in range(3)):
        return "下降3連", -1
    return "無し", 0

def detect_dow_theory(df):
    highs = df["high"].rolling(3).apply(lambda x: x[1] > x[0] and x[1] > x[2])
    lows = df["low"].rolling(3).apply(lambda x: x[1] < x[0] and x[1] < x[2])
    if highs.iloc[-1] == 1 and lows.iloc[-1] == 0:
        return "高値切り上げ", 1
    elif lows.iloc[-1] == 1 and highs.iloc[-1] == 0:
        return "安値切り下げ", -1
    return "無し", 0

# --- ダウ理論判定 ---
def dow_theory_signal(df):
    # 高値・安値の切り上げ・切り下げ判定
    highs = df['high'].rolling(window=3).apply(lambda x: x[2] > x[1] > x[0])
    lows = df['low'].rolling(window=3).apply(lambda x: x[2] > x[1] > x[0])
    last_high_trend = highs.iloc[-1]
    last_low_trend = lows.iloc[-1]
    if np.isnan(last_high_trend) or np.isnan(last_low_trend):
        return "待ち", []
    if last_high_trend and last_low_trend:
        return "買い", ["🟢 ダウ理論：高値・安値の切り上げ確認"]
    if not last_high_trend and not last_low_trend:
        return "売り", ["🔴 ダウ理論：高値・安値の切り下げ確認"]
    return "待ち", ["⚪ ダウ理論：トレンド未確認"]

# --- プライスアクション判定（単純な陽線・陰線判定例） ---
def price_action_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    logs = []
    if last["close"] > last["open"]:
        logs.append("🟢 プライスアクション：陽線")
        if last["close"] > prev["high"]:
            logs.append("🟢 プライスアクション：上抜け陽線")
            return "買い", logs
        return "買い", logs
    elif last["close"] < last["open"]:
        logs.append("🔴 プライスアクション：陰線")
        if last["close"] < prev["low"]:
            logs.append("🔴 プライスアクション：下抜け陰線")
            return "売り", logs
        return "売り", logs
    else:
        logs.append("⚪ プライスアクション：方向感なし")
        return "待ち", logs

# --- 総合シグナル判定 ---
def combined_signal(df):
    signals = []
    buy_score = 0
    sell_score = 0
    logs = []

    # 既存のMACD/SMA/BB/RCI判定を実施
    sig_main, logs_main, b_main, s_main = extract_signal(df)
    signals.append(sig_main)
    buy_score += b_main
    sell_score += s_main
    logs.extend(logs_main)

    # ダウ理論判定
    sig_dow, logs_dow = dow_theory_signal(df)
    signals.append(sig_dow)
    if sig_dow == "買い":
        buy_score += 1
    elif sig_dow == "売り":
        sell_score += 1
    logs.extend(logs_dow)

    # プライスアクション判定
    sig_pa, logs_pa = price_action_signal(df)
    signals.append(sig_pa)
    if sig_pa == "買い":
        buy_score += 1
    elif sig_pa == "売り":
        sell_score += 1
    logs.extend(logs_pa)

    # 判定基準
    if buy_score >= 3 and buy_score > sell_score:
        final_signal = "買い"
    elif sell_score >= 3 and sell_score > buy_score:
        final_signal = "売り"
    else:
        final_signal = "待ち"

    return final_signal, logs, buy_score, sell_score

# --- UI表示での各時間足シグナル詳細 ---
def show_signals(symbol, style, use_dummy):
    timeframes = tf_map[style]
    total_buy_score = 0
    total_sell_score = 0
    score_log = []
    df_dict = {}
    main_df = None
    main_tf = ""

    st.subheader(f"📊 通貨ペア：{symbol} | スタイル：{style}")
    st.markdown("### ⏱ 各時間足シグナル詳細\n\n凡例：🟢=買い、🔴=売り、⚪=未達")

    for tf in timeframes:
        if use_dummy:
            df = get_dummy_data(symbol, tf)  # ダミーデータ関数はパート5で定義
        else:
            df = fetch_data(symbol.replace("/", ""), tf)
            if df is None:
                st.error(f"{tf}のデータ取得に失敗")
                continue
            df = calc_indicators(df)

        sig, logs, b, s = combined_signal(df)
        weight = tf_weights[tf]
        total_buy_score += b * weight
        total_sell_score += s * weight
        score_log.append((tf, b, s, weight))
        df_dict[tf] = df

        st.markdown(f"⏱ {tf} 判定：{sig}（スコア：{max(b, s):.1f}）")
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

    return decision, main_df, main_tf

import random

# --- ダミーデータ生成 ---
def get_dummy_data(symbol, tf):
    # 日付の生成
    periods = 500
    freq_map = {
        "5min": "5T",
        "15min": "15T",
        "1h": "60T",
        "4h": "240T",
        "1day": "1D"
    }
    freq = freq_map.get(tf, "1D")
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq=freq)

    base_price = {
        "USDJPY": 140,
        "EURUSD": 1.12,
        "GBPJPY": 160,
        "AUDUSD": 0.75
    }
    base = base_price.get(symbol.replace("/", ""), 1.0)

    # ロジックで買いシグナル出るように調整（単純な上昇トレンド）
    prices = [base + i*0.01 for i in range(periods)]

    data = {
        "datetime": dates,
        "open": prices,
        "high": [p + 0.005 for p in prices],
        "low": [p - 0.005 for p in prices],
        "close": [p + 0.003 for p in prices],
        "volume": [100 + random.randint(-10, 10) for _ in range(periods)]
    }
    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)
    return df

# --- メイン ---
def main():
    st.title("FXトレード分析ツール")
    symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=0)
    style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=2)
    use_dummy = st.checkbox("ダミーデータを使用する（API制限回避）", value=False)

    if st.button("実行"):
        try:
            decision, main_df, main_tf = show_signals(symbol, style, use_dummy)
            # トレードプラン表示は今は保留（後日追加可能）
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
