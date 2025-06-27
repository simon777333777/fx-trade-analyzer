import streamlit as st
import requests
import pandas as pd
import numpy as np

# --- 設定 ---
API_KEY = "YOUR_TWELVE_DATA_API_KEY"
SYMBOL = "GBP/JPY"
INTERVALS = ["15min", "1h", "4h", "1day"]

# トレードスタイル別重み
STYLE_WEIGHTS = {
    "スキャル": {"rsi": 0.4, "macd": 0.3, "bollinger": 0.3},
    "デイトレ": {"rsi": 0.3, "macd": 0.4, "bollinger": 0.3},
    "スイング": {"rsi": 0.3, "macd": 0.3, "bollinger": 0.4},
}

# インジケーター判定閾値（例）
THRESHOLDS = {
    "rsi": {"buy": 30, "sell": 70},
    "macd": {"buy": 0, "sell": 0},
    "bollinger": {"buy": -1, "sell": 1},  # 乖離のzスコア例
}

# 進捗バー用にスコアを0〜1に正規化
def normalize_score(score, max_score=3):
    return min(max(score / max_score, 0), 1)

# Twelve Data API でOHLC取得
def fetch_ohlc(symbol, interval, outputsize=100):
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "format": "JSON",
        "outputsize": outputsize
    }
    r = requests.get(url, params=params)
    data = r.json()
    if "values" in data:
        df = pd.DataFrame(data["values"])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    else:
        st.error("API取得エラーまたはデータなし")
        return None

# 簡単なインジケーター計算
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calc_bollinger(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    # ボリンジャーバンド幅のzスコア（例）
    band_width = (series - sma) / std
    return upper, lower, band_width

# トレンド・レンジ判定（例：ADXや乖離率で単純判定）
def trend_or_range(df):
    # 簡易的にSMA乖離率を使い、乖離が大きければトレンド、小さければレンジと判定
    sma20 = df["close"].rolling(window=20).mean()
    deviation = abs(df["close"] - sma20) / sma20
    avg_dev = deviation.rolling(window=20).mean().iloc[-1]
    if avg_dev > 0.03:
        return "トレンド"
    else:
        return "レンジ"

# 各インジケーター判定（買い=1、売り=-1、中立=0）
def indicator_signal_rsi(rsi):
    if rsi < THRESHOLDS["rsi"]["buy"]:
        return 1
    elif rsi > THRESHOLDS["rsi"]["sell"]:
        return -1
    else:
        return 0

def indicator_signal_macd(macd_line, signal_line):
    if macd_line > signal_line:
        return 1
    elif macd_line < signal_line:
        return -1
    else:
        return 0

def indicator_signal_bollinger(band_width):
    if band_width < THRESHOLDS["bollinger"]["buy"]:
        return 1
    elif band_width > THRESHOLDS["bollinger"]["sell"]:
        return -1
    else:
        return 0

# スコア計算（スタイル別重み込み）
def calc_score(signals, style):
    weights = STYLE_WEIGHTS[style]
    score = 0
    total_weight = 0
    for ind, val in signals.items():
        w = weights.get(ind, 0)
        total_weight += w
        score += val * w
    # 最大スコアはtotal_weight * 1（買い）と仮定して正規化
    normalized = (score + total_weight) / (2 * total_weight)  # -total_weight〜+total_weight → 0〜1に変換
    return normalized * 3  # 最大3点満点スコア

# Streamlit UI
st.title("FXトレード分析ツール（トレンド／レンジ判定＋スコア＋視覚化）")

style = st.selectbox("トレードスタイルを選択", list(STYLE_WEIGHTS.keys()))
interval = st.selectbox("時間足を選択", INTERVALS)

if st.button("分析開始"):
    df = fetch_ohlc(SYMBOL, interval)
    if df is not None and len(df) > 30:
        st.write(f"最新データ日時: {df['datetime'].iloc[-1]}")
        
        trend_status = trend_or_range(df)
        st.markdown(f"### 相場構造判定: **{trend_status}相場**")
        
        # インジケーター計算（最新1本のみ判定）
        rsi = calc_rsi(df["close"]).iloc[-1]
        macd_line, signal_line, _ = calc_macd(df["close"])
        macd_val = macd_line.iloc[-1]
        macd_signal = signal_line.iloc[-1]
        upper, lower, band_width = calc_bollinger(df["close"])
        bb_val = band_width.iloc[-1]
        
        # シグナル取得
        signals = {
            "rsi": indicator_signal_rsi(rsi),
            "macd": indicator_signal_macd(macd_val, macd_signal),
            "bollinger": indicator_signal_bollinger(bb_val)
        }
        
        # スコア計算
        score = calc_score(signals, style)
        
        # スコア視覚化バー
        st.write("### スコア（0〜3点）")
        progress_val = normalize_score(score)
        bar_color = "green" if score > 2 else "orange" if score > 1 else "red"
        st.progress(progress_val)
        st.markdown(f"<div style='color:{bar_color}; font-weight:bold; font-size:24px;'>スコア: {score:.2f}</div>", unsafe_allow_html=True)
        
        # 詳細シグナル表示
        st.write("### インジケーター判定詳細")
        col1, col2, col3 = st.columns(3)
        col1.metric("RSI", f"{rsi:.1f}", "買い" if signals["rsi"]==1 else "売り" if signals["rsi"]==-1 else "中立")
        col2.metric("MACD", f"{macd_val:.4f}", "買い" if signals["macd"]==1 else "売り" if signals["macd"]==-1 else "中立")
        col3.metric("Bollinger Band Z", f"{bb_val:.2f}", "買い" if signals["bollinger"]==1 else "売り" if signals["bollinger"]==-1 else "中立")
        
        # トレードアドバイス
        if score >= 2.0:
            advice = "買いのチャンスが強いです。"
        elif score <= 1.0:
            advice = "売りのチャンスまたは様子見を検討してください。"
        else:
            advice = "中立。大きな動きを待つのが良いでしょう。"
        st.write(f"### トレードアドバイス: {advice}")
