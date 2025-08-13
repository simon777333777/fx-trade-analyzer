import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import List

st.set_page_config(page_title="RCI主軸FXトレード分析（一覧）", layout="centered")
API_KEY = st.secrets["API_KEY"]

st.title("📈 RCI主軸FXトレード分析ツール（一覧）")

# ---------- UI: 基本設定 ----------
pairs_all = ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"]
pairs = st.multiselect("監視する通貨ペア（複数選択可）", pairs_all, default=pairs_all)

style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=1)
use_dummy = st.checkbox("📦 ダミーデータで実行（テストモード）", value=False)

tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}

all_tfs = sorted({tf for v in tf_map.values() for tf in v})
selected_tfs = st.multiselect("時間足（複数選択可）", all_tfs, default=tf_map[style])

st.markdown("---")

# ---------- 内部：キャッシュ用ダミーデータ / データ取得 ----------
def get_dummy_data():
    idx = pd.date_range(end=pd.Timestamp.now(), periods=500, freq="H")
    price = np.cumsum(np.random.randn(len(idx))) + 150
    df = pd.DataFrame({
        "datetime": idx,
        "open": price + np.random.randn(len(idx)) * 0.2,
        "high": price + 0.5 + np.abs(np.random.randn(len(idx)) * 0.2),
        "low": price - 0.5 - np.abs(np.random.randn(len(idx)) * 0.2),
        "close": price + np.random.randn(len(idx)) * 0.1,
        "volume": 1000
    }).set_index("datetime")
    return df

@st.cache_data(ttl=300)
def fetch_data(symbol: str, interval: str, use_dummy_flag: bool) -> pd.DataFrame:
    if use_dummy_flag:
        return get_dummy_data()
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
    r = requests.get(url, timeout=10)
    data = r.json()
    if "values" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

# ---------- 指標計算 ----------
def calc_rci(series):
    n = len(series)
    if series.isna().any() or n < 2:
        return np.nan
    price_rank = series.rank(method="average")
    time_rank = pd.Series(np.arange(1, n+1), index=series.index)
    d = price_rank - time_rank
    denom = n * (n**2 - 1)
    if denom == 0:
        return np.nan
    rho = 1 - (6 * (d**2).sum()) / denom
    return rho

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    for period in [9, 26, 52]:
        df[f"RCI_{period}"] = df["close"].rolling(period).apply(lambda x: calc_rci(pd.Series(x)) if len(x)==period else np.nan, raw=False)
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["ATR"] = df["high"].combine(df["low"], max) - df["low"].combine(df["close"].shift(), min)
    df["ATR"] = df["ATR"].rolling(14).mean()
    return df

def get_thresholds(style):
    if style == "スキャルピング":
        return 0.7, 0.4
    elif style == "デイトレード":
        return 0.6, 0.3
    else:
        return 0.5, 0.2

def determine_trend(df, style):
    short_thr, long_thr = get_thresholds(style)
    last = df.iloc[-1]
    rci_52 = last.get("RCI_52", 0)
    if rci_52 > long_thr:
        return "上昇"
    elif rci_52 < -long_thr:
        return "下降"
    else:
        return "ニュートラル"

def rci_based_signal(df, style, higher_trends: List[str]):
    if df.empty or len(df)<10:
        return 0, None, None, ["データ不足"], {}
    last = df.iloc[-1]
    short_thr, long_thr = get_thresholds(style)
    rci_9 = last.get("RCI_9", np.nan)
    rci_26_now = last.get("RCI_26", np.nan)
    rci_26_prev = df["RCI_26"].iloc[-2] if len(df)>=2 else np.nan
    rci_52 = last.get("RCI_52", np.nan)
    macd = last.get("MACD", np.nan)
    signal = last.get("Signal", np.nan)
    macd_diff = df["MACD"].diff().iloc[-1] if "MACD" in df.columns else 0
    macd_cross_up = (macd > signal) and (macd_diff > 0)
    macd_cross_down = (macd < signal) and (macd_diff < 0)
    close = last["close"]
    atr = last.get("ATR", 0)
    logs = []

    # --- 条件判定（簡易例） ---
    cond_buy = rci_9>short_thr and rci_26_now>rci_26_prev and rci_52>long_thr and macd_cross_up
    cond_sell = rci_9<-short_thr and rci_26_now<rci_26_prev and rci_52<-long_thr and macd_cross_down

    # --- 上位足整合 ---
    def alignment(direction):
        if not higher_trends:
            return False
        if direction=="買い":
            return all(t=="上昇" for t in higher_trends)
        else:
            return all(t=="下降" for t in higher_trends)

    score = 0
    signal_type = None
    mode = None
    if cond_buy:
        signal_type = "買い"
        mode = "順張り"
        # スコア連続化: RCI_9の大きさ × 上位足整合
        base_score = min(max(rci_9*10,2),7)
        score = base_score if alignment("買い") else base_score*0.6
        logs.append(f"買い判定: RCI9={rci_9:.2f}, 上位足整合={alignment('買い')}, score={score:.1f}")
    elif cond_sell:
        signal_type = "売り"
        mode = "順張り"
        base_score = min(max(abs(rci_9)*10,2),7)
        score = -base_score if alignment("売り") else -base_score*0.6
        logs.append(f"売り判定: RCI9={rci_9:.2f}, 上位足整合={alignment('売り')}, score={score:.1f}")
    else:
        logs.append("条件未成立（保留）")
    return score, signal_type, mode, logs, {}

def generate_trade_plan(df, signal_score, signal_type, mode, higher_trends):
    entry = df["close"].iloc[-1]
    atr = df["ATR"].iloc[-1] if "ATR" in df.columns else 0
    if signal_type=="買い":
        tp = entry + atr*2.0
        sl = entry - atr*1.0
    elif signal_type=="売り":
        tp = entry - atr*2.0
        sl = entry + atr*1.0
    else:
        return {}
    rr = round(abs((tp-entry)/(entry-sl)),2) if (entry-sl)!=0 else 0
    comment = "強" if abs(signal_score)>=6 else ("中" if abs(signal_score)>=4 else "弱")
    alignment_str = "整合" if signal_type and all(
        t==("上昇" if signal_type=="買い" else "下降") for t in higher_trends) else "不整合"
    return {
        "エントリー価格": round(entry,4),
        "利確（TP）": round(tp,4),
        "損切（SL）": round(sl,4),
        "RR": rr,
        "コメント": comment,
        "シグナル種類": f"{signal_type} ({mode})",
        "上位足整合": alignment_str
    }

# ---------- 実行ボタン ----------
if st.button("🔍 一覧スキャン実行"):
    if not pairs:
        st.warning("監視する通貨ペアを1つ以上選んでください。")
    elif not selected_tfs:
        st.warning("時間足を1つ以上選んでください。")
    else:
        results=[]
        progress=st.progress(0)
        total=len(pairs)*len(selected_tfs)
        i=0
        for pair in pairs:
            for tf in selected_tfs:
                i+=1
                progress.progress(int(i/total*100))
                df = fetch_data(pair, tf, use_dummy)
                if df.empty:
                    results.append({"ペア":pair,"時間足":tf,"シグナル":"データ取得失敗","スコア":None})
                    continue
                df=calc_indicators(df)

                # 上位足トレンド取得
                higher_trends = []
                for htf in selected_tfs:
                    if htf==tf: continue
                    hdf = fetch_data(pair, htf, use_dummy)
                    if hdf.empty: continue
                    hdf=calc_indicators(hdf)
                    higher_trends.append(determine_trend(hdf, style))

                score, signal_type, mode, logs, _ = rci_based_signal(df, style, higher_trends)
                plan = generate_trade_plan(df, score, signal_type, mode, higher_trends) if signal_type else {}
                results.append({
                    "ペア": pair,
                    "時間足": tf,
                    "シグナル": signal_type if signal_type else "なし",
                    "スコア": round(score,1) if score else None,
                    "エントリー": plan.get("エントリー価格"),
                    "TP": plan.get("利確（TP）"),
                    "SL": plan.get("損切（SL）"),
                    "RR": plan.get("RR"),
                    "備考": "; ".join(logs[:2])
                })
        progress.empty()
        df_res=pd.DataFrame(results)
        st.subheader("📋 シグナル一覧")
        st.dataframe(df_res, use_container_width=True)
