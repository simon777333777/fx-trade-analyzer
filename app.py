import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="FXトレード分析", layout="centered")

API_KEY = st.secrets["API_KEY"]

# --- キャッシュクリアボタン ---
if st.button("🧹 キャッシュをクリア"):
    st.cache_data.clear()
    st.success("キャッシュをクリアしました。")

# --- UI ---
st.title("FXトレード分析ツール")
symbol = st.selectbox("通貨ペアを選択", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"], index=2)
style = st.selectbox("トレードスタイルを選択", ["スキャルピング", "デイトレード", "スイング"], index=2)
use_dummy = st.checkbox("📦 ダミーデータで実行", value=False)

# --- 時間足と重み ---
tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}
tf_weights = {"5min": 0.2, "15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.4}

# --- ダミーデータ生成 ---
def get_dummy_data():
    idx = pd.date_range(end=pd.Timestamp.now(), periods=500, freq="H")
    np.random.seed(0)
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
    df["volume"] = df.get("volume", 1000)
    return df.apply(pd.to_numeric, errors="coerce")

def calc_indicators(df):
    df["SMA_5"] = df["close"].rolling(5).mean()
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Upper"] = df["SMA_20"] + 2 * df["close"].rolling(20).std()
    df["Lower"] = df["SMA_20"] - 2 * df["close"].rolling(20).std()
    df["RCI"] = df["close"].rank().rolling(9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0,1])
    df["ADX"] = (df["MACD"] - df["Signal"]).abs().rolling(14).mean()
    df["STD"] = df["close"].rolling(20).std()
    return df

def detect_market_structure(last):
    t=0
    if last["ADX"]>25: t+=1
    if abs(last["SMA_5"]-last["SMA_20"])/last["close"]>0.015: t+=1
    if last["STD"]>last["close"]*0.005: t+=1
    return "トレンド" if t>=2 else "レンジ"

def detect_dow(df):
    highs = df["high"].iloc[-3:]
    lows = df["low"].iloc[-3:]
    is_hh = highs.iloc[2] > highs.iloc[1] > highs.iloc[0]
    is_ll = lows.iloc[2] < lows.iloc[1] < lows.iloc[0]
    if is_hh and is_ll: return "保ち合い","⚪ ダウ理論：保ち合い"
    if is_hh: return "上昇","🟢 高値切り上げ"
    if is_ll: return "下降","🔴 安値切り下げ"
    return "不明","⚪ ダウ理論未達"

def detect_price_action(df):
    l2, l1 = df.iloc[-2], df.iloc[-1]
    if l2.close<l2.open and l1.close>l1.open and l1.close>l2.open and l1.open<l2.close:
        return "🟢 陽線包み足"
    if l2.close>l2.open and l1.close<l1.open and l1.close<l2.open and l1.open>l2.close:
        return "🔴 陰線包み足"
    return "⚪ プライスアクション未達"

def extract_signal(df):
    last = df.iloc[-1]
    market = detect_market_structure(last)
    logs=[f"• 市場判定：{market}"]
    buy=sell=0
    tw=2 if market=="トレンド" else 1
    rw=2 if market=="レンジ" else 1
    macd, sig = df["MACD"].iloc[-3:], df["Signal"].iloc[-3:]
    if macd.iloc[-1]>sig.iloc[-1] and macd.is_monotonic_increasing: buy+=tw; logs.append("🟢 MACDゴールデンクロス")
    elif macd.iloc[-1]<sig.iloc[-1] and macd.is_monotonic_decreasing: sell+=tw; logs.append("🔴 MACDデッドクロス")
    else: logs.append("⚪ MACD微妙")
    sma5,sma20 = df["SMA_5"].iloc[-3:], df["SMA_20"].iloc[-3:]
    if sma5.iloc[-1]>sma20.iloc[-1] and sma5.is_monotonic_increasing: buy+=tw; logs.append("🟢 SMA上昇")
    elif sma5.iloc[-1]<sma20.iloc[-1] and sma5.is_monotonic_decreasing: sell+=tw; logs.append("🔴 SMA下降")
    else: logs.append("⚪ SMA微妙")
    if last.close<last.Lower: buy+=rw; logs.append("🟢 BB下限反発")
    elif last.close>last.Upper: sell+=rw; logs.append("🔴 BB上限反発")
    else: logs.append("⚪ BBなし")
    if last.RCI>0.5: buy+=rw; logs.append("🟢 RCI上昇")
    elif last.RCI< -0.5: sell+=rw; logs.append("🔴 RCI下降")
    else: logs.append("⚪ RCI微妙")
    _, dlog = detect_dow(df)
    if "高値" in dlog: buy+=1
    elif "安値" in dlog: sell+=1
    logs.append(dlog)
    pa = detect_price_action(df)
    if "陽線" in pa: buy+=1
    elif "陰線" in pa: sell+=1
    logs.append(pa)
    sigtype = "買い" if buy>=4 and buy>sell else "売り" if sell>=4 and sell>buy else "待ち"
    return sigtype, logs

def get_hi_lo(df, style):
    if style=="スキャルピング":
        hi, lo = df.high.iloc[-12:].max(), df.low.iloc[-12:].min()
    elif style=="デイトレード":
        hi, lo = df.high.iloc[-20:].max(), df.low.iloc[-20:].min()
    else:
        hi, lo = df.high.iloc[-10:].max(), df.low.iloc[-10:].min()
    return hi,lo

def suggest_trade(price, atr, decision, df, style):
    hi,lo = get_hi_lo(df,style)
    am=1.5; br=False
    if decision=="買い":
        if price>hi:
            tp=price+atr*am; sl=price-atr*am; br=True
        else:
            tp=hi*0.997; diff=tp-price; sl=price-abs(diff)/1.7
    elif decision=="売り":
        if price<lo:
            tp=price-atr*am; sl=price+atr*am; br=True
        else:
            tp=lo*0.997; diff=price-tp; sl=price+abs(diff)/1.7
    else:
        return None
    rr=abs((tp-price)/(sl-price)); p_tp=abs(tp-price)*(100 if "JPY" in symbol else 10000)
    p_sl=abs(sl-price)*(100 if "JPY" in symbol else 10000)
    return {"tp":tp,"sl":sl,"rr":rr,"ptp":p_tp,"psl":p_sl,"br":br}

def backtest(df, style):
    wins=losses=0; total_pips=0; trades=0
    for i in range(100, len(df)):
        sub = df.iloc[:i].copy()
        sig, _ = extract_signal(sub)
        if sig=="待ち": continue
        price = sub.close.iloc[-1]; atr=sub.close.rolling(14).std().iloc[-1]
        tr = suggest_trade(price,atr,sig,sub,style)
        if not tr: continue
        trades+=1
        fut = df.iloc[i+1:i+50]  # 次50本以内でTP/SL判定
        if not len(fut): break
        res = fut.apply(lambda r: r.high>=tr["tp"] if sig=="買い" else r.low<=tr["tp"],axis=1)
        if res.any():
            wins+=1; total_pips += tr["ptp"]
        else:
            total_pips -= tr["psl"]; losses+=1
    if trades==0:
        return {"trades":0,"win":0,"lr":0,"ev":0}
    win_rate = wins/trades
    ev = total_pips/trades
    return {"trades":trades,"win":win_rate,"lr":(wins,losses),"ev":ev}

if st.button("実行"):
    tf_list = tf_map[style]
    st.subheader(f"📊 {symbol}｜{style}｜バックテスト結果（過去100本）")
    df = fetch_data(symbol, tf_list[-1], use_dummy)
    df = calc_indicators(df)
    res = backtest(df, style)
    st.write(f"✔️ トレード回数: {res['trades']}")
    st.write(f"✔️ 勝率: {res['win']*100:.1f}%")
    st.write(f"✔️ 期待値 (平均pips): {res['ev']:.2f}")
