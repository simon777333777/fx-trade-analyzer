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

def get_dummy_data(freq="1H", periods=500):
    idx = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq=freq)
    np.random.seed(0)
    price = np.cumsum(np.random.randn(len(idx))) + 150
    return pd.DataFrame({"open": price+np.random.randn(len(idx)),
                         "high": price+1, "low": price-1,
                         "close": price, "volume":1000}, index=idx)

@st.cache_data(ttl=300)
def fetch_data(symbol, interval, use_dummy):
    if use_dummy:
        return get_dummy_data(freq=interval_map[interval], periods=500)
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
    data = requests.get(url).json()
    if "values" not in data:
        raise ValueError(f"APIデータ取得エラー: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index().astype(float)
    return df

interval_map = {"5min":"5min","15min":"15min","1h":"1H","4h":"4H","1day":"1D"}

def calc_indicators(df):
    df["SMA_5"]=df["close"].rolling(5).mean()
    df["SMA_20"]=df["close"].rolling(20).mean()
    df["MACD"]=df["close"].ewm(span=12).mean()-df["close"].ewm(span=26).mean()
    df["Signal"]=df["MACD"].ewm(span=9).mean()
    df["Upper"]=df["SMA_20"]+2*df["close"].rolling(20).std()
    df["Lower"]=df["SMA_20"]-2*df["close"].rolling(20).std()
    df["RCI"]=df["close"].rank().rolling(9).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0,1])
    df["ADX"]=abs(df["MACD"]-df["Signal"]).rolling(14).mean()
    df["STD"]=df["close"].rolling(20).std()
    return df

def detect_market_structure(last):
    trend=sum([last["ADX"]>25,
               abs(last["SMA_5"]-last["SMA_20"])/last["close"]>0.015,
               last["STD"]>last["close"]*0.005])
    return "トレンド" if trend>=2 else "レンジ"

def detect_dow(df):
    highs, lows=df["high"].iloc[-3:], df["low"].iloc[-3:]
    is_hh=highs.iloc[2]>highs.iloc[1]>highs.iloc[0]
    is_ll=lows.iloc[2]<lows.iloc[1]<lows.iloc[0]
    if is_hh and is_ll: return "保ち合い","⚪ ダウ理論：保ち合い"
    if is_hh: return "上昇","🟢 高値切り上げ"
    if is_ll: return "下降","🔴 安値切り下げ"
    return "不明","⚪ ダウ理論未達"

def detect_price_action(df):
    a,b=df.iloc[-2],df.iloc[-1]
    if a["close"]<a["open"]<b["open"]<b["close"] and b["close"]>a["open"]:
        return "🟢 陽線包み足"
    if a["close"]>a["open"]>b["open"]>b["close"] and b["close"]<a["open"]:
        return "🔴 陰線包み足"
    return "⚪ プライスアクション未達"

def extract_signal(df):
    last=df.iloc[-1]
    market=detect_market_structure(last)
    logs=[f"• 市場判定：{market}"]
    buy=sell=0
    tw=2 if market=="トレンド" else 1
    rw=2 if market=="レンジ" else 1

    mt,st=df["MACD"].iloc[-3:],df["Signal"].iloc[-3:]
    if mt.iloc[-1]>st.iloc[-1] and mt.is_monotonic_increasing:
        buy+=tw; logs.append("🟢 MACDゴールデンクロス + 上昇傾向")
    elif mt.iloc[-1]<st.iloc[-1] and mt.is_monotonic_decreasing:
        sell+=tw; logs.append("🔴 MACDデッドクロス + 下降傾向")
    else: logs.append("⚪ MACD判定微妙")

    s5,s20=df["SMA_5"].iloc[-3:],df["SMA_20"].iloc[-3:]
    if s5.iloc[-1]>s20.iloc[-1] and s5.is_monotonic_increasing:
        buy+=tw; logs.append("🟢 SMA短期>長期 + 上昇傾向")
    elif s5.iloc[-1]<s20.iloc[-1] and s5.is_monotonic_decreasing:
        sell+=tw; logs.append("🔴 SMA短期<長期 + 下降傾向")
    else: logs.append("⚪ SMA判定微妙")

    if last["close"]<last["Lower"]:
        buy+=rw; logs.append("🟢 BB下限反発の可能性")
    elif last["close"]>last["Upper"]:
        sell+=rw; logs.append("🔴 BB上限反発の可能性")
    else: logs.append("⚪ BB反発無し")

    if last["RCI"]>0.5:
        buy+=rw; logs.append("🟢 RCI上昇傾向")
    elif last["RCI"]<-0.5:
        sell+=rw; logs.append("🔴 RCI下降傾向")
    else: logs.append("⚪ RCI未達")

    _,ld=detect_dow(df)
    if "高値" in ld: buy+=1
    if "安値" in ld: sell+=1
    logs.append(ld)

    pa=detect_price_action(df)
    if "陽線" in pa: buy+=1
    if "陰線" in pa: sell+=1
    logs.append(pa)

    sig="買い" if buy>=4 and buy>sell else ("売り" if sell>=4 and sell>buy else "待ち")
    return sig, logs, buy, sell

def get_hi_lo(df,style):
    if style=="スキャルピング":
        return df["high"].iloc[-12:].max(), df["low"].iloc[-12:].min()
    if style=="デイトレード":
        return df["high"].iloc[-20:].max(), df["low"].iloc[-20:].min()
    return df["high"].iloc[-10:].max(), df["low"].iloc[-10:].min()

def suggest_trade_plan(price,atr,decision,df,style):
    hi,lo=get_hi_lo(df,style)
    atr_mult=1.5; is_bo=False
    if decision=="買い":
        if price>hi:
            tp=price+atr*atr_mult; sl=price-atr*atr_mult; is_bo=True
        else:
            tp=hi*0.997; sl=price-(tp-price)/1.7
    elif decision=="売り":
        if price<lo:
            tp=price-atr*atr_mult; sl=price+atr*atr_mult; is_bo=True
        else:
            tp=lo*0.997; sl=price+(price-tp)/1.7
    else:
        return price,price,price,0,0,0,False
    rr=abs((tp-price)/(sl-price)) if sl!=price else 0
    pips_tp=abs(tp-price)*(100 if "JPY" in symbol else 10000)
    pips_sl=abs(sl-price)*(100 if "JPY" in symbol else 10000)
    return price,tp,sl,rr,pips_tp,pips_sl,is_bo

def backtest(df,style):
    results=[]
    for i in range(100,len(df)):
        sub=df.iloc[:i]
        sig,_,_,_=extract_signal(sub)
        if sig=="待ち": continue
        price=sub["close"].iloc[-1]
        atr=sub["close"].rolling(14).std().iloc[-1]
        _,tp,sl,rr,p_tp,p_sl,_=suggest_trade_plan(price,atr,sig,sub,style)
        future=df["low"].iloc[i+1:i+6].min() if sig=="買い" else df["high"].iloc[i+1:i+6].max()
        hit= (future>=tp if sig=="買い" else future<=tp)
        profit=p_tp if hit else -p_sl
        results.append(profit)
    return results

if st.button("実行"):
    timeframes=tf_map[style]
    total_b=total_s=0; logs_all=[]; main_df=None; main_tf=""
    st.subheader(f"📊 通貨ペア：{symbol} | スタイル：{style}")
    st.markdown("### ⏱ 各時間足シグナル詳細")
    for tf in timeframes:
        df_tf=fetch_data(symbol,tf,use_dummy)
        df_tf=calc_indicators(df_tf)
        sig,logs,b,s=extract_signal(df_tf)
        wt=tf_weights[tf]; total_b+=b*wt; total_s+=s*wt
        st.markdown(f"⏱ {tf} 判定：{sig}（買 {b} 売 {s}）")
        for l in logs: st.markdown(l)
        main_df=df_tf; main_tf=tf
    st.markdown("---")
    decision="買い" if total_b>=2.4 and total_b>total_s else ("売り" if total_s>=2.4 and total_s>total_b else ("買い" if abs(total_b-total_s)>=1 else "待ち"))
    st.markdown(f"総合スコア：{total_b:.2f}（買） / {total_s:.2f}（売） → {decision}")
    if decision!="待ち":
        price=main_df["close"].iloc[-1]; atr=main_df["close"].rolling(14).std().iloc[-1]
        _,tp,sl,rr,p_tp,p_sl,is_bo=suggest_trade_plan(price,atr,decision,main_df,style)
        st.markdown("### 🎯 トレードプラン")
        st.markdown(f"• Entry: {price:.5f} • TP: {tp:.5f} • SL: {sl:.5f} • RR: {rr:.2f} • Breakout: {is_bo}")
    else:
        st.markdown("エントリー見送り")
    st.markdown("---")
    df_bt=fetch_data(symbol,tf_map[style][-1],use_dummy)
    df_bt=calc_indicators(df_bt)
    profits=backtest(df_bt,style)
    if profits:
        win=sum(1 for p in profits if p>0)
        total=len(profits)
        ev=np.mean(profits)
        st.markdown("### 📈 バックテスト結果（過去100本）")
        st.markdown(f"• トレード数：{total} • 勝率：{win/total*100:.1f}% • 期待値(pips)：{ev:.2f}")
    else:
        st.markdown("バックテストに該当トレードなし")
