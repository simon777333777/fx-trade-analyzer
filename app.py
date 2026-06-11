import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="RCI FX Scanner", layout="wide")
API_KEY = st.secrets["API_KEY"]

pairs=["USD/JPY","EUR/USD","GBP/JPY","AUD/USD","EUR/JPY","GBP/USD"]
tfs=["15min","1h","4h","1day"]

st.title("RCI FX Scanner")
selected_pairs=st.multiselect("通貨ペア",pairs,default=["USD/JPY","GBP/JPY","EUR/USD"])
selected_tfs=st.multiselect("時間足",tfs,default=["1h","4h"])

@st.cache_data(ttl=300)
def fetch_data(symbol,interval):
    url=f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=300&apikey={API_KEY}"
    try:
        r=requests.get(url,timeout=15).json()
        if "values" not in r:
            return pd.DataFrame()
        df=pd.DataFrame(r["values"])
        df["datetime"]=pd.to_datetime(df["datetime"])
        df=df.set_index("datetime").sort_index()
        return df.apply(pd.to_numeric,errors="coerce")
    except:
        return pd.DataFrame()

def rci(x):
    n=len(x)
    if n<2:
        return np.nan
    p=pd.Series(x).rank()
    t=np.arange(1,n+1)
    d=p-t
    return (1-6*(d**2).sum()/(n*(n**2-1)))*100

def indicators(df):
    for p in (9,26,52):
        df[f"RCI{p}"]=df["close"].rolling(p).apply(rci,raw=False)
    df["EMA20"]=df["close"].ewm(span=20,adjust=False).mean()
    df["EMA200"]=df["close"].ewm(span=200,adjust=False).mean()
    tr=pd.concat([
        df["high"]-df["low"],
        (df["high"]-df["close"].shift()).abs(),
        (df["low"]-df["close"].shift()).abs()
    ],axis=1).max(axis=1)
    df["ATR"]=tr.rolling(14).mean()
    return df

def analyze(df):
    last=df.iloc[-1]
    prev=df.iloc[-2]
    score=0
    reasons=[]

    if last.RCI52>20:
        score+=2
        reasons.append("RCI52↑")
    elif last.RCI52<-20:
        score-=2
        reasons.append("RCI52↓")

    if last.RCI26>20:
        score+=2
        reasons.append("RCI26↑")
    elif last.RCI26<-20:
        score-=2
        reasons.append("RCI26↓")

    if last.RCI9>last.RCI26:
        score+=2
        reasons.append("RCI9優勢")
    else:
        score-=2

    if last.RCI9>prev.RCI9:
        score+=1
        reasons.append("RCI9上昇")
    else:
        score-=1

    if last.close>last.EMA200:
        score+=2
        reasons.append("EMA上")
    else:
        score-=2

    confidence=round(abs(score)/9*100)

    if confidence>=85:
        rank="S"
    elif confidence>=70:
        rank="A"
    elif confidence>=50:
        rank="B"
    else:
        rank="C"

    ratio=abs(last.close-last.EMA20)/last.ATR if pd.notna(last.ATR) and last.ATR>0 else 999

    if ratio<0.5:
        timing="今すぐ可"
    elif ratio<1.0:
        timing="やや待ち"
    else:
        timing="押し目待ち"

    if score>=7:
        signal="強買い"
        sl_price=round(last.close-last.ATR,4)
        tp_price=round(last.close+last.ATR*2,4)
    elif score>=4:
        signal="買い"
        sl_price=round(last.close-last.ATR,4)
        tp_price=round(last.close+last.ATR*1.5,4)
    elif score<=-7:
        signal="強売り"
        sl_price=round(last.close+last.ATR,4)
        tp_price=round(last.close-last.ATR*2,4)
    elif score<=-4:
        signal="売り"
        sl_price=round(last.close+last.ATR,4)
        tp_price=round(last.close-last.ATR*1.5,4)
    else:
        signal="監視"
        sl_price=None
        tp_price=None

    return signal,score,confidence,rank,timing,sl_price,tp_price,",".join(reasons)

if st.button("スキャン"):
    rows=[]
    for pair in selected_pairs:
        for tf in selected_tfs:
            df=fetch_data(pair,tf)
            if len(df)<220:
                continue
            df=indicators(df)
            signal,score,confidence,rank,timing,sl_price,tp_price,reason=analyze(df)
            rows.append({
                "ランク":rank,
                "ペア":pair,
                "時間足":tf,
                "判定":signal,
                "点数":score,
                "信頼度":f"{confidence}%",
                "タイミング":timing,
                "現在価格":round(df["close"].iloc[-1],4),
                "SL価格":sl_price,
                "TP価格":tp_price,
                "根拠":reason
            })

    if rows:
        res=pd.DataFrame(rows)
        res["sort"]=res["点数"].abs()
        res=res.sort_values("sort",ascending=False).drop(columns="sort")
        st.dataframe(res,use_container_width=True)
    else:
        st.warning("データ取得失敗")
