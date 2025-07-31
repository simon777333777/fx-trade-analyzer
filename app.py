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
    # Bollinger Bands
    df["BB_Mid"] = df["close"].rolling(20).mean()
    df["BB_Std"] = df["close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    return df

def get_thresholds(style):
    # (短期RCI閾値, 長期RCI閾値)
    if style == "スキャルピング":
        return 0.8, 0.5
    elif style == "デイトレード":
        return 0.7, 0.4
    else:  # スイング
        return 0.6, 0.3

def rci_based_signal(df, style):
    last = df.iloc[-1]
    short_thr, long_thr = get_thresholds(style)

    rci_9 = last["RCI_9"]
    rci_26_now = last["RCI_26"]
    rci_26_prev = df["RCI_26"].iloc[-2]
    rci_52 = last["RCI_52"]

    macd = last["MACD"]
    signal = last["Signal"]
    macd_diff = df["MACD"].diff().iloc[-1]
    macd_cross_up = macd > signal and macd_diff > 0
    macd_cross_down = macd < signal and macd_diff < 0

    close = last["close"]
    bb_upper = last["BB_Upper"]
    bb_lower = last["BB_Lower"]
    bb_mid = last["BB_Mid"]
    std = last["BB_Std"]
    std_mean = df["BB_Std"].mean()

    logs = []

    # パターン判定順張り買い
    if (
        rci_9 > short_thr
        and rci_26_now > rci_26_prev
        and rci_52 > long_thr
        and macd_cross_up
        and close > bb_mid
        and 0 < std < std_mean * 1.5
    ):
        logs.append("✅ 買いシグナル（順張り）: 短期/中期/長期RCI上向き, MACD GC, BB順行, 安定ボラ")
        score = 7
        signal_type = "買い"
        mode = "順張り"
        return score, signal_type, mode, logs

    # 逆張り買い（押し目反発のニュアンス：RCI短期が低いが底で反転＋MACD支持＋BB下限付近）
    if (
        rci_9 < -short_thr
        and rci_26_now < rci_26_prev  # 中期が下降からの反転期待は別実装だざい。簡易に下降
        and rci_52 < -long_thr
        and macd_cross_up
        and close < bb_lower
    ):
        logs.append("✅ 買いシグナル（逆張り）: RCI反転想定, MACD GC, BB下限反発狙い")
        score = 7
        signal_type = "買い"
        mode = "逆張り"
        return score, signal_type, mode, logs

    # 順張り売り
    if (
        rci_9 < -short_thr
        and rci_26_now < rci_26_prev
        and rci_52 < -long_thr
        and macd_cross_down
        and close < bb_mid
        and 0 < std < std_mean * 1.5
    ):
        logs.append("🟥 売りシグナル（順張り）: 短期/中期/長期RCI下向き, MACD DC, BB順行, 安定ボラ")
        score = -7
        signal_type = "売り"
        mode = "順張り"
        return score, signal_type, mode, logs

    # 逆張り売り（天井反転狙い）
    if (
        rci_9 > short_thr
        and rci_26_now > rci_26_prev
        and rci_52 > long_thr
        and macd_cross_down
        and close > bb_upper
    ):
        logs.append("🟥 売りシグナル（逆張り）: RCI反転想定, MACD DC, BB上限反発狙い")
        score = -7
        signal_type = "売り"
        mode = "逆張り"
        return score, signal_type, mode, logs

    # 否定・保留
    # どこが足りないか詳細に出す
    if rci_9 > short_thr:
        logs.append(f"• 短期RCI（9）: 高水準 {round(rci_9,2)}")
    elif rci_9 < -short_thr:
        logs.append(f"• 短期RCI（9）: 低水準 {round(rci_9,2)}")
    else:
        logs.append(f"• 短期RCI（9）: 中立 {round(rci_9,2)}")

    logs.append(f"• 中期RCI（26）: {'上昇中' if rci_26_now > rci_26_prev else '下降中'} ({round(rci_26_now,2)})")
    logs.append(f"• 長期RCI（52）: {round(rci_52,2)}")

    logs.append(f"• MACD: {'GC' if macd_cross_up else ('DC' if macd_cross_down else 'なし')}")
    logs.append(f"• BB位置: close={round(close,3)}, 上限={round(bb_upper,3)}, 下限={round(bb_lower,3)}, 中間={round(bb_mid,3)}")
    logs.append(f"• ボラティリティSTD: {round(std,4)} (平均比 {std_mean:.2f})")

    return 0, None, None, logs

def generate_trade_plan(df, signal_score, signal_type, mode):
    entry = df["close"].iloc[-1]
    std = df["BB_Std"].iloc[-1]
    bb_upper = df["BB_Upper"].iloc[-1]
    bb_lower = df["BB_Lower"].iloc[-1]
    bb_mid = df["BB_Mid"].iloc[-1]

    # 順張りはボラをベースに幅、逆張りは反転付近を狙う想定
    if signal_type == "買い":
        if mode == "順張り":
            tp = entry + std * 2.0
            sl = entry - std * 1.0
        else:  # 逆張り
            tp = entry + (entry - bb_lower) * 0.8  # 小さめ利確、反発期待
            sl = entry - std * 1.2  # 少し広め
    elif signal_type == "売り":
        if mode == "順張り":
            tp = entry - std * 2.0
            sl = entry + std * 1.0
        else:
            tp = entry - (bb_upper - entry) * 0.8
            sl = entry + std * 1.2
    else:
        return {}

    rr = round(abs((tp - entry) / (entry - sl)), 2) if (entry - sl) != 0 else 0
    comment = "🟢 良好なRR" if rr >= 1.5 else ("🟡 平均的" if rr >= 1.0 else "⚠️ RR注意")

    return {
        "エントリー価格": round(entry, 3),
        "利確（TP）": round(tp, 3),
        "損切り（SL）": round(sl, 3),
        "リスクリワード比（RR）": rr,
        "コメント": comment,
        "シグナル種類": f"{signal_type} ({mode})"
    }

if st.button("実行"):
    for tf in tf_map[style]:
        st.subheader(f"⏱ 時間足：{tf}")
        df = fetch_data(symbol, tf, use_dummy)
        df = calc_indicators(df)
        score, signal_type, mode, logs = rci_based_signal(df, style)

        if score == 7:
            decision = "🟢 エントリー判定：買い"
        elif score == -7:
            decision = "🟥 エントリー判定：売り"
        else:
            decision = "⚪ 判定保留"

        st.markdown(f"**{decision}**")
        for log in logs:
            st.markdown(log)
        st.markdown(f"**シグナルスコア：{score} / ±7点**")

        if score in (7, -7):
            plan = generate_trade_plan(df, score, signal_type, mode)
            st.subheader("🧮 トレードプラン（RCI主軸型）")
            for k, v in plan.items():
                st.write(f"{k}: {v}")
