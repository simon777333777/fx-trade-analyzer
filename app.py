import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import List

st.set_page_config(page_title="RCI主軸FXトレード分析（改善版）", layout="centered")

API_KEY = st.secrets["API_KEY"]

st.title("📈 RCI主軸FXトレード分析ツール（改善版）")

# =========================================================
# UI
# =========================================================
pairs_all = ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/USD"]

pairs = st.multiselect(
    "監視する通貨ペア（複数選択可）",
    pairs_all,
    default=["GBP/JPY", "EUR/USD"]
)

style = st.selectbox(
    "トレードスタイルを選択",
    ["スキャルピング", "デイトレード", "スイング"],
    index=1
)

use_dummy = st.checkbox(
    "📦 ダミーデータで実行（テストモード）",
    value=False
)

tf_map = {
    "スキャルピング": ["5min", "15min", "1h"],
    "デイトレード": ["15min", "1h", "4h"],
    "スイング": ["1h", "4h", "1day"]
}

all_tfs = sorted({tf for v in tf_map.values() for tf in v})

selected_tfs = st.multiselect(
    "時間足（複数選択可）",
    all_tfs,
    default=tf_map[style]
)

st.markdown("---")

# =========================================================
# ダミーデータ
# =========================================================
def get_dummy_data():

    idx = pd.date_range(
        end=pd.Timestamp.now(),
        periods=700,
        freq="H"
    )

    trend = np.linspace(0, 10, len(idx))

    noise = np.random.randn(len(idx)) * 0.5

    price = 150 + trend + np.cumsum(noise)

    df = pd.DataFrame({
        "datetime": idx,
        "open": price + np.random.randn(len(idx))*0.2,
        "high": price + np.abs(np.random.randn(len(idx))*0.4),
        "low": price - np.abs(np.random.randn(len(idx))*0.4),
        "close": price,
        "volume": 1000
    })

    df.set_index("datetime", inplace=True)

    return df

# =========================================================
# データ取得
# =========================================================
@st.cache_data(ttl=300)
def fetch_data(symbol, interval, use_dummy_flag):

    if use_dummy_flag:
        return get_dummy_data()

    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}"
        f"&interval={interval}"
        f"&outputsize=700"
        f"&apikey={API_KEY}"
    )

    try:
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

    except Exception:
        return pd.DataFrame()

# =========================================================
# RCI
# =========================================================
def calc_rci(series):

    n = len(series)

    if series.isna().any() or n < 2:
        return np.nan

    price_rank = series.rank(method="average")

    time_rank = pd.Series(
        np.arange(1, n+1),
        index=series.index
    )

    d = price_rank - time_rank

    denom = n * (n**2 - 1)

    if denom == 0:
        return np.nan

    rho = 1 - (6 * (d**2).sum()) / denom

    return rho * 100

# =========================================================
# 指標計算
# =========================================================
def calc_indicators(df):

    # -------------------------------
    # RCI
    # -------------------------------
    for period in [9, 26, 52]:

        df[f"RCI_{period}"] = (
            df["close"]
            .rolling(period)
            .apply(
                lambda x: calc_rci(pd.Series(x)),
                raw=False
            )
        )

    # -------------------------------
    # MACD
    # -------------------------------
    ema12 = df["close"].ewm(span=12).mean()

    ema26 = df["close"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26

    df["Signal"] = df["MACD"].ewm(span=9).mean()

    # -------------------------------
    # EMA200
    # -------------------------------
    df["EMA200"] = df["close"].ewm(span=200).mean()

    # -------------------------------
    # ATR
    # -------------------------------
    high_low = df["high"] - df["low"]

    high_close = np.abs(df["high"] - df["close"].shift())

    low_close = np.abs(df["low"] - df["close"].shift())

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    # -------------------------------
    # ADX
    # -------------------------------
    plus_dm = df["high"].diff()

    minus_dm = -df["low"].diff()

    plus_dm[plus_dm < 0] = 0

    minus_dm[minus_dm < 0] = 0

    tr_smooth = tr.rolling(14).sum()

    plus_di = 100 * (
        plus_dm.rolling(14).sum() / tr_smooth
    )

    minus_di = 100 * (
        minus_dm.rolling(14).sum() / tr_smooth
    )

    dx = (
        abs(plus_di - minus_di)
        / (plus_di + minus_di)
    ) * 100

    df["ADX"] = dx.rolling(14).mean()

    return df

# =========================================================
# スタイル別閾値
# =========================================================
def get_thresholds(style):

    if style == "スキャルピング":
        return 80, 40

    elif style == "デイトレード":
        return 70, 30

    else:
        return 60, 20

# =========================================================
# 上位足判定
# =========================================================
def determine_trend(df, style):

    if df.empty:
        return "ニュートラル"

    last = df.iloc[-1]

    rci52 = last.get("RCI_52", 0)

    ema200 = last.get("EMA200", 0)

    close = last.get("close", 0)

    if rci52 > 40 and close > ema200:
        return "上昇"

    elif rci52 < -40 and close < ema200:
        return "下降"

    else:
        return "ニュートラル"

# =========================================================
# シグナル判定
# =========================================================
def rci_based_signal(df, style, higher_trends):

    if df.empty or len(df) < 60:
        return 0, None, None, ["データ不足"], {}

    last = df.iloc[-1]

    prev = df.iloc[-2]

    logs = []

    short_thr, long_thr = get_thresholds(style)

    # -------------------------------
    # 各指標
    # -------------------------------
    rci9 = last["RCI_9"]
    rci26 = last["RCI_26"]
    rci52 = last["RCI_52"]

    rci9_prev = prev["RCI_9"]
    rci26_prev = prev["RCI_26"]

    macd = last["MACD"]
    signal = last["Signal"]

    close = last["close"]

    ema200 = last["EMA200"]

    atr = last["ATR"]

    adx = last["ADX"]

    # -------------------------------
    # ATRフィルター
    # -------------------------------
    atr_ratio = atr / close if close != 0 else 0

    volatility_ok = atr_ratio > 0.0025

    # -------------------------------
    # ADXフィルター
    # -------------------------------
    trend_ok = adx > 20

    # -------------------------------
    # EMA方向
    # -------------------------------
    ema_bull = close > ema200

    ema_bear = close < ema200

    # -------------------------------
    # MACD
    # -------------------------------
    macd_bull = macd > signal

    macd_bear = macd < signal

    # -------------------------------
    # RCIクロス
    # -------------------------------
    bullish_cross = (
        rci9_prev < rci26_prev
        and rci9 > rci26
    )

    bearish_cross = (
        rci9_prev > rci26_prev
        and rci9 < rci26
    )

    # -------------------------------
    # 上位足整合
    # -------------------------------
    buy_alignment = all(
        t == "上昇"
        for t in higher_trends
    ) if higher_trends else False

    sell_alignment = all(
        t == "下降"
        for t in higher_trends
    ) if higher_trends else False

    # =====================================================
    # 買い条件（押し目）
    # =====================================================
    cond_buy = (
        rci52 > long_thr
        and rci26 > 0
        and rci9_prev < -80
        and bullish_cross
        and macd_bull
        and ema_bull
        and volatility_ok
        and trend_ok
    )

    # =====================================================
    # 売り条件
    # =====================================================
    cond_sell = (
        rci52 < -long_thr
        and rci26 < 0
        and rci9_prev > 80
        and bearish_cross
        and macd_bear
        and ema_bear
        and volatility_ok
        and trend_ok
    )

    # =====================================================
    # スコアリング
    # =====================================================
    score = 0

    signal_type = None

    mode = None

    if cond_buy:

        signal_type = "買い"

        mode = "押し目買い"

        score = 4

        if buy_alignment:
            score += 2

        if adx > 30:
            score += 1

        logs.append(
            f"買い成立 "
            f"RCI9={rci9:.1f} "
            f"ADX={adx:.1f} "
            f"ATR比={atr_ratio:.4f}"
        )

    elif cond_sell:

        signal_type = "売り"

        mode = "戻り売り"

        score = -4

        if sell_alignment:
            score -= 2

        if adx > 30:
            score -= 1

        logs.append(
            f"売り成立 "
            f"RCI9={rci9:.1f} "
            f"ADX={adx:.1f} "
            f"ATR比={atr_ratio:.4f}"
        )

    else:

        logs.append(
            f"条件未成立 "
            f"ADX={adx:.1f} "
            f"ATR比={atr_ratio:.4f}"
        )

    return score, signal_type, mode, logs, {}

# =========================================================
# トレードプラン
# =========================================================
def generate_trade_plan(
    df,
    signal_score,
    signal_type,
    mode,
    higher_trends
):

    if signal_type is None:
        return {}

    entry = df["close"].iloc[-1]

    atr = df["ATR"].iloc[-1]

    recent_high = df["high"].rolling(20).max().iloc[-1]

    recent_low = df["low"].rolling(20).min().iloc[-1]

    # =====================================================
    # BUY
    # =====================================================
    if signal_type == "買い":

        sl = recent_low - (atr * 0.3)

        risk = entry - sl

        tp = entry + (risk * 1.5)

    # =====================================================
    # SELL
    # =====================================================
    else:

        sl = recent_high + (atr * 0.3)

        risk = sl - entry

        tp = entry - (risk * 1.5)

    # =====================================================
    # RR
    # =====================================================
    rr = round(
        abs((tp - entry) / (entry - sl)),
        2
    ) if (entry - sl) != 0 else 0

    # =====================================================
    # コメント
    # =====================================================
    abs_score = abs(signal_score)

    if abs_score >= 6:
        comment = "強"

    elif abs_score >= 4:
        comment = "中"

    else:
        comment = "弱"

    alignment_str = "整合" if (
        signal_type == "買い"
        and all(t == "上昇" for t in higher_trends)
    ) or (
        signal_type == "売り"
        and all(t == "下降" for t in higher_trends)
    ) else "不整合"

    return {
        "エントリー価格": round(entry, 4),
        "利確（TP）": round(tp, 4),
        "損切（SL）": round(sl, 4),
        "RR": rr,
        "コメント": comment,
        "シグナル種類": f"{signal_type} ({mode})",
        "上位足整合": alignment_str
    }

# =========================================================
# 実行
# =========================================================
if st.button("🔍 一覧スキャン実行"):

    if not pairs:

        st.warning(
            "監視する通貨ペアを選択してください。"
        )

    elif not selected_tfs:

        st.warning(
            "時間足を選択してください。"
        )

    else:

        results = []

        progress = st.progress(0)

        total = len(pairs) * len(selected_tfs)

        i = 0

        for pair in pairs:

            for tf in selected_tfs:

                i += 1

                progress.progress(int(i / total * 100))

                df = fetch_data(
                    pair,
                    tf,
                    use_dummy
                )

                if df.empty:

                    results.append({
                        "ペア": pair,
                        "時間足": tf,
                        "シグナル": "取得失敗",
                        "スコア": None
                    })

                    continue

                df = calc_indicators(df)

                # ----------------------------------------
                # 上位足
                # ----------------------------------------
                higher_trends = []

                for htf in selected_tfs:

                    if htf == tf:
                        continue

                    hdf = fetch_data(
                        pair,
                        htf,
                        use_dummy
                    )

                    if hdf.empty:
                        continue

                    hdf = calc_indicators(hdf)

                    trend = determine_trend(
                        hdf,
                        style
                    )

                    higher_trends.append(trend)

                # ----------------------------------------
                # シグナル
                # ----------------------------------------
                score, signal_type, mode, logs, _ = (
                    rci_based_signal(
                        df,
                        style,
                        higher_trends
                    )
                )

                # ----------------------------------------
                # トレードプラン
                # ----------------------------------------
                if signal_type:

                    plan = generate_trade_plan(
                        df,
                        score,
                        signal_type,
                        mode,
                        higher_trends
                    )

                else:
                    plan = {}

                # ----------------------------------------
                # RRフィルター
                # ----------------------------------------
                rr = plan.get("RR", 0)

                if signal_type and rr < 1.2:

                    logs.append(
                        f"RR不足({rr})"
                    )

                # ----------------------------------------
                # 結果
                # ----------------------------------------
                results.append({
                    "ペア": pair,
                    "時間足": tf,
                    "シグナル": signal_type if signal_type else "なし",
                    "スコア": round(score, 1) if score else None,
                    "エントリー": plan.get("エントリー価格"),
                    "TP": plan.get("利確（TP）"),
                    "SL": plan.get("損切（SL）"),
                    "RR": plan.get("RR"),
                    "備考": "; ".join(logs[:3])
                })

        progress.empty()

        df_res = pd.DataFrame(results)

        # ----------------------------------------
        # スコア順
        # ----------------------------------------
        if not df_res.empty and "スコア" in df_res.columns:

            df_res["abs_score"] = (
                df_res["スコア"]
                .fillna(0)
                .abs()
            )

            df_res = df_res.sort_values(
                "abs_score",
                ascending=False
            )

            df_res = df_res.drop(
                columns=["abs_score"]
            )

        st.subheader("📋 シグナル一覧")

        st.dataframe(
            df_res,
            use_container_width=True
        )
