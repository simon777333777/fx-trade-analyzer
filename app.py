import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import List, Tuple

st.set_page_config(page_title="RCI主軸FXトレード分析（一覧＋通知）", layout="centered")
API_KEY = st.secrets["API_KEY"]

st.title("📈 RCI主軸FXトレード分析ツール（一覧＋通知対応）")

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

# timeframes選択：デフォルトはスタイルに合わせるが変更可能
all_tfs = sorted({tf for v in tf_map.values() for tf in v})
selected_tfs = st.multiselect("時間足（複数選択可）", all_tfs, default=tf_map[style])

st.markdown("---")

# ---------- 通知設定（LINE Notify） ----------
st.subheader("🔔 通知設定（LINE Notify）")
use_notify = st.checkbox("LINE通知を有効にする", value=False)
line_token = st.text_input("LINE Notify トークン（有効化時のみ）", type="password") if use_notify else ""
notify_threshold = st.slider("通知のスコア閾値（絶対値）: このスコア以上で通知", min_value=2, max_value=7, value=4, step=1)

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
    # TwelveData のエンドポイント（ベースコード準拠）
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=500&apikey={API_KEY}"
    r = requests.get(url, timeout=10)
    data = r.json()
    if "values" not in data:
        # raise ValueError(f"APIエラー: {data}")
        return pd.DataFrame()  # 失敗時は空DFを返して呼び出し側で処理
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

# ---------- 指標計算（ベースロジック踏襲） ----------
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
        df[f"RCI_{period}"] = df["close"].rolling(period).apply(
            lambda x: calc_rci(pd.Series(x)) if len(x) == period else np.nan,
            raw=False
        )
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["BB_Mid"] = df["close"].rolling(20).mean()
    df["BB_Std"] = df["close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    return df

# 閾値緩和（ベースコードと同様に緩和済）
def get_thresholds(style):
    if style == "スキャルピング":
        return 0.7, 0.4
    elif style == "デイトレード":
        return 0.6, 0.3
    else:
        return 0.5, 0.2

def determine_tf_trend(df, style):
    short_thr, long_thr = get_thresholds(style)
    last = df.iloc[-1]
    rci_52 = last.get("RCI_52", 0)
    if rci_52 > long_thr:
        return "上昇"
    elif rci_52 < -long_thr:
        return "下降"
    else:
        return "ニュートラル"

# rci_based_signal と generate_trade_plan はベースのロジックを踏襲（少し出力用に調整）
def rci_based_signal(df, style, higher_trends: List[str]):
    if df.empty or len(df) < 10:
        return 0, None, None, ["データ不足または足数不足"], {}
    last = df.iloc[-1]
    short_thr, long_thr = get_thresholds(style)

    rci_9 = last.get("RCI_9", np.nan)
    rci_26_now = last.get("RCI_26", np.nan)
    rci_26_prev = df["RCI_26"].iloc[-2] if len(df) >= 2 else np.nan
    rci_52 = last.get("RCI_52", np.nan)

    macd = last.get("MACD", np.nan)
    signal = last.get("Signal", np.nan)
    macd_diff = df["MACD"].diff().iloc[-1] if "MACD" in df.columns else 0
    macd_cross_up = (macd > signal) and (macd_diff > 0)
    macd_cross_down = (macd < signal) and (macd_diff < 0)

    close = last["close"]
    bb_upper = last.get("BB_Upper", np.nan)
    bb_lower = last.get("BB_Lower", np.nan)
    bb_mid = last.get("BB_Mid", np.nan)
    std = last.get("BB_Std", np.nan)
    std_mean = df["BB_Std"].mean() if "BB_Std" in df.columns else np.nan

    logs = []
    strong = False
    signal_type = None
    mode = None

    # 条件判定（ベースに忠実）
    cond_buy_trend = (
        not np.isnan(rci_9) and not np.isnan(rci_26_now) and not np.isnan(rci_52)
        and rci_9 > short_thr
        and rci_26_now > rci_26_prev
        and rci_52 > long_thr
        and macd_cross_up
        and close > bb_mid
        and (0 < std < std_mean * 1.5)
    )
    mid_reversal_buy = False
    if len(df) >= 3:
        rci_26_prev2 = df["RCI_26"].iloc[-3]
        mid_reversal_buy = (rci_26_prev2 > rci_26_prev) and (rci_26_now >= rci_26_prev)
    cond_buy_reversal = (
        not np.isnan(rci_9) and not np.isnan(rci_26_now) and not np.isnan(rci_52)
        and rci_9 < -short_thr
        and mid_reversal_buy
        and rci_52 < -long_thr
        and macd_cross_up
        and close < bb_lower
    )

    cond_sell_trend = (
        not np.isnan(rci_9) and not np.isnan(rci_26_now) and not np.isnan(rci_52)
        and rci_9 < -short_thr
        and rci_26_now < rci_26_prev
        and rci_52 < -long_thr
        and macd_cross_down
        and close < bb_mid
        and (0 < std < std_mean * 1.5)
    )
    mid_reversal_sell = False
    if len(df) >= 3:
        rci_26_prev2 = df["RCI_26"].iloc[-3]
        mid_reversal_sell = (rci_26_prev2 < rci_26_prev) and (rci_26_now <= rci_26_prev)
    cond_sell_reversal = (
        not np.isnan(rci_9) and not np.isnan(rci_26_now) and not np.isnan(rci_52)
        and rci_9 > short_thr
        and mid_reversal_sell
        and rci_52 > long_thr
        and macd_cross_down
        and close > bb_upper
    )

    def aligned_with_higher(expected_direction):
        if not higher_trends:
            return True
        if expected_direction == "買い":
            return all(t == "上昇" for t in higher_trends)
        elif expected_direction == "売り":
            return all(t == "下降" for t in higher_trends)
        return False

    # 判定（ベースに準拠）
    if cond_buy_trend:
        if aligned_with_higher("買い"):
            logs.append("強い買い（順張り）: 全条件一致＋上位足整合")
            signal_type = "買い"
            mode = "順張り"
            score = 7
        else:
            logs.append("弱い買い（順張り）: 条件は揃うが上位足とズレ")
            signal_type = "買い"
            mode = "順張り"
            score = 4
        return score, signal_type, mode, logs, {}
    if cond_buy_reversal:
        if aligned_with_higher("買い"):
            logs.append("強い買い（逆張り）: 反転兆候＋上位足整合")
            signal_type = "買い"
            mode = "逆張り"
            score = 7
        else:
            logs.append("弱い買い（逆張り）: 反転条件有り、上位足整合弱")
            signal_type = "買い"
            mode = "逆張り"
            score = 4
        return score, signal_type, mode, logs, {}
    if cond_sell_trend:
        if aligned_with_higher("売り"):
            logs.append("強い売り（順張り）: 全条件一致＋上位足整合")
            signal_type = "売り"
            mode = "順張り"
            score = -7
        else:
            logs.append("弱い売り（順張り）: 条件は揃うが上位足とズレ")
            signal_type = "売り"
            mode = "順張り"
            score = -4
        return score, signal_type, mode, logs, {}
    if cond_sell_reversal:
        if aligned_with_higher("売り"):
            logs.append("強い売り（逆張り）: 反転兆候＋上位足整合")
            signal_type = "売り"
            mode = "逆張り"
            score = -7
        else:
            logs.append("弱い売り（逆張り）: 反転条件有り、上位足整合弱")
            signal_type = "売り"
            mode = "逆張り"
            score = -4
        return score, signal_type, mode, logs, {}

    # 保留時の詳細ログ
    logs.append("条件未成立（保留）:")
    try:
        logs.append(f"短期RCI9: {round(rci_9,3)} 中期RCI26: {round(rci_26_now,3)} 長期RCI52: {round(rci_52,3)}")
    except:
        pass
    return 0, None, None, logs, {}

def generate_trade_plan(df, signal_score, signal_type, mode, higher_trends):
    # ベースのロジック踏襲
    entry = df["close"].iloc[-1]
    std = df["BB_Std"].iloc[-1] if "BB_Std" in df.columns else 0
    bb_upper = df["BB_Upper"].iloc[-1] if "BB_Upper" in df.columns else entry
    bb_lower = df["BB_Lower"].iloc[-1] if "BB_Lower" in df.columns else entry
    recent_high = df["high"].rolling(50).max().iloc[-1] if "high" in df.columns else entry
    recent_low = df["low"].rolling(50).min().iloc[-1] if "low" in df.columns else entry

    if signal_type == "買い":
        if mode == "順張り":
            tp = entry + std * 2.5
            sl = max(entry - std * 1.0, recent_low)
        else:
            tp = entry + (entry - bb_lower) * 0.9
            sl = entry - std * 1.3
    elif signal_type == "売り":
        if mode == "順張り":
            tp = entry - std * 2.5
            sl = min(entry + std * 1.0, recent_high)
        else:
            tp = entry - (bb_upper - entry) * 0.9
            sl = entry + std * 1.3
    else:
        return {}

    rr = round(abs((tp - entry) / (entry - sl)), 2) if (entry - sl) != 0 else 0
    comment = "強" if abs(signal_score) >= 7 else ("中" if abs(signal_score) >= 4 else "弱")

    alignment = "整合" if signal_type and all(t == ("上昇" if signal_type=="買い" else "下降") for t in higher_trends) else "不整合"

    return {
        "エントリー価格": round(entry, 4),
        "利確（TP）": round(tp, 4),
        "損切（SL）": round(sl, 4),
        "RR": rr,
        "コメント": comment,
        "シグナル種類": f"{signal_type} ({mode})",
        "上位足整合": alignment
    }

# ---------- 通知送信関数 ----------
def send_line_notify(token: str, message: str):
    if not token:
        return False
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    try:
        res = requests.post(url, headers=headers, data=data, timeout=10)
        return res.status_code == 200
    except Exception:
        return False

# セッションステート：通知済みキー保存
if "notified_keys" not in st.session_state:
    st.session_state["notified_keys"] = set()

# ---------- 実行ボタン ----------
if st.button("🔍 一覧スキャンと通知実行"):
    if not pairs:
        st.warning("監視する通貨ペアを1つ以上選んでください。")
    elif not selected_tfs:
        st.warning("時間足を1つ以上選んでください。")
    else:
        results = []
        progress = st.progress(0)
        total = len(pairs) * len(selected_tfs)
        i = 0
        for pair in pairs:
            for tf in selected_tfs:
                i += 1
                progress.progress(int(i/total*100))
                # fetch
                try:
                    df = fetch_data(pair, tf, use_dummy)
                except Exception as e:
                    df = pd.DataFrame()
                if df is None or df.empty:
                    # 失敗なら空行で埋める
                    results.append({
                        "ペア": pair,
                        "時間足": tf,
                        "シグナル": "データ取得失敗",
                        "スコア": None,
                        "エントリー": None,
                        "TP": None,
                        "SL": None,
                        "RR": None,
                        "備考": "データ不足"
                    })
                    continue

                # calc indicators & detect
                df = calc_indicators(df)
                # For一覧目的、higher_trendsは空にして判定（簡易）
                score, signal_type, mode, logs, _ = rci_based_signal(df, style, [])
                row_note = "; ".join(logs[:3]) if logs else ""
                # generate plan when at least 弱〜中
                if signal_type in ("買い", "売り") and abs(score) >= 2:
                    plan = generate_trade_plan(df, score, signal_type, mode, [])
                    entry = plan.get("エントリー価格")
                    tp = plan.get("利確（TP）")
                    sl = plan.get("損切（SL）")
                    rr = plan.get("RR")
                else:
                    entry = None; tp = None; sl = None; rr = None

                results.append({
                    "ペア": pair,
                    "時間足": tf,
                    "シグナル": signal_type if signal_type else "なし",
                    "スコア": score,
                    "エントリー": entry,
                    "TP": tp,
                    "SL": sl,
                    "RR": rr,
                    "備考": row_note
                })

                # 通知判定（条件を満たし、かつ未通知）
                if use_notify and line_token and signal_type in ("買い", "売り") and abs(score) >= notify_threshold:
                    notify_key = f"{pair}_{tf}_{signal_type}_{score}"
                    if notify_key not in st.session_state["notified_keys"]:
                        # compose message
                        decimals = 2 if "JPY" in pair else 4
                        entry_s = f"{entry:.{decimals}f}" if entry is not None else "N/A"
                        tp_s = f"{tp:.{decimals}f}" if tp is not None else "N/A"
                        sl_s = f"{sl:.{decimals}f}" if sl is not None else "N/A"
                        msg = f"{pair} {tf} {signal_type}シグナル（スコア {score}）\nエントリー: {entry_s}\nTP: {tp_s} / SL: {sl_s} (RR={rr})\n備考: {row_note}"
                        ok = send_line_notify(line_token, msg)
                        if ok:
                            st.success(f"LINE通知送信済: {pair} {tf} {signal_type}")
                            st.session_state["notified_keys"].add(notify_key)
                        else:
                            st.error(f"LINE通知送信失敗: {pair} {tf}")

        progress.empty()
        df_res = pd.DataFrame(results)

        # 色付けスタイル：シグナル列に基づく
        def color_signal(val):
            if val is None:
                return ""
            if isinstance(val, str):
                if "強い買い" in val or val == "買い":
                    return "background-color: #d4f7d4"  # light green
                if "強い売り" in val or val == "売り":
                    return "background-color: #f7d4d4"  # light red
                if val in ("買い", "売り"):
                    return "background-color: #fff2cc"
                if val == "なし" or val == "見送り":
                    return ""
            return ""

        # 表示
        st.subheader("📋 シグナル一覧")
        st.write(f"監視ペア: {', '.join(pairs)} / 時間足: {', '.join(selected_tfs)}")
        # 列並びを見やすく
        display_cols = ["ペア", "時間足", "シグナル", "スコア", "エントリー", "TP", "SL", "RR", "備考"]
        df_show = df_res[display_cols].copy()

        # 数値の丸め（JPYは小数2）
        for col in ["エントリー", "TP", "SL"]:
            if col in df_show.columns:
                df_show[col] = df_show[col].apply(lambda x: (round(x, 2) if (pd.notna(x) and isinstance(x, (int, float))) else x))

        try:
            styled = df_show.style.applymap(lambda v: "background-color: #d4f7d4" if v == "買い" else (
                "background-color: #f7d4d4" if v == "売り" else ("background-color: #fff2cc" if v in ("買い（弱）","売り（弱）") else "")
            ), subset=["シグナル"])
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(df_show, use_container_width=True)
