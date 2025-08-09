import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="RCI主軸FXトレード分析", layout="centered")
API_KEY = st.secrets["API_KEY"]

st.title("📈 RCI主軸FXトレード分析ツール2025/08/04（改良版）")
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

def calc_rci(series):
    # 本来のRCI（順位相関）：価格の順位と時間の順位の相関（Spearman風）
    n = len(series)
    if series.isna().any() or n < 2:
        return np.nan
    price_rank = series.rank(method="average")
    time_rank = pd.Series(np.arange(1, n+1), index=series.index)
    d = price_rank - time_rank
    # Spearman-like
    denom = n * (n**2 - 1)
    if denom == 0:
        return np.nan
    rho = 1 - (6 * (d**2).sum()) / denom
    return rho  # in [-1,1]

def calc_indicators(df):
    # RCI: 短期・中期・長期
    for period in [9, 26, 52]:
        df[f"RCI_{period}"] = df["close"].rolling(period).apply(
            lambda x: calc_rci(pd.Series(x)) if len(x) == period else np.nan,
            raw=False
        )
    # MACD components (difference)
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    # Bollinger Bands
    df["BB_Mid"] = df["close"].rolling(20).mean()
    df["BB_Std"] = df["close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    return df

def get_thresholds(style):
    # ※ 閾値をやや緩和しました（短期・長期ともに -0.1 程度）
    # (短期RCI閾値, 長期RCI閾値)
    if style == "スキャルピング":
        return 0.7, 0.4  # was 0.8,0.5
    elif style == "デイトレード":
        return 0.6, 0.3  # was 0.7,0.4
    else:  # スイング
        return 0.5, 0.2  # was 0.6,0.3

def determine_tf_trend(df, style):
    # 上位足の方向性（簡易）：長期RCIによるトレンド判定
    short_thr, long_thr = get_thresholds(style)
    last = df.iloc[-1]
    rci_52 = last.get("RCI_52", 0)
    if rci_52 > long_thr:
        return "上昇"
    elif rci_52 < -long_thr:
        return "下降"
    else:
        return "ニュートラル"

def rci_based_signal(df, style, higher_trends):
    # higher_trends: list of trend strings from higher timeframes (e.g., ["上昇","上昇"]) for alignment
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
    signal_type = None  # "買い"/"売り"
    mode = None  # "順張り"/"逆張り"

    # ----- 順張り買い ----- #
    cond_buy_trend = (
        not np.isnan(rci_9) and not np.isnan(rci_26_now) and not np.isnan(rci_52)
        and rci_9 > short_thr
        and rci_26_now > rci_26_prev
        and rci_52 > long_thr
        and macd_cross_up
        and close > bb_mid
        and (0 < std < std_mean * 1.5)
    )
    # ----- 逆張り買い：反転兆候（中期RCIが下降から横ばい/上向きへ変化）＋短期底＋BB下限付近 ----- #
    mid_reversal_buy = False
    if len(df) >= 3:
        rci_26_prev2 = df["RCI_26"].iloc[-3]
        # 下降→横ばいor上昇の転換
        mid_reversal_buy = (rci_26_prev2 > rci_26_prev) and (rci_26_now >= rci_26_prev)
    cond_buy_reversal = (
        not np.isnan(rci_9) and not np.isnan(rci_26_now) and not np.isnan(rci_52)
        and rci_9 < -short_thr  # 短期が底域（反転の兆し）
        and mid_reversal_buy
        and rci_52 < -long_thr
        and macd_cross_up
        and close < bb_lower
    )

    # ----- 順張り売り ----- #
    cond_sell_trend = (
        not np.isnan(rci_9) and not np.isnan(rci_26_now) and not np.isnan(rci_52)
        and rci_9 < -short_thr
        and rci_26_now < rci_26_prev
        and rci_52 < -long_thr
        and macd_cross_down
        and close < bb_mid
        and (0 < std < std_mean * 1.5)
    )
    # ----- 逆張り売り：天井反転（中期RCIが上昇から横ばい/下降へ変化）＋短期天井＋BB上限付近 ----- #
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

    # 上位足との整合性フィルタ（強シグナルには必要、弱シグナルなら緩和）
    def aligned_with_higher(expected_direction):
        # expected_direction: "買い" -> higher_trends should be all "上昇"
        if not higher_trends:
            return True
        if expected_direction == "買い":
            return all(t == "上昇" for t in higher_trends)
        elif expected_direction == "売り":
            return all(t == "下降" for t in higher_trends)
        return False

    # 判定
    if cond_buy_trend:
        if aligned_with_higher("買い"):
            logs.append("✅ 強い買いシグナル（順張り）: 全条件一致＋上位足整合") 
            strong = True
            signal_type = "買い"
            mode = "順張り"
            score = 7
        else:
            logs.append("🟡 弱い買いシグナル（順張り）: 条件は揃うが上位足と方向がズレ") 
            strong = False
            signal_type = "買い"
            mode = "順張り"
            score = 4
        return score, signal_type, mode, logs

    if cond_buy_reversal:
        if aligned_with_higher("買い"):
            logs.append("✅ 強い買いシグナル（逆張りリバース）: 反転兆候＋上位足整合") 
            strong = True
            signal_type = "買い"
            mode = "逆張り"
            score = 7
        else:
            logs.append("🟡 弱い買いシグナル（逆張り）: 反転条件はあるが上位足との整合不十分") 
            strong = False
            signal_type = "買い"
            mode = "逆張り"
            score = 4
        return score, signal_type, mode, logs

    if cond_sell_trend:
        if aligned_with_higher("売り"):
            logs.append("✅ 強い売りシグナル（順張り）: 全条件一致＋上位足整合") 
            strong = True
            signal_type = "売り"
            mode = "順張り"
            score = -7
        else:
            logs.append("🟡 弱い売りシグナル（順張り）: 条件は揃うが上位足と方向がズレ") 
            strong = False
            signal_type = "売り"
            mode = "順張り"
            score = -4
        return score, signal_type, mode, logs

    if cond_sell_reversal:
        if aligned_with_higher("売り"):
            logs.append("✅ 強い売りシグナル（逆張りリバース）: 反転兆候＋上位足整合") 
            strong = True
            signal_type = "売り"
            mode = "逆張り"
            score = -7
        else:
            logs.append("🟡 弱い売りシグナル（逆張り）: 反転条件はあるが上位足との整合不十分") 
            strong = False
            signal_type = "売り"
            mode = "逆張り"
            score = -4
        return score, signal_type, mode, logs

    # 否定・保留（どこが足りないか詳細に出す）
    logs.append("⚪ シグナル条件未成立（保留）詳細:")
    if rci_9 > short_thr:
        logs.append(f"• 短期RCI（9）: 高水準 {round(rci_9,2)}")
    elif rci_9 < -short_thr:
        logs.append(f"• 短期RCI（9）: 低水準 {round(rci_9,2)}")
    else:
        logs.append(f"• 短期RCI（9）: 中立 {round(rci_9,2)}")

    # 中期の状態表現
    if len(df) >= 3:
        if mid_reversal_buy or mid_reversal_sell:
            logs.append(f"• 中期RCI（26）: 反転兆候 ({round(rci_26_now,2)})")
        else:
            logs.append(f"• 中期RCI（26）: {'上昇中' if rci_26_now > rci_26_prev else '下降中'} ({round(rci_26_now,2)})")
    else:
        logs.append(f"• 中期RCI（26）: {round(rci_26_now,2)}")

    logs.append(f"• 長期RCI（52）: {round(rci_52,2)}")
    logs.append(f"• MACD: {'GC' if macd_cross_up else ('DC' if macd_cross_down else 'なし')}")
    logs.append(f"• BB位置: close={round(close,3)}, 上限={round(bb_upper,3)}, 下限={round(bb_lower,3)}, 中間={round(bb_mid,3)}")
    # ボラの文脈化：収縮→拡張/過熱の目安
    vol_context = "通常"
    if std > std_mean * 1.5:
        vol_context = "拡張（過熱気味）"
    elif std < std_mean * 0.5:
        vol_context = "収縮（動き出し前）"
    logs.append(f"• ボラティリティSTD: {round(std,4)} ({vol_context}, 平均比 {std_mean:.2f})")

    return 0, None, None, logs

def generate_trade_plan(df, signal_score, signal_type, mode, higher_trends):
    entry = df["close"].iloc[-1]
    std = df["BB_Std"].iloc[-1]
    bb_upper = df["BB_Upper"].iloc[-1]
    bb_lower = df["BB_Lower"].iloc[-1]
    bb_mid = df["BB_Mid"].iloc[-1]

    # 上位足の直近構造（簡易）：直近高値安値を参考にTP/SL調整
    recent_high = df["high"].rolling(50).max().iloc[-1]
    recent_low = df["low"].rolling(50).min().iloc[-1]

    # 順張りはボラをベースに幅、逆張りは反転付近を狙う想定
    if signal_type == "買い":
        if mode == "順張り":
            tp = entry + std * 2.5  # 多少広めにとってトレンド伸びを取りに行く
            sl = max(entry - std * 1.0, recent_low)  # 直近安値近くをSL下限に
        else:  # 逆張り
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
    comment = ""
    if abs(signal_score) >= 7:
        comment = "🟢 強シグナル＋構造的に整合性あり"
    elif abs(signal_score) >= 4:
        comment = "🟡 弱シグナル：上位足とのズレまたは補助条件欠け"
    else:
        comment = "⚪ 保留：条件不十分"

    # 追加の根拠表示（上位足方向一致の有無）
    alignment = "整合" if signal_type and all(t == ("上昇" if signal_type=="買い" else "下降") for t in higher_trends) else "不整合"

    return {
        "エントリー価格": round(entry, 3),
        "利確（TP）": round(tp, 3),
        "損切り（SL）": round(sl, 3),
        "リスクリワード比（RR）": rr,
        "コメント": comment,
        "シグナル種類": f"{signal_type} ({mode})",
        "上位足方向との整合": alignment
    }

# ----------------- 実行 -----------------
if st.button("実行"):
    # まずすべての時間足のデータを取って環境認識用にトレンドを取る
    tf_list = tf_map[style]
    # 上位から下位の順にトレンドを得る（最後の足がエントリー足想定）
    tf_dfs = {}
    tf_trends = {}
    for tf in tf_list:
        df = fetch_data(symbol, tf, use_dummy)
        df = calc_indicators(df)
        tf_dfs[tf] = df
        tf_trends[tf] = determine_tf_trend(df, style)

    # エントリー足は最後のtf_list
    entry_tf = tf_list[-1]
    higher_trends = [tf_trends[tf] for tf in tf_list[:-1]]  # 上位足の方向
    for tf in tf_list:
        st.subheader(f"⏱ 時間足：{tf}")
        df = tf_dfs[tf]
        score, signal_type, mode, logs = rci_based_signal(df, style, higher_trends if tf == entry_tf else [])
        if score == 7:
            decision = "🟢 エントリー判定：買い" if signal_type == "買い" else "🟥 エントリー判定：売り"
        elif score == -7:
            decision = "🟥 エントリー判定：売り"
        elif abs(score) in (4,):
            decision = "🟡 弱シグナル（保留寄り）"
        else:
            decision = "⚪ 判定保留"

        st.markdown(f"**{decision}**")
        for log in logs:
            st.markdown(log)
        st.markdown(f"**シグナルスコア：{score} / ±7点**")
        # ---- 表示条件を緩和：弱〜中レベルもトレードプランを表示 ----
        if tf == entry_tf:
            if signal_type in ("買い", "売り") and abs(score) >= 2:
                plan = generate_trade_plan(df, score, signal_type, mode, higher_trends)
                st.subheader("🧮 トレードプラン（RCI主軸型）")
                # 短縮スタイル表示（スキャル/デイトレ/スイング）
                short_style = "スキャル" if style == "スキャルピング" else ("デイトレ" if style == "デイトレード" else "スイング")
                # 表示フォーマット（JPYは小数2桁表示を目安に）
                decimals = 2 if "JPY" in symbol else 4
                entry_str = f"{plan['エントリー価格']:.{decimals}f}円前後" if "JPY" in symbol else f"{plan['エントリー価格']:.{decimals}f}"
                tp_str = f"{plan['利確（TP）']:.{decimals}f}"
                sl_str = f"{plan['損切り（SL）']:.{decimals}f}"
                rr = plan.get("リスクリワード比（RR）", "")
                st.markdown(f"**現在のおすすめ：{signal_type}（{short_style}）**")
                st.write(f"- エントリー価格：{entry_str}")
                st.write(f"- 利確目標（TP）：{tp_str} / 損切目安（SL）：{sl_str}（RR={rr}）")
                # 主軸説明（順張り or 逆張り）
                main_axis = "RCI順張りが主軸となっています" if mode == "順張り" else "RCI逆張りが主軸となっています"
                st.write(f"- {main_axis}")
                st.write(f"- スタイル：{short_style}")
                st.write(f"- コメント：{plan.get('コメント','')}")
                st.write(f"- 上位足方向との整合：{plan.get('上位足方向との整合','')}")
            else:
                st.info("シグナル条件を満たしていないため、トレードプランは表示されません。")
