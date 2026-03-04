"""
Streamlit リアルタイムダッシュボード
ポジション・リスク・セッション・パフォーマンスを可視化

起動: streamlit run monitoring/dashboard.py
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz

from config_manager import load_config, get
from session.detector import get_current_session, Session
from session.weights import get_engine_weights
from database import get_trade_log, get_daily_pnl, init_db

JST = pytz.timezone("Asia/Tokyo")


def main():
    st.set_page_config(
        page_title="FX LLM Bot Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 FX LLM Bot - リアルタイムダッシュボード")
    st.markdown("**EUR/USD H1** | LLM × テクニカル自動売買")

    # config読み込み
    try:
        load_config()
    except Exception:
        st.error("config.yaml の読み込みに失敗しました")
        return

    # DB初期化
    try:
        init_db()
    except Exception:
        pass

    # サイドバー
    with st.sidebar:
        st.header("⚙️ システム情報")
        st.write(f"**Version:** {get('meta.version', 'N/A')}")
        st.write(f"**最終最適化:** {get('meta.last_optimized', 'N/A')}")
        st.write(f"**LLMフィルター:** {'✅ 有効' if get('llm.enabled') else '❌ 無効'}")

        st.divider()
        st.header("📋 現在のパラメータ")
        st.json({
            "SL倍率": get("risk.sl_multiplier"),
            "BEトリガー": get("position.be_trigger"),
            "部分利確": get("position.partial_trigger"),
            "TP倍率": get("position.tp_multiplier"),
            "トレール": get("position.trail_multiplier"),
        })

    # メインエリア - 4カラム
    col1, col2, col3, col4 = st.columns(4)

    # セッション情報
    session = get_current_session()
    session_names = {
        Session.ASIA: "🌏 アジア",
        Session.LONDON_PREP: "🔄 ロンドン準備",
        Session.LONDON: "🇬🇧 ロンドン",
        Session.OVERLAP: "🔥 ロンドン×NY",
        Session.NY_LATE: "🇺🇸 NY後半",
        Session.CLOSED: "🔒 市場クローズ",
    }

    with col1:
        st.metric("現在セッション", session_names.get(session, "不明"))

    with col2:
        weights = get_engine_weights(session)
        active_engines = [k for k, v in weights.items() if v > 0]
        st.metric("アクティブエンジン", f"{len(active_engines)}個")

    with col3:
        now = datetime.now(JST)
        st.metric("時刻 (JST)", now.strftime("%H:%M:%S"))

    with col4:
        st.metric("DD上限", f"{get('risk.max_dd_pct', 20)}%")

    st.divider()

    # エンジン重み可視化
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🎛️ 現在のエンジン重み")
        weights = get_engine_weights(session)
        fig_weights = go.Figure(data=[
            go.Bar(
                x=list(weights.keys()),
                y=list(weights.values()),
                marker_color=["#2196F3", "#4CAF50", "#FF9800"],
            )
        ])
        fig_weights.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="重み",
            xaxis_title="エンジン",
        )
        st.plotly_chart(fig_weights, use_container_width=True)

    with col_right:
        st.subheader("⏰ セッション別重みマップ")
        session_data = []
        for s in [Session.ASIA, Session.LONDON, Session.OVERLAP, Session.NY_LATE]:
            w = get_engine_weights(s)
            session_data.append({
                "セッション": s.value,
                "Trend": w["trend"],
                "MeanRev": w["mean_rev"],
                "Breakout": w["breakout"],
            })

        df_sessions = pd.DataFrame(session_data).set_index("セッション")
        st.dataframe(
            df_sessions.style.background_gradient(cmap="YlOrRd", axis=None),
            use_container_width=True,
        )

    st.divider()

    # トレードログ
    st.subheader("📝 直近トレードログ")
    try:
        trades = get_trade_log(days=7)
        if trades:
            df_trades = pd.DataFrame(trades)
            display_cols = [
                "ticket", "direction", "entry_price", "exit_price",
                "lot_size", "pnl", "pnl_pips", "session", "status", "close_reason",
            ]
            available_cols = [c for c in display_cols if c in df_trades.columns]
            st.dataframe(df_trades[available_cols], use_container_width=True)
        else:
            st.info("直近7日間のトレードデータはありません")
    except Exception as e:
        st.warning(f"トレードログの取得に失敗: {e}")

    # リスク管理状態
    st.divider()
    st.subheader("🛡️ リスク管理パラメータ")

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    with risk_col1:
        st.metric("1トレードMaxロス", f"{get('risk.max_loss_per_trade', 0.01)*100}%")
        st.metric("SL倍率 (ATR)", f"×{get('risk.sl_multiplier', 1.5)}")

    with risk_col2:
        st.metric("最大DD制限", f"{get('risk.max_dd_pct', 20)}%")
        st.metric("日次ロス上限", f"{get('risk.daily_loss_limit_pct', 5)}%")

    with risk_col3:
        st.metric("最大同時保有", f"{get('risk.max_positions', 3)}ポジ")
        st.metric("エクスポージャー上限", f"{get('risk.max_exposure_pct', 5)}%")

    # フッター
    st.divider()
    st.caption(
        f"最終更新: {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')} | "
        f"FX LLM Bot v{get('meta.version', '1.2')}"
    )


if __name__ == "__main__":
    main()
