"""
FX LLM Bot - メインエントリーポイント
EUR/USD H1足 デイトレード自動売買システム

起動方法:
  python main.py              # 通常実行
  python main.py --optimize   # 手動で週次最適化を実行
  python main.py --backtest   # バックテストモード
  python main.py --dashboard  # ダッシュボード起動

設計思想:
  「LLMは優位なところのみ使う。システムの質を最優先」
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytz
import pandas_ta as ta

from config_manager import load_config, get, reload_config
from logger_setup import setup_logger, get_logger
from database import init_db
from session.detector import get_current_session, is_market_open, Session
from session.weights import get_engine_weights
from engine.composite import calc_composite_signal
from llm.filter_manager import LLMFilterManager
from position.no_hedge import NoHedgeController
from position.manager import PositionManager, ManagedPosition
from risk.risk_manager import RiskManager
from execution.mt5_executor import MT5Executor
from optimizer.scheduler import OptimizationScheduler
from notifier import DiscordNotifier
from filters.entry_filters import EntryFilterManager
from db_maintenance import DBMaintenance

JST = pytz.timezone("Asia/Tokyo")

# グローバル停止フラグ
_running = True


def signal_handler(sig, frame):
    """Ctrl+C ハンドラ"""
    global _running
    _running = False
    print("\n⏹️ 停止要求を受信しました。安全にシャットダウンします...")


class FXTradingBot:
    """FX自動売買ボット メインクラス"""

    def __init__(self):
        # 設定読み込み
        load_config()
        self.logger = setup_logger("fx_bot")
        init_db()

        # 各モジュール初期化
        self.symbol = get("meta.symbol", "EURUSD")
        self.timeframe = get("meta.timeframe", "H1")

        self.executor = MT5Executor(self.symbol)
        self.risk_manager = RiskManager()
        self.position_manager = PositionManager(self.executor)
        self.no_hedge = NoHedgeController(self.executor)
        self.llm_filter = LLMFilterManager()
        self.entry_filter = EntryFilterManager()
        self.notifier = DiscordNotifier()
        self.optimizer = OptimizationScheduler(self.executor, self.notifier)
        self.db_maintenance = DBMaintenance()

        self._last_bar_time: Optional[datetime] = None

    def start(self):
        """メインループを開始"""
        global _running

        self.logger.info("=" * 60)
        self.logger.info("FX LLM Bot 起動")
        self.logger.info(f"通貨ペア: {self.symbol} | タイムフレーム: {self.timeframe}")
        self.logger.info(f"LLMフィルター: {'有効' if get('llm.enabled') else '無効'}")
        self.logger.info(f"追加フィルター: ボラ={'ON' if get('filters.volatility_enabled', True) else 'OFF'} "
                         f"スプレッド={'ON' if get('filters.spread_enabled', True) else 'OFF'} "
                         f"時間帯={'ON' if get('filters.time_performance_enabled', True) else 'OFF'} "
                         f"アダプティブ={'ON' if get('filters.adaptive_enabled', True) else 'OFF'}")
        self.logger.info("=" * 60)

        # MT5接続
        if not self.executor.initialize():
            self.logger.critical("MT5接続失敗 → 起動中止")
            self.notifier.send_error_alert("MT5接続失敗 - ボット起動中止")
            return

        # 口座情報取得
        account = self.executor.get_account_info()
        if account:
            self.logger.info(
                f"口座情報: 残高={account['balance']} "
                f"有効証拠金={account['equity']} "
                f"レバレッジ={account['leverage']}"
            )
            self.initial_balance = account["balance"]
        else:
            self.logger.warning("口座情報取得失敗")
            self.initial_balance = 1_000_000  # フォールバック

        self.notifier.send("🚀 FX LLM Bot 起動しました")

        # DB統計ログ
        db_stats = self.db_maintenance.get_db_stats()
        self.logger.info(
            f"DB統計: トレード{db_stats.get('trades_count', 0)}件 "
            f"(オープン{db_stats.get('open_trades', 0)}) | "
            f"サイズ{db_stats.get('db_size_mb', 0):.1f}MB"
        )

        # メインループ
        signal.signal(signal.SIGINT, signal_handler)
        try:
            while _running:
                try:
                    self._main_cycle()
                except Exception as e:
                    self.logger.error(f"メインサイクルエラー: {e}", exc_info=True)
                    self.notifier.send_error_alert(f"メインサイクルエラー: {e}")

                # DBメンテナンス（毎日自動実行）
                try:
                    if self.db_maintenance.should_run():
                        now_jst = datetime.now(JST)
                        run_hour = get("db_maintenance.run_hour_jst", 6)
                        if now_jst.hour == run_hour:
                            maint_result = self.db_maintenance.run()
                            if maint_result.get("error"):
                                self.logger.warning(f"DBメンテナンスエラー: {maint_result['error']}")
                except Exception as e:
                    self.logger.error(f"DBメンテナンス例外: {e}")

                # H1足の更新を待つ（60秒間隔でチェック）
                time.sleep(60)

        finally:
            self._shutdown()

    def _main_cycle(self):
        """メインサイクル（H1足クローズごとに実行）"""
        # 市場クローズ中はスキップ
        if not is_market_open():
            return

        # H1足データ取得
        df = self.executor.get_ohlcv(self.timeframe, bars=200)
        if df is None or len(df) < 60:
            return

        # 新しいH1足の確認（同じ足を重複処理しない）
        current_bar_time = df["datetime"].iloc[-1]
        if self._last_bar_time is not None and current_bar_time <= self._last_bar_time:
            # 新しいバーなし → ポジション管理のみ実行
            self._manage_existing_positions()
            return

        self._last_bar_time = current_bar_time
        session = get_current_session()
        self.logger.info(f"--- 新しいH1足: {current_bar_time} | セッション: {session.value} ---")

        # 準備フェーズ or 市場クローズ → スキップ
        if session in (Session.LONDON_PREP, Session.CLOSED):
            self.logger.info(f"セッション {session.value}: エントリー待機中")
            self._manage_existing_positions()
            return

        # リスクチェック
        account = self.executor.get_account_info()
        if account:
            positions = self.executor.get_positions(self.symbol)
            risk_check = self.risk_manager.check_risk_limits(
                balance=account["balance"],
                equity=account["equity"],
                open_positions=len(positions),
                initial_balance=self.initial_balance,
            )

            if not risk_check["can_trade"]:
                self.logger.warning(f"リスク制限: {risk_check['reasons']}")
                self._manage_existing_positions()
                return

        # H4データ取得（マルチタイムフレーム確認用）
        h4_df = self.executor.get_ohlcv("H4", bars=100)

        # アダプティブ閾値取得（直近成績に応じて動的調整）
        adaptive_threshold, adaptive_details = self.entry_filter.get_adjusted_threshold()

        # シグナル算出（アダプティブ閾値適用）
        signal_result = calc_composite_signal(df, h4_df, entry_threshold=adaptive_threshold)
        direction = signal_result["direction"]
        score = signal_result["score"]

        if direction == "NONE":
            self._manage_existing_positions()
            return

        # LLMフィルター
        filter_result = self.llm_filter.should_allow_entry(df, direction=direction)
        if not filter_result["allowed"]:
            self.logger.info(f"LLMフィルターでブロック: {filter_result['reason']}")
            self._manage_existing_positions()
            return

        # 両建て制御
        hedge_check = self.no_hedge.pre_entry_check(self.symbol, direction)
        if not hedge_check["can_entry"]:
            self.logger.info(f"エントリー不可: {hedge_check['reason']}")
            self._manage_existing_positions()
            return

        # ATR算出
        atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
        current_atr = atr_series.iloc[-1]

        if current_atr is None or current_atr <= 0:
            self.logger.warning("ATRが無効 → エントリー見送り")
            return

        # スプレッド取得（フィルター用）
        price_info_pre = self.executor.get_current_price()
        current_spread = price_info_pre["spread"] if price_info_pre else 0.0

        # 追加フィルターチェック（ボラティリティ + スプレッド + 時間帯）
        filter_check = self.entry_filter.pre_entry_check(
            df=df,
            current_atr=current_atr,
            current_spread=current_spread,
            symbol=self.symbol,
        )
        if not filter_check["allowed"]:
            self.logger.info(f"エントリーフィルター: {filter_check['reason']}")
            self._manage_existing_positions()
            return

        # ロットサイズ算出
        lot_size = self.risk_manager.calculate_lot_size(
            balance=account["balance"] if account else self.initial_balance,
            atr=current_atr,
        )

        if lot_size <= 0:
            return

        # SL/TP算出
        price_info = self.executor.get_current_price()
        if price_info is None:
            return

        entry_price = price_info["ask"] if direction == "BUY" else price_info["bid"]
        sl = self.risk_manager.calculate_sl(entry_price, current_atr, direction)
        tp = self.risk_manager.calculate_tp(entry_price, current_atr, direction)

        # 発注
        order_result = self.executor.execute_order(
            direction=direction,
            lot_size=lot_size,
            sl=sl,
            tp=tp,
            comment=f"s={score:.2f}_{session.value}",
        )

        if order_result:
            # ポジション管理に登録
            managed_pos = ManagedPosition(
                ticket=order_result["ticket"],
                symbol=self.symbol,
                direction=direction,
                entry_price=order_result["price"],
                lot_size=lot_size,
                sl=sl,
                tp=tp,
                atr_at_entry=current_atr,
                spread=order_result["spread"],
            )
            self.position_manager.register(managed_pos)

            # 通知
            self.notifier.send_trade_alert({
                "direction": direction,
                "lot_size": lot_size,
                "price": order_result["price"],
                "sl": sl,
                "tp": tp,
                "signal_score": score,
                "session": session.value,
            })

            self.logger.info(
                f"✅ エントリー完了: {direction} {lot_size}lots @ {order_result['price']:.5f} "
                f"SL={sl:.5f} TP={tp:.5f} | スコア={score:.3f}"
            )

        # 既存ポジション管理
        self._manage_existing_positions()

    def _manage_existing_positions(self):
        """既存ポジションの4ステップ管理"""
        active_positions = self.position_manager.get_active_positions()

        for pos in active_positions:
            price_info = self.executor.get_current_price()
            if price_info is None:
                continue

            current_price = price_info["bid"] if pos.direction == "BUY" else price_info["ask"]
            action = self.position_manager.update(pos.ticket, current_price)

            if action is None:
                continue

            # アクション実行
            if action["action"] == "modify_sl":
                self.executor.modify_position(
                    ticket=action["ticket"],
                    sl=action["new_sl"],
                )

            elif action["action"] == "partial_close":
                success = self.executor.partial_close(
                    ticket=action["ticket"],
                    lot_size=action["close_lot"],
                )
                if success:
                    self.logger.info(f"部分利確: ticket={action['ticket']} {action['close_lot']}lots")

            elif action["action"] == "modify_tp":
                self.executor.modify_position(
                    ticket=action["ticket"],
                    tp=action["new_tp"],
                )

            elif action["action"] == "close":
                success = self.executor.close_position(
                    ticket=action["ticket"],
                    reason=action.get("reason", "manager"),
                )
                if success:
                    self.position_manager.mark_closed(action["ticket"], action.get("reason", ""))
                    self.notifier.send_close_alert({
                        "ticket": action["ticket"],
                        "pnl": 0,  # 実際のPnLはMT5から取得
                        "reason": action.get("reason", "manager"),
                    })

    def _shutdown(self):
        """安全なシャットダウン"""
        self.logger.info("シャットダウン処理開始...")
        self.executor.shutdown()
        self.notifier.send("⏹️ FX LLM Bot を停止しました")
        self.logger.info("シャットダウン完了")


def run_backtest_mode():
    """バックテストモードで実行"""
    from optimizer.backtest_runner import BacktestRunner
    from optimizer.grid_search import GridSearchOptimizer

    logger = get_logger("backtest")
    load_config()

    logger.info("=== バックテストモード ===")
    logger.info("注意: MT5からデータを取得するにはMT5を起動してください")

    executor = MT5Executor()
    if not executor.initialize():
        logger.error("MT5接続失敗。CSVファイルからのデータ読み込みを検討してください。")
        return

    # 8週分のデータ取得
    df = executor.get_ohlcv("H1", bars=8 * 5 * 24)
    executor.shutdown()

    if df is None or len(df) < 200:
        logger.error("データ不足")
        return

    logger.info(f"データ取得完了: {len(df)}本")

    # グリッドサーチ最適化
    optimizer = GridSearchOptimizer()
    result = optimizer.optimize(df)

    if result["best_params"]:
        logger.info(f"最適パラメータ: {result['best_params']}")
        logger.info(f"Sharpe: {result['best_result']['sharpe_ratio']:.3f}")
        logger.info(f"Max DD: {result['best_result']['max_dd']:.1f}%")
        logger.info(f"PF: {result['best_result']['profit_factor']:.2f}")
        logger.info(f"勝率: {result['best_result']['win_rate']*100:.1f}%")
        logger.info(f"トレード数: {result['best_result']['total_trades']}")

        # 上位5結果
        logger.info("\n--- 上位5パターン ---")
        for i, r in enumerate(result["top_results"], 1):
            logger.info(f"{i}. {r['params']} → Sharpe={r['result']['sharpe_ratio']:.3f}")
    else:
        logger.warning("有効な結果が見つかりませんでした")


def run_optimize_mode():
    """手動最適化モードで実行"""
    load_config()
    init_db()

    executor = MT5Executor()
    notifier = DiscordNotifier()

    if not executor.initialize():
        print("MT5接続失敗")
        return

    scheduler = OptimizationScheduler(executor, notifier)
    result = scheduler.run_weekly_optimization()

    executor.shutdown()

    if result["success"]:
        print(f"\n✅ 最適化完了: {result['best_params']}")
    else:
        print(f"\n❌ 最適化失敗: {result.get('reason', 'Unknown')}")


def run_dashboard():
    """Streamlitダッシュボードを起動"""
    import subprocess
    dashboard_path = Path(__file__).parent / "monitoring" / "dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_path)])


def main():
    parser = argparse.ArgumentParser(
        description="FX LLM Bot - EUR/USD H1 自動売買システム"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="手動で週次最適化を実行",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="バックテストモードで実行",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Streamlitダッシュボードを起動",
    )

    args = parser.parse_args()

    if args.optimize:
        run_optimize_mode()
    elif args.backtest:
        run_backtest_mode()
    elif args.dashboard:
        run_dashboard()
    else:
        bot = FXTradingBot()
        bot.start()


if __name__ == "__main__":
    main()
