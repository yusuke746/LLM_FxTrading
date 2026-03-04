"""
週次自動最適化スケジューラー
APSchedulerで毎週日曜 07:00 JST に実行

フロー:
  1. 直近8週のH1データ取得
  2. グリッドサーチ（全組み合わせでバックテスト）
  3. 最優秀パラメータ選定（Sharpe最大 + DD上限20%）
  4. 安全ガードチェック（変化量が大きければ人間確認）
  5. config.yaml 自動書き込み
  6. LLMによる週次レポート生成（オプション）
  7. Discordに結果通知
"""

from datetime import datetime, timedelta
from typing import Optional

import pytz

from config_manager import get, reload_config
from optimizer.grid_search import GridSearchOptimizer
from optimizer.backtest_runner import BacktestRunner
from optimizer.config_updater import ConfigUpdater
from database import insert_optimization_result
from logger_setup import get_logger

logger = get_logger("optimizer.scheduler")

JST = pytz.timezone("Asia/Tokyo")


class OptimizationScheduler:
    """週次最適化スケジューラー"""

    def __init__(self, executor=None, notifier=None):
        """
        Args:
            executor: MT5Executorインスタンス（データ取得用）
            notifier: Discord通知インスタンス
        """
        self.executor = executor
        self.notifier = notifier
        self.grid_search = GridSearchOptimizer()
        self.config_updater = ConfigUpdater()
        self.backtest_runner = BacktestRunner()

    def run_weekly_optimization(self) -> dict:
        """
        週次最適化を実行
        
        Returns:
            dict: 最適化結果
        """
        logger.info("=== 週次自動最適化 開始 ===")
        start_time = datetime.now(JST)

        try:
            # 1. 直近16週のH1データ取得
            lookback_weeks = get("optimizer.lookback_weeks", 16)
            data = self._fetch_data(lookback_weeks)
            if data is None or len(data) < 200:
                msg = f"データ不足: {len(data) if data is not None else 0}本"
                logger.error(msg)
                self._notify(f"❌ 週次最適化失敗: {msg}")
                return {"success": False, "reason": msg}

            logger.info(f"データ取得完了: {len(data)}本 ({lookback_weeks}週分)")

            # 2. グリッドサーチ実行
            search_result = self.grid_search.optimize(data)

            if not search_result["best_params"]:
                msg = "有効なパラメータが見つかりませんでした"
                logger.warning(msg)
                self._notify(f"⚠️ 週次最適化: {msg}")
                return {"success": False, "reason": msg}

            best_params = search_result["best_params"]
            best_result = search_result["best_result"]

            logger.info(
                f"最適パラメータ: {best_params} | "
                f"Sharpe={best_result['sharpe_ratio']:.3f} "
                f"DD={best_result['max_dd']:.1f}% "
                f"PF={best_result['profit_factor']:.2f} "
                f"勝率={best_result['win_rate']*100:.1f}%"
            )

            # 3. ウォークフォワード検証（過学習チェック）
            wf_result = self.backtest_runner.run_walk_forward(data, best_params)
            logger.info(
                f"ウォークフォワード検証: Sharpe={wf_result['sharpe_ratio']:.3f} "
                f"安定性={wf_result.get('stability_score', 0):.3f} "
                f"IS={wf_result.get('is_sharpe', 'N/A')} "
                f"OOS={wf_result.get('oos_sharpe', 'N/A')}"
            )

            # 安定性が低い場合は警告（OOS/IS比率ベース）
            stability = wf_result.get("stability_score", 0)
            if stability < 0.3:
                logger.warning(
                    f"ウォークフォワード安定性が低い({stability:.3f}) → 過学習の可能性"
                )

            # 4. config.yaml 更新
            update_result = self.config_updater.update(
                best_params,
                backtest_result=best_result,
            )

            # 5. データベースに記録
            try:
                insert_optimization_result({
                    "run_date": start_time.isoformat(),
                    "lookback_weeks": lookback_weeks,
                    "sl_multiplier": best_params.get("sl_multiplier"),
                    "be_trigger": best_params.get("be_trigger"),
                    "partial_trigger": best_params.get("partial_trigger"),
                    "tp_multiplier": best_params.get("tp_multiplier"),
                    "trail_multiplier": best_params.get("trail_multiplier"),
                    "sharpe_ratio": best_result.get("sharpe_ratio"),
                    "max_drawdown": best_result.get("max_dd"),
                    "profit_factor": best_result.get("profit_factor"),
                    "win_rate": best_result.get("win_rate"),
                    "total_trades": best_result.get("total_trades"),
                    "applied": 1 if update_result["updated"] else 0,
                    "notes": f"WF stability={wf_result.get('stability_score', 'N/A')}",
                })
            except Exception as e:
                logger.error(f"最適化結果のDB記録失敗: {e}")

            # 6. 設定を再読み込み
            if update_result["updated"]:
                reload_config()

            # 7. Discord通知
            elapsed = (datetime.now(JST) - start_time).total_seconds()
            self._notify(self._format_result(best_params, best_result, update_result, elapsed))

            logger.info(f"=== 週次自動最適化 完了 ({elapsed:.0f}秒) ===")

            return {
                "success": True,
                "best_params": best_params,
                "best_result": best_result,
                "walk_forward": wf_result,
                "update_result": update_result,
                "elapsed_seconds": elapsed,
            }

        except Exception as e:
            logger.error(f"週次最適化エラー: {e}", exc_info=True)
            self._notify(f"❌ 週次最適化エラー: {e}")
            return {"success": False, "reason": str(e)}

    def _fetch_data(self, weeks: int):
        """MT5からOHLCVデータを取得"""
        if self.executor is None:
            logger.error("MT5 Executorが未設定 → データ取得不可")
            return None

        bars = weeks * 5 * 24  # 週5日 × 24時間
        return self.executor.get_ohlcv("H1", bars=bars)

    def _notify(self, message: str):
        """Discord通知"""
        if self.notifier:
            try:
                self.notifier.send(message)
            except Exception as e:
                logger.error(f"通知送信失敗: {e}")
        logger.info(f"通知: {message}")

    def _format_result(
        self,
        params: dict,
        result: dict,
        update: dict,
        elapsed: float,
    ) -> str:
        """結果をフォーマット"""
        status = "✅ 適用済み" if update["updated"] else "⚠️ 手動確認要"
        return (
            f"📊 **週次自動最適化結果** ({status})\n"
            f"```\n"
            f"SL倍率:     {params.get('sl_multiplier', '-')}\n"
            f"BEトリガー: {params.get('be_trigger', '-')}\n"
            f"部分利確:   {params.get('partial_trigger', '-')}\n"
            f"TP倍率:     {params.get('tp_multiplier', '-')}\n"
            f"トレール:   {params.get('trail_multiplier', '-')}\n"
            f"─────────────────────\n"
            f"Sharpe:     {result.get('sharpe_ratio', 0):.3f}\n"
            f"Max DD:     {result.get('max_dd', 0):.1f}%\n"
            f"PF:         {result.get('profit_factor', 0):.2f}\n"
            f"勝率:       {result.get('win_rate', 0)*100:.1f}%\n"
            f"トレード数: {result.get('total_trades', 0)}\n"
            f"処理時間:   {elapsed:.0f}秒\n"
            f"```"
        )

    def start_scheduler(self):
        """APSchedulerで定期実行を開始"""
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler

            run_day = get("optimizer.run_day", "sun")
            run_hour = get("optimizer.run_hour_jst", 7)

            scheduler = BlockingScheduler(timezone="Asia/Tokyo")
            scheduler.add_job(
                self.run_weekly_optimization,
                "cron",
                day_of_week=run_day,
                hour=run_hour,
                minute=0,
                id="weekly_optimization",
                name="週次自動最適化",
            )

            logger.info(f"週次最適化スケジューラー起動: 毎週{run_day} {run_hour}:00 JST")
            scheduler.start()

        except Exception as e:
            logger.error(f"スケジューラー起動失敗: {e}")
            raise
