"""
データベースメンテナンスモジュール
長期運用時のDB肥大化を防止するための定期クリーンアップ

機能:
  - クローズ済みトレードの古いレコードを削除（保持日数設定可）
  - 最適化履歴の古いレコードを削除
  - 日次パフォーマンスの古いレコードを削除
  - VACUUM実行でDBファイルサイズを縮小
  - 毎日1回、指定時刻に自動実行

注意:
  - オープン中のトレードは決して削除しない
  - 削除前にレコード数をログに記録
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

from config_manager import get
from database import get_connection, get_db_path
from logger_setup import get_logger

logger = get_logger("db_maintenance")


class DBMaintenance:
    """データベース定期メンテナンス"""

    def __init__(self):
        self.enabled = get("db_maintenance.enabled", True)
        self.trades_retention = get("db_maintenance.trades_retention_days", 180)
        self.optimization_retention = get("db_maintenance.optimization_retention_days", 365)
        self.daily_perf_retention = get("db_maintenance.daily_perf_retention_days", 365)
        self.vacuum_enabled = get("db_maintenance.vacuum_enabled", True)
        self._last_run_date: str = ""

    def should_run(self) -> bool:
        """今日まだ実行していなければTrue"""
        if not self.enabled:
            return False
        today = datetime.now().strftime("%Y-%m-%d")
        return today != self._last_run_date

    def run(self) -> dict:
        """
        メンテナンスを実行
        
        Returns:
            dict: 各テーブルの削除件数とDB情報
        """
        if not self.enabled:
            return {"skipped": True, "reason": "disabled"}

        logger.info("=== DBメンテナンス開始 ===")
        result = {
            "trades_deleted": 0,
            "optimization_deleted": 0,
            "daily_perf_deleted": 0,
            "vacuum": False,
            "db_size_before": 0,
            "db_size_after": 0,
        }

        try:
            db_path = get_db_path()
            result["db_size_before"] = self._get_file_size_mb(db_path)

            # バックアップ作成（月初のみ）
            if datetime.now().day == 1:
                self._create_backup(db_path)

            with get_connection() as conn:
                cursor = conn.cursor()

                # 1. クローズ済みトレードの古いレコードを削除
                result["trades_deleted"] = self._cleanup_trades(cursor)

                # 2. 最適化履歴の古いレコードを削除
                result["optimization_deleted"] = self._cleanup_optimization(cursor)

                # 3. 日次パフォーマンスの古いレコードを削除
                result["daily_perf_deleted"] = self._cleanup_daily_performance(cursor)

            # 4. VACUUM（接続外で実行）
            if self.vacuum_enabled:
                result["vacuum"] = self._run_vacuum()

            result["db_size_after"] = self._get_file_size_mb(db_path)

            self._last_run_date = datetime.now().strftime("%Y-%m-%d")

            total_deleted = (
                result["trades_deleted"]
                + result["optimization_deleted"]
                + result["daily_perf_deleted"]
            )
            logger.info(
                f"=== DBメンテナンス完了: {total_deleted}件削除 | "
                f"DB: {result['db_size_before']:.1f}MB → {result['db_size_after']:.1f}MB ==="
            )

            return result

        except Exception as e:
            logger.error(f"DBメンテナンスエラー: {e}", exc_info=True)
            return {"error": str(e)}

    def _cleanup_trades(self, cursor) -> int:
        """クローズ済みトレードの古いレコードを削除"""
        # まずレコード数を確認
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed'")
        total_closed = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM trades WHERE status = 'closed' "
            "AND exit_time < datetime('now', ?)",
            (f"-{self.trades_retention} days",),
        )
        to_delete = cursor.fetchone()[0]

        if to_delete > 0:
            cursor.execute(
                "DELETE FROM trades WHERE status = 'closed' "
                "AND exit_time < datetime('now', ?)",
                (f"-{self.trades_retention} days",),
            )
            logger.info(
                f"トレード削除: {to_delete}件 "
                f"(残り{total_closed - to_delete}件, "
                f"保持期間: {self.trades_retention}日)"
            )

        # オープンポジションは絶対に削除しない
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'open'")
        open_count = cursor.fetchone()[0]
        if open_count > 0:
            logger.info(f"オープンポジション: {open_count}件 (削除対象外)")

        return to_delete

    def _cleanup_optimization(self, cursor) -> int:
        """最適化履歴の古いレコードを削除"""
        cursor.execute(
            "SELECT COUNT(*) FROM optimization_history WHERE created_at < datetime('now', ?)",
            (f"-{self.optimization_retention} days",),
        )
        to_delete = cursor.fetchone()[0]

        if to_delete > 0:
            cursor.execute(
                "DELETE FROM optimization_history WHERE created_at < datetime('now', ?)",
                (f"-{self.optimization_retention} days",),
            )
            logger.info(f"最適化履歴削除: {to_delete}件 (保持期間: {self.optimization_retention}日)")

        return to_delete

    def _cleanup_daily_performance(self, cursor) -> int:
        """日次パフォーマンスの古いレコードを削除"""
        cursor.execute(
            "SELECT COUNT(*) FROM daily_performance WHERE date < date('now', ?)",
            (f"-{self.daily_perf_retention} days",),
        )
        to_delete = cursor.fetchone()[0]

        if to_delete > 0:
            cursor.execute(
                "DELETE FROM daily_performance WHERE date < date('now', ?)",
                (f"-{self.daily_perf_retention} days",),
            )
            logger.info(f"日次パフォーマンス削除: {to_delete}件 (保持期間: {self.daily_perf_retention}日)")

        return to_delete

    def _run_vacuum(self) -> bool:
        """VACUUM実行でDBファイルを圧縮"""
        try:
            import sqlite3
            conn = sqlite3.connect(str(get_db_path()))
            conn.execute("VACUUM")
            conn.close()
            logger.info("VACUUM実行完了")
            return True
        except Exception as e:
            logger.error(f"VACUUMエラー: {e}")
            return False

    def _create_backup(self, db_path: Path):
        """月1回のDBバックアップを作成"""
        try:
            backup_dir = db_path.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_name = f"trades_{datetime.now().strftime('%Y%m')}.db"
            backup_path = backup_dir / backup_name

            if not backup_path.exists():
                shutil.copy2(str(db_path), str(backup_path))
                logger.info(f"DBバックアップ作成: {backup_path}")

                # 古いバックアップを削除（6ヶ月超）
                for old_backup in sorted(backup_dir.glob("trades_*.db")):
                    if old_backup != backup_path:
                        # ファイル名から年月を取得
                        try:
                            name = old_backup.stem.replace("trades_", "")
                            backup_date = datetime.strptime(name, "%Y%m")
                            age_days = (datetime.now() - backup_date).days
                            if age_days > 180:
                                old_backup.unlink()
                                logger.info(f"古いバックアップ削除: {old_backup.name}")
                        except (ValueError, OSError):
                            pass

        except Exception as e:
            logger.error(f"バックアップ作成エラー: {e}")

    @staticmethod
    def _get_file_size_mb(path: Path) -> float:
        """ファイルサイズをMBで取得"""
        try:
            if path.exists():
                return path.stat().st_size / (1024 * 1024)
        except OSError:
            pass
        return 0.0

    def get_db_stats(self) -> dict:
        """現在のDB統計情報を取得"""
        try:
            with get_connection() as conn:
                cursor = conn.cursor()

                stats = {}
                for table in ["trades", "optimization_history", "daily_performance"]:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'open'")
                stats["open_trades"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed'")
                stats["closed_trades"] = cursor.fetchone()[0]

                cursor.execute("SELECT MIN(entry_time) FROM trades")
                row = cursor.fetchone()
                stats["oldest_trade"] = row[0] if row and row[0] else "N/A"

                stats["db_size_mb"] = self._get_file_size_mb(get_db_path())

                return stats

        except Exception as e:
            logger.error(f"DB統計取得エラー: {e}")
            return {"error": str(e)}
