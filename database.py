"""
データベースモジュール
トレードログ・最適化履歴のSQLite管理
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config_manager import get_project_root
from logger_setup import get_logger

logger = get_logger("database")

DB_PATH = get_project_root() / "data" / "trade_logs" / "trades.db"


def get_db_path() -> Path:
    """データベースファイルパスを取得（ディレクトリも作成）"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


@contextmanager
def get_connection():
    """SQLite接続のコンテキストマネージャ"""
    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """データベースの初期化（テーブル作成）"""
    with get_connection() as conn:
        cursor = conn.cursor()

        # トレードログテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket INTEGER UNIQUE,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,         -- 'BUY' or 'SELL'
                entry_price REAL NOT NULL,
                exit_price REAL,
                lot_size REAL NOT NULL,
                sl REAL,
                tp REAL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                session TEXT,                    -- 'asia', 'london', etc.
                signal_score REAL,
                signal_engines TEXT,             -- JSON: エンジン別スコア
                pnl REAL,
                pnl_pips REAL,
                status TEXT DEFAULT 'open',      -- 'open', 'closed', 'partial'
                close_reason TEXT,               -- 'tp', 'sl', 'trailing', 'reverse', 'manual'
                llm_sentiment REAL,
                llm_event_risk TEXT,
                notes TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # 最適化履歴テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT NOT NULL,
                lookback_weeks INTEGER,
                sl_multiplier REAL,
                be_trigger REAL,
                partial_trigger REAL,
                tp_multiplier REAL,
                trail_multiplier REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                profit_factor REAL,
                win_rate REAL,
                total_trades INTEGER,
                applied INTEGER DEFAULT 0,       -- 1=適用済み, 0=不適用
                notes TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # 日次パフォーマンステーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                total_pnl REAL DEFAULT 0,
                trade_count INTEGER DEFAULT 0,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                max_dd_pct REAL DEFAULT 0,
                balance_start REAL,
                balance_end REAL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        logger.info("データベース初期化完了")


def insert_trade(trade: Dict[str, Any]) -> int:
    """トレードを挿入"""
    with get_connection() as conn:
        cursor = conn.cursor()
        columns = ", ".join(trade.keys())
        placeholders = ", ".join(["?"] * len(trade))
        cursor.execute(
            f"INSERT INTO trades ({columns}) VALUES ({placeholders})",
            list(trade.values()),
        )
        return cursor.lastrowid


def update_trade(ticket: int, updates: Dict[str, Any]) -> None:
    """トレードを更新"""
    with get_connection() as conn:
        cursor = conn.cursor()
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        updates["updated_at"] = datetime.now().isoformat()
        cursor.execute(
            f"UPDATE trades SET {set_clause}, updated_at = ? WHERE ticket = ?",
            list(updates.values()) + [updates["updated_at"], ticket],
        )


def get_open_positions(symbol: Optional[str] = None) -> List[dict]:
    """オープンポジションを取得"""
    with get_connection() as conn:
        cursor = conn.cursor()
        if symbol:
            cursor.execute(
                "SELECT * FROM trades WHERE status = 'open' AND symbol = ?",
                (symbol,),
            )
        else:
            cursor.execute("SELECT * FROM trades WHERE status = 'open'")
        return [dict(row) for row in cursor.fetchall()]


def get_trade_log(days: Optional[int] = None, weeks: Optional[int] = None) -> List[dict]:
    """トレードログを取得"""
    with get_connection() as conn:
        cursor = conn.cursor()
        if weeks:
            cursor.execute(
                "SELECT * FROM trades WHERE entry_time >= datetime('now', ?) ORDER BY entry_time DESC",
                (f"-{weeks * 7} days",),
            )
        elif days:
            cursor.execute(
                "SELECT * FROM trades WHERE entry_time >= datetime('now', ?) ORDER BY entry_time DESC",
                (f"-{days} days",),
            )
        else:
            cursor.execute("SELECT * FROM trades ORDER BY entry_time DESC LIMIT 500")
        return [dict(row) for row in cursor.fetchall()]


def get_daily_pnl(date: str) -> float:
    """特定日のP&Lを取得"""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COALESCE(SUM(pnl), 0) as total_pnl FROM trades "
            "WHERE DATE(exit_time) = ? AND status = 'closed'",
            (date,),
        )
        row = cursor.fetchone()
        return row["total_pnl"] if row else 0.0


def insert_optimization_result(result: Dict[str, Any]) -> int:
    """最適化結果を挿入"""
    with get_connection() as conn:
        cursor = conn.cursor()
        columns = ", ".join(result.keys())
        placeholders = ", ".join(["?"] * len(result))
        cursor.execute(
            f"INSERT INTO optimization_history ({columns}) VALUES ({placeholders})",
            list(result.values()),
        )
        return cursor.lastrowid
