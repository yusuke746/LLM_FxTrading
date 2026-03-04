"""
ロギングモジュール
アプリケーション全体のログ管理
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config_manager import get, get_project_root


def setup_logger(
    name: str = "fx_bot",
    log_file: Optional[str] = None,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    ロガーをセットアップ
    
    Args:
        name: ロガー名
        log_file: ログファイルパス（Noneの場合はconfig.yamlから取得）
        level: ログレベル（Noneの場合はconfig.yamlから取得）
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = level or get("logging.level", "INFO")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # フォーマッタ
    formatter = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラ
    if log_file is None:
        log_file = get("logging.file", "data/logs/trading.log")

    log_path = get_project_root() / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = get("logging.max_bytes", 10485760)
    backup_count = get("logging.backup_count", 5)

    file_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "fx_bot") -> logging.Logger:
    """既存のロガーを取得（なければ新規作成）"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
