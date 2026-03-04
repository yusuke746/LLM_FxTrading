"""
設定管理モジュール
config.yaml の読み込みとグローバルアクセスを提供
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional

_config: Optional[dict] = None
_config_path: Optional[Path] = None


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得"""
    return Path(__file__).parent


def load_config(config_path: Optional[str] = None) -> dict:
    """config.yaml を読み込む"""
    global _config, _config_path

    if config_path is None:
        _config_path = get_project_root() / "config.yaml"
    else:
        _config_path = Path(config_path)

    if not _config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {_config_path}")

    with open(_config_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)

    return _config


def get_config() -> dict:
    """現在の設定を取得（未読み込みの場合は自動読み込み）"""
    global _config
    if _config is None:
        load_config()
    return _config


def get(key_path: str, default: Any = None) -> Any:
    """
    ドット区切りのキーパスで設定値を取得
    例: get("risk.sl_multiplier") → 1.5
    """
    config = get_config()
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def save_config(config: dict, config_path: Optional[str] = None) -> None:
    """config.yaml に書き込む"""
    global _config, _config_path

    if config_path:
        path = Path(config_path)
    elif _config_path:
        path = _config_path
    else:
        path = get_project_root() / "config.yaml"

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    _config = config


def reload_config() -> dict:
    """設定を再読み込み"""
    global _config
    _config = None
    return load_config(str(_config_path) if _config_path else None)
