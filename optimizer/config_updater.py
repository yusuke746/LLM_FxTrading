"""
config.yaml 自動更新モジュール
週次最適化で得た最適パラメータを安全にconfig.yamlに書き込む

安全ガード:
  - パラメータ変化が大きすぎる場合は自動更新せずアラートのみ送信
  - 更新前のバックアップを作成
  - 更新履歴をデータベースに記録
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml
import pytz

from config_manager import get, get_project_root, load_config, save_config
from logger_setup import get_logger

logger = get_logger("optimizer.config_updater")

JST = pytz.timezone("Asia/Tokyo")


class ConfigUpdater:
    """config.yaml 安全更新マネージャー"""

    def __init__(self):
        self.config_path = get_project_root() / "config.yaml"
        self.backup_dir = get_project_root() / "data" / "config_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 安全ガード: 最大変化量
        self.change_limits = {
            "sl_multiplier": get("optimizer.max_param_change.sl_multiplier", 0.5),
            "tp_multiplier": get("optimizer.max_param_change.tp_multiplier", 0.8),
            "trail_multiplier": get("optimizer.max_param_change.trail_multiplier", 0.5),
        }

    def update(
        self,
        new_params: Dict[str, float],
        backtest_result: Optional[dict] = None,
        force: bool = False,
    ) -> dict:
        """
        config.yaml を新しいパラメータで更新
        
        Args:
            new_params: 新しいパラメータ値
            backtest_result: バックテスト結果（ログ用）
            force: 安全ガードを無視して強制更新
            
        Returns:
            dict: {"updated": bool, "reason": str, "backup_path": str}
        """
        # 現在のパラメータを取得
        current_config = load_config(str(self.config_path))
        current_params = self._extract_current_params(current_config)

        # 安全ガードチェック
        if not force:
            safety_check = self.is_safe_to_update(new_params, current_params)
            if not safety_check["safe"]:
                logger.warning(
                    f"安全ガード発動: {safety_check['reason']} "
                    f"手動確認が必要です"
                )
                return {
                    "updated": False,
                    "reason": safety_check["reason"],
                    "unsafe_params": safety_check["unsafe_params"],
                    "backup_path": "",
                }

        # バックアップ作成
        backup_path = self._create_backup()

        # config.yaml 更新
        try:
            config = current_config.copy()

            config["risk"]["sl_multiplier"] = new_params.get(
                "sl_multiplier", current_params["sl_multiplier"]
            )
            config["position"]["be_trigger"] = new_params.get(
                "be_trigger", current_params["be_trigger"]
            )
            config["position"]["partial_trigger"] = new_params.get(
                "partial_trigger", current_params["partial_trigger"]
            )
            config["position"]["tp_multiplier"] = new_params.get(
                "tp_multiplier", current_params["tp_multiplier"]
            )
            config["position"]["trail_multiplier"] = new_params.get(
                "trail_multiplier", current_params["trail_multiplier"]
            )
            config["meta"]["last_optimized"] = datetime.now(JST).isoformat()

            save_config(config, str(self.config_path))

            logger.info(
                f"config.yaml 更新完了: {new_params} "
                f"(バックアップ: {backup_path})"
            )

            return {
                "updated": True,
                "reason": "OK",
                "backup_path": str(backup_path),
                "old_params": current_params,
                "new_params": new_params,
                "backtest_result": backtest_result,
            }

        except Exception as e:
            logger.error(f"config.yaml 更新失敗: {e}")
            # バックアップから復元
            self._restore_backup(backup_path)
            return {
                "updated": False,
                "reason": f"更新失敗: {e}",
                "backup_path": str(backup_path),
            }

    def is_safe_to_update(
        self,
        new_params: Dict[str, float],
        current_params: Dict[str, float],
    ) -> dict:
        """
        パラメータ変化量の安全チェック
        
        Returns:
            dict: {"safe": bool, "reason": str, "unsafe_params": list}
        """
        unsafe_params = []

        for key, limit in self.change_limits.items():
            if key in new_params and key in current_params:
                change = abs(new_params[key] - current_params[key])
                if change > limit:
                    unsafe_params.append({
                        "param": key,
                        "current": current_params[key],
                        "new": new_params[key],
                        "change": round(change, 3),
                        "limit": limit,
                    })

        if unsafe_params:
            reason = "; ".join(
                f"{p['param']}: {p['current']}→{p['new']} (変化{p['change']} > 制限{p['limit']})"
                for p in unsafe_params
            )
            return {"safe": False, "reason": reason, "unsafe_params": unsafe_params}

        return {"safe": True, "reason": "OK", "unsafe_params": []}

    def _extract_current_params(self, config: dict) -> Dict[str, float]:
        """config dictから最適化対象パラメータを抽出"""
        return {
            "sl_multiplier": config.get("risk", {}).get("sl_multiplier", 1.5),
            "be_trigger": config.get("position", {}).get("be_trigger", 1.0),
            "partial_trigger": config.get("position", {}).get("partial_trigger", 1.2),
            "tp_multiplier": config.get("position", {}).get("tp_multiplier", 2.5),
            "trail_multiplier": config.get("position", {}).get("trail_multiplier", 1.0),
        }

    def _create_backup(self) -> Path:
        """config.yaml のバックアップを作成"""
        timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"config_backup_{timestamp}.yaml"
        shutil.copy2(str(self.config_path), str(backup_path))
        logger.info(f"バックアップ作成: {backup_path}")
        return backup_path

    def _restore_backup(self, backup_path: Path):
        """バックアップからconfig.yamlを復元"""
        try:
            shutil.copy2(str(backup_path), str(self.config_path))
            logger.info(f"バックアップから復元: {backup_path}")
        except Exception as e:
            logger.error(f"バックアップ復元失敗: {e}")
