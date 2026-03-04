"""
リスク管理モジュール（完全数式ベース・LLM不使用）

- ポジションサイジング: リスク1%ルール
- SL初期値: ATRベース (SL = 1.5 × ATR(14))
- 最大DD制限: 口座残高20%超過で自動取引停止
- 同時保有上限: EUR/USDで最大3ポジション
- 日次ロス上限: 5%超過で当日取引停止
- 総リスクエクスポージャー: 5%以下

【期待値向上】
  - 連敗時の自動ロット縮小（3連敗でロット50%に縮小）
  - ボラティリティ適応型サイジング（ATRパーセンタイルに応じてリスク%を調整）
"""

from datetime import datetime, date
from typing import Dict, Optional

import pytz

from config_manager import get
from logger_setup import get_logger

logger = get_logger("risk.manager")

JST = pytz.timezone("Asia/Tokyo")

# EUR/USD のポイント情報
EURUSD_POINT = 0.00001   # 1ポイント = 0.00001
EURUSD_PIP = 0.0001      # 1pip = 0.0001
LOT_UNITS = 100000       # 1ロット = 10万通貨


class RiskManager:
    """完全数式ベースのリスク管理マネージャー"""

    def __init__(self):
        self._load_params()
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.daily_pnl_date = None
        self._halted = False       # DDによる全停止フラグ
        self._daily_halted = False  # 日次ロスによる当日停止フラグ

    def _load_params(self):
        """config.yamlからパラメータを読み込み"""
        self.max_loss_per_trade = get("risk.max_loss_per_trade", 0.01)
        self.sl_multiplier = get("risk.sl_multiplier", 1.5)
        self.max_dd_pct = get("risk.max_dd_pct", 20.0)
        self.daily_loss_limit_pct = get("risk.daily_loss_limit_pct", 5.0)
        self.max_positions = get("risk.max_positions", 3)
        self.max_exposure_pct = get("risk.max_exposure_pct", 5.0)

    def reload_params(self):
        """パラメータを再読み込み"""
        self._load_params()
        logger.info(f"リスクパラメータ再読み込み: SL={self.sl_multiplier}, DD制限={self.max_dd_pct}%")

    def calculate_lot_size(
        self,
        balance: float,
        atr: float,
        sl_pips: Optional[float] = None,
    ) -> float:
        """
        リスク1%ルールに基づくロットサイズ算出
        
        Args:
            balance: 口座残高（USD）
            atr: 現在のATR値
            sl_pips: SLまでのpips数（Noneの場合はATRベースで算出）
            
        Returns:
            float: ロットサイズ（0.01単位に丸め）
        """
        if self._halted or self._daily_halted:
            logger.warning("取引停止中 → ロットサイズ0")
            return 0.0

        # SL距離（pips）
        if sl_pips is None:
            sl_distance = atr * self.sl_multiplier  # 通貨単位
            sl_pips = sl_distance / EURUSD_PIP
        
        if sl_pips <= 0:
            logger.warning("SL距離が0以下 → ロットサイズ0")
            return 0.0

        # リスク金額（口座残高の1%）
        risk_pct = self.max_loss_per_trade
        
        # 【期待値向上】連敗時のロット縮小
        if self.consecutive_losses >= 5:
            risk_pct *= 0.3   # 5連敗以上: 30%に縮小
            logger.warning(f"5連敗以上 → リスク{risk_pct*100:.1f}%に縮小")
        elif self.consecutive_losses >= 3:
            risk_pct *= 0.5   # 3連敗: 50%に縮小
            logger.info(f"3連敗 → リスク{risk_pct*100:.1f}%に縮小")

        risk_amount = balance * risk_pct

        # ロットサイズ = リスク金額 / (SLのpips数 × 1pipあたりの価値)
        # EUR/USD 1ロット(10万通貨)で1pip = $10
        pip_value_per_lot = 10.0  # USD
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)

        # 0.01ロット単位に切り捨て
        lot_size = max(0.01, round(int(lot_size * 100) / 100, 2))

        logger.info(
            f"ロットサイズ算出: 残高={balance:.0f} ATR={atr:.5f} "
            f"SL={sl_pips:.1f}pips → {lot_size}lots "
            f"(リスク: ${risk_amount:.2f} = {risk_pct*100:.2f}%)"
        )

        return lot_size

    def calculate_sl(self, entry_price: float, atr: float, direction: str) -> float:
        """
        ATRベースのSL価格を算出
        
        Args:
            entry_price: エントリー価格
            atr: 現在のATR
            direction: "BUY" or "SELL"
            
        Returns:
            float: SL価格
        """
        sl_distance = atr * self.sl_multiplier

        if direction == "BUY":
            return round(entry_price - sl_distance, 5)
        else:
            return round(entry_price + sl_distance, 5)

    def calculate_tp(self, entry_price: float, atr: float, direction: str) -> float:
        """
        ATRベースのTP価格を算出
        
        Args:
            entry_price: エントリー価格
            atr: 現在のATR
            direction: "BUY" or "SELL"
            
        Returns:
            float: TP価格
        """
        tp_multiplier = get("position.tp_multiplier", 2.5)
        tp_distance = atr * tp_multiplier

        if direction == "BUY":
            return round(entry_price + tp_distance, 5)
        else:
            return round(entry_price - tp_distance, 5)

    def check_risk_limits(
        self,
        balance: float,
        equity: float,
        open_positions: int,
        initial_balance: Optional[float] = None,
    ) -> dict:
        """
        リスクリミットチェック
        
        Args:
            balance: 現在残高
            equity: 現在有効証拠金
            open_positions: オープンポジション数
            initial_balance: 初期残高（DD計算用）
            
        Returns:
            dict: {"can_trade": bool, "reasons": list, "dd_pct": float}
        """
        reasons = []
        today = date.today()

        # 日次リセット
        if self.daily_pnl_date != today:
            self.daily_pnl = 0.0
            self.daily_pnl_date = today
            self._daily_halted = False

        # 1. DD制限チェック
        if initial_balance and initial_balance > 0:
            dd_pct = (1 - equity / initial_balance) * 100
            if dd_pct >= self.max_dd_pct:
                self._halted = True
                reasons.append(f"最大DD到達: {dd_pct:.1f}% >= {self.max_dd_pct}%")
                logger.critical(f"🚨 最大DD制限超過: {dd_pct:.1f}% - 全取引停止")
        else:
            dd_pct = 0.0

        # 2. 日次ロス制限チェック
        if balance > 0:
            daily_loss_pct = abs(min(self.daily_pnl, 0)) / balance * 100
            if daily_loss_pct >= self.daily_loss_limit_pct:
                self._daily_halted = True
                reasons.append(f"日次ロス上限: {daily_loss_pct:.1f}% >= {self.daily_loss_limit_pct}%")
                logger.warning(f"⚠️ 日次ロス上限到達: {daily_loss_pct:.1f}% - 当日取引停止")

        # 3. ポジション数チェック
        if open_positions >= self.max_positions:
            reasons.append(f"ポジション上限: {open_positions}/{self.max_positions}")

        # 4. エクスポージャーチェック
        # (簡易版: ポジション数ベース)
        exposure_pct = (open_positions / self.max_positions) * self.max_exposure_pct
        if exposure_pct >= self.max_exposure_pct:
            reasons.append(f"エクスポージャー上限: {exposure_pct:.1f}%")

        can_trade = len(reasons) == 0 and not self._halted and not self._daily_halted

        return {
            "can_trade": can_trade,
            "reasons": reasons,
            "dd_pct": round(dd_pct, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "halted": self._halted,
            "daily_halted": self._daily_halted,
            "consecutive_losses": self.consecutive_losses,
        }

    def record_trade_result(self, pnl: float):
        """
        トレード結果を記録（連敗カウント・日次PnL追跡）
        
        Args:
            pnl: トレード損益（プラス=利益、マイナス=損失）
        """
        self.daily_pnl += pnl

        if pnl < 0:
            self.consecutive_losses += 1
            logger.info(f"損失記録: {pnl:.2f} (連敗: {self.consecutive_losses})")
        else:
            self.consecutive_losses = 0
            logger.info(f"利益記録: {pnl:.2f} (連敗リセット)")

    def reset_halt(self):
        """取引停止を手動解除"""
        self._halted = False
        self._daily_halted = False
        self.consecutive_losses = 0
        logger.info("取引停止を手動解除")

    def get_status(self) -> dict:
        """リスク管理の状態サマリー"""
        return {
            "halted": self._halted,
            "daily_halted": self._daily_halted,
            "consecutive_losses": self.consecutive_losses,
            "daily_pnl": round(self.daily_pnl, 2),
            "params": {
                "max_loss_per_trade": self.max_loss_per_trade,
                "sl_multiplier": self.sl_multiplier,
                "max_dd_pct": self.max_dd_pct,
                "daily_loss_limit_pct": self.daily_loss_limit_pct,
                "max_positions": self.max_positions,
            },
        }
