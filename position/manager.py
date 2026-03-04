"""
ポジション管理モジュール（4ステップ管理）

STEP 1: Break Even設定 → SLを建値±スプレッド×1.5に移動
STEP 2: 含み益50%を利益確定 → ロットの50%をクローズ
STEP 3: 残り50%にTP再設定 → TP = 建値 + ATR × TP_MULTIPLIER
STEP 4: トレーリングストップ → SL = 現在価格 - ATR × TRAIL_MULTIPLIER

【期待値向上】時間経過による利確判断:
  ポジション保有時間が長すぎる場合（例: 24H以上）、
  含み損が一定範囲内なら早期クローズして資金効率を改善
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import pytz

from config_manager import get
from logger_setup import get_logger

logger = get_logger("position.manager")

JST = pytz.timezone("Asia/Tokyo")


class PositionStep(Enum):
    """ポジション管理ステップ"""
    INITIAL = "initial"              # エントリー直後
    BREAK_EVEN = "break_even"        # STEP1: BE設定済み
    PARTIAL_CLOSED = "partial_closed" # STEP2: 部分利確済み
    TRAILING = "trailing"            # STEP3&4: TP再設定＋トレーリング中
    CLOSED = "closed"                # 決済済み


@dataclass
class ManagedPosition:
    """管理対象ポジション"""
    ticket: int
    symbol: str
    direction: str                   # "BUY" or "SELL"
    entry_price: float
    lot_size: float
    sl: float
    tp: float
    atr_at_entry: float              # エントリー時のATR
    spread: float                    # エントリー時のスプレッド
    entry_time: datetime = field(default_factory=lambda: datetime.now(JST))
    step: PositionStep = PositionStep.INITIAL
    partial_closed: bool = False
    remaining_lot: float = 0.0
    original_lot: float = 0.0
    trail_sl: float = 0.0
    
    def __post_init__(self):
        self.remaining_lot = self.lot_size
        self.original_lot = self.lot_size


class PositionManager:
    """4ステップポジション管理マネージャー"""

    def __init__(self, executor=None):
        self.executor = executor
        self.positions: Dict[int, ManagedPosition] = {}
        
        # config.yamlからパラメータ取得
        self._load_params()

    def _load_params(self):
        """config.yamlからパラメータを読み込み"""
        self.be_trigger = get("position.be_trigger", 1.0)
        self.partial_trigger = get("position.partial_trigger", 1.2)
        self.partial_ratio = get("position.partial_ratio", 0.5)
        self.tp_multiplier = get("position.tp_multiplier", 2.5)
        self.trail_multiplier = get("position.trail_multiplier", 1.0)
        self.max_hold_hours = 48  # 【期待値向上】最大保有時間

    def reload_params(self):
        """パラメータを再読み込み（週次最適化後に呼ぶ）"""
        self._load_params()
        logger.info(
            f"ポジション管理パラメータ再読み込み: "
            f"BE={self.be_trigger}, Partial={self.partial_trigger}, "
            f"TP={self.tp_multiplier}, Trail={self.trail_multiplier}"
        )

    def register(self, position: ManagedPosition):
        """ポジションを管理対象に登録"""
        self.positions[position.ticket] = position
        logger.info(
            f"ポジション登録: ticket={position.ticket} "
            f"{position.direction} {position.lot_size} lots @ {position.entry_price}"
        )

    def update(self, ticket: int, current_price: float) -> Optional[dict]:
        """
        ポジションの状態を更新し、必要なアクションを返す
        
        Args:
            ticket: ポジションチケット
            current_price: 現在価格
            
        Returns:
            dict: アクション指示（None=アクションなし）
                {"action": "modify_sl"/"partial_close"/"modify_tp"/"close", ...}
        """
        pos = self.positions.get(ticket)
        if pos is None or pos.step == PositionStep.CLOSED:
            return None

        # 含み損益計算
        if pos.direction == "BUY":
            unrealized_pnl = current_price - pos.entry_price
        else:  # SELL
            unrealized_pnl = pos.entry_price - current_price

        atr = pos.atr_at_entry

        # === STEP 1: Break Even設定 ===
        if pos.step == PositionStep.INITIAL:
            be_threshold = atr * self.be_trigger
            if unrealized_pnl >= be_threshold:
                return self._step1_break_even(pos)

        # === STEP 2: 部分利確 ===
        elif pos.step == PositionStep.BREAK_EVEN:
            partial_threshold = atr * self.partial_trigger
            if unrealized_pnl >= partial_threshold:
                return self._step2_partial_close(pos, current_price)

        # === STEP 3 & 4: TP再設定 + トレーリング ===
        elif pos.step == PositionStep.PARTIAL_CLOSED:
            return self._step3_set_tp_and_start_trail(pos, current_price)

        elif pos.step == PositionStep.TRAILING:
            return self._step4_trailing_stop(pos, current_price)

        # 【期待値向上】時間経過チェック
        time_action = self._check_time_based_exit(pos, unrealized_pnl)
        if time_action:
            return time_action

        return None

    def _step1_break_even(self, pos: ManagedPosition) -> dict:
        """STEP1: SLを建値±スプレッド×1.5に移動"""
        if pos.direction == "BUY":
            new_sl = pos.entry_price + pos.spread * 1.5
        else:
            new_sl = pos.entry_price - pos.spread * 1.5

        pos.step = PositionStep.BREAK_EVEN
        pos.sl = new_sl
        logger.info(f"STEP1 BE設定: ticket={pos.ticket} SL→{new_sl:.5f}")

        return {
            "action": "modify_sl",
            "ticket": pos.ticket,
            "new_sl": new_sl,
            "step": "break_even",
        }

    def _step2_partial_close(self, pos: ManagedPosition, current_price: float) -> dict:
        """STEP2: ロットの50%をクローズ"""
        close_lot = round(pos.remaining_lot * self.partial_ratio, 2)
        close_lot = max(close_lot, 0.01)  # 最小ロット

        pos.step = PositionStep.PARTIAL_CLOSED
        pos.partial_closed = True
        pos.remaining_lot = round(pos.remaining_lot - close_lot, 2)

        logger.info(
            f"STEP2 部分利確: ticket={pos.ticket} "
            f"{close_lot}lot @ {current_price:.5f} (残り{pos.remaining_lot}lot)"
        )

        return {
            "action": "partial_close",
            "ticket": pos.ticket,
            "close_lot": close_lot,
            "remaining_lot": pos.remaining_lot,
            "step": "partial_close",
        }

    def _step3_set_tp_and_start_trail(self, pos: ManagedPosition, current_price: float) -> dict:
        """STEP3: 残り50%にTP再設定し、STEP4トレーリングを開始"""
        atr = pos.atr_at_entry

        if pos.direction == "BUY":
            new_tp = pos.entry_price + atr * self.tp_multiplier
        else:
            new_tp = pos.entry_price - atr * self.tp_multiplier

        pos.step = PositionStep.TRAILING
        pos.tp = new_tp
        pos.trail_sl = pos.sl  # 現在のSL（BE位置）からトレーリング開始

        logger.info(
            f"STEP3 TP再設定+トレーリング開始: ticket={pos.ticket} "
            f"TP→{new_tp:.5f}"
        )

        return {
            "action": "modify_tp",
            "ticket": pos.ticket,
            "new_tp": new_tp,
            "step": "tp_reset_and_trail_start",
        }

    def _step4_trailing_stop(self, pos: ManagedPosition, current_price: float) -> Optional[dict]:
        """STEP4: トレーリングストップで追随"""
        atr = pos.atr_at_entry
        trail_distance = atr * self.trail_multiplier

        if pos.direction == "BUY":
            new_trail_sl = current_price - trail_distance
            if new_trail_sl > pos.trail_sl:
                pos.trail_sl = new_trail_sl
                pos.sl = new_trail_sl
                logger.debug(f"STEP4 トレーリング更新: ticket={pos.ticket} SL→{new_trail_sl:.5f}")
                return {
                    "action": "modify_sl",
                    "ticket": pos.ticket,
                    "new_sl": new_trail_sl,
                    "step": "trailing",
                }
        else:  # SELL
            new_trail_sl = current_price + trail_distance
            if new_trail_sl < pos.trail_sl or pos.trail_sl == 0:
                pos.trail_sl = new_trail_sl
                pos.sl = new_trail_sl
                logger.debug(f"STEP4 トレーリング更新: ticket={pos.ticket} SL→{new_trail_sl:.5f}")
                return {
                    "action": "modify_sl",
                    "ticket": pos.ticket,
                    "new_sl": new_trail_sl,
                    "step": "trailing",
                }

        return None

    def _check_time_based_exit(self, pos: ManagedPosition, unrealized_pnl: float) -> Optional[dict]:
        """
        【期待値向上】時間経過による早期クローズ判断
        保有時間が長すぎてPnLがほぼゼロの場合、資金効率改善のため早期クローズ
        """
        now = datetime.now(JST)
        hold_hours = (now - pos.entry_time).total_seconds() / 3600

        if hold_hours >= self.max_hold_hours:
            # ATRの0.3倍以内の含み損なら早期クローズ
            if abs(unrealized_pnl) < pos.atr_at_entry * 0.3:
                logger.info(
                    f"時間ベース早期クローズ: ticket={pos.ticket} "
                    f"保有{hold_hours:.1f}時間 PnL={unrealized_pnl:.5f}"
                )
                return {
                    "action": "close",
                    "ticket": pos.ticket,
                    "reason": "time_based_exit",
                    "hold_hours": hold_hours,
                }

        return None

    def mark_closed(self, ticket: int, reason: str = ""):
        """ポジションをクローズ済みにマーク"""
        pos = self.positions.get(ticket)
        if pos:
            pos.step = PositionStep.CLOSED
            logger.info(f"ポジションクローズ: ticket={ticket} 理由={reason}")

    def get_active_positions(self) -> List[ManagedPosition]:
        """アクティブなポジションを取得"""
        return [
            pos for pos in self.positions.values()
            if pos.step != PositionStep.CLOSED
        ]

    def get_status(self) -> dict:
        """ポジション管理の状態サマリー"""
        active = self.get_active_positions()
        return {
            "total_managed": len(self.positions),
            "active": len(active),
            "by_step": {
                step.value: sum(1 for p in active if p.step == step)
                for step in PositionStep
                if step != PositionStep.CLOSED
            },
            "params": {
                "be_trigger": self.be_trigger,
                "partial_trigger": self.partial_trigger,
                "partial_ratio": self.partial_ratio,
                "tp_multiplier": self.tp_multiplier,
                "trail_multiplier": self.trail_multiplier,
            },
        }
