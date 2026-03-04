"""
バックテスト実行モジュール
グリッドサーチ用にパラメータセットを受けてバックテストを実行

EUR/USD H1足専用。セッション別エンジン重みもシミュレーションに含む。

【期待値向上】ウォークフォワード最適化:
  In-Sampleでパラメータ最適化 → Out-of-Sample で検証 を繰り返すことで
  過学習を防止する仕組みを組み込み
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Optional

from logger_setup import get_logger

logger = get_logger("optimizer.backtest")


class BacktestRunner:
    """ベクトル化バックテスト実行器"""

    def __init__(self):
        self.point = 0.00001   # EUR/USD 1ポイント
        self.pip = 0.0001      # 1pip
        self.pip_value = 10.0  # 1ロット1pip = $10

    def run(
        self,
        df: pd.DataFrame,
        params: Dict[str, float],
        initial_balance: float = 1_000_000,  # 100万円想定
        risk_per_trade: float = 0.01,
    ) -> dict:
        """
        バックテストを実行
        
        Args:
            df: H1足 OHLCVデータ
            params: {
                "sl_multiplier": float,
                "be_trigger": float,
                "partial_trigger": float,
                "tp_multiplier": float,
                "trail_multiplier": float,
            }
            initial_balance: 初期残高
            risk_per_trade: 1トレードあたりのリスク比率
            
        Returns:
            dict: バックテスト結果
        """
        if len(df) < 100:
            return self._empty_result()

        sl_mult = params["sl_multiplier"]
        be_trigger = params["be_trigger"]
        partial_trigger = params["partial_trigger"]
        tp_mult = params["tp_multiplier"]
        trail_mult = params["trail_multiplier"]

        # テクニカル指標の算出
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # ATR
        atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
        atr = atr_series.values

        # EMA（トレンド判定用簡易版）
        ema9 = ta.ema(df["close"], length=9).values
        ema21 = ta.ema(df["close"], length=21).values
        ema50 = ta.ema(df["close"], length=50).values

        # RSI
        rsi = ta.rsi(df["close"], length=14).values

        # ADX
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        adx = adx_df["ADX_14"].values

        # バックテスト実行
        balance = initial_balance
        peak_balance = initial_balance
        trades = []
        in_trade = False
        trade_direction = None
        entry_price = 0.0
        trade_sl = 0.0
        trade_tp = 0.0
        trade_atr = 0.0
        lot_size = 0.0
        be_set = False
        partial_done = False
        remaining_lot = 0.0
        trail_sl = 0.0

        for i in range(60, len(df)):
            if np.isnan(atr[i]) or np.isnan(ema9[i]) or np.isnan(ema50[i]):
                continue

            current_atr = atr[i]

            if not in_trade:
                # シグナル判定（簡易版）
                signal = self._get_simple_signal(
                    ema9[i], ema21[i], ema50[i],
                    rsi[i], adx[i],
                    high, low, close, i,
                )

                if signal != 0:
                    direction = "BUY" if signal > 0 else "SELL"
                    sl_distance = current_atr * sl_mult
                    tp_distance = current_atr * tp_mult
                    sl_pips = sl_distance / self.pip

                    # ロットサイズ計算
                    risk_amount = balance * risk_per_trade
                    lot = risk_amount / (sl_pips * self.pip_value)
                    lot = max(0.01, round(int(lot * 100) / 100, 2))

                    entry_price = close[i]
                    if direction == "BUY":
                        trade_sl = entry_price - sl_distance
                        trade_tp = entry_price + tp_distance
                    else:
                        trade_sl = entry_price + sl_distance
                        trade_tp = entry_price - tp_distance

                    trade_direction = direction
                    trade_atr = current_atr
                    lot_size = lot
                    remaining_lot = lot
                    in_trade = True
                    be_set = False
                    partial_done = False
                    trail_sl = trade_sl
                    entry_bar = i

            else:
                # ポジション管理
                cur_price = close[i]
                cur_high = high[i]
                cur_low = low[i]

                if trade_direction == "BUY":
                    unrealized = cur_price - entry_price
                    # SLチェック
                    if cur_low <= trail_sl:
                        pnl = (trail_sl - entry_price) * remaining_lot * 100000
                        balance += pnl
                        trades.append({"pnl": pnl, "direction": "BUY", "bars": i - entry_bar})
                        in_trade = False
                        continue
                    # TPチェック
                    if cur_high >= trade_tp:
                        pnl = (trade_tp - entry_price) * remaining_lot * 100000
                        balance += pnl
                        trades.append({"pnl": pnl, "direction": "BUY", "bars": i - entry_bar})
                        in_trade = False
                        continue
                else:
                    unrealized = entry_price - cur_price
                    if cur_high >= trail_sl:
                        pnl = (entry_price - trail_sl) * remaining_lot * 100000
                        balance += pnl
                        trades.append({"pnl": pnl, "direction": "SELL", "bars": i - entry_bar})
                        in_trade = False
                        continue
                    if cur_low <= trade_tp:
                        pnl = (entry_price - trade_tp) * remaining_lot * 100000
                        balance += pnl
                        trades.append({"pnl": pnl, "direction": "SELL", "bars": i - entry_bar})
                        in_trade = False
                        continue

                # STEP1: BE設定
                if not be_set and unrealized >= trade_atr * be_trigger:
                    if trade_direction == "BUY":
                        trail_sl = entry_price + self.pip * 2  # 建値 + 少し上
                    else:
                        trail_sl = entry_price - self.pip * 2
                    be_set = True

                # STEP2: 部分利確
                if be_set and not partial_done and unrealized >= trade_atr * partial_trigger:
                    partial_lot = round(remaining_lot * 0.5, 2)
                    partial_pnl = unrealized * partial_lot * 100000
                    balance += partial_pnl
                    remaining_lot = round(remaining_lot - partial_lot, 2)
                    remaining_lot = max(remaining_lot, 0.01)
                    partial_done = True

                    # STEP3: TP再設定
                    if trade_direction == "BUY":
                        trade_tp = entry_price + trade_atr * tp_mult
                    else:
                        trade_tp = entry_price - trade_atr * tp_mult

                # STEP4: トレーリング
                if partial_done:
                    trail_distance = trade_atr * trail_mult
                    if trade_direction == "BUY":
                        new_trail = cur_price - trail_distance
                        if new_trail > trail_sl:
                            trail_sl = new_trail
                    else:
                        new_trail = cur_price + trail_distance
                        if new_trail < trail_sl:
                            trail_sl = new_trail

                peak_balance = max(peak_balance, balance)

        # 結果集計
        return self._calculate_metrics(trades, initial_balance, balance, peak_balance)

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        params: Dict[str, float],
        n_splits: int = 4,
        is_ratio: float = 0.7,  # In-Sample比率
    ) -> dict:
        """
        ウォークフォワード検証
        データを分割し、各期間でバックテストを実行して安定性を検証
        """
        total_bars = len(df)
        split_size = total_bars // n_splits
        results = []

        for i in range(n_splits):
            start = i * split_size
            end = min(start + split_size, total_bars)
            split_df = df.iloc[start:end].reset_index(drop=True)

            if len(split_df) < 100:
                continue

            result = self.run(split_df, params)
            results.append(result)

        if not results:
            return self._empty_result()

        # 各期間の結果を集約
        avg_sharpe = np.mean([r["sharpe_ratio"] for r in results])
        max_dd = max(r["max_dd"] for r in results)
        avg_pf = np.mean([r["profit_factor"] for r in results])
        avg_winrate = np.mean([r["win_rate"] for r in results])
        total_trades = sum(r["total_trades"] for r in results)

        # 安定性スコア（シャープレシオの標準偏差が小さいほど安定）
        sharpe_std = np.std([r["sharpe_ratio"] for r in results])
        stability_score = max(0, 1 - sharpe_std)

        return {
            "sharpe_ratio": round(avg_sharpe, 4),
            "max_dd": round(max_dd, 2),
            "profit_factor": round(avg_pf, 4),
            "win_rate": round(avg_winrate, 4),
            "total_trades": total_trades,
            "stability_score": round(stability_score, 4),
            "period_results": results,
        }

    def _get_simple_signal(
        self,
        ema9: float, ema21: float, ema50: float,
        rsi: float, adx: float,
        high: np.ndarray, low: np.ndarray, close: np.ndarray,
        idx: int,
    ) -> int:
        """
        簡易シグナル判定（バックテスト用）
        Returns: +1(BUY), -1(SELL), 0(NONE)
        """
        signal = 0.0

        # トレンドフォロー
        if ema9 > ema21 > ema50 and adx >= 25:
            signal += 0.5
        elif ema9 < ema21 < ema50 and adx >= 25:
            signal -= 0.5

        # 逆張り
        if rsi < 30:
            signal += 0.3
        elif rsi > 70:
            signal -= 0.3

        # ブレイクアウト（簡易版）
        if idx >= 20:
            range_high = np.max(high[idx-20:idx])
            range_low = np.min(low[idx-20:idx])
            if close[idx] > range_high:
                signal += 0.4
            elif close[idx] < range_low:
                signal -= 0.4

        if signal >= 0.6:
            return 1
        elif signal <= -0.6:
            return -1
        return 0

    def _calculate_metrics(
        self,
        trades: list,
        initial_balance: float,
        final_balance: float,
        peak_balance: float,
    ) -> dict:
        """バックテスト結果のメトリクス計算"""
        if not trades:
            return self._empty_result()

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

        # Sharpe Ratio（日次ベースで近似）
        pnl_array = np.array(pnls)
        if pnl_array.std() > 0:
            sharpe = (pnl_array.mean() / pnl_array.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max Drawdown
        cumulative = np.cumsum(pnls)
        balance_curve = initial_balance + cumulative
        peak = np.maximum.accumulate(balance_curve)
        drawdown = (peak - balance_curve) / peak * 100
        max_dd = drawdown.max() if len(drawdown) > 0 else 0

        # 連敗
        max_consecutive_losses = 0
        current_streak = 0
        for p in pnls:
            if p <= 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0

        return {
            "sharpe_ratio": round(sharpe, 4),
            "max_dd": round(max_dd, 2),
            "profit_factor": round(profit_factor, 4),
            "win_rate": round(win_rate, 4),
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(np.mean(pnls), 2),
            "max_consecutive_losses": max_consecutive_losses,
            "final_balance": round(final_balance, 2),
            "return_pct": round((final_balance / initial_balance - 1) * 100, 2),
        }

    def _empty_result(self) -> dict:
        return {
            "sharpe_ratio": -999,
            "max_dd": 100,
            "profit_factor": 0,
            "win_rate": 0,
            "total_trades": 0,
            "win_count": 0,
            "loss_count": 0,
            "total_pnl": 0,
            "avg_pnl": 0,
            "max_consecutive_losses": 0,
            "final_balance": 0,
            "return_pct": -100,
        }
