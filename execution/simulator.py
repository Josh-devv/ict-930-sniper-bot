"""
Institutional-Grade NQ Futures Intrabar Execution Simulator.
Accurately models:
- Dynamic 25% Breakeven progression mid-trade and intrabar
- Intrabar stop-loss / take-profit collision handling (worst-case and path-based)
- CME NQ futures tick arithmetic ($20/pt, 0.25 tick)
- Realistic commissions, regulatory fees, and market slippage
- R-multiple expectancy, trade duration, and trade logs
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from config import StrategyConfig, ExecutionConfig, NQContractSpec
from data.contract import FuturesCalculator
from strategy.rules import TradeSignal


@dataclass
class SimulatedTrade:
    trade_id: int
    direction: int                # 1 = Long, -1 = Short
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    initial_sl: float = 0.0
    current_sl: float = 0.0
    target_price: float = 0.0
    be_trigger_price: float = 0.0
    risk_points: float = 0.0      # 1R in points
    contracts: int = 1
    be_activated: bool = False
    be_activated_time: Optional[pd.Timestamp] = None
    exit_reason: str = ""         # "TARGET_3R", "BREAKEVEN", "STOP_LOSS", "SQUARE_OFF"
    gross_points: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0
    realized_r: float = 0.0       # Net or gross return in R


class NQIntrabarSimulator:
    """
    Event-driven bar-by-bar futures simulator.
    Solves limitations of backtesting.py by accurately managing intrabar state transitions.
    """

    def __init__(
        self,
        strategy_config: Optional[StrategyConfig] = None,
        execution_config: Optional[ExecutionConfig] = None,
        contract_spec: Optional[NQContractSpec] = None,
    ):
        self.strat_cfg = strategy_config or StrategyConfig()
        self.exec_cfg = execution_config or ExecutionConfig()
        self.contract = contract_spec or NQContractSpec()
        self.calc = FuturesCalculator(contract=self.contract, execution=self.exec_cfg)

    def run(self, df: pd.DataFrame, signals: List[TradeSignal]) -> List[SimulatedTrade]:
        """
        Executes simulated trading over historical 1-minute bars.
        Orders are entered at next bar's Open following the signal to prevent lookahead bias.
        """
        trades: List[SimulatedTrade] = []
        active_trade: Optional[SimulatedTrade] = None
        trade_id_seq = 1

        # Map signal triggers by index
        signal_map = {sig.index: sig for sig in signals}

        total_bars = len(df)
        opens = df["Open"].values
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        timestamps = df.index

        # Slippage adjustment on entry
        entry_slippage_points = self.calc.ticks_to_points(self.exec_cfg.slippage_ticks)

        for i in range(total_bars):
            current_time = timestamps[i]
            bar_open = opens[i]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]

            # 1. Enter New Trade on Open of bar i if signal triggered at bar i - 1
            prev_bar_idx = i - 1
            if active_trade is None and prev_bar_idx in signal_map:
                sig = signal_map[prev_bar_idx]
                fill_price = sig.entry_price
                sl = sig.stop_loss
                tp = sig.target_price
                be = sig.breakeven_trigger_price
                risk = sig.risk_points

                if risk > 0 and self.strat_cfg.min_risk_points <= risk <= self.strat_cfg.max_risk_points:
                    active_trade = SimulatedTrade(
                        trade_id=trade_id_seq,
                        direction=sig.direction,
                        entry_time=current_time,
                        entry_price=fill_price,
                        initial_sl=sl,
                        current_sl=sl,
                        target_price=tp,
                        be_trigger_price=be,
                        risk_points=round(risk, 4),
                        contracts=self.strat_cfg.fixed_contracts,
                    )
                    trade_id_seq += 1

            # 2. Manage Active Trade during bar i (High, Low, Close)
            if active_trade is not None:
                # Time exit 16:00
                if current_time.hour >= 16:
                    self._close_trade(active_trade, current_time, bar_open, "TIME_EXIT_1600")
                    trades.append(active_trade)
                    active_trade = None
                    continue

                trade_closed = self._process_bar_for_active_trade(
                    active_trade=active_trade,
                    timestamp=current_time,
                    bar_open=bar_open,
                    bar_high=bar_high,
                    bar_low=bar_low,
                    bar_close=bar_close,
                )
                if trade_closed:
                    trades.append(active_trade)
                    active_trade = None

        # Close any lingering trade at the end of the dataset
        if active_trade is not None:
            self._close_trade(
                trade=active_trade,
                exit_time=timestamps[-1],
                exit_price=closes[-1],
                reason="END_OF_DATA",
            )
            trades.append(active_trade)

        return trades

    def _process_bar_for_active_trade(
        self,
        active_trade: SimulatedTrade,
        timestamp: pd.Timestamp,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
    ) -> bool:
        """
        Evaluates intrabar price progression for active trade using causal OHLC path modeling.
        If Close >= Open: Open -> Low -> High -> Close.
        If Close < Open:  Open -> High -> Low -> Close.
        """
        target_label = f"TARGET_{int(self.strat_cfg.target_r_multiple)}R"
        buffer_pts = self.calc.ticks_to_points(self.strat_cfg.be_buffer_ticks)
        exit_slippage = self.calc.ticks_to_points(self.exec_cfg.slippage_ticks)

        if active_trade.direction == 1:
            # LONG TRADE
            is_green_bar = bar_close >= bar_open

            if is_green_bar:
                # Path: Open -> Low -> High -> Close
                # 1. Check Low against current stop loss BEFORE evaluating High
                if bar_low <= active_trade.current_sl:
                    exit_price = self.calc.round_to_tick(active_trade.current_sl - exit_slippage)
                    reason = "BREAKEVEN" if active_trade.be_activated else "STOP_LOSS"
                    self._close_trade(active_trade, timestamp, exit_price, reason)
                    return True

                # 2. Check High for TP and BE trigger
                if bar_high >= active_trade.target_price:
                    self._close_trade(active_trade, timestamp, active_trade.target_price, target_label)
                    return True

                if not active_trade.be_activated and bar_high >= active_trade.be_trigger_price:
                    active_trade.be_activated = True
                    active_trade.be_activated_time = timestamp
                    new_sl = self.calc.round_to_tick(active_trade.entry_price + buffer_pts)
                    # Check if price pulled back from High to Close below new BE stop
                    if bar_close <= new_sl:
                        exit_price = self.calc.round_to_tick(new_sl - exit_slippage)
                        self._close_trade(active_trade, timestamp, exit_price, "BREAKEVEN")
                        return True
                    active_trade.current_sl = max(active_trade.current_sl, new_sl)

            else:
                # Path: Open -> High -> Low -> Close (Red bar)
                # 1. Check High for TP or BE trigger first
                if bar_high >= active_trade.target_price:
                    self._close_trade(active_trade, timestamp, active_trade.target_price, target_label)
                    return True

                if not active_trade.be_activated and bar_high >= active_trade.be_trigger_price:
                    active_trade.be_activated = True
                    active_trade.be_activated_time = timestamp
                    new_sl = self.calc.round_to_tick(active_trade.entry_price + buffer_pts)
                    active_trade.current_sl = max(active_trade.current_sl, new_sl)

                # 2. Check Low against current SL (which may now be BE)
                if bar_low <= active_trade.current_sl:
                    exit_price = self.calc.round_to_tick(active_trade.current_sl - exit_slippage)
                    reason = "BREAKEVEN" if active_trade.be_activated else "STOP_LOSS"
                    self._close_trade(active_trade, timestamp, exit_price, reason)
                    return True

        elif active_trade.direction == -1:
            # SHORT TRADE
            is_red_bar = bar_close <= bar_open

            if is_red_bar:
                # Path: Open -> High -> Low -> Close (Favorable for short)
                # 1. Check High against current SL first
                if bar_high >= active_trade.current_sl:
                    exit_price = self.calc.round_to_tick(active_trade.current_sl + exit_slippage)
                    reason = "BREAKEVEN" if active_trade.be_activated else "STOP_LOSS"
                    self._close_trade(active_trade, timestamp, exit_price, reason)
                    return True

                # 2. Check Low for TP and BE trigger
                if bar_low <= active_trade.target_price:
                    self._close_trade(active_trade, timestamp, active_trade.target_price, target_label)
                    return True

                if not active_trade.be_activated and bar_low <= active_trade.be_trigger_price:
                    active_trade.be_activated = True
                    active_trade.be_activated_time = timestamp
                    new_sl = self.calc.round_to_tick(active_trade.entry_price - buffer_pts)
                    if bar_close >= new_sl:
                        exit_price = self.calc.round_to_tick(new_sl + exit_slippage)
                        self._close_trade(active_trade, timestamp, exit_price, "BREAKEVEN")
                        return True
                    active_trade.current_sl = min(active_trade.current_sl, new_sl)

            else:
                # Path: Open -> Low -> High -> Close (Green bar against short)
                # 1. Check Low for TP or BE trigger first
                if bar_low <= active_trade.target_price:
                    self._close_trade(active_trade, timestamp, active_trade.target_price, target_label)
                    return True

                if not active_trade.be_activated and bar_low <= active_trade.be_trigger_price:
                    active_trade.be_activated = True
                    active_trade.be_activated_time = timestamp
                    new_sl = self.calc.round_to_tick(active_trade.entry_price - buffer_pts)
                    active_trade.current_sl = min(active_trade.current_sl, new_sl)

                # 2. Check High against current SL
                if bar_high >= active_trade.current_sl:
                    exit_price = self.calc.round_to_tick(active_trade.current_sl + exit_slippage)
                    reason = "BREAKEVEN" if active_trade.be_activated else "STOP_LOSS"
                    self._close_trade(active_trade, timestamp, exit_price, reason)
                    return True

        return False

    def _close_trade(
        self,
        trade: SimulatedTrade,
        exit_time: pd.Timestamp,
        exit_price: float,
        reason: str,
    ) -> None:
        """
        Finalizes trade accounting, gross/net P&L, fees, and realized R.
        """
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason

        pnl_stats = self.calc.calculate_trade_pnl(
            direction=trade.direction,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            contracts=trade.contracts,
            include_costs=True,
        )

        trade.gross_points = pnl_stats["gross_points"]
        trade.gross_pnl = pnl_stats["gross_pnl"]
        trade.fees = pnl_stats["commission_and_fees"]
        trade.slippage = pnl_stats["slippage_cost"]
        trade.net_pnl = pnl_stats["net_pnl"]

        if trade.risk_points > 0:
            # Realized R based on gross points vs initial risk points
            trade.realized_r = round(trade.gross_points / trade.risk_points, 2)
        else:
            trade.realized_r = 0.0
