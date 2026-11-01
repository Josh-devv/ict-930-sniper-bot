"""
backtesting.py Framework Adapter for NQ Futures.
Enables running native backtesting.py Backtest instances, parameter optimization,
and interactive Bokeh HTML visualizations while incorporating NQ price action rules.
"""

from backtesting import Strategy
import pandas as pd
import numpy as np
from strategy.rules import ICTStrategyLogic
from config import StrategyConfig, SessionConfig, NQContractSpec


class BacktestingPyNQStrategy(Strategy):
    """
    Subclasses backtesting.py Strategy.
    Translates institutional NQ signals into native orders.
    """

    # Strategy Parameters exposed for optimization
    swing_lookback = 5
    fvg_min_size = 1.0
    target_r = 4.0
    breakeven_pct = 0.25
    use_session_filter = True

    def init(self):
        strat_cfg = StrategyConfig(
            target_r_multiple=self.target_r,
            breakeven_trigger_pct=self.breakeven_pct,
            fvg_min_size_points=self.fvg_min_size,
            mss_swing_lookback=self.swing_lookback,
            use_session_filter=self.use_session_filter,
        )
        self.logic = ICTStrategyLogic(strategy_config=strat_cfg)
        # Precompute signal flags on whole dataframe
        df = self.data.df.copy()
        signals = self.logic.generate_signals(df)

        signal_directions = np.zeros(len(df))
        entry_prices = np.zeros(len(df))
        stop_losses = np.zeros(len(df))
        take_profits = np.zeros(len(df))
        be_triggers = np.zeros(len(df))

        for sig in signals:
            idx = sig.index
            if 0 <= idx < len(df):
                signal_directions[idx] = sig.direction
                entry_prices[idx] = sig.entry_price
                stop_losses[idx] = sig.stop_loss
                take_profits[idx] = sig.target_price
                be_triggers[idx] = sig.breakeven_trigger_price

        self.sig_dir = self.I(lambda: signal_directions, name="Signal_Direction")
        self.sig_entry = self.I(lambda: entry_prices, name="Signal_Entry")
        self.sig_sl = self.I(lambda: stop_losses, name="Signal_SL")
        self.sig_tp = self.I(lambda: take_profits, name="Signal_TP")
        self.sig_be = self.I(lambda: be_triggers, name="Signal_BE")

        self.be_active = False

    def next(self):
        current_idx = len(self.data.Close) - 1
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]

        # 1. Check Breakeven Progression for Open Position
        if self.position:
            if self.position.is_long:
                if not self.be_active and current_high >= self.active_be_trigger:
                    self.be_active = True
                    for trade in self.trades:
                        trade.sl = max(trade.sl or 0, trade.entry_price)
            elif self.position.is_short:
                if not self.be_active and current_low <= self.active_be_trigger:
                    self.be_active = True
                    for trade in self.trades:
                        trade.sl = min(trade.sl or 999999, trade.entry_price)
            return

        # 2. Check for New Entry Signal
        direction = self.sig_dir[-1]
        if direction != 0:
            limit = self.sig_entry[-1]
            sl = self.sig_sl[-1]
            tp = self.sig_tp[-1]
            self.active_be_trigger = self.sig_be[-1]
            self.be_active = False

            if direction == 1:
                if sl < limit < tp and current_close > sl:
                    self.buy(limit=limit, sl=sl, tp=tp)
            elif direction == -1:
                if sl > limit > tp and current_close < sl:
                    self.sell(limit=limit, sl=sl, tp=tp)
