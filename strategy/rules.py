"""
10:00 AM Open Bias & Price Action Block Strategy for NQ / MNQ Futures.

Specifications:
- Mark blocks: 09:30–09:59 NY time.
- Bias: 10:00 Open.
- Long setup: Price trades below 10:00 Open, sweeps liquidity, and enters into a block to go long.
- Short setup: Price trades above 10:00 Open, sweeps liquidity, and enters into a block to go short.
- Block types:
    1. Order Block (OB): Last opposite-color candle before displacement.
    2. Breaker Block: Failed order block resulting from a liquidity run.
    3. Rejection Block: Untouched rejection wick (only if NO OB or Breaker is closer to 10:00 open).
- Selection hierarchy:
    Valid rejection block ONLY if closer than any OB/Breaker.
    OB > Breaker.
    Tie-breaking: Earliest formed, then closest to 10:00 open.
- Untouched rule: Block must not be touched from formation until entry.
- Invalidation rule: If price crosses the far edge after 10:00 before entry, the block is dead.
- Entry: First touch of the block (near edge).
- Long: Buy near edge; stop below far edge + 1 tick; target 4R.
- Short: Sell near edge; stop above far edge + 1 tick; target 4R.
- One trade per day.
- Time exit: 16:00.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np


from config import StrategyConfig, SessionConfig, NQContractSpec
from strategy.indicators import StructureEngine, Block, SwingPoint
from data.contract import FuturesCalculator


@dataclass
class TradeSignal:
    index: int
    timestamp: pd.Timestamp
    direction: int            # 1 for Long, -1 for Short
    entry_price: float
    stop_loss: float
    target_price: float       # 4R target
    breakeven_trigger_price: float  # 25% of the distance to target (1R)
    risk_points: float        # Stop distance in points (1R)
    reason: str
    block_type: str = ""


class ICTStrategyLogic:
    """
    Evaluates market conditions daily and executes the 10:00 AM Open Bias Block Strategy.
    """

    def __init__(
        self,
        strategy_config: Optional[StrategyConfig] = None,
        session_config: Optional[SessionConfig] = None,
        contract_spec: Optional[NQContractSpec] = None,
    ):
        self.strat_cfg = strategy_config or StrategyConfig()
        self.sess_cfg = session_config or SessionConfig()
        self.contract = contract_spec or NQContractSpec()
        self.calc = FuturesCalculator(contract=self.contract)
        self.engine = StructureEngine(
            swing_lookback=self.strat_cfg.mss_swing_lookback,
            fvg_min_size_points=self.strat_cfg.fvg_min_size_points,
            expansion_multiplier=self.strat_cfg.expansion_body_multiplier,
            atr_period=self.strat_cfg.expansion_atr_period,
            min_displacement_points=self.strat_cfg.min_displacement_points,
        )

    def calculate_trade_levels(
        self,
        direction: int,
        entry_price: float,
        far_edge: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates: (stop_loss, target_price, breakeven_trigger_price, risk_points).
        - Long: Stop below far edge + 1 tick (0.25 pts), Target 4R.
        - Short: Stop above far edge + 1 tick (0.25 pts), Target 4R.
        """
        tick = self.contract.tick_size
        entry_price = self.calc.round_to_tick(entry_price)

        if direction == 1:
            # Long: stop below far edge + 1 tick
            stop_loss = self.calc.round_to_tick(far_edge - tick)
            risk_points = entry_price - stop_loss
            if risk_points <= 0 or risk_points < self.strat_cfg.min_risk_points or risk_points > self.strat_cfg.max_risk_points:
                return None
            target_price = self.calc.round_to_tick(entry_price + (risk_points * self.strat_cfg.target_r_multiple))
            # 25% progress toward 4R target = entry + 0.25 * 4R = entry + 1R
            be_trigger = self.calc.round_to_tick(entry_price + (target_price - entry_price) * self.strat_cfg.breakeven_trigger_pct)

        elif direction == -1:
            # Short: stop above far edge + 1 tick
            stop_loss = self.calc.round_to_tick(far_edge + tick)
            risk_points = stop_loss - entry_price
            if risk_points <= 0 or risk_points < self.strat_cfg.min_risk_points or risk_points > self.strat_cfg.max_risk_points:
                return None
            target_price = self.calc.round_to_tick(entry_price - (risk_points * self.strat_cfg.target_r_multiple))
            be_trigger = self.calc.round_to_tick(entry_price - (entry_price - target_price) * self.strat_cfg.breakeven_trigger_pct)

        else:
            return None

        return (stop_loss, target_price, be_trigger, round(risk_points, 4))

    def _select_best_block(self, blocks: List[Block], open_10am: float) -> Optional[Block]:
        """
        Applies user selection hierarchy:
        1. Valid rejection block ONLY if there is NO order block or breaker block closer to 10:00 open.
        2. Priority: OB > Breaker.
        3. Tie-breaking: Earliest formed, then closest to 10:00 open.
        """
        if not blocks:
            return None

        obs = [b for b in blocks if b.block_type == "OB"]
        breakers = [b for b in blocks if b.block_type == "BREAKER"]
        rejections = [b for b in blocks if b.block_type == "REJECTION"]

        # Check if rejection block is valid (only if closer than any OB or breaker)
        valid_rejections = []
        min_other_dist = float("inf")
        for b in obs + breakers:
            d = abs(b.near_edge - open_10am)
            if d < min_other_dist:
                min_other_dist = d

        for r in rejections:
            d_rej = abs(r.near_edge - open_10am)
            if d_rej < min_other_dist:
                valid_rejections.append(r)

        # Priority 1: Order Blocks
        if obs:
            obs.sort(key=lambda b: (b.formation_idx, abs(b.near_edge - open_10am)))
            return obs[0]

        # Priority 2: Breaker Blocks
        if breakers:
            breakers.sort(key=lambda b: (b.formation_idx, abs(b.near_edge - open_10am)))
            return breakers[0]

        # Priority 3: Rejection Blocks (only if eligible)
        if valid_rejections:
            valid_rejections.sort(key=lambda b: (b.formation_idx, abs(b.near_edge - open_10am)))
            return valid_rejections[0]

        return None

    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        """
        Processes historical bars day-by-day and outputs trade signals adhering strictly
        to the 10:00 AM Open Bias and block rules.
        """
        signals: List[TradeSignal] = []

        # Group data by trading date (America/New_York)
        dates = pd.Series(df.index.date, index=df.index)
        unique_days = dates.unique()

        for day in unique_days:
            day_mask = (dates == day)
            day_indices = np.where(day_mask)[0]
            if len(day_indices) == 0:
                continue

            df_day = df.iloc[day_indices]
            day_timestamps = df_day.index

            # Find 09:30 to 09:59 window (Mark blocks window)
            mark_mask = (day_timestamps.time >= pd.to_datetime(self.strat_cfg.mark_blocks_start).time()) & \
                        (day_timestamps.time <= pd.to_datetime(self.strat_cfg.mark_blocks_end).time())

            mark_indices = [day_indices[j] for j, m in enumerate(mark_mask) if m]
            if len(mark_indices) < 5:
                continue

            df_mark = df.iloc[mark_indices]

            # Detect all candidate blocks in the 09:30-09:59 window
            raw_obs = self.engine.detect_order_blocks(df_mark, mark_indices)
            raw_breakers = self.engine.detect_breaker_blocks(df_mark, mark_indices)
            raw_rejections = self.engine.detect_rejection_blocks(df_mark, mark_indices)

            all_raw_blocks = raw_obs + raw_breakers + raw_rejections

            # Untouched check: block must NOT have been touched between formation
            # and the end of the 09:59 mark window.
            # Entry is the FIRST touch of the block (near_edge).
            valid_pre_10am_blocks: List[Block] = []
            for block in all_raw_blocks:
                is_touched = False
                for g_idx in mark_indices:
                    if g_idx > block.formation_idx:
                        b_high = df["High"].iloc[g_idx]
                        b_low = df["Low"].iloc[g_idx]

                        if block.direction == 1:
                            # Bullish block: touched if price reaches near_edge
                            if b_low <= block.near_edge:
                                is_touched = True
                                break
                        elif block.direction == -1:
                            # Bearish block: touched if price reaches near_edge
                            if b_high >= block.near_edge:
                                is_touched = True
                                break

                if not is_touched:
                    valid_pre_10am_blocks.append(block)

            # Locate 10:00 candle and restrict entries to entry_window_end (10:00 to 10:30 macro)
            entry_end_time = getattr(self.strat_cfg, "entry_window_end", "10:30")
            post_10am_mask = (day_timestamps.time >= pd.to_datetime(self.strat_cfg.bias_time).time()) & \
                             (day_timestamps.time <= pd.to_datetime(entry_end_time).time())

            post_10am_indices = [day_indices[j] for j, m in enumerate(post_10am_mask) if m]
            if len(post_10am_indices) == 0:
                continue

            open_10am_idx = post_10am_indices[0]
            open_10am = df["Open"].iloc[open_10am_idx]

            # Bias: 10:00 Open
            # Long setup: Price trades below 10:00 open sweeps liquidity and enters into a block to long
            # (Candidate bullish block must be at or below 10:00 open)
            # Short setup: Price trades above 10:00 open sweeps liquidity and enters into a block to short
            # (Candidate bearish block must be at or above 10:00 open)
            bullish_candidates = [b for b in valid_pre_10am_blocks if b.direction == 1 and b.near_edge <= open_10am]
            bearish_candidates = [b for b in valid_pre_10am_blocks if b.direction == -1 and b.near_edge >= open_10am]

            # Select Best Long block and Best Short block according to selection rules
            best_long_block = self._select_best_block(bullish_candidates, open_10am)
            best_short_block = self._select_best_block(bearish_candidates, open_10am)

            # State tracking after 10:00
            traded_today = False
            long_block_dead = False
            short_block_dead = False

            # Scan bars from 10:00 onwards
            for g_idx in post_10am_indices:
                if traded_today:
                    break

                b_dt = df.index[g_idx]
                b_open = df["Open"].iloc[g_idx]
                b_high = df["High"].iloc[g_idx]
                b_low = df["Low"].iloc[g_idx]
                b_close = df["Close"].iloc[g_idx]

                # Check Invalidation Rule before entry:
                # "if price crosses the far edge after 10:00 before entry, block is dead."
                if best_long_block and not long_block_dead:
                    if b_open < best_long_block.far_edge:
                        long_block_dead = True
                    elif b_low <= best_long_block.near_edge:
                        if b_open >= best_long_block.near_edge:
                            # Valid limit / touch entry at near_edge
                            levels = self.calculate_trade_levels(
                                direction=1,
                                entry_price=best_long_block.near_edge,
                                far_edge=best_long_block.far_edge,
                            )
                            if levels:
                                sl, tp, be, risk = levels
                                signals.append(
                                    TradeSignal(
                                        index=g_idx,
                                        timestamp=b_dt,
                                        direction=1,
                                        entry_price=best_long_block.near_edge,
                                        stop_loss=sl,
                                        target_price=tp,
                                        breakeven_trigger_price=be,
                                        risk_points=risk,
                                        reason=f"Long touch {best_long_block.block_type} [{best_long_block.near_edge:.2f}] below 10am open ({open_10am:.2f})",
                                        block_type=best_long_block.block_type,
                                    )
                                )
                                traded_today = True
                                break
                        else:
                            if b_low < best_long_block.far_edge:
                                long_block_dead = True

                if best_short_block and not short_block_dead:
                    if b_open > best_short_block.far_edge:
                        short_block_dead = True
                    elif b_high >= best_short_block.near_edge:
                        if b_open <= best_short_block.near_edge:
                            levels = self.calculate_trade_levels(
                                direction=-1,
                                entry_price=best_short_block.near_edge,
                                far_edge=best_short_block.far_edge,
                            )
                            if levels:
                                sl, tp, be, risk = levels
                                signals.append(
                                    TradeSignal(
                                        index=g_idx,
                                        timestamp=b_dt,
                                        direction=-1,
                                        entry_price=best_short_block.near_edge,
                                        stop_loss=sl,
                                        target_price=tp,
                                        breakeven_trigger_price=be,
                                        risk_points=risk,
                                        reason=f"Short touch {best_short_block.block_type} [{best_short_block.near_edge:.2f}] above 10am open ({open_10am:.2f})",
                                        block_type=best_short_block.block_type,
                                    )
                                )
                                traded_today = True
                                break
                        else:
                            if b_high > best_short_block.far_edge:
                                short_block_dead = True

        return signals
