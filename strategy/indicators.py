"""
Institutional Price Action & Market Structure Engine.
Calculates Swing Highs/Lows (strictly causal, zero lookahead),
Fair Value Gaps (FVGs), Market Structure Shifts (MSS),
Expansion Candles, and Session Time Windows.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    is_high: bool  # True for Swing High, False for Swing Low
    confirmed_index: int  # Exact index when the swing was confirmed (no lookahead)


@dataclass
class FVG:
    index: int
    timestamp: pd.Timestamp
    top: float
    bottom: float
    is_bullish: bool  # True for Bullish FVG, False for Bearish FVG
    size: float
    mitigated: bool = False
    mitigated_index: Optional[int] = None


@dataclass
class Block:
    block_type: str         # "OB", "BREAKER", "REJECTION"
    direction: int          # 1 for Bullish (Long), -1 for Bearish (Short)
    formation_time: pd.Timestamp
    formation_idx: int      # Global bar index of formation
    near_edge: float        # Entry price (Top for Bullish, Bottom for Bearish)
    far_edge: float         # Stop anchor (Bottom for Bullish, Top for Bearish)
    candle_idx: int


class StructureEngine:
    """
    Computes market structure indicators on 1-minute OHLCV data.
    Strictly causal: bar t can only use data from <= t.
    """

    def __init__(
        self,
        swing_lookback: int = 3,
        fvg_min_size_points: float = 1.0,
        expansion_multiplier: float = 1.5,
        atr_period: int = 14,
        min_displacement_points: float = 3.0,
    ):
        self.swing_lookback = swing_lookback
        self.fvg_min_size_points = fvg_min_size_points
        self.expansion_multiplier = expansion_multiplier
        self.atr_period = atr_period
        self.min_displacement_points = min_displacement_points

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        True Range & Exponential ATR calculation.
        """
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=self.atr_period, adjust=False).mean()
        return atr

    def detect_confirmed_swings(self, df: pd.DataFrame) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Identifies swing highs and swing lows without lookahead.
        """
        swing_highs: List[SwingPoint] = []
        swing_lows: List[SwingPoint] = []
        n = self.swing_lookback
        total_bars = len(df)

        highs = df["High"].values
        lows = df["Low"].values
        timestamps = df.index

        for i in range(n, total_bars - n):
            is_sh = True
            current_h = highs[i]
            for j in range(1, n + 1):
                if highs[i - j] >= current_h or highs[i + j] >= current_h:
                    is_sh = False
                    break
            if is_sh:
                swing_highs.append(
                    SwingPoint(
                        index=i,
                        timestamp=timestamps[i],
                        price=current_h,
                        is_high=True,
                        confirmed_index=i + n,
                    )
                )

            is_sl = True
            current_l = lows[i]
            for j in range(1, n + 1):
                if lows[i - j] <= current_l or lows[i + j] <= current_l:
                    is_sl = False
                    break
            if is_sl:
                swing_lows.append(
                    SwingPoint(
                        index=i,
                        timestamp=timestamps[i],
                        price=current_l,
                        is_high=False,
                        confirmed_index=i + n,
                    )
                )

        return swing_highs, swing_lows

    def detect_order_blocks(self, df_window: pd.DataFrame, global_indices: List[int]) -> List[Block]:
        """
        Detects Order Blocks (OB): last opposite-color candle before a significant displacement.

        Displacement criteria (both must hold):
          1. Displacement body >= expansion_multiplier * average body in the window.
          2. Displacement body >= min_displacement_points (absolute floor).

        Only ONE OB (the last qualified opposite-color candle) is kept per
        displacement event to avoid duplicating overlapping blocks.
        """
        blocks: List[Block] = []
        if len(df_window) < 3:
            return blocks

        opens = df_window["Open"].values
        highs = df_window["High"].values
        lows = df_window["Low"].values
        closes = df_window["Close"].values
        timestamps = df_window.index

        body_sizes = np.abs(closes - opens)
        avg_body = np.mean(body_sizes) if len(body_sizes) > 0 else 2.0
        # Adaptive floor: at least expansion_multiplier * avg, AND absolute minimum
        rel_threshold = avg_body * self.expansion_multiplier

        for i in range(1, len(df_window)):
            disp_open = opens[i]
            disp_close = closes[i]
            disp_body = abs(disp_close - disp_open)

            # Must exceed BOTH relative and absolute thresholds
            if disp_body < rel_threshold or disp_body < self.min_displacement_points:
                continue

            # Bullish Displacement: find the last down (or doji) candle before this bar
            if disp_close > disp_open:
                # Walk back to find the LAST bearish/doji candle before i
                ob_i = None
                for k in range(i - 1, -1, -1):
                    if closes[k] <= opens[k]:  # down or doji
                        ob_i = k
                        break
                if ob_i is not None:
                    g_idx = global_indices[i]
                    c_idx = global_indices[ob_i]
                    blocks.append(
                        Block(
                            block_type="OB",
                            direction=1,  # Bullish
                            formation_time=timestamps[i],
                            formation_idx=g_idx,
                            near_edge=highs[ob_i],   # Top of OB candle (first touched from above)
                            far_edge=lows[ob_i],     # Bottom (stop anchor)
                            candle_idx=c_idx,
                        )
                    )

            # Bearish Displacement: find the last up (or doji) candle before this bar
            elif disp_close < disp_open:
                ob_i = None
                for k in range(i - 1, -1, -1):
                    if closes[k] >= opens[k]:  # up or doji
                        ob_i = k
                        break
                if ob_i is not None:
                    g_idx = global_indices[i]
                    c_idx = global_indices[ob_i]
                    blocks.append(
                        Block(
                            block_type="OB",
                            direction=-1,  # Bearish
                            formation_time=timestamps[i],
                            formation_idx=g_idx,
                            near_edge=lows[ob_i],    # Bottom of OB candle (first touched from below)
                            far_edge=highs[ob_i],    # Top (stop anchor)
                            candle_idx=c_idx,
                        )
                    )

        return blocks

    def detect_breaker_blocks(self, df_window: pd.DataFrame, global_indices: List[int]) -> List[Block]:
        """
        Detects Breaker Blocks:
        - Bullish Breaker: Low -> High (H1) -> Lower Low (sweep) -> Displacement breaking above H1.
          The up candle at/preceding H1 that was broken flips into support.
        - Bearish Breaker: High -> Low (L1) -> Higher High (sweep) -> Displacement breaking below L1.
          The down candle at/preceding L1 that was broken flips into resistance.
        """
        blocks: List[Block] = []
        if len(df_window) < 6:
            return blocks

        opens = df_window["Open"].values
        highs = df_window["High"].values
        lows = df_window["Low"].values
        closes = df_window["Close"].values
        timestamps = df_window.index

        # Find local peaks and valleys within the window
        for i in range(2, len(df_window) - 2):
            # Check Bearish Breaker sequence: High (H) at i-2, Low (L) at i-1, Higher High at i, break below L at i+1..
            # Or Bearish Breaker: L1 at i-2, H1 at i-1, L2 < L1 (sweep) at i, Break above H1 at j > i
            # Let's check Bearish Breaker (for Short):
            # 1. H1 at i-2, L1 at i-1 (down candle), H2 > H1 (liquidity sweep of H1) at i.
            # 2. Then a subsequent bar j > i drops and closes below L1.
            # The down-close candle around L1 that was violated becomes the Bearish Breaker!
            if i >= 2:
                # Find swing high H1
                h1 = highs[i - 2]
                l1 = lows[i - 1]
                h2 = highs[i]
                if h2 > h1:  # Liquidity sweep of high
                    # Check if subsequent bar breaks below l1
                    for j in range(i + 1, min(i + 5, len(df_window))):
                        if closes[j] < l1:  # Broken structure
                            # Candle at i-1 is the breaker block
                            blocks.append(
                                Block(
                                    block_type="BREAKER",
                                    direction=-1,  # Bearish Breaker for Short
                                    formation_time=timestamps[j],
                                    formation_idx=global_indices[j],
                                    near_edge=lows[i - 1],
                                    far_edge=highs[i - 1],
                                    candle_idx=global_indices[i - 1],
                                )
                            )
                            break

            # Bullish Breaker (for Long):
            # 1. L1 at i-2, H1 at i-1 (up candle), L2 < L1 (liquidity sweep of L1) at i.
            # 2. Then a subsequent bar j > i rallies and closes above H1.
            # The up-close candle around H1 that was violated becomes the Bullish Breaker!
            if i >= 2:
                l1 = lows[i - 2]
                h1 = highs[i - 1]
                l2 = lows[i]
                if l2 < l1:  # Liquidity sweep of low
                    for j in range(i + 1, min(i + 5, len(df_window))):
                        if closes[j] > h1:  # Broken structure
                            blocks.append(
                                Block(
                                    block_type="BREAKER",
                                    direction=1,  # Bullish Breaker for Long
                                    formation_time=timestamps[j],
                                    formation_idx=global_indices[j],
                                    near_edge=highs[i - 1],
                                    far_edge=lows[i - 1],
                                    candle_idx=global_indices[i - 1],
                                )
                            )
                            break

        return blocks

    def detect_rejection_blocks(self, df_window: pd.DataFrame, global_indices: List[int]) -> List[Block]:
        """
        Detects Rejection Blocks at the lowest swing low (Bullish) or highest swing high (Bearish)
        in the marking window where a clear rejection wick exists.
        """
        blocks: List[Block] = []
        if len(df_window) < 3:
            return blocks

        opens = df_window["Open"].values
        highs = df_window["High"].values
        lows = df_window["Low"].values
        closes = df_window["Close"].values
        timestamps = df_window.index

        # Bullish Rejection Block at the lowest point of the window
        min_idx = np.argmin(lows)
        # Check if candle at min_idx has a lower wick: Low < min(Open, Close)
        body_bottom = min(opens[min_idx], closes[min_idx])
        if body_bottom > lows[min_idx] + 0.5:  # Valid lower wick >= 0.5 pt
            blocks.append(
                Block(
                    block_type="REJECTION",
                    direction=1,  # Bullish
                    formation_time=timestamps[min_idx],
                    formation_idx=global_indices[min_idx],
                    near_edge=body_bottom,
                    far_edge=lows[min_idx],
                    candle_idx=global_indices[min_idx],
                )
            )

        # Bearish Rejection Block at the highest point of the window
        max_idx = np.argmax(highs)
        body_top = max(opens[max_idx], closes[max_idx])
        if highs[max_idx] > body_top + 0.5:  # Valid upper wick >= 0.5 pt
            blocks.append(
                Block(
                    block_type="REJECTION",
                    direction=-1,  # Bearish
                    formation_time=timestamps[max_idx],
                    formation_idx=global_indices[max_idx],
                    near_edge=body_top,
                    far_edge=highs[max_idx],
                    candle_idx=global_indices[max_idx],
                )
            )

        return blocks

    def detect_fvgs(self, df: pd.DataFrame) -> List[FVG]:
        """
        Detects 3-candle Fair Value Gaps (FVGs).
        """
        fvgs: List[FVG] = []
        highs = df["High"].values
        lows = df["Low"].values
        timestamps = df.index
        total_bars = len(df)

        for i in range(2, total_bars):
            bar1_high = highs[i - 2]
            bar3_low = lows[i]
            if bar3_low > bar1_high:
                gap_size = bar3_low - bar1_high
                if gap_size >= self.fvg_min_size_points:
                    fvgs.append(
                        FVG(
                            index=i,
                            timestamp=timestamps[i],
                            top=bar3_low,
                            bottom=bar1_high,
                            is_bullish=True,
                            size=gap_size,
                        )
                    )

            bar1_low = lows[i - 2]
            bar3_high = highs[i]
            if bar1_low > bar3_high:
                gap_size = bar1_low - bar3_high
                if gap_size >= self.fvg_min_size_points:
                    fvgs.append(
                        FVG(
                            index=i,
                            timestamp=timestamps[i],
                            top=bar1_low,
                            bottom=bar3_high,
                            is_bullish=False,
                            size=gap_size,
                        )
                    )

        return fvgs

    def detect_expansion_candles(self, df: pd.DataFrame) -> pd.Series:
        body = (df["Close"] - df["Open"]).abs()
        avg_body = body.rolling(window=20, min_periods=5).mean()
        return body > (avg_body * self.expansion_multiplier)

    def is_in_session(
        self,
        dt: pd.Timestamp,
        start_time_str: str = "09:30",
        end_time_str: str = "11:30",
    ) -> bool:
        t = dt.time()
        start_h, start_m = map(int, start_time_str.split(":"))
        end_h, end_m = map(int, end_time_str.split(":"))

        current_minutes = t.hour * 60 + t.minute
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m

        return start_minutes <= current_minutes <= end_minutes
