"""
Unit tests for market structure engine, swing detection causality, and FVG detection.
"""

import pytest
import pandas as pd
import numpy as np
from strategy.indicators import StructureEngine


def test_swing_causality_no_lookahead():
    """
    Verifies that a swing point is confirmed strictly at index i + lookback,
    meaning at index t < i + lookback, the swing does not exist.
    """
    engine = StructureEngine(swing_lookback=2)
    # 7 bars with a peak at bar 2
    highs = [10.0, 12.0, 20.0, 15.0, 11.0, 10.0, 9.0]
    lows = [8.0, 9.0, 10.0, 10.0, 8.0, 7.0, 6.0]
    dates = pd.date_range("2024-01-08 09:30", periods=7, freq="1min", tz="America/New_York")
    df = pd.DataFrame({"High": highs, "Low": lows, "Open": lows, "Close": highs}, index=dates)

    swing_highs, _ = engine.detect_confirmed_swings(df)
    assert len(swing_highs) == 1
    sh = swing_highs[0]
    assert sh.index == 2
    assert sh.price == 20.0
    # Confirmed strictly at index 2 + 2 = 4
    assert sh.confirmed_index == 4


def test_fvg_detection():
    engine = StructureEngine(fvg_min_size_points=2.0)
    # Create 3-bar Bullish FVG:
    # Bar 0: High = 100
    # Bar 1: Huge green bar
    # Bar 2: Low = 105
    # Gap: 105 - 100 = 5 points
    highs = [100.0, 110.0, 108.0]
    lows = [95.0, 98.0, 105.0]
    dates = pd.date_range("2024-01-08 09:30", periods=3, freq="1min", tz="America/New_York")
    df = pd.DataFrame({"High": highs, "Low": lows, "Open": lows, "Close": highs}, index=dates)

    fvgs = engine.detect_fvgs(df)
    assert len(fvgs) == 1
    f = fvgs[0]
    assert f.is_bullish is True
    assert f.bottom == 100.0
    assert f.top == 105.0
    assert f.size == 5.0
