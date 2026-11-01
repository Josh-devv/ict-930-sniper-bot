"""
Unit tests for quantitative metrics and performance analytics.
"""

import pytest
import pandas as pd
from execution.simulator import SimulatedTrade
from analytics.metrics import PerformanceAnalytics


def test_performance_analytics_calculations():
    t1 = SimulatedTrade(
        trade_id=1,
        direction=1,
        entry_time=pd.Timestamp("2024-01-08 09:31:00"),
        exit_time=pd.Timestamp("2024-01-08 09:45:00"),
        entry_price=18000.0,
        exit_price=18030.0,
        risk_points=10.0,
        gross_points=30.0,
        gross_pnl=600.0,
        fees=5.20,
        slippage=10.0,
        net_pnl=584.80,
        realized_r=3.0,
        exit_reason="TARGET_3R",
    )
    t2 = SimulatedTrade(
        trade_id=2,
        direction=-1,
        entry_time=pd.Timestamp("2024-01-08 10:15:00"),
        exit_time=pd.Timestamp("2024-01-08 10:25:00"),
        entry_price=18020.0,
        exit_price=18030.0,
        risk_points=10.0,
        gross_points=-10.0,
        gross_pnl=-200.0,
        fees=5.20,
        slippage=10.0,
        net_pnl=-215.20,
        realized_r=-1.0,
        exit_reason="STOP_LOSS",
    )

    analytics = PerformanceAnalytics([t1, t2], initial_capital=100_000.0)
    stats = analytics.compute_summary_statistics()

    assert stats["total_trades"] == 2
    assert stats["win_count"] == 1
    assert stats["loss_count"] == 1
    assert stats["win_rate_pct"] == 50.0
    assert stats["net_pnl"] == round(584.80 - 215.20, 2)
    assert stats["average_r"] == 1.0  # (3.0 + (-1.0)) / 2 = 1.0
    assert stats["expectancy_r"] > 0
    assert stats["profit_factor"] > 2.0
