"""
Unit tests for intrabar futures simulation, 3R target, and 25% breakeven rule.
"""

import pytest
import pandas as pd
from execution.simulator import NQIntrabarSimulator
from strategy.rules import TradeSignal
from config import StrategyConfig, ExecutionConfig, NQContractSpec


def test_3r_target_hit():
    """
    Test long trade reaching 3R target without triggering stop loss.
    """
    strat_cfg = StrategyConfig(target_r_multiple=3.0, breakeven_trigger_pct=0.25)
    # Zero slippage for deterministic testing
    exec_cfg = ExecutionConfig(slippage_ticks=0, commission_per_side=0.0, exchange_fee_per_side=0.0)
    sim = NQIntrabarSimulator(strategy_config=strat_cfg, execution_config=exec_cfg)

    # 4 bars:
    # Bar 0: Signal generated at close
    # Bar 1: Entry at Open (18000.0)
    # Bar 2: Moves toward target
    # Bar 3: Hits 3R Target (18030.0)
    dates = pd.date_range("2024-01-08 09:30", periods=4, freq="1min", tz="America/New_York")
    df = pd.DataFrame({
        "Open": [17990.0, 18000.0, 18015.0, 18025.0],
        "High": [18000.0, 18015.0, 18025.0, 18035.0],  # Bar 3 High reaches 18035 >= 18030 TP
        "Low": [17985.0, 17998.0, 18010.0, 18020.0],
        "Close": [18000.0, 18010.0, 18020.0, 18030.0],
    }, index=dates)

    # Signal at Bar 0: Entry 18000, SL 17990 (Risk = 10 pts). 3R Target = 18030. BE trigger = 18007.5
    signal = TradeSignal(
        index=0,
        timestamp=dates[0],
        direction=1,
        entry_price=18000.0,
        stop_loss=17990.0,
        target_price=18030.0,
        breakeven_trigger_price=18007.5,
        risk_points=10.0,
        reason="Test Long",
    )

    trades = sim.run(df, [signal])
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "TARGET_3R"
    assert t.entry_price == 18000.0
    assert t.exit_price == 18030.0
    assert t.gross_points == 30.0
    assert t.gross_pnl == 600.0  # 30 pts * $20 = $600
    assert t.realized_r == 3.0


def test_25_pct_breakeven_trigger():
    """
    Test long trade reaching 25% of target (triggering BE), then reversing down.
    The trade should exit at Breakeven (entry + buffer), NOT at the original stop loss!
    """
    strat_cfg = StrategyConfig(target_r_multiple=3.0, breakeven_trigger_pct=0.25, be_buffer_ticks=1)
    exec_cfg = ExecutionConfig(slippage_ticks=0, commission_per_side=0.0, exchange_fee_per_side=0.0)
    sim = NQIntrabarSimulator(strategy_config=strat_cfg, execution_config=exec_cfg)

    # Signal: Entry 18000, SL 17990 (Risk = 10). TP = 18030.
    # 25% progress toward 3R = 18000 + 0.25 * 30 = 18007.5
    # BE stop level = Entry + 1 tick (0.25) = 18000.25
    dates = pd.date_range("2024-01-08 09:30", periods=4, freq="1min", tz="America/New_York")
    df = pd.DataFrame({
        "Open": [17990.0, 18000.0, 18005.0, 18002.0],
        "High": [18000.0, 18009.0, 18006.0, 18003.0],  # Bar 1 High reaches 18009 >= 18007.5 (BE triggered!)
        "Low": [17985.0, 17998.0, 17999.0, 17995.0],   # Bar 2 Low drops to 17999 <= 18000.25 BE Stop
        "Close": [18000.0, 18004.0, 18000.0, 17996.0],
    }, index=dates)

    signal = TradeSignal(
        index=0,
        timestamp=dates[0],
        direction=1,
        entry_price=18000.0,
        stop_loss=17990.0,
        target_price=18030.0,
        breakeven_trigger_price=18007.5,
        risk_points=10.0,
        reason="Test BE",
    )

    trades = sim.run(df, [signal])
    assert len(trades) == 1
    t = trades[0]
    assert t.be_activated is True
    assert t.exit_reason == "BREAKEVEN"
    # Should exit at 18000.25, NOT at 17990.0!
    assert t.exit_price == 18000.25
    assert t.gross_points == 0.25
    assert t.realized_r >= 0.0
