"""
Unit tests for the 10:00 AM Open Bias & Price Action Block Strategy rules.
"""

import pytest
import pandas as pd
from config import StrategyConfig, SessionConfig, NQContractSpec
from strategy.rules import ICTStrategyLogic
from strategy.indicators import Block


def test_block_selection_hierarchy():
    """
    Validates user hierarchy:
    - Rejection block only if closer to 10:00 open than any OB or Breaker.
    - OB > Breaker.
    - Earliest formed, then closest to 10:00 open.
    """
    logic = ICTStrategyLogic()
    open_10am = 18000.0

    # Case 1: OB at 17980 (dist 20), Breaker at 17985 (dist 15).
    # Since OB has higher priority than Breaker, OB is chosen.
    ob = Block(block_type="OB", direction=1, formation_time=pd.Timestamp("2024-01-08 09:35"), formation_idx=5, near_edge=17980.0, far_edge=17970.0, candle_idx=5)
    breaker = Block(block_type="BREAKER", direction=1, formation_time=pd.Timestamp("2024-01-08 09:40"), formation_idx=10, near_edge=17985.0, far_edge=17975.0, candle_idx=10)

    chosen = logic._select_best_block([ob, breaker], open_10am)
    assert chosen is not None
    assert chosen.block_type == "OB"
    assert chosen.near_edge == 17980.0

    # Case 2: Rejection block at 17990 (dist 10) vs OB at 17970 (dist 30).
    # Rejection block IS closer to 10am open than OB (dist 10 < 30).
    rej = Block(block_type="REJECTION", direction=1, formation_time=pd.Timestamp("2024-01-08 09:32"), formation_idx=2, near_edge=17990.0, far_edge=17982.0, candle_idx=2)
    # But rule says: OB > breaker. Let's verify rejection rule:
    # "an untouched Rejection block only if there are not orderblock or breaker block closer"
    # If rejection block is closer, does it qualify?
    # When rej is at 17990 (dist 10) and OB is at 17970 (dist 30):
    # There is NO OB closer than rej!
    # However between OB and Breaker, OB > Breaker. If an OB exists, does OB take priority over Rejection?
    # "selection: valid rejection block,; OB > breaker; earliest formed, then closest to the 10:00 open."
    # If Rejection block is further than OB (e.g. dist 40 vs dist 20), Rejection block is completely eliminated!
    rej_far = Block(block_type="REJECTION", direction=1, formation_time=pd.Timestamp("2024-01-08 09:32"), formation_idx=2, near_edge=17950.0, far_edge=17945.0, candle_idx=2)
    chosen_far = logic._select_best_block([ob, rej_far], open_10am)
    assert chosen_far.block_type == "OB"


def test_4r_target_and_stop_levels():
    """
    Validates:
    - Long: stop below far edge + 1 tick (0.25 pts), target 4R.
    - Short: stop above far edge + 1 tick (0.25 pts), target 4R.
    """
    strat_cfg = StrategyConfig(target_r_multiple=4.0, breakeven_trigger_pct=0.25)
    logic = ICTStrategyLogic(strategy_config=strat_cfg)

    # Long setup: entry 18000.0, far edge 17990.0
    # Stop: 17990.0 - 0.25 = 17989.75
    # Risk: 18000.0 - 17989.75 = 10.25 pts
    # Target: 18000.0 + (10.25 * 4) = 18041.0
    # BE trigger: 18000.0 + 10.25 = 18010.25 (25% progress toward 4R)
    res_long = logic.calculate_trade_levels(direction=1, entry_price=18000.0, far_edge=17990.0)
    assert res_long is not None
    sl, tp, be, risk = res_long
    assert sl == 17989.75
    assert risk == 10.25
    assert tp == 18041.0
    assert be == 18010.25

    # Short setup: entry 18000.0, far edge 18010.0
    # Stop: 18010.0 + 0.25 = 18010.25
    # Risk: 18010.25 - 18000.0 = 10.25 pts
    # Target: 18000.0 - (10.25 * 4) = 17959.0
    res_short = logic.calculate_trade_levels(direction=-1, entry_price=18000.0, far_edge=18010.0)
    assert res_short is not None
    sl_s, tp_s, be_s, risk_s = res_short
    assert sl_s == 18010.25
    assert risk_s == 10.25
    assert tp_s == 17959.0
