"""
Unit tests for futures contract specifications, tick rounding, and cost calculations.
"""

import pytest
from config import NQContractSpec, ExecutionConfig
from data.contract import FuturesCalculator


def test_nq_contract_spec_defaults():
    spec = NQContractSpec()
    assert spec.symbol == "NQ"
    assert spec.tick_size == 0.25
    assert spec.point_value == 20.0
    assert spec.tick_value == 5.0
    assert spec.contract_multiplier == 20.0


def test_tick_rounding():
    calc = FuturesCalculator()
    assert calc.round_to_tick(17500.12) == 17500.00
    assert calc.round_to_tick(17500.13) == 17500.25
    assert calc.round_to_tick(17500.38) == 17500.50
    assert calc.round_to_tick(17500.87) == 17500.75


def test_points_to_dollars():
    calc = FuturesCalculator()
    # 10 points on 1 NQ contract = 10 * 20 = $200
    assert calc.points_to_dollars(10.0, contracts=1) == 200.0
    # 5.5 points on 2 NQ contracts = 5.5 * 20 * 2 = $220
    assert calc.points_to_dollars(5.5, contracts=2) == 220.0


def test_trade_pnl_with_costs():
    exec_cfg = ExecutionConfig(
        commission_per_side=2.25,
        exchange_fee_per_side=0.35,
        slippage_ticks=1,
    )
    calc = FuturesCalculator(execution=exec_cfg)
    # Long trade from 18000.00 to 18010.00 (+10 points)
    # Gross P&L: 10 * $20 = $200.00
    # Fees: (2.25 + 0.35) * 2 = $5.20
    # Slippage: 2 ticks = 0.5 points = $10.00
    # Net P&L: 200.00 - 5.20 - 10.00 = $184.80
    res = calc.calculate_trade_pnl(direction=1, entry_price=18000.0, exit_price=18010.0, contracts=1)
    assert res["gross_points"] == 10.0
    assert res["gross_pnl"] == 200.0
    assert res["commission_and_fees"] == 5.20
    assert res["slippage_cost"] == 10.0
    assert res["net_pnl"] == 184.80


def test_short_trade_pnl():
    calc = FuturesCalculator()
    # Short trade from 18010.00 to 18000.00 (+10 points gain)
    res = calc.calculate_trade_pnl(direction=-1, entry_price=18010.0, exit_price=18000.0, contracts=1)
    assert res["gross_points"] == 10.0
    assert res["gross_pnl"] == 200.0
