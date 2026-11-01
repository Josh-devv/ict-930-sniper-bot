"""
Futures contract arithmetic, tick alignment, and realistic P&L calculations.
Accurately models tick size (0.25), point value ($20.00 for NQ), commissions,
exchange/regulatory fees, and slippage.
"""

import math
from typing import Optional
from config import NQContractSpec, ExecutionConfig


class FuturesCalculator:
    """
    Computes precise futures P&L, tick rounding, and execution costs.
    """

    def __init__(
        self,
        contract: Optional[NQContractSpec] = None,
        execution: Optional[ExecutionConfig] = None,
    ):
        self.contract = contract or NQContractSpec()
        self.execution = execution or ExecutionConfig()

    def round_to_tick(self, price: float) -> float:
        """
        Rounds a price to the nearest valid contract tick (e.g. 0.25).
        """
        ticks = round(price / self.contract.tick_size)
        return round(ticks * self.contract.tick_size, 4)

    def points_to_ticks(self, points: float) -> int:
        """
        Converts index points to integer ticks.
        """
        return int(round(points / self.contract.tick_size))

    def ticks_to_points(self, ticks: int) -> float:
        """
        Converts integer ticks to index points.
        """
        return round(ticks * self.contract.tick_size, 4)

    def points_to_dollars(self, points: float, contracts: int = 1) -> float:
        """
        Converts price points into dollar value.
        For NQ: 1 point = $20.00 per contract.
        """
        return round(points * self.contract.point_value * contracts, 2)

    def calculate_round_turn_cost(self, contracts: int = 1) -> float:
        """
        Calculates total round-turn execution costs (commission + fees + slippage).
        """
        # Commission + exchange/NFA fees (both entry and exit = 2 sides)
        fees = (
            (self.execution.commission_per_side + self.execution.exchange_fee_per_side)
            * 2
            * contracts
        )
        # Slippage cost in dollars (assumed per round-turn, entry + exit)
        slippage_points = self.ticks_to_points(self.execution.slippage_ticks * 2)
        slippage_cost = self.points_to_dollars(slippage_points, contracts)
        return round(fees + slippage_cost, 2)

    def calculate_trade_pnl(
        self,
        direction: int,  # 1 for Long, -1 for Short
        entry_price: float,
        exit_price: float,
        contracts: int = 1,
        include_costs: bool = True,
    ) -> dict:
        """
        Calculates gross and net P&L for a completed trade.

        Returns:
            dict containing:
                - gross_points: gross price difference in points
                - gross_pnl: gross dollar P&L
                - commission_and_fees: total brokerage/exchange fees
                - slippage_cost: dollar cost of assumed slippage
                - net_pnl: net dollar P&L after all frictions
        """
        if direction not in (1, -1):
            raise ValueError(f"Invalid direction {direction}; expected 1 (long) or -1 (short)")

        gross_points = (exit_price - entry_price) * direction
        gross_pnl = self.points_to_dollars(gross_points, contracts)

        if not include_costs:
            return {
                "gross_points": gross_points,
                "gross_pnl": gross_pnl,
                "commission_and_fees": 0.0,
                "slippage_cost": 0.0,
                "net_pnl": gross_pnl,
            }

        fees = (
            (self.execution.commission_per_side + self.execution.exchange_fee_per_side)
            * 2
            * contracts
        )
        slippage_points = self.ticks_to_points(self.execution.slippage_ticks * 2)
        slippage_cost = self.points_to_dollars(slippage_points, contracts)
        net_pnl = round(gross_pnl - fees - slippage_cost, 2)

        return {
            "gross_points": round(gross_points, 4),
            "gross_pnl": gross_pnl,
            "commission_and_fees": round(fees, 2),
            "slippage_cost": round(slippage_cost, 2),
            "net_pnl": net_pnl,
        }
