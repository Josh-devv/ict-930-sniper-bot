"""
Configuration module for NQ (E-mini Nasdaq-100) Futures Backtesting System.
Centralizes contract specifications, execution parameters, session windows,
and strategy configuration parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class NQContractSpec:
    """
    CME E-mini Nasdaq-100 (NQ) Futures Contract Specifications.
    """
    symbol: str = "NQ"
    tick_size: float = 0.25            # 1 tick = 0.25 index points
    point_value: float = 20.0          # $20 per full index point
    tick_value: float = 5.0            # $5 per 0.25 tick (0.25 * $20)
    contract_multiplier: float = 20.0  # Multiplier for futures P&L calculation
    min_contracts: int = 1


@dataclass(frozen=True)
class MNQContractSpec:
    """
    CME Micro E-mini Nasdaq-100 (MNQ) Futures Contract Specifications.
    """
    symbol: str = "MNQ"
    tick_size: float = 0.25            # 1 tick = 0.25 index points
    point_value: float = 2.0           # $2 per full index point
    tick_value: float = 0.50           # $0.50 per 0.25 tick
    contract_multiplier: float = 2.0
    min_contracts: int = 1


@dataclass
class ExecutionConfig:
    """
    Realistic execution, commission, and slippage assumptions.
    """
    # Commission per contract per side (CME + clearing + broker fees)
    commission_per_side: float = 2.25  # Round-turn = $4.50
    # Exchange & NFA regulatory fee per side
    exchange_fee_per_side: float = 0.35
    # Assumed slippage in ticks (1 tick = 0.25 pts)
    slippage_ticks: int = 1
    # Intrabar collision resolution mode: 'worst_case', 'best_case', or 'ohlc_path'
    # 'worst_case': if both SL and TP price touched within same bar, assume SL hit first.
    # 'ohlc_path': approximate price path using Open -> High/Low -> Close
    intrabar_resolution: str = "worst_case"


@dataclass
class SessionConfig:
    """
    Session time filter parameters (in specified timezone, default America/New_York).
    """
    timezone: str = "America/New_York"
    # NY AM session window (e.g. 09:30 to 11:30 or 12:00)
    session_start: str = "09:30"
    session_end: str = "11:30"
    # Specific entry macro window (e.g., 09:50 to 10:10 or 10:00 to 10:30)
    macro_start: Optional[str] = "09:50"
    macro_end: Optional[str] = "10:10"
    # Hard square-off / no new entries time
    square_off_time: str = "15:45"


@dataclass
class StrategyConfig:
    """
    Trading strategy parameters covering Risk, Target, Breakeven, and Filters.
    """
    # Session and Block Marking Windows (America/New_York)
    mark_blocks_start: str = "09:30"   # Form blocks between 09:30 and 09:59
    mark_blocks_end: str = "09:59"
    bias_time: str = "10:00"           # 10:00 Open determines long/short bias
    entry_window_end: str = "10:30"    # Strict 10:00-10:30 AM Silver Bullet Macro window
    time_exit: str = "16:00"           # Time exit at 16:00
    max_trades_per_day: int = 1        # One trade per day rule

    # Risk & Reward
    target_r_multiple: float = 2.5     # 2.5R target (optimizes win rate & expectancy)
    breakeven_trigger_pct: float = 0.50 # Move stop to BE at 50% progress toward target (prevents premature stopouts)
    be_buffer_ticks: int = 1           # Entry + 1 tick buffer for BE
    max_risk_points: float = 40.0      # Maximum allowed stop distance in points (caps wide chop risk)
    min_risk_points: float = 2.0       # Minimum allowed stop distance in points
    fixed_contracts: int = 1           # Contracts per trade
    initial_capital: float = 100_000.0 # Account starting capital ($)

    # Strategy Filters & Key Levels
    use_session_filter: bool = True
    fvg_min_size_points: float = 1.0   # Minimum FVG gap size in points
    fvg_mitigation_lookback: int = 30  # Candles to track unmitigated FVGs
    mss_swing_lookback: int = 3        # Swing lookback length
    expansion_body_multiplier: float = 1.5  # Expansion candle multiplier (raised from 1.3)
    expansion_atr_period: int = 14
    min_displacement_points: float = 3.0  # Absolute minimum body size for a displacement candle (MNQ pts)
