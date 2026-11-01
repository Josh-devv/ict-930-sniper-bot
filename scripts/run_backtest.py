"""
Command-Line Backtesting Runner for NQ 1-Minute Strategy.
Supports high-precision intrabar futures execution simulation as well as
native backtesting.py execution.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import StrategyConfig, SessionConfig, ExecutionConfig, NQContractSpec
from data.loader import NQDataLoader
from strategy.rules import ICTStrategyLogic
from strategy.backtesting_py_adapter import BacktestingPyNQStrategy
from execution.simulator import NQIntrabarSimulator
from analytics.metrics import PerformanceAnalytics
from analytics.reporting import PerformanceReporter


def run_intrabar_simulation(args, df):
    """
    Runs high-precision institutional intrabar futures simulation.
    """
    strat_cfg = StrategyConfig(
        target_r_multiple=args.target_r,
        breakeven_trigger_pct=args.breakeven_pct,
        initial_capital=args.capital,
        entry_window_end=args.entry_end,
        max_risk_points=args.max_risk,
    )
    exec_cfg = ExecutionConfig()
    from config import MNQContractSpec
    contract = MNQContractSpec() if getattr(args, "symbol", "NQ").upper() == "MNQ" else NQContractSpec()

    print(f"\n[RunBacktest] Mode: High-Precision Intrabar Simulator")
    print(f"[RunBacktest] Symbol: {contract.symbol} | Tick: {contract.tick_size} | Multiplier: ${contract.point_value}/pt")
    print(f"[RunBacktest] Strategy: {args.target_r}R Target | {int(args.breakeven_pct * 100)}% Breakeven Progression | NY Session Filters")

    logic = ICTStrategyLogic(strategy_config=strat_cfg, contract_spec=contract)
    signals = logic.generate_signals(df)
    print(f"[RunBacktest] Generated {len(signals)} verified trade signals.")

    simulator = NQIntrabarSimulator(strategy_config=strat_cfg, execution_config=exec_cfg, contract_spec=contract)
    trades = simulator.run(df, signals)

    analytics = PerformanceAnalytics(trades, initial_capital=args.capital)
    reporter = PerformanceReporter(analytics)

    reporter.print_terminal_summary()

    if args.export:
        reporter.export_trade_log("trade_log.csv")

    if args.plot and trades:
        reporter.save_equity_curve_plot("equity_curve.png")

    return analytics.compute_summary_statistics()


def run_backtesting_py(args, df):
    """
    Runs native backtesting.py engine with Bokeh visualization.
    """
    from backtesting import Backtest

    print(f"\n[RunBacktest] Mode: backtesting.py Framework")
    # backtesting.py requires un-localized datetime index or UTC
    df_bt = df.copy()
    if df_bt.index.tz is not None:
        df_bt.index = df_bt.index.tz_localize(None)

    bt = Backtest(
        df_bt,
        BacktestingPyNQStrategy,
        cash=args.capital,
        commission=0.0001,  # fractional fee proxy in backtesting.py
        exclusive_orders=True,
    )

    stats = bt.run()
    print("\n" + "=" * 60)
    print("           BACKTESTING.PY NATIVE RESULTS")
    print("=" * 60)
    print(stats)
    print("=" * 60 + "\n")

    if args.plot:
        try:
            bt.plot(filename="backtesting_py_result.html", open_browser=False)
            print("[RunBacktest] Bokeh interactive plot saved to: backtesting_py_result.html")
        except Exception as e:
            print(f"[RunBacktest] Notice: bokeh plot generation skipped ({e})")

    return stats


def main():
    parser = argparse.ArgumentParser(description="NQ 1-Minute Futures Backtesting System")
    parser.add_argument("--data", type=str, default="sample_data/nq_1min_sample.csv", help="Path to 1-minute OHLCV CSV")
    parser.add_argument("--mode", type=str, choices=["intrabar", "backtesting_py"], default="intrabar", help="Execution mode")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Starting capital ($)")
    parser.add_argument("--target_r", type=float, default=2.5, help="Target in R (default: 2.5)")
    parser.add_argument("--breakeven_pct", type=float, default=0.50, help="Breakeven trigger % (default: 0.50)")
    parser.add_argument("--entry_end", type=str, default="10:30", help="Entry macro window end time (default: 10:30)")
    parser.add_argument("--max_risk", type=float, default=40.0, help="Maximum allowed stop loss points (default: 40.0)")
    parser.add_argument("--symbol", type=str, choices=["NQ", "MNQ"], default="MNQ", help="Contract symbol (default: MNQ)")
    parser.add_argument("--source_tz", type=str, default="UTC", help="Source data timezone (default: UTC)")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate performance charts")
    parser.add_argument("--export", action="store_true", default=True, help="Export trade log CSV")

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[RunBacktest] Data file not found: {data_path}. Generating synthetic sample dataset...")
        from scripts.generate_sample_data import generate_synthetic_nq_data
        data_path = generate_synthetic_nq_data(output_path=str(data_path))

    loader = NQDataLoader(target_timezone="America/New_York")
    df = loader.load_csv(data_path, source_timezone=args.source_tz)
    print(f"[RunBacktest] Successfully loaded {len(df)} validated bars from {data_path.name}")

    if args.mode == "intrabar":
        run_intrabar_simulation(args, df)
    else:
        run_backtesting_py(args, df)


if __name__ == "__main__":
    main()
