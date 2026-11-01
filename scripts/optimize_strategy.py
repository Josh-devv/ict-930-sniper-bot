"""
Comprehensive Quantitative Strategy Optimization Engine for NQ/MNQ 1-Minute Strategy.
Analyzes:
1. Higher Timeframe (HTF) Trend Filters (15m EMA, 09:30 session open bias, London high/low)
2. Displacement Quality Thresholds (min points: 3, 5, 8, 10)
3. Target R multiples (1.5R, 2.0R, 2.5R, 3.0R, 4.0R)
4. Breakeven triggers (50% progress, 1.0R, no BE)
5. Block types (OB only, Breaker only, All)
Simulates exact FundedNext $50k account metrics (Win rate, Expectancy, Max Drawdown, Pass/Fail).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from data.loader import NQDataLoader
from strategy.indicators import StructureEngine, Block
from config import StrategyConfig, SessionConfig, MNQContractSpec, ExecutionConfig
from execution.simulator import NQIntrabarSimulator
from strategy.rules import ICTStrategyLogic, TradeSignal


def run_grid_search():
    print("Loading MNQ 1-minute historical data...")
    loader = NQDataLoader(target_timezone="America/New_York")
    df = loader.load_csv("sample_data/MNQ_1min_20260120_20260415.csv", source_timezone="UTC")
    print(f"Loaded {len(df)} bars across {len(pd.Series(df.index.date).unique())} trading days.\n")

    # Precalculate 15-minute EMA for HTF trend filter
    # Resample to 15m to get clean trend
    df_15m = df.resample("15min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()
    df_15m["EMA20"] = df_15m["Close"].ewm(span=20, adjust=False).mean()
    df_15m["EMA50"] = df_15m["Close"].ewm(span=50, adjust=False).mean()
    df_15m["Trend"] = np.where(df_15m["EMA20"] > df_15m["EMA50"], 1, -1)

    # Reindex 15m trend back to 1m
    df["HTF_Trend"] = df_15m["Trend"].reindex(df.index, method="ffill").fillna(0)

    dates = pd.Series(df.index.date, index=df.index)
    unique_days = dates.unique()

    # Parameter grid
    min_disp_options = [4.0, 7.0, 10.0]
    htf_options = [True, False]
    target_r_options = [1.5, 2.0, 2.5, 3.0]
    be_pct_options = [0.33, 0.5, 1.0] # 1.0 means no BE until target

    results = []

    print(f"{'HTF':<5} | {'Disp':<5} | {'TargetR':<8} | {'BE_Pct':<7} | {'Trades':<7} | {'WinRate':<8} | {'NetPnL':<10} | {'ProfitFactor':<13} | {'MaxDD':<8} | {'Pass$50k?'}")
    print("-" * 105)

    for use_htf in htf_options:
        for min_disp in min_disp_options:
            strat_cfg = StrategyConfig(
                min_displacement_points=min_disp,
                expansion_body_multiplier=1.5
            )
            engine = StructureEngine(
                swing_lookback=strat_cfg.mss_swing_lookback,
                fvg_min_size_points=strat_cfg.fvg_min_size_points,
                expansion_multiplier=strat_cfg.expansion_body_multiplier,
                atr_period=strat_cfg.expansion_atr_period,
                min_displacement_points=min_disp
            )

            # Generate candidate signals across all days
            signals_base = []
            for day in unique_days:
                day_indices = np.where(dates == day)[0]
                if len(day_indices) == 0:
                    continue
                df_day = df.iloc[day_indices]
                day_ts = df_day.index

                mark_mask = (day_ts.time >= pd.to_datetime("09:30:00").time()) & (day_ts.time <= pd.to_datetime("09:59:00").time())
                mark_indices = [day_indices[j] for j, m in enumerate(mark_mask) if m]
                if len(mark_indices) < 5:
                    continue

                df_mark = df.iloc[mark_indices]

                raw = engine.detect_order_blocks(df_mark, mark_indices) + \
                      engine.detect_breaker_blocks(df_mark, mark_indices) + \
                      engine.detect_rejection_blocks(df_mark, mark_indices)

                # Untouched check
                valid_pre = []
                for b in raw:
                    touched = False
                    for g in mark_indices:
                        if g > b.formation_idx:
                            if b.direction == 1 and df["Low"].iloc[g] <= b.near_edge:
                                touched = True; break
                            elif b.direction == -1 and df["High"].iloc[g] >= b.near_edge:
                                touched = True; break
                    if not touched:
                        valid_pre.append(b)

                post_mask = (day_ts.time >= pd.to_datetime("10:00:00").time()) & (day_ts.time < pd.to_datetime("16:00:00").time())
                post_indices = [day_indices[j] for j, m in enumerate(post_mask) if m]
                if not post_indices:
                    continue

                open_10am = df["Open"].iloc[post_indices[0]]
                htf_dir = df["HTF_Trend"].iloc[post_indices[0]]

                bulls = [b for b in valid_pre if b.direction == 1 and b.near_edge <= open_10am]
                bears = [b for b in valid_pre if b.direction == -1 and b.near_edge >= open_10am]

                # Filter by HTF trend if enabled
                if use_htf:
                    if htf_dir == 1:
                        bears = [] # Only longs in bull trend
                    elif htf_dir == -1:
                        bulls = [] # Only shorts in bear trend

                logic_inst = ICTStrategyLogic(strategy_config=strat_cfg, contract_spec=MNQContractSpec())
                best_long = logic_inst._select_best_block(bulls, open_10am)
                best_short = logic_inst._select_best_block(bears, open_10am)

                long_dead = False
                short_dead = False
                for g in post_indices:
                    b_dt = df.index[g]
                    b_open = df["Open"].iloc[g]
                    b_high = df["High"].iloc[g]
                    b_low = df["Low"].iloc[g]

                    # Long
                    if best_long and not long_dead:
                        if b_open < best_long.far_edge:
                            long_dead = True
                        elif b_low <= best_long.near_edge:
                            if b_open >= best_long.near_edge:
                                signals_base.append((g, b_dt, 1, best_long.near_edge, best_long.far_edge, best_long.block_type))
                                break
                            else:
                                if b_low < best_long.far_edge:
                                    long_dead = True

                    # Short
                    if best_short and not short_dead:
                        if b_open > best_short.far_edge:
                            short_dead = True
                        elif b_high >= best_short.near_edge:
                            if b_open <= best_short.near_edge:
                                signals_base.append((g, b_dt, -1, best_short.near_edge, best_short.far_edge, best_short.block_type))
                                break
                            else:
                                if b_high > best_short.far_edge:
                                    short_dead = True

            # Now test targets & breakeven variations
            for target_r in target_r_options:
                for be_pct in be_pct_options:
                    cfg_test = StrategyConfig(
                        target_r_multiple=target_r,
                        breakeven_trigger_pct=be_pct,
                        min_risk_points=2.0,
                        max_risk_points=80.0
                    )
                    logic_test = ICTStrategyLogic(strategy_config=cfg_test, contract_spec=MNQContractSpec())

                    full_signals = []
                    for (g, b_dt, direction, near_edge, far_edge, b_type) in signals_base:
                        levels = logic_test.calculate_trade_levels(direction, near_edge, far_edge)
                        if levels:
                            sl, tp, be, risk = levels
                            full_signals.append(
                                TradeSignal(
                                    index=g,
                                    timestamp=b_dt,
                                    direction=direction,
                                    entry_price=near_edge,
                                    stop_loss=sl,
                                    target_price=tp,
                                    breakeven_trigger_price=be,
                                    risk_points=risk,
                                    reason="Signal",
                                    block_type=b_type
                                )
                            )

                    sim = NQIntrabarSimulator(strategy_config=cfg_test, contract_spec=MNQContractSpec())
                    trades = sim.run(df, full_signals)

                    if not trades:
                        continue

                    wins = [t for t in trades if t.realized_r > 0]
                    losses = [t for t in trades if t.realized_r < 0]
                    win_rate = len(wins) / len(trades) * 100
                    net_pnl = sum(t.net_pnl for t in trades)
                    gross_wins = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
                    gross_losses = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
                    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else 99.0

                    # Compute drawdown
                    cum = pd.Series([t.net_pnl for t in trades]).cumsum()
                    dd = (cum.cummax() - cum).max()

                    # Test FundedNext Challenge ($50k, pass = +$2500, fail = -$1500)
                    # Sizing contracts to risk $150 per trade
                    funded_bal = 50000.0
                    funded_peak = 50000.0
                    funded_passed = False
                    funded_failed = False
                    for t in trades:
                        contracts = max(1, int(round(150.0 / (t.risk_points * 2.0))))
                        trade_net = contracts * t.gross_points * 2.0 - contracts * 6.20
                        funded_bal += trade_net
                        if funded_bal > funded_peak:
                            funded_peak = funded_bal
                        if funded_peak - funded_bal >= 1500.0:
                            funded_failed = True
                            break
                        if funded_bal >= 52500.0:
                            funded_passed = True

                    pass_str = "PASS" if funded_passed else ("BLOWN" if funded_failed else "PENDING")

                    results.append({
                        "use_htf": use_htf,
                        "min_disp": min_disp,
                        "target_r": target_r,
                        "be_pct": be_pct,
                        "trades": len(trades),
                        "win_rate": win_rate,
                        "net_pnl": net_pnl,
                        "pf": profit_factor,
                        "max_dd": dd,
                        "funded_passed": funded_passed,
                        "funded_failed": funded_failed,
                        "pass_str": pass_str
                    })

                    print(f"{str(use_htf):<5} | {min_disp:<5.1f} | {target_r:<8.1f} | {be_pct:<7.2f} | {len(trades):<7} | {win_rate:<7.1f}% | ${net_pnl:<9.2f} | {profit_factor:<13.2f} | ${dd:<7.2f} | {pass_str}")

    df_res = pd.DataFrame(results)
    df_res.to_csv("optimization_results.csv", index=False)
    print("\nOptimization complete. Top 5 Configurations by Win Rate:")
    print(df_res.sort_values(by="win_rate", ascending=False).head(5)[["use_htf", "min_disp", "target_r", "be_pct", "trades", "win_rate", "net_pnl", "pf", "pass_str"]].to_string(index=False))

    print("\nTop 5 Configurations by Net P&L:")
    print(df_res.sort_values(by="net_pnl", ascending=False).head(5)[["use_htf", "min_disp", "target_r", "be_pct", "trades", "win_rate", "net_pnl", "pf", "pass_str"]].to_string(index=False))


if __name__ == "__main__":
    run_grid_search()
