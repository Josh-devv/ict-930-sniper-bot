"""
Daye's Quarterly Theory (AMDX 90-Minute Cycle) Trading Engine.
Implements:
1. NY AM Session (06:00 - 12:00) 90-Minute Cycles:
   - Q1: 06:00 - 07:30 (Accumulation)
   - Q2: 07:30 - 09:00 (Manipulation - 07:30 True Session Open)
   - Q3: 09:00 - 10:30 (Distribution - Main Trend Delivery)
2. Q3 Micro-Cycle (AMDX within 09:00 - 10:30):
   - Micro-Q1 (09:00 - 09:22): Accumulation range
   - Micro-Q2 (09:22 - 09:45): 09:30 Equity Open Manipulation & Liquidity Sweep
   - Micro-Q3 (09:45 - 10:10): Distribution Entry (MSS + FVG/OB in direction of true delivery)
   - Micro-Q4 (10:10 - 10:30): Trend Delivery to Opposing Liquidity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from data.loader import NQDataLoader
from strategy.indicators import StructureEngine
from config import StrategyConfig, MNQContractSpec, ExecutionConfig
from execution.simulator import NQIntrabarSimulator, SimulatedTrade
from strategy.rules import TradeSignal


def backtest_daye_quarterly():
    print("Loading MNQ 1-minute historical data for Daye's Quarterly Theory Backtest...")
    loader = NQDataLoader(target_timezone="America/New_York")
    df = loader.load_csv("sample_data/MNQ_1min_20260120_20260415.csv", source_timezone="UTC")

    engine = StructureEngine(swing_lookback=3, fvg_min_size_points=1.0, min_displacement_points=4.0)
    contract = MNQContractSpec()
    exec_cfg = ExecutionConfig()

    dates = pd.Series(df.index.date, index=df.index)
    unique_days = dates.unique()

    target_r_options = [1.5, 2.0, 2.5]
    
    for target_r in target_r_options:
        signals = []

        for day in unique_days:
            day_indices = np.where(dates == day)[0]
            if len(day_indices) == 0:
                continue
            df_day = df.iloc[day_indices]
            day_ts = df_day.index

            # 1. 07:30 True Session Open
            mask_0730 = (day_ts.time >= pd.to_datetime("07:30:00").time()) & (day_ts.time <= pd.to_datetime("07:35:00").time())
            idx_0730 = [day_indices[j] for j, m in enumerate(mask_0730) if m]
            true_open_0730 = df["Open"].iloc[idx_0730[0]] if idx_0730 else None

            # 2. Micro-Q1: Accumulation (09:00 - 09:22)
            mask_q1 = (day_ts.time >= pd.to_datetime("09:00:00").time()) & (day_ts.time <= pd.to_datetime("09:22:00").time())
            idx_q1 = [day_indices[j] for j, m in enumerate(mask_q1) if m]
            if len(idx_q1) < 5:
                continue
            q1_high = df["High"].iloc[idx_q1].max()
            q1_low = df["Low"].iloc[idx_q1].min()

            # 3. Micro-Q2: Manipulation & 09:30 Sweep (09:22 - 09:45)
            mask_q2 = (day_ts.time >= pd.to_datetime("09:22:00").time()) & (day_ts.time <= pd.to_datetime("09:45:00").time())
            idx_q2 = [day_indices[j] for j, m in enumerate(mask_q2) if m]
            if len(idx_q2) < 5:
                continue
            q2_high = df["High"].iloc[idx_q2].max()
            q2_low = df["Low"].iloc[idx_q2].min()

            swept_sellside = q2_low < q1_low
            swept_buyside = q2_high > q1_high

            # 4. Micro-Q3: Distribution Entry (09:45 - 10:15)
            mask_q3 = (day_ts.time >= pd.to_datetime("09:45:00").time()) & (day_ts.time <= pd.to_datetime("10:15:00").time())
            idx_q3 = [day_indices[j] for j, m in enumerate(mask_q3) if m]
            if not idx_q3:
                continue

            # Detect FVGs and Order Blocks formed during Q2/Q3 (09:30 to 10:15)
            mask_entry_blocks = (day_ts.time >= pd.to_datetime("09:30:00").time()) & (day_ts.time <= pd.to_datetime("10:15:00").time())
            idx_entry_blocks = [day_indices[j] for j, m in enumerate(mask_entry_blocks) if m]
            df_blocks = df.iloc[idx_entry_blocks]
            
            raw_obs = engine.detect_order_blocks(df_blocks, idx_entry_blocks)
            bullish_obs = [b for b in raw_obs if b.direction == 1]
            bearish_obs = [b for b in raw_obs if b.direction == -1]

            traded_day = False
            for g in idx_q3:
                if traded_day:
                    break
                b_dt = df.index[g]
                b_open = df["Open"].iloc[g]
                b_high = df["High"].iloc[g]
                b_low = df["Low"].iloc[g]
                b_close = df["Close"].iloc[g]

                # Long condition: Swept sellside in Q2 manipulation, enters bullish OB in Q3
                if swept_sellside and bullish_obs:
                    best_ob = bullish_obs[-1] # latest formed OB
                    if best_ob.formation_idx < g:
                        if b_low <= best_ob.near_edge and b_open >= best_ob.near_edge:
                            entry_price = best_ob.near_edge
                            stop_loss = best_ob.far_edge - 0.25
                            risk = entry_price - stop_loss
                            if 3.0 <= risk <= 35.0:
                                target = entry_price + (risk * target_r)
                                be = entry_price + (target - entry_price) * 0.50
                                signals.append(
                                    TradeSignal(
                                        index=g,
                                        timestamp=b_dt,
                                        direction=1,
                                        entry_price=entry_price,
                                        stop_loss=stop_loss,
                                        target_price=target,
                                        breakeven_trigger_price=be,
                                        risk_points=round(risk, 4),
                                        reason="Daye QT Q3 Bullish Distribution",
                                        block_type="OB"
                                    )
                                )
                                traded_day = True
                                break

                # Short condition: Swept buyside in Q2 manipulation, enters bearish OB in Q3
                if swept_buyside and bearish_obs and not traded_day:
                    best_ob = bearish_obs[-1]
                    if best_ob.formation_idx < g:
                        if b_high >= best_ob.near_edge and b_open <= best_ob.near_edge:
                            entry_price = best_ob.near_edge
                            stop_loss = best_ob.far_edge + 0.25
                            risk = stop_loss - entry_price
                            if 3.0 <= risk <= 35.0:
                                target = entry_price - (risk * target_r)
                                be = entry_price - (entry_price - target) * 0.50
                                signals.append(
                                    TradeSignal(
                                        index=g,
                                        timestamp=b_dt,
                                        direction=-1,
                                        entry_price=entry_price,
                                        stop_loss=stop_loss,
                                        target_price=target,
                                        breakeven_trigger_price=be,
                                        risk_points=round(risk, 4),
                                        reason="Daye QT Q3 Bearish Distribution",
                                        block_type="OB"
                                    )
                                )
                                traded_day = True
                                break

        strat_cfg = StrategyConfig(target_r_multiple=target_r, breakeven_trigger_pct=0.50, max_risk_points=35.0)
        sim = NQIntrabarSimulator(strategy_config=strat_cfg, contract_spec=contract)
        trades = sim.run(df, signals)

        wins = [t for t in trades if t.realized_r > 0]
        losses = [t for t in trades if t.exit_reason == "STOP_LOSS"]
        be_trades = [t for t in trades if t.exit_reason == "BREAKEVEN"]
        wr = len(wins) / len(trades) * 100 if trades else 0
        net_pnl = sum(t.net_pnl for t in trades)
        gross_w = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
        gross_l = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
        pf = gross_w / gross_l if gross_l > 0 else 0

        cum = pd.Series([t.net_pnl for t in trades]).cumsum()
        max_dd = (cum.cummax() - cum).max() if not cum.empty else 0

        print(f"\n=======================================================")
        print(f"  Daye's Quarterly Theory Backtest (Target: {target_r}R)")
        print(f"=======================================================")
        print(f"Total Trades        : {len(trades)}")
        print(f"Winning Trades      : {len(wins)} ({wr:.1f}%)")
        print(f"Breakeven Trades    : {len(be_trades)} ({len(be_trades)/len(trades)*100:.1f}%)" if trades else "0")
        print(f"Losing Trades       : {len(losses)} ({len(losses)/len(trades)*100:.1f}%)" if trades else "0")
        print(f"Profit Factor       : {pf:.2f}")
        print(f"Net Realized P&L    : ${net_pnl:.2f}")
        print(f"Max Drawdown (1 MNQ): ${max_dd:.2f}")

        # Export log
        if target_r == 2.0:
            exp_df = pd.DataFrame([{
                'trade_id': t.trade_id,
                'direction': 'LONG' if t.direction == 1 else 'SHORT',
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'risk_points': t.risk_points,
                'gross_points': t.gross_points,
                'gross_pnl': t.gross_pnl,
                'fees': t.fees,
                'slippage': t.slippage,
                'net_pnl': t.net_pnl,
                'realized_r': t.realized_r,
                'exit_reason': t.exit_reason
            } for t in trades])
            exp_df.to_csv("trade_log_daye_quarterly.csv", index=False)
            print("Saved trade log to trade_log_daye_quarterly.csv")


if __name__ == "__main__":
    backtest_daye_quarterly()
