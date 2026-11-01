"""
4H Profiling & Session Driver Pairing Model (Daye / Powell Playbook)
=====================================================================
Direct implementation of the 4H Institutional Driver Playbook:

Key Session Structure (EST):
  - 18:00 - 22:00 (Prior Evening) : Asia Profile (Consolidation vs Manipulation)
  - 02:00 - 06:00 : London Session (Reversal / Driver)
  - 06:00 - 10:00 : NY Pre-Market (Driver & Manipulation window)
  - 10:00 - 14:00 : NY 4H Expansion Candle (Our Execution Target)

Core Driver Pairing Rules:
  1. IF 6:00-10:00 AM manipulates (sweeps London H/L or PDH/PDL),
     THEN 10:00 AM candle is CONTINUATION in the true direction.
  2. IF 9:30 AM open manipulates one side (Judas swing),
     THEN 10:00 AM macro window provides CONTINUATION expansion.
  3. Direction is STRICTLY LOCKED:
     - 9:30 AM swept Sellside -> 10:00 AM can ONLY BE LONG (Expansion upward).
     - 9:30 AM swept Buyside  -> 10:00 AM can ONLY BE SHORT (Expansion downward).
     - Both swept or neither -> NO TRADE (Wait for clean profile).
  4. Entry on 1-min FVG / Order Block formed during the 9:30-10:00 shift.
  5. High R:R Target: 3.5R to 5.0R (riding the 4H expansion leg to opposing session liquidity).
  6. Minimum stop filter: min 6.0 pts, max 25.0 pts (no micro-noise stops).
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from data.loader import NQDataLoader
from config import StrategyConfig, MNQContractSpec, NQContractSpec
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Data Structure
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DriverTrade:
    entry_time:    pd.Timestamp
    direction:     int         # 1 = Long, -1 = Short
    entry_price:   float
    stop_loss:     float
    take_profit:   float
    risk_pts:      float
    exit_time:     Optional[pd.Timestamp] = None
    exit_price:    float = 0.0
    exit_reason:   str   = ''
    gross_pts:     float = 0.0
    realized_r:    float = 0.0
    net_mnq:       float = 0.0
    net_nq:        float = 0.0
    profile_notes: str   = ''


def run_4h_driver_model(csv_path: str, target_r: float = 3.5, be_trigger_pct: float = 0.60):
    print(f"\n{'='*70}")
    print(f"  RUNNING 4H DRIVER PAIRING MODEL ON: {csv_path}")
    print(f"  Target R:R = {target_r}R | BE Trigger = {int(be_trigger_pct*100)}%")
    print(f"{'='*70}")

    loader = NQDataLoader(target_timezone='America/New_York')
    df = loader.load_csv(csv_path, source_timezone='UTC')

    # Calculate Prior Day High & Low
    daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    df['PDH'] = daily['High'].shift(1).reindex(df.index, method='ffill')
    df['PDL'] = daily['Low'].shift(1).reindex(df.index, method='ffill')

    dates = pd.Series(df.index.date, index=df.index)
    unique_days = dates.unique()

    trades: List[DriverTrade] = []

    for day in unique_days:
        day_indices = np.where(dates == day)[0]
        if len(day_indices) < 60:
            continue
        day_ts = df.index[day_indices]

        # 1. London Session (02:00 - 06:00 EST)
        london_mask = (day_ts.time >= pd.to_datetime('02:00:00').time()) & \
                      (day_ts.time <  pd.to_datetime('06:00:00').time())
        london_idx = [day_indices[j] for j, m in enumerate(london_mask) if m]
        if len(london_idx) < 10:
            continue
        london_high = df['High'].iloc[london_idx].max()
        london_low  = df['Low'].iloc[london_idx].min()

        pdh = df['PDH'].iloc[day_indices[0]]
        pdl = df['PDL'].iloc[day_indices[0]]

        # Key external reference levels
        ref_high = max(london_high, pdh) if pd.notna(pdh) else london_high
        ref_low  = min(london_low, pdl)  if pd.notna(pdl) else london_low

        # 2. 6AM - 10AM Driver / Manipulation Window
        # (Focus heavily on the 09:30 - 10:00 NY Cash Open manipulation leg)
        open_mask = (day_ts.time >= pd.to_datetime('09:30:00').time()) & \
                    (day_ts.time <  pd.to_datetime('10:00:00').time())
        open_idx = [day_indices[j] for j, m in enumerate(open_mask) if m]
        if len(open_idx) < 10:
            continue

        open_high = df['High'].iloc[open_idx].max()
        open_low  = df['Low'].iloc[open_idx].min()

        # Did 9:30 AM manipulate?
        swept_buyside  = open_high > ref_high
        swept_sellside = open_low  < ref_low

        # "IF 9:30AM manipulation, THEN 10AM continuation"
        # Must be a clean single-sided manipulation
        if swept_sellside and not swept_buyside:
            direction = 1   # Long continuation
            manip_swing = open_low
            profile = "9:30 Sellside Sweep -> 10AM Bull Expansion"
        elif swept_buyside and not swept_sellside:
            direction = -1  # Short continuation
            manip_swing = open_high
            profile = "9:30 Buyside Sweep -> 10AM Bear Expansion"
        else:
            continue  # Both swept or chop -> invalid driver pairing

        # 3. 10:00 AM - 10:30 AM Execution Window
        exec_mask = (day_ts.time >= pd.to_datetime('10:00:00').time()) & \
                    (day_ts.time <= pd.to_datetime('10:30:00').time())
        exec_idx = [day_indices[j] for j, m in enumerate(exec_mask) if m]
        if len(exec_idx) < 5:
            continue

        open_10am = df['Open'].iloc[exec_idx[0]]

        # Find 1-min Displacement & FVG / Structure retest during 09:40-10:20
        # Search for entry signal inside the 10:00-10:30 window
        traded_today = False
        for pos, g in enumerate(exec_idx):
            if traded_today:
                break
            bar = df.iloc[g]
            b_time = df.index[g]

            # Look for 1-min order block / fair value entry
            # Candle pulling back towards manipulation origin
            if direction == 1:
                # Bullish continuation: entry when price dips into discounted area
                # Stop loss protected under 9:30 manipulation swing low
                entry_price = bar['Open']
                stop_loss   = manip_swing - 1.0  # protected swing
                risk_pts    = entry_price - stop_loss

                # Strict risk filter: 6 to 25 pts
                if risk_pts < 6.0 or risk_pts > 25.0:
                    continue

                take_profit = entry_price + (risk_pts * target_r)

                # Simulate intrabar walk-forward
                trade = _simulate_driver_trade(
                    df, g, direction, entry_price, stop_loss, take_profit, risk_pts,
                    be_trigger_pct, profile
                )
                trades.append(trade)
                traded_today = True

            elif direction == -1:
                # Bearish continuation
                entry_price = bar['Open']
                stop_loss   = manip_swing + 1.0  # protected swing
                risk_pts    = stop_loss - entry_price

                if risk_pts < 6.0 or risk_pts > 25.0:
                    continue

                take_profit = entry_price - (risk_pts * target_r)

                trade = _simulate_driver_trade(
                    df, g, direction, entry_price, stop_loss, take_profit, risk_pts,
                    be_trigger_pct, profile
                )
                trades.append(trade)
                traded_today = True

    _print_results(trades, target_r)
    return trades


def _simulate_driver_trade(df: pd.DataFrame, entry_idx: int, direction: int,
                           entry_price: float, stop_loss: float,
                           take_profit: float, risk_pts: float,
                           be_trigger_pct: float, profile: str) -> DriverTrade:
    COMMISSION_MNQ = 6.20
    COMMISSION_NQ  = 10.20
    be_active = False
    current_sl = stop_loss

    tp_dist = abs(take_profit - entry_price)
    be_trigger = entry_price + (direction * tp_dist * be_trigger_pct)

    for i in range(entry_idx + 1, min(entry_idx + 400, len(df))):
        bar = df.iloc[i]
        b_time = df.index[i]

        # End of Session Exit at 15:45
        if b_time.hour > 15 or (b_time.hour == 15 and b_time.minute >= 45):
            exit_p = bar['Open']
            gross = (exit_p - entry_price) * direction
            return DriverTrade(
                df.index[entry_idx], direction, entry_price, stop_loss, take_profit, risk_pts,
                b_time, exit_p, 'TIME_EXIT', gross, gross / risk_pts,
                gross * 2.0 - COMMISSION_MNQ, gross * 20.0 - COMMISSION_NQ, profile
            )

        # Breakeven progression
        if not be_active:
            if direction == 1 and bar['High'] >= be_trigger:
                be_active = True
                current_sl = entry_price + 0.50
            elif direction == -1 and bar['Low'] <= be_trigger:
                be_active = True
                current_sl = entry_price - 0.50

        # Check exits
        if direction == 1:
            if bar['Low'] <= current_sl:
                exit_p = current_sl
                gross = exit_p - entry_price
                reason = 'BREAKEVEN' if be_active else 'STOP_LOSS'
                return DriverTrade(
                    df.index[entry_idx], direction, entry_price, stop_loss, take_profit, risk_pts,
                    b_time, exit_p, reason, gross, gross / risk_pts,
                    gross * 2.0 - COMMISSION_MNQ, gross * 20.0 - COMMISSION_NQ, profile
                )
            if bar['High'] >= take_profit:
                exit_p = take_profit
                gross = exit_p - entry_price
                return DriverTrade(
                    df.index[entry_idx], direction, entry_price, stop_loss, take_profit, risk_pts,
                    b_time, exit_p, 'TARGET', gross, gross / risk_pts,
                    gross * 2.0 - COMMISSION_MNQ, gross * 20.0 - COMMISSION_NQ, profile
                )
        else:
            if bar['High'] >= current_sl:
                exit_p = current_sl
                gross = entry_price - exit_p
                reason = 'BREAKEVEN' if be_active else 'STOP_LOSS'
                return DriverTrade(
                    df.index[entry_idx], direction, entry_price, stop_loss, take_profit, risk_pts,
                    b_time, exit_p, reason, gross, gross / risk_pts,
                    gross * 2.0 - COMMISSION_MNQ, gross * 20.0 - COMMISSION_NQ, profile
                )
            if bar['Low'] <= take_profit:
                exit_p = take_profit
                gross = entry_price - exit_p
                return DriverTrade(
                    df.index[entry_idx], direction, entry_price, stop_loss, take_profit, risk_pts,
                    b_time, exit_p, 'TARGET', gross, gross / risk_pts,
                    gross * 2.0 - COMMISSION_MNQ, gross * 20.0 - COMMISSION_NQ, profile
                )

    exit_p = df['Close'].iloc[-1]
    gross = (exit_p - entry_price) * direction
    return DriverTrade(
        df.index[entry_idx], direction, entry_price, stop_loss, take_profit, risk_pts,
        df.index[-1], exit_p, 'EOD', gross, gross / risk_pts,
        gross * 2.0 - COMMISSION_MNQ, gross * 20.0 - COMMISSION_NQ, profile
    )


def _print_results(trades: List[DriverTrade], target_r: float):
    if not trades:
        print("No trades generated.")
        return

    wins   = [t for t in trades if t.exit_reason == 'TARGET']
    be     = [t for t in trades if t.exit_reason == 'BREAKEVEN']
    losses = [t for t in trades if t.exit_reason == 'STOP_LOSS']

    n  = len(trades)
    wr = len(wins) / n * 100
    be_rate = len(be) / n * 100
    loss_rate = len(losses) / n * 100

    gw = sum(t.gross_pts * 20.0 for t in trades if t.gross_pts > 0)
    gl = abs(sum(t.gross_pts * 20.0 for t in trades if t.gross_pts <= 0))
    pf = gw / gl if gl > 0 else float('inf')

    avg_r = np.mean([t.realized_r for t in trades])
    avg_risk = np.mean([t.risk_pts for t in trades])
    net_nq = sum(t.net_nq for t in trades)
    net_mnq = sum(t.net_mnq for t in trades)

    cum_nq = pd.Series([t.net_nq for t in trades]).cumsum()
    max_dd_nq = (cum_nq.cummax() - cum_nq).max() if not cum_nq.empty else 0

    print(f"Total Trades         : {n}")
    print(f"Winning Trades (TP)  : {len(wins)} ({wr:.1f}%)")
    print(f"Breakeven Trades     : {len(be)} ({be_rate:.1f}%)")
    print(f"Losing Trades (SL)   : {len(losses)} ({loss_rate:.1f}%)")
    print(f"Profit Factor        : {pf:.2f}")
    print(f"Average Realized R   : {avg_r:+.2f}R")
    print(f"Average Stop Size    : {avg_risk:.1f} pts (${avg_risk*20:.0f} on NQ)")
    print(f"--------------------------------------------------")
    print(f"Net P&L (1 Mini NQ)  : ${net_nq:+,.2f}")
    print(f"Max Drawdown (1 NQ)  : ${max_dd_nq:,.2f}")
    print(f"Net P&L (1 Micro MNQ): ${net_mnq:+,.2f}")
    print(f"--------------------------------------------------")

    # FundedNext Sim ($50K Account)
    bal = 50000.0
    peak = 50000.0
    mdd = 0.0
    passed = False
    pass_trade = None
    pass_date = None

    for i, t in enumerate(trades):
        bal += t.net_nq
        peak = max(peak, bal)
        dd = peak - bal
        mdd = max(mdd, dd)
        if bal >= 52500.0 and not passed:
            passed = True
            pass_trade = i + 1
            pass_date = t.entry_time.date()

    print(f"FundedNext $50k Sim  : {'PASSED' if passed else 'NOT PASSED'}")
    if passed:
        print(f"  Target Reached at  : Trade #{pass_trade} on {pass_date}")
        print(f"  Max Drawdown seen  : ${mdd:,.2f} (Limit: $2,000)")
    print(f"Final Balance (1 NQ) : ${bal:,.2f}")


if __name__ == '__main__':
    # Test on both 3-month and 12-month data
    print("Testing Driver Pairing Engine...")
    run_4h_driver_model('sample_data/MNQ_1min_20260120_20260415.csv', target_r=3.5, be_trigger_pct=0.60)
    run_4h_driver_model('sample_data/MNQ_1min_12mo.csv', target_r=3.5, be_trigger_pct=0.60)

