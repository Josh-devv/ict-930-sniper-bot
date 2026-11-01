"""
4H Driver Pairing Engine — Calibrated Targets (2.2R Full / 1.5R Partials)
==========================================================================
Adjustments:
  1. Primary Target: 2.2R (achievable within 10:00 - 12:00 NY expansion).
  2. Trailing / Partials: Book 50% at 1.5R, move stop to entry (breakeven).
  3. Minimum stop size: 8.0 pts (no micro noise stops).
  4. Compare performance across both datasets.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from data.loader import NQDataLoader
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CalibratedTrade:
    entry_time:   pd.Timestamp
    direction:    int
    entry_price:  float
    stop_loss:    float
    target_price: float
    risk_pts:     float
    exit_time:    Optional[pd.Timestamp] = None
    exit_price:   float = 0.0
    exit_reason:  str   = ''
    gross_pts:    float = 0.0
    realized_r:   float = 0.0
    net_nq:       float = 0.0
    net_mnq:      float = 0.0


def run_calibrated_model(csv_path: str, target_r: float = 2.2, partial_r: float = 1.5):
    loader = NQDataLoader(target_timezone='America/New_York')
    df = loader.load_csv(csv_path, source_timezone='UTC')

    daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    df['PDH'] = daily['High'].shift(1).reindex(df.index, method='ffill')
    df['PDL'] = daily['Low'].shift(1).reindex(df.index, method='ffill')

    dates = pd.Series(df.index.date, index=df.index)
    unique_days = dates.unique()

    trades: List[CalibratedTrade] = []
    COMMISSION_NQ = 10.20
    COMMISSION_MNQ = 6.20

    for day in unique_days:
        day_indices = np.where(dates == day)[0]
        if len(day_indices) < 60:
            continue
        day_ts = df.index[day_indices]

        # London Session (02:00 - 06:00 EST)
        london_mask = (day_ts.time >= pd.to_datetime('02:00:00').time()) & \
                      (day_ts.time <  pd.to_datetime('06:00:00').time())
        london_idx = [day_indices[j] for j, m in enumerate(london_mask) if m]
        if len(london_idx) < 10:
            continue
        london_high = df['High'].iloc[london_idx].max()
        london_low  = df['Low'].iloc[london_idx].min()

        pdh = df['PDH'].iloc[day_indices[0]]
        pdl = df['PDL'].iloc[day_indices[0]]

        ref_high = max(london_high, pdh) if pd.notna(pdh) else london_high
        ref_low  = min(london_low, pdl)  if pd.notna(pdl) else london_low

        # 9:30 - 10:00 AM NY Open Manipulation
        open_mask = (day_ts.time >= pd.to_datetime('09:30:00').time()) & \
                    (day_ts.time <  pd.to_datetime('10:00:00').time())
        open_idx = [day_indices[j] for j, m in enumerate(open_mask) if m]
        if len(open_idx) < 10:
            continue

        open_high = df['High'].iloc[open_idx].max()
        open_low  = df['Low'].iloc[open_idx].min()

        swept_buyside  = open_high > ref_high
        swept_sellside = open_low  < ref_low

        # Driver Pairing: Lock direction
        if swept_sellside and not swept_buyside:
            direction = 1
            manip_swing = open_low
        elif swept_buyside and not swept_sellside:
            direction = -1
            manip_swing = open_high
        else:
            continue

        # 10:00 - 10:30 AM Entry Window
        exec_mask = (day_ts.time >= pd.to_datetime('10:00:00').time()) & \
                    (day_ts.time <= pd.to_datetime('10:30:00').time())
        exec_idx = [day_indices[j] for j, m in enumerate(exec_mask) if m]
        if len(exec_idx) < 5:
            continue

        bar_idx = exec_idx[0]
        entry_price = df['Open'].iloc[bar_idx]

        if direction == 1:
            stop_loss = manip_swing - 1.0
            risk_pts = entry_price - stop_loss
            if risk_pts < 8.0 or risk_pts > 25.0:
                continue
            take_profit = entry_price + (risk_pts * target_r)
            partial_tp  = entry_price + (risk_pts * partial_r)
        else:
            stop_loss = manip_swing + 1.0
            risk_pts = stop_loss - entry_price
            if risk_pts < 8.0 or risk_pts > 25.0:
                continue
            take_profit = entry_price - (risk_pts * target_r)
            partial_tp  = entry_price - (risk_pts * partial_r)

        # Walk-forward simulation with 50% partial at partial_tp + BE
        partial_hit = False
        current_sl = stop_loss
        exit_found = False

        for i in range(bar_idx + 1, min(bar_idx + 360, len(df))):
            bar = df.iloc[i]
            b_time = df.index[i]

            # 15:45 Time Exit
            if b_time.hour > 15 or (b_time.hour == 15 and b_time.minute >= 45):
                exit_p = bar['Open']
                if partial_hit:
                    gross = 0.5 * (partial_r * risk_pts) + 0.5 * ((exit_p - entry_price) * direction)
                else:
                    gross = (exit_p - entry_price) * direction
                t = CalibratedTrade(
                    df.index[bar_idx], direction, entry_price, stop_loss, take_profit, risk_pts,
                    b_time, exit_p, 'TIME_EXIT', gross, gross / risk_pts,
                    gross * 20.0 - COMMISSION_NQ, gross * 2.0 - COMMISSION_MNQ
                )
                trades.append(t)
                exit_found = True
                break

            if direction == 1:
                # Check Partial Trigger
                if not partial_hit and bar['High'] >= partial_tp:
                    partial_hit = True
                    current_sl = entry_price + 0.50  # move to breakeven + buffer

                # Check Stop
                if bar['Low'] <= current_sl:
                    exit_p = current_sl
                    if partial_hit:
                        gross = 0.5 * (partial_r * risk_pts) + 0.5 * (exit_p - entry_price)
                        reason = 'PARTIAL_BE'
                    else:
                        gross = exit_p - entry_price
                        reason = 'STOP_LOSS'
                    t = CalibratedTrade(
                        df.index[bar_idx], direction, entry_price, stop_loss, take_profit, risk_pts,
                        b_time, exit_p, reason, gross, gross / risk_pts,
                        gross * 20.0 - COMMISSION_NQ, gross * 2.0 - COMMISSION_MNQ
                    )
                    trades.append(t)
                    exit_found = True
                    break

                # Check Full TP
                if bar['High'] >= take_profit:
                    exit_p = take_profit
                    if partial_hit:
                        gross = 0.5 * (partial_r * risk_pts) + 0.5 * (target_r * risk_pts)
                    else:
                        gross = target_r * risk_pts
                    t = CalibratedTrade(
                        df.index[bar_idx], direction, entry_price, stop_loss, take_profit, risk_pts,
                        b_time, exit_p, 'FULL_TP', gross, gross / risk_pts,
                        gross * 20.0 - COMMISSION_NQ, gross * 2.0 - COMMISSION_MNQ
                    )
                    trades.append(t)
                    exit_found = True
                    break

            else:  # direction == -1
                if not partial_hit and bar['Low'] <= partial_tp:
                    partial_hit = True
                    current_sl = entry_price - 0.50

                if bar['High'] >= current_sl:
                    exit_p = current_sl
                    if partial_hit:
                        gross = 0.5 * (partial_r * risk_pts) + 0.5 * (entry_price - exit_p)
                        reason = 'PARTIAL_BE'
                    else:
                        gross = entry_price - exit_p
                        reason = 'STOP_LOSS'
                    t = CalibratedTrade(
                        df.index[bar_idx], direction, entry_price, stop_loss, take_profit, risk_pts,
                        b_time, exit_p, reason, gross, gross / risk_pts,
                        gross * 20.0 - COMMISSION_NQ, gross * 2.0 - COMMISSION_MNQ
                    )
                    trades.append(t)
                    exit_found = True
                    break

                if bar['Low'] <= take_profit:
                    exit_p = take_profit
                    if partial_hit:
                        gross = 0.5 * (partial_r * risk_pts) + 0.5 * (target_r * risk_pts)
                    else:
                        gross = target_r * risk_pts
                    t = CalibratedTrade(
                        df.index[bar_idx], direction, entry_price, stop_loss, take_profit, risk_pts,
                        b_time, exit_p, 'FULL_TP', gross, gross / risk_pts,
                        gross * 20.0 - COMMISSION_NQ, gross * 2.0 - COMMISSION_MNQ
                    )
                    trades.append(t)
                    exit_found = True
                    break

    _print_calibrated_summary(csv_path, trades, target_r, partial_r)


def _print_calibrated_summary(csv_path: str, trades: List[CalibratedTrade], target_r: float, partial_r: float):
    if not trades:
        print(f"\n{csv_path}: No trades generated.")
        return

    n = len(trades)
    full_tp = [t for t in trades if t.exit_reason == 'FULL_TP']
    part_be = [t for t in trades if t.exit_reason == 'PARTIAL_BE']
    stopped = [t for t in trades if t.exit_reason == 'STOP_LOSS']
    time_ex = [t for t in trades if t.exit_reason == 'TIME_EXIT']

    # Total profitable trades = Full TP + Partial BE
    profitable_trades = len(full_tp) + len(part_be)
    win_rate = (profitable_trades / n) * 100

    gw = sum(t.gross_pts * 20.0 for t in trades if t.gross_pts > 0)
    gl = abs(sum(t.gross_pts * 20.0 for t in trades if t.gross_pts <= 0))
    pf = gw / gl if gl > 0 else float('inf')

    net_nq = sum(t.net_nq for t in trades)
    avg_r = np.mean([t.realized_r for t in trades])
    avg_stop = np.mean([t.risk_pts for t in trades])

    cum_nq = pd.Series([t.net_nq for t in trades]).cumsum()
    max_dd = (cum_nq.cummax() - cum_nq).max() if not cum_nq.empty else 0

    print(f"\n{'='*70}")
    print(f"  RESULTS: {csv_path}")
    print(f"  Target: {target_r}R (Full) | {partial_r}R (50% Partial + Breakeven)")
    print(f"{'='*70}")
    print(f"  Total Trades             : {n}")
    print(f"  Profitable Trades        : {profitable_trades} ({win_rate:.1f}%)")
    print(f"    - Full TP ({target_r}R)       : {len(full_tp)} ({len(full_tp)/n*100:.1f}%)")
    print(f"    - Partial Taken + BE   : {len(part_be)} ({len(part_be)/n*100:.1f}%)")
    print(f"  Full Losses (SL)         : {len(stopped)} ({len(stopped)/n*100:.1f}%)")
    print(f"  Time / EOD Exits         : {len(time_ex)} ({len(time_ex)/n*100:.1f}%)")
    print(f"  --------------------------------------------------")
    print(f"  Profit Factor            : {pf:.2f}")
    print(f"  Average Realized R       : {avg_r:+.2f}R")
    print(f"  Average Stop Size        : {avg_stop:.1f} pts (${avg_stop*20:.0f} on NQ)")
    print(f"  Net P&L (1 Mini NQ)      : ${net_nq:+,.2f}")
    print(f"  Max Drawdown (1 Mini NQ) : ${max_dd:,.2f}")

    # FundedNext $50K challenge check
    bal = 50000.0
    peak = 50000.0
    mdd = 0.0
    passed = False
    pass_trade = None

    for i, t in enumerate(trades):
        bal += t.net_nq
        peak = max(peak, bal)
        dd = peak - bal
        mdd = max(mdd, dd)
        if bal >= 52500.0 and not passed:
            passed = True
            pass_trade = i + 1

    print(f"  FundedNext $50K ($2,500) : {'PASSED' if passed else 'NOT PASSED'}")
    if passed:
        print(f"    -> Passed on Trade #{pass_trade} | Max DD seen: ${mdd:,.2f}")
    print(f"  Final Account Balance    : ${bal:,.2f}")


if __name__ == '__main__':
    run_calibrated_model('sample_data/MNQ_1min_20260120_20260415.csv', target_r=2.2, partial_r=1.5)
    run_calibrated_model('sample_data/MNQ_1min_12mo.csv', target_r=2.2, partial_r=1.5)

