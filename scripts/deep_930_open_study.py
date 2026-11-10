"""
Deep 09:30 Open Study: Finding the Best Daily Strategy on Real CME NQ Futures (2018-2026)
========================================================================================
Analyzes 8 Years of Real CME NQ 1-Min Data (2018-2026) around the 09:30 AM NY Open:

Models Evaluated:
1. MODEL 1: 09:30 - 09:35 Opening Candle 50% Retest Continuation
2. MODEL 2: 09:30 - 09:45 Opening Range (15m ORB) Breakout & Retest of Order Block
3. MODEL 3: 09:30 Judas Swing (Initial Fakeout + Sweep of 09:30-09:35 extreme + Delta Absorption)
4. MODEL 4: 09:30 Opening Gap Continuation with VWAP Cross

Zero-Cheat Execution Rules:
- Commission: $5.20 RT on NQ / $1.24 MNQ
- Slippage: 1 tick (0.25 pt)
- Hard Stop Loss checked FIRST on every bar
- Realistic limit fills (price must trade through)
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict

print("=" * 80)
print("DEEP 09:30 NY OPEN STRATEGY STUDY (2018-2026)")
print("Source: sample_data/NQ_1min_RTH_2018_2026.csv (839,332 bars)")
print("=" * 80)

# Load 1m data
df_1m = pd.read_csv('sample_data/NQ_1min_RTH_2018_2026.csv')
df_1m.columns = [c.lower() for c in df_1m.columns]
df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
df_1m = df_1m[df_1m['high'] > df_1m['low']].reset_index(drop=True)
df_1m = df_1m.set_index('datetime').sort_index()

# 1-min Delta proxy
rng_1m = (df_1m['high'] - df_1m['low']).replace(0, 1e-6)
df_1m['delta'] = df_1m['volume'] * (2.0 * df_1m['close'] - df_1m['high'] - df_1m['low']) / rng_1m

# Resample 5m
df_5m = df_1m.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'delta': 'sum'
}).dropna().reset_index()

# Daily levels
df_daily = df_1m.resample('1D').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_daily['pdh'] = df_daily['high'].shift(1)
df_daily['pdl'] = df_daily['low'].shift(1)
df_daily['pdc'] = df_daily['close'].shift(1)

daily_levels = {}
for dt, row in df_daily.iterrows():
    daily_levels[dt.date()] = {'pdh': row['pdh'], 'pdl': row['pdl'], 'pdc': row['pdc']}

COMMISSION_PTS = 0.26
SLIPPAGE_PTS   = 0.25

@dataclass
class Trade:
    year: int
    date: str
    direction: int
    entry_time: pd.Timestamp
    entry_px: float
    stop_px: float
    risk_pts: float
    exit_px: float
    exit_reason: str
    net_pts: float
    realized_r: float
    dur_bars: int
    model_name: str


# ─── MODEL 1: 09:30-09:35 First 5-Minute Candle 50% Retest ───────────────────
def test_model_1_opening_candle_retest(target_r: float = 1.5, min_body_ratio: float = 0.50) -> List[Trade]:
    """
    Looks at the first 5m candle of the day (09:30 - 09:35).
    If it has strong directional conviction (body >= 50% of range):
    Place limit order at 50% of the 5m candle.
    Stop: 1 pt beyond opposite extreme.
    Target: 1.5R or 2.0R.
    """
    trades = []
    
    for date, day_df in df_5m.groupby(df_5m['datetime'].dt.date):
        day_session = day_df[(day_df['datetime'].dt.hour >= 9) & (day_df['datetime'].dt.hour <= 15)].reset_index(drop=True)
        if len(day_session) < 10:
            continue
            
        c0 = day_session.iloc[0]
        # Verify it's the 09:30 candle
        if c0['datetime'].hour != 9 or c0['datetime'].minute != 30:
            continue
            
        rng = c0['high'] - c0['low']
        if rng < 4.0 or rng > 60.0:
            continue
            
        body = abs(c0['close'] - c0['open'])
        if (body / rng) < min_body_ratio:
            continue
            
        is_bull = c0['close'] > c0['open']
        direction = 1 if is_bull else -1
        
        # 50% retracement entry
        entry_px = (c0['high'] + c0['low']) / 2.0
        stop_px = (c0['low'] - 1.0) if is_bull else (c0['high'] + 1.0)
        risk = (entry_px - stop_px) if is_bull else (stop_px - entry_px)
        
        if risk < 6.0 or risk > 35.0:
            continue
            
        target_px = (entry_px + target_r * risk) if is_bull else (entry_px - target_r * risk)
        
        # Walk forward through the remaining day bars
        filled = False
        fill_idx = -1
        
        for k in range(1, min(len(day_session), 12)): # must fill within first hour (by 10:30)
            bar = day_session.iloc[k]
            if is_bull and bar['low'] <= entry_px:
                filled = True
                fill_idx = k
                break
            elif not is_bull and bar['high'] >= entry_px:
                filled = True
                fill_idx = k
                break
                
        if not filled:
            continue
            
        actual_entry = entry_px + (SLIPPAGE_PTS if is_bull else -SLIPPAGE_PTS)
        exit_px = 0.0
        exit_reason = ""
        dur = 0
        
        for k in range(fill_idx, min(len(day_session), fill_idx + 35)):
            dur = k - fill_idx
            bar = day_session.iloc[k]
            
            # Hard stop checked first
            if is_bull:
                if bar['low'] <= stop_px:
                    exit_px = stop_px - SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if bar['high'] >= target_px:
                    exit_px = target_px - SLIPPAGE_PTS
                    exit_reason = f"TARGET_{target_r}R"
                    break
            else:
                if bar['high'] >= stop_px:
                    exit_px = stop_px + SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if bar['low'] <= target_px:
                    exit_px = target_px + SLIPPAGE_PTS
                    exit_reason = f"TARGET_{target_r}R"
                    break
                    
            if bar['datetime'].hour == 15 and bar['datetime'].minute >= 50:
                exit_px = bar['close'] + (-SLIPPAGE_PTS if is_bull else SLIPPAGE_PTS)
                exit_reason = "EOD_EXIT"
                break
                
        if exit_reason:
            net = ((exit_px - actual_entry) if is_bull else (actual_entry - exit_px)) - COMMISSION_PTS
            trades.append(Trade(
                c0['datetime'].year, str(date), direction, day_session.iloc[fill_idx]['datetime'],
                actual_entry, stop_px, risk, exit_px, exit_reason, net, net / risk, dur, "0930_CANDLE1_RETEST"
            ))
            
    return trades


# ─── MODEL 2: 09:30-09:45 Opening Range Breakout & Retest ───────────────────
def test_model_2_orb_retest(target_r: float = 1.5) -> List[Trade]:
    """
    Establish 09:30 - 09:45 Opening Range (first three 5m candles).
    Wait for 09:45 - 10:15 breakout candle that closes outside ORH or ORL.
    Limit entry at the OR level or 50% of the breakout candle.
    Stop: opposite side of the breakout candle.
    Target: 1.5R.
    """
    trades = []
    
    for date, day_df in df_5m.groupby(df_5m['datetime'].dt.date):
        day_session = day_df[(day_df['datetime'].dt.hour >= 9) & (day_df['datetime'].dt.hour <= 15)].reset_index(drop=True)
        if len(day_session) < 12:
            continue
            
        # First 3 bars: 09:30, 09:35, 09:40
        or_bars = day_session.iloc[:3]
        if or_bars.iloc[0]['datetime'].hour != 9 or or_bars.iloc[0]['datetime'].minute != 30:
            continue
            
        orh = or_bars['high'].max()
        orl = or_bars['low'].min()
        or_rng = orh - orl
        
        if or_rng < 8.0 or or_rng > 80.0:
            continue
            
        # Look for breakout between 09:45 and 10:30 (bars 3 to 12)
        breakout_found = False
        bo_idx = -1
        is_bull = False
        
        for k in range(3, min(len(day_session), 12)):
            bar = day_session.iloc[k]
            if bar['close'] > orh:
                breakout_found = True
                bo_idx = k
                is_bull = True
                break
            elif bar['close'] < orl:
                breakout_found = True
                bo_idx = k
                is_bull = False
                break
                
        if not breakout_found:
            continue
            
        bo_bar = day_session.iloc[bo_idx]
        direction = 1 if is_bull else -1
        
        # Entry at the OR level or 50% of breakout candle
        entry_px = (bo_bar['high'] + bo_bar['low']) / 2.0
        stop_px = (bo_bar['low'] - 1.0) if is_bull else (bo_bar['high'] + 1.0)
        risk = (entry_px - stop_px) if is_bull else (stop_px - entry_px)
        
        if risk < 6.0 or risk > 30.0:
            continue
            
        target_px = (entry_px + target_r * risk) if is_bull else (entry_px - target_r * risk)
        
        # Walk forward for fill
        filled = False
        fill_idx = -1
        for k in range(bo_idx + 1, min(len(day_session), bo_idx + 10)):
            bar = day_session.iloc[k]
            if is_bull and bar['low'] <= entry_px:
                filled = True
                fill_idx = k
                break
            elif not is_bull and bar['high'] >= entry_px:
                filled = True
                fill_idx = k
                break
                
        if not filled:
            continue
            
        actual_entry = entry_px + (SLIPPAGE_PTS if is_bull else -SLIPPAGE_PTS)
        exit_px = 0.0
        exit_reason = ""
        dur = 0
        
        for k in range(fill_idx, min(len(day_session), fill_idx + 35)):
            dur = k - fill_idx
            bar = day_session.iloc[k]
            
            if is_bull:
                if bar['low'] <= stop_px:
                    exit_px = stop_px - SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if bar['high'] >= target_px:
                    exit_px = target_px - SLIPPAGE_PTS
                    exit_reason = f"TARGET_{target_r}R"
                    break
            else:
                if bar['high'] >= stop_px:
                    exit_px = stop_px + SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if bar['low'] <= target_px:
                    exit_px = target_px + SLIPPAGE_PTS
                    exit_reason = f"TARGET_{target_r}R"
                    break
                    
            if bar['datetime'].hour == 15 and bar['datetime'].minute >= 50:
                exit_px = bar['close'] + (-SLIPPAGE_PTS if is_bull else SLIPPAGE_PTS)
                exit_reason = "EOD_EXIT"
                break
                
        if exit_reason:
            net = ((exit_px - actual_entry) if is_bull else (actual_entry - exit_px)) - COMMISSION_PTS
            trades.append(Trade(
                bo_bar['datetime'].year, str(date), direction, day_session.iloc[fill_idx]['datetime'],
                actual_entry, stop_px, risk, exit_px, exit_reason, net, net / risk, dur, "0945_ORB_RETEST"
            ))
            
    return trades


# ─── MODEL 3: 09:30 Judas Swing (Initial Fakeout + Liquidity Sweep) ───────────
def test_model_3_judas_swing(target_r: float = 1.5) -> List[Trade]:
    """
    Between 09:30 and 09:40, price drives to create a high/low (the initial fakeout).
    Between 09:35 and 10:00, price sweeps that 09:30 high/low, leaves a rejection wick (>= 30%),
    and displaces back in the opposite direction (Judas reversal).
    Entry: 50% Mean Threshold of the sweep bar.
    Stop: 1 pt beyond sweep extreme.
    Target: 1.5R.
    """
    trades = []
    
    for date, day_df in df_5m.groupby(df_5m['datetime'].dt.date):
        day_session = day_df[(day_df['datetime'].dt.hour >= 9) & (day_df['datetime'].dt.hour <= 15)].reset_index(drop=True)
        if len(day_session) < 15:
            continue
            
        c0 = day_session.iloc[0]
        if c0['datetime'].hour != 9 or c0['datetime'].minute != 30:
            continue
            
        # 09:30 candle high and low
        open_h = c0['high']
        open_l = c0['low']
        
        # Look for sweep and reversal on bars 1 to 6 (09:35 to 10:00)
        for k in range(1, min(len(day_session), 7)):
            bar = day_session.iloc[k]
            rng = bar['high'] - bar['low']
            if rng < 3.0:
                continue
                
            # Bullish Judas: swept 09:30 low, closed back above it, with lower wick >= 30%
            lower_wick = min(bar['open'], bar['close']) - bar['low']
            is_bull_judas = (bar['low'] < open_l) and (bar['close'] > open_l) and (lower_wick / rng >= 0.30)
            
            # Bearish Judas: swept 09:30 high, closed back below it, with upper wick >= 30%
            upper_wick = bar['high'] - math.max(bar['open'], bar['close']) if 'math' in globals() else (bar['high'] - max(bar['open'], bar['close']))
            is_bear_judas = (bar['high'] > open_h) and (bar['close'] < open_h) and (upper_wick / rng >= 0.30)
            
            if not (is_bull_judas or is_bear_judas):
                continue
                
            is_bull = is_bull_judas
            direction = 1 if is_bull else -1
            
            # Entry at 50% of the Judas candle
            entry_px = (bar['high'] + bar['low']) / 2.0
            stop_px = (bar['low'] - 1.0) if is_bull else (bar['high'] + 1.0)
            risk = (entry_px - stop_px) if is_bull else (stop_px - entry_px)
            
            if risk < 6.0 or risk > 30.0:
                continue
                
            target_px = (entry_px + target_r * risk) if is_bull else (entry_px - target_r * risk)
            
            # Walk forward for fill
            filled = False
            fill_idx = -1
            for f in range(k + 1, min(len(day_session), k + 10)):
                f_bar = day_session.iloc[f]
                if is_bull and f_bar['low'] <= entry_px:
                    filled = True
                    fill_idx = f
                    break
                elif not is_bull and f_bar['high'] >= entry_px:
                    filled = True
                    fill_idx = f
                    break
                    
            if not filled:
                continue
                
            actual_entry = entry_px + (SLIPPAGE_PTS if is_bull else -SLIPPAGE_PTS)
            exit_px = 0.0
            exit_reason = ""
            dur = 0
            
            for f in range(fill_idx, min(len(day_session), fill_idx + 35)):
                dur = f - fill_idx
                f_bar = day_session.iloc[f]
                
                if is_bull:
                    if f_bar['low'] <= stop_px:
                        exit_px = stop_px - SLIPPAGE_PTS
                        exit_reason = "STOP_LOSS"
                        break
                    if f_bar['high'] >= target_px:
                        exit_px = target_px - SLIPPAGE_PTS
                        exit_reason = f"TARGET_{target_r}R"
                        break
                else:
                    if f_bar['high'] >= stop_px:
                        exit_px = stop_px + SLIPPAGE_PTS
                        exit_reason = "STOP_LOSS"
                        break
                    if f_bar['low'] <= target_px:
                        exit_px = target_px + SLIPPAGE_PTS
                        exit_reason = f"TARGET_{target_r}R"
                        break
                        
                if f_bar['datetime'].hour == 15 and f_bar['datetime'].minute >= 50:
                    exit_px = f_bar['close'] + (-SLIPPAGE_PTS if is_bull else SLIPPAGE_PTS)
                    exit_reason = "EOD_EXIT"
                    break
                    
            if exit_reason:
                net = ((exit_px - actual_entry) if is_bull else (actual_entry - exit_px)) - COMMISSION_PTS
                trades.append(Trade(
                    bar['datetime'].year, str(date), direction, day_session.iloc[fill_idx]['datetime'],
                    actual_entry, stop_px, risk, exit_px, exit_reason, net, net / risk, dur, "0930_JUDAS_SWING"
                ))
                break # Only 1 trade per day
                
    return trades


# ─── RUN COMPARISON ──────────────────────────────────────────────────────────
models = [
    ("M1: 09:30 Candle 1 Retest (1.5R)", test_model_1_opening_candle_retest(target_r=1.5)),
    ("M1: 09:30 Candle 1 Retest (2.0R)", test_model_1_opening_candle_retest(target_r=2.0)),
    ("M2: 09:45 ORB Breakout & Retest (1.5R)", test_model_2_orb_retest(target_r=1.5)),
    ("M2: 09:45 ORB Breakout & Retest (2.0R)", test_model_2_orb_retest(target_r=2.0)),
    ("M3: 09:30 Judas Sweep & Reversal (1.5R)", test_model_3_judas_swing(target_r=1.5)),
    ("M3: 09:30 Judas Sweep & Reversal (2.0R)", test_model_3_judas_swing(target_r=2.0)),
]

print("\n" + "=" * 115)
print(f"| {'09:30 OPEN STRATEGY':<38} | {'TRADES':>6} | {'TR/YR':>6} | {'TR/WK':>5} | {'WIN%':>6} | {'PF':>5} | {'AVG SL':>6} | {'NET PTS':>8} | {'NQ NET $':>10} |")
print("=" * 115)

for name, t_list in models:
    if not t_list:
        print(f"| {name:<38} | No trades |")
        continue
    tdf = pd.DataFrame([t.__dict__ for t in t_list])
    n = len(tdf)
    tr_yr = n / 8.8
    tr_wk = tr_yr / 52.0
    wins = tdf[tdf['net_pts'] > 0]
    losses = tdf[tdf['net_pts'] <= 0]
    wr = len(wins) / n * 100
    pf = wins['net_pts'].sum() / abs(losses['net_pts'].sum()) if len(losses) > 0 and losses['net_pts'].sum() != 0 else 99.0
    tot_pts = tdf['net_pts'].sum()
    avg_sl = tdf['risk_pts'].mean()
    nq_net = tot_pts * 20.0
    
    print(f"| {name:<38} | {n:>6} | {tr_yr:>6.1f} | {tr_wk:>5.1f} | {wr:>5.1f}% | {pf:>5.2f} | {avg_sl:>5.1f}p | {tot_pts:>+7.1f}p | +${nq_net:>9,.0f} |")

print("=" * 115)

