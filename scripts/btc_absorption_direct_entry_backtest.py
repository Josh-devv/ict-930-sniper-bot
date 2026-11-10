"""
BTC Orderflow Absorption Strategy: Direct Entry vs 50% Retest
Fetches LIVE historical 5-minute BTC data from Binance (free, no API key)
===========================================================================

GOAL: Backtest YOUR exact TradingView/Telegram absorption setup:
  - Volume Climax (>= 1.3x 20-period SMA)
  - Liquidity Sweep (low <= 20-bar local low)
  - 30%+ Lower Rejection Wick (institutional absorption)
  - Active Sessions: London (07-11 UTC) & NY (13-18 UTC)

Two Entry Modes Tested:
  A) DIRECT ENTRY: Market buy at close of absorption candle (how you're trading live)
  B) 50% WICK RETEST: Limit order at 50% of the lower wick (old Python script method)

CORRECTED ACCOUNTING: Stop losses are properly negative.
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime, timezone, timedelta

def p(msg):
    print(msg)
    sys.stdout.flush()
from dataclasses import dataclass
from typing import List

p("=" * 80)
p("BTC ABSORPTION BACKTEST: DIRECT ENTRY vs 50% RETEST (CORRECTED ACCOUNTING)")
p("=" * 80)

# ── LOAD LOCAL DUKASCOPY BTC DATA ─────────────────────────────────────────────
DATA_PATH = r'C:\Users\JOSH\Desktop\2xjosh\CRTbot\data\crypto\BTCUSD-duka.json'

p(f"Loading BTC data from {DATA_PATH}...")
df_raw = pd.read_json(DATA_PATH)
p(f"JSON parsed, {len(df_raw):,} rows loaded")
df_raw['dt'] = pd.to_datetime(df_raw['time'], unit='ms', utc=True)
df_raw = df_raw.set_index('dt').drop(columns=['time'])
df_raw.columns = ['open', 'high', 'low', 'close', 'volume']

# Dukascopy volume scaling
df_raw['volume'] = df_raw['volume'] * 1_000_000
df_raw['volume'] = df_raw['volume'].clip(lower=1)

p(f"Raw bars: {len(df_raw):,} | {df_raw.index[0].date()} to {df_raw.index[-1].date()}")

# Resample to 5m
df = df_raw.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna(subset=['open'])

df = df[~df.index.duplicated(keep='first')]
p(f"Resampled to {len(df):,} 5m bars | {df.index[0].date()} to {df.index[-1].date()}")

# ── COMPUTE INDICATORS ────────────────────────────────────────────────────────
hours = df.index.hour
df['active_session'] = ((hours >= 7) & (hours <= 11)) | ((hours >= 13) & (hours <= 18))

df['vol_ma'] = df['volume'].rolling(20).mean()
df['vol_climax'] = df['volume'] >= (df['vol_ma'] * 1.3)

df['local_high'] = df['high'].rolling(20).max().shift(1)
df['local_low']  = df['low'].rolling(20).min().shift(1)

df['range'] = (df['high'] - df['low']).clip(lower=1.0)
df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
df['upper_wick_ratio'] = df['upper_wick'] / df['range']
df['lower_wick_ratio'] = df['lower_wick'] / df['range']

tr1 = df['high'] - df['low']
tr2 = (df['high'] - df['close'].shift(1)).abs()
tr3 = (df['low'] - df['close'].shift(1)).abs()
df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().bfill()

# Signals
df['bull_absorption'] = (
    df['active_session'] &
    df['vol_climax'] &
    (df['low'] <= df['local_low']) &
    (df['lower_wick_ratio'] >= 0.30)
)

df['bear_absorption'] = (
    df['active_session'] &
    df['vol_climax'] &
    (df['high'] >= df['local_high']) &
    (df['upper_wick_ratio'] >= 0.30)
)

p(f"\nBull Absorption signals: {df['bull_absorption'].sum()}")
p(f"Bear Absorption signals: {df['bear_absorption'].sum()}")

# ── BACKTEST ENGINE (CORRECTED ACCOUNTING) ────────────────────────────────────
BTC_SIZE = 0.1
COMMISSION = 5.0  # $5 RT

@dataclass
class Trade:
    date: str
    direction: str
    entry_mode: str
    entry_px: float
    stop_px: float
    tp1_px: float
    tp2_px: float
    exit_px: float
    result: str
    pnl: float
    risk_pts: float

def run_absorption_backtest(entry_mode: str = 'direct', long_only: bool = True) -> List[Trade]:
    """
    entry_mode: 'direct' = market buy at absorption candle close
                'retest_50' = limit at 50% of lower wick
    """
    trades: List[Trade] = []
    in_pos = False
    pos = None
    
    vals = df.reset_index()
    n = len(vals)
    
    for i in range(25, n - 2):
        row = vals.iloc[i]
        
        if not in_pos:
            # ── BULL ABSORPTION SIGNAL ──
            if row['bull_absorption']:
                lower_wick = row['lower_wick']
                atr = row['atr']
                
                if entry_mode == 'direct':
                    entry = row['close']  # Market buy at candle close
                else:
                    entry = row['low'] + 0.50 * lower_wick  # 50% wick retest
                
                stop = row['low'] - 1.0 * atr
                stop_dist = entry - stop
                
                if stop_dist < 40:
                    continue
                
                tp1 = entry + 1.5 * stop_dist
                tp2 = entry + 3.0 * stop_dist
                
                # Check fill
                if entry_mode == 'direct':
                    # Direct entry: filled immediately on next bar open
                    fill_bar = i + 1
                    filled = True
                else:
                    # Limit order: need price to retrace to entry level
                    filled = False
                    fill_bar = -1
                    for k in range(i + 1, min(n - 1, i + 15)):
                        if vals.iloc[k]['low'] <= entry:
                            filled = True
                            fill_bar = k
                            break
                        if vals.iloc[k]['high'] >= tp1:
                            break  # Ran away
                
                if not filled or fill_bar >= n - 1:
                    continue
                
                in_pos = True
                pos = {
                    'date': str(row['dt'].date()),
                    'entry': entry,
                    'stop': stop,
                    'stop_dist': stop_dist,
                    'tp1': tp1,
                    'tp2': tp2,
                    'tp1_hit': False,
                    'tp1_pnl': 0.0,
                    'runner_stop': None,
                    'fill_bar': fill_bar,
                    'bars_held': 0
                }
            
            # ── BEAR ABSORPTION SIGNAL ──
            elif (not long_only) and row['bear_absorption']:
                upper_wick = row['upper_wick']
                atr = row['atr']
                
                if entry_mode == 'direct':
                    entry = row['close']
                else:
                    entry = row['high'] - 0.50 * upper_wick
                
                stop = row['high'] + 1.0 * atr
                stop_dist = stop - entry
                
                if stop_dist < 40:
                    continue
                
                tp1 = entry - 1.5 * stop_dist
                tp2 = entry - 3.0 * stop_dist
                
                if entry_mode == 'direct':
                    fill_bar = i + 1
                    filled = True
                else:
                    filled = False
                    fill_bar = -1
                    for k in range(i + 1, min(n - 1, i + 15)):
                        if vals.iloc[k]['high'] >= entry:
                            filled = True
                            fill_bar = k
                            break
                        if vals.iloc[k]['low'] <= tp1:
                            break
                
                if not filled or fill_bar >= n - 1:
                    continue
                
                in_pos = True
                pos = {
                    'date': str(row['dt'].date()),
                    'entry': entry,
                    'stop': stop,
                    'stop_dist': stop_dist,
                    'tp1': tp1,
                    'tp2': tp2,
                    'tp1_hit': False,
                    'tp1_pnl': 0.0,
                    'runner_stop': None,
                    'fill_bar': fill_bar,
                    'bars_held': 0,
                    'dir': 'short'
                }
        
        else:
            pos['bars_held'] += 1
            h = row['high']
            l = row['low']
            c = row['close']
            is_long = pos.get('dir', 'long') != 'short'
            
            tp1_qty = 0.06  # 60%
            run_qty = 0.04  # 40%
            
            if not pos['tp1_hit']:
                if is_long:
                    # STOP LOSS: Correctly NEGATIVE
                    if l <= pos['stop']:
                        pnl = -((pos['entry'] - pos['stop']) * BTC_SIZE) - COMMISSION
                        trades.append(Trade(
                            date=pos['date'], direction='long', entry_mode=entry_mode,
                            entry_px=pos['entry'], stop_px=pos['stop'],
                            tp1_px=pos['tp1'], tp2_px=pos['tp2'],
                            exit_px=pos['stop'], result='STOP',
                            pnl=pnl, risk_pts=pos['stop_dist']
                        ))
                        in_pos = False; pos = None; continue
                    elif h >= pos['tp1']:
                        pos['tp1_hit'] = True
                        pos['tp1_pnl'] = (pos['tp1'] - pos['entry']) * tp1_qty
                        pos['runner_stop'] = pos['entry'] + 30.0
                else:
                    if h >= pos['stop']:
                        pnl = -((pos['stop'] - pos['entry']) * BTC_SIZE) - COMMISSION
                        trades.append(Trade(
                            date=pos['date'], direction='short', entry_mode=entry_mode,
                            entry_px=pos['entry'], stop_px=pos['stop'],
                            tp1_px=pos['tp1'], tp2_px=pos['tp2'],
                            exit_px=pos['stop'], result='STOP',
                            pnl=pnl, risk_pts=pos['stop_dist']
                        ))
                        in_pos = False; pos = None; continue
                    elif l <= pos['tp1']:
                        pos['tp1_hit'] = True
                        pos['tp1_pnl'] = (pos['entry'] - pos['tp1']) * tp1_qty
                        pos['runner_stop'] = pos['entry'] - 30.0
            else:
                runner_stop = pos['runner_stop']
                if is_long:
                    if l <= runner_stop:
                        run_pnl = (runner_stop - pos['entry']) * run_qty
                        trades.append(Trade(
                            date=pos['date'], direction='long', entry_mode=entry_mode,
                            entry_px=pos['entry'], stop_px=pos['stop'],
                            tp1_px=pos['tp1'], tp2_px=pos['tp2'],
                            exit_px=runner_stop, result='TP1+BE',
                            pnl=pos['tp1_pnl'] + run_pnl - COMMISSION,
                            risk_pts=pos['stop_dist']
                        ))
                        in_pos = False; pos = None; continue
                    elif h >= pos['tp2']:
                        run_pnl = (pos['tp2'] - pos['entry']) * run_qty
                        trades.append(Trade(
                            date=pos['date'], direction='long', entry_mode=entry_mode,
                            entry_px=pos['entry'], stop_px=pos['stop'],
                            tp1_px=pos['tp1'], tp2_px=pos['tp2'],
                            exit_px=pos['tp2'], result='TP2_FULL',
                            pnl=pos['tp1_pnl'] + run_pnl - COMMISSION,
                            risk_pts=pos['stop_dist']
                        ))
                        in_pos = False; pos = None; continue
                else:
                    if h >= runner_stop:
                        run_pnl = (pos['entry'] - runner_stop) * run_qty
                        trades.append(Trade(
                            date=pos['date'], direction='short', entry_mode=entry_mode,
                            entry_px=pos['entry'], stop_px=pos['stop'],
                            tp1_px=pos['tp1'], tp2_px=pos['tp2'],
                            exit_px=runner_stop, result='TP1+BE',
                            pnl=pos['tp1_pnl'] + run_pnl - COMMISSION,
                            risk_pts=pos['stop_dist']
                        ))
                        in_pos = False; pos = None; continue
                    elif l <= pos['tp2']:
                        run_pnl = (pos['entry'] - pos['tp2']) * run_qty
                        trades.append(Trade(
                            date=pos['date'], direction='short', entry_mode=entry_mode,
                            entry_px=pos['entry'], stop_px=pos['stop'],
                            tp1_px=pos['tp1'], tp2_px=pos['tp2'],
                            exit_px=pos['tp2'], result='TP2_FULL',
                            pnl=pos['tp1_pnl'] + run_pnl - COMMISSION,
                            risk_pts=pos['stop_dist']
                        ))
                        in_pos = False; pos = None; continue
            
            # Time exit after 30 bars
            if pos and pos['bars_held'] >= 30:
                if is_long:
                    if not pos['tp1_hit']:
                        pnl = (c - pos['entry']) * BTC_SIZE - COMMISSION
                    else:
                        pnl = pos['tp1_pnl'] + (c - pos['entry']) * run_qty - COMMISSION
                else:
                    if not pos['tp1_hit']:
                        pnl = (pos['entry'] - c) * BTC_SIZE - COMMISSION
                    else:
                        pnl = pos['tp1_pnl'] + (pos['entry'] - c) * run_qty - COMMISSION
                trades.append(Trade(
                    date=pos['date'], direction='long' if is_long else 'short',
                    entry_mode=entry_mode,
                    entry_px=pos['entry'], stop_px=pos['stop'],
                    tp1_px=pos['tp1'], tp2_px=pos['tp2'],
                    exit_px=c, result='TIME_EXIT',
                    pnl=pnl, risk_pts=pos['stop_dist']
                ))
                in_pos = False; pos = None
    
    return trades

def print_bt_results(trades: List[Trade], title: str):
    if not trades:
        p(f"\n{title}: NO TRADES")
        return
    tdf = pd.DataFrame([t.__dict__ for t in trades])
    n = len(tdf)
    wins = tdf[tdf['pnl'] > 0]
    losses = tdf[tdf['pnl'] <= 0]
    wr = len(wins) / n * 100.0
    gp = wins['pnl'].sum()
    gl = abs(losses['pnl'].sum())
    pf = gp / gl if gl > 0 else 99.0
    net = tdf['pnl'].sum()
    
    # Monthly breakdown
    tdf['month'] = pd.to_datetime(tdf['date']).dt.to_period('M')
    monthly = tdf.groupby('month')['pnl'].sum()
    green_months = (monthly > 0).sum()
    
    p(f"\n{'='*75}")
    p(f"RESULT: {title}")
    p(f"{'='*75}")
    p(f"Total Trades:       {n}")
    p(f"Win Rate:           {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    p(f"Profit Factor:      {pf:.2f}")
    p(f"Net Profit (0.1BTC): ${net:,.2f}")
    p(f"Scaled to 1.0 BTC:  ${net*10:,.2f}")
    p(f"Avg Win:            ${wins['pnl'].mean():,.2f}" if len(wins) > 0 else "")
    p(f"Avg Loss:           ${losses['pnl'].mean():,.2f}" if len(losses) > 0 else "")
    p(f"Green Months:       {green_months}/{len(monthly)}")
    
    p("\n  Trade Outcome Breakdown:")
    for res, grp in tdf.groupby('result'):
        p(f"    {res:<12}: {len(grp):>4} trades | Avg ${grp['pnl'].mean():>8.2f} | Total ${grp['pnl'].sum():>10,.2f}")
    
    p(f"\n  Monthly PnL:")
    for m, pnl in monthly.items():
        marker = "✅" if pnl > 0 else "❌"
        m_trades = tdf[tdf['month'] == m]
        m_wr = (m_trades['pnl'] > 0).sum() / len(m_trades) * 100
        p(f"    {m} : {marker} ${pnl:>+10,.2f} ({len(m_trades)} trades, {m_wr:.0f}% WR)")

# ── RUN ALL CONFIGURATIONS ────────────────────────────────────────────────────

p("\n--- Running Config 1/4: Direct Entry, Long Only ---")
t_direct_long = run_absorption_backtest(entry_mode='direct', long_only=True)
print_bt_results(t_direct_long, "1. DIRECT ENTRY (Market Buy at Absorption Close) - LONG ONLY")

p("\n--- Running Config 2/4: 50% Retest, Long Only ---")
t_retest_long = run_absorption_backtest(entry_mode='retest_50', long_only=True)
print_bt_results(t_retest_long, "2. 50% WICK RETEST (Limit Order) - LONG ONLY")

p("\n--- Running Config 3/4: Direct Entry, Both Directions ---")
t_direct_both = run_absorption_backtest(entry_mode='direct', long_only=False)
print_bt_results(t_direct_both, "3. DIRECT ENTRY (Market Buy/Sell at Absorption Close) - BOTH DIRECTIONS")

p("\n--- Running Config 4/4: 50% Retest, Both Directions ---")
t_retest_both = run_absorption_backtest(entry_mode='retest_50', long_only=False)
print_bt_results(t_retest_both, "4. 50% WICK RETEST (Limit Order) - BOTH DIRECTIONS")

p("\n✅ ALL BACKTESTS COMPLETE")

