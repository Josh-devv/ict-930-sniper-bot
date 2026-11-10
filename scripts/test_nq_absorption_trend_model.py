"""
NQ Institutional Absorption + HTF Trend Filter Engine (8.8-Year CME Continuous Data)
====================================================================================
Adapting the proven BTC Absorption Initiative to NQ / MNQ Futures:
  - 1H 50 EMA Macro Trend Filter (Gating Direction)
  - 20-bar Local Liquidity Sweep + Volume Climax + >= 30% Rejection Wick
  - Testing Direct Entry (Market at Candle Close) vs 50% Retest Entry
  - Testing 2m, 3m, 5m, and 15m timeframes
  - Strict Institutional Accounting: Hard stops first, $5.20 NQ commission, 1-tick slippage.
"""

import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
from typing import List

def p(msg):
    print(msg)
    sys.stdout.flush()

p("=" * 80)
p("NQ ABSORPTION + HTF TREND INITIATIVE BACKTEST (2018 - 2026)")
p("=" * 80)

DATA_PATH = r'sample_data\NQ_1min_RTH_2018_2026.csv'
p(f"Loading continuous NQ data from {DATA_PATH}...")

df_raw = pd.read_csv(DATA_PATH)
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
df_raw = df_raw.set_index('datetime').sort_index()

p(f"Loaded {len(df_raw):,} 1-minute bars: {df_raw.index[0]} to {df_raw.index[-1]}")

# ── COMPUTE 1-HOUR HTF TREND (1H 50 EMA) ───────────────────────────────────────
df_1h = df_raw.resample('1h', label='left', closed='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna(subset=['open'])

df_1h['ema50'] = df_1h['close'].ewm(span=50, adjust=False).mean()
df_1h['ema20'] = df_1h['close'].ewm(span=20, adjust=False).mean()

# Shift 1H values by 1 to ensure zero future lookahead
df_1h_shifted = df_1h.shift(1)

@dataclass
class Trade:
    date: str
    direction: str
    tf: str
    entry_mode: str
    entry_px: float
    stop_px: float
    tp1_px: float
    tp2_px: float
    exit_px: float
    result: str
    pnl: float
    risk_pts: float
    year: int

def run_nq_absorption_test(tf_str='5min', entry_mode='direct', trend_filter='ema50', rr_target=1.5):
    # Resample to target timeframe
    df = df_raw.resample(tf_str, label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open'])
    
    # Map HTF trend to LTF bars (using shifted 1H bar to prevent lookahead)
    df['htf_ema'] = df_1h_shifted[trend_filter].reindex(df.index, method='ffill')
    df['htf_close'] = df_1h_shifted['close'].reindex(df.index, method='ffill')
    
    df['htf_bull'] = df['close'] > df['htf_ema']
    df['htf_bear'] = df['close'] < df['htf_ema']
    
    # Session filter: RTH 09:30 - 15:45 EST
    hours = df.index.hour
    mins = df.index.minute
    time_val = hours * 60 + mins
    df['rth_active'] = (time_val >= 9 * 60 + 35) & (time_val <= 15 * 60 + 45)
    
    # Indicators
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_climax'] = df['volume'] >= (df['vol_ma'] * 1.3)
    
    df['local_high'] = df['high'].rolling(20).max().shift(1)
    df['local_low'] = df['low'].rolling(20).min().shift(1)
    
    df['range'] = (df['high'] - df['low']).clip(lower=0.5)
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
        df['rth_active'] &
        df['vol_climax'] &
        (df['low'] <= df['local_low']) &
        (df['lower_wick_ratio'] >= 0.30) &
        df['htf_bull']
    )
    
    df['bear_absorption'] = (
        df['rth_active'] &
        df['vol_climax'] &
        (df['high'] >= df['local_high']) &
        (df['upper_wick_ratio'] >= 0.30) &
        df['htf_bear']
    )
    
    # Simulation
    POINT_VAL = 20.0     # $20 per point on Mini NQ ($2 per point on Micro MNQ = 1/10th)
    COMMISSION = 5.20    # $5.20 RT
    SLIPPAGE = 5.00      # 1 tick slippage ($5)
    
    trades: List[Trade] = []
    in_pos = False
    pos = None
    
    vals = df.reset_index()
    n = len(vals)
    
    for i in range(25, n - 2):
        row = vals.iloc[i]
        
        if not in_pos:
            if row['bull_absorption']:
                if entry_mode == 'direct':
                    entry = row['close']
                else:
                    entry = row['low'] + 0.50 * row['lower_wick']
                
                stop = row['low'] - 1.0 * row['atr']
                stop_dist = entry - stop
                
                if stop_dist < 6.0:  # Min stop distance 6 pts
                    continue
                
                tp1 = entry + rr_target * stop_dist
                tp2 = entry + (rr_target * 2.0) * stop_dist
                
                if entry_mode == 'direct':
                    filled = True
                    fill_bar = i + 1
                else:
                    filled = False
                    fill_bar = -1
                    for k in range(i + 1, min(n - 1, i + 12)):
                        if vals.iloc[k]['low'] <= entry:
                            filled = True
                            fill_bar = k
                            break
                        if vals.iloc[k]['high'] >= tp1:
                            break
                
                if not filled or fill_bar >= n - 1:
                    continue
                
                in_pos = True
                pos = {
                    'date': str(row['datetime'].date()),
                    'year': row['datetime'].year,
                    'dir': 'long',
                    'entry': entry,
                    'stop': stop,
                    'stop_dist': stop_dist,
                    'tp1': tp1,
                    'tp2': tp2,
                    'bars_held': 0
                }
            
            elif row['bear_absorption']:
                if entry_mode == 'direct':
                    entry = row['close']
                else:
                    entry = row['high'] - 0.50 * row['upper_wick']
                
                stop = row['high'] + 1.0 * row['atr']
                stop_dist = stop - entry
                
                if stop_dist < 6.0:
                    continue
                
                tp1 = entry - rr_target * stop_dist
                tp2 = entry - (rr_target * 2.0) * stop_dist
                
                if entry_mode == 'direct':
                    filled = True
                    fill_bar = i + 1
                else:
                    filled = False
                    fill_bar = -1
                    for k in range(i + 1, min(n - 1, i + 12)):
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
                    'date': str(row['datetime'].date()),
                    'year': row['datetime'].year,
                    'dir': 'short',
                    'entry': entry,
                    'stop': stop,
                    'stop_dist': stop_dist,
                    'tp1': tp1,
                    'tp2': tp2,
                    'bars_held': 0
                }
        
        else:
            pos['bars_held'] += 1
            h, l, c = row['high'], row['low'], row['close']
            is_long = pos['dir'] == 'long'
            
            # HARD STOPS EVALUATED FIRST
            if is_long:
                if l <= pos['stop']:
                    pnl = -(pos['stop_dist'] * POINT_VAL) - COMMISSION - SLIPPAGE
                    trades.append(Trade(
                        date=pos['date'], direction='long', tf=tf_str, entry_mode=entry_mode,
                        entry_px=pos['entry'], stop_px=pos['stop'], tp1_px=pos['tp1'], tp2_px=pos['tp2'],
                        exit_px=pos['stop'], result='STOP', pnl=pnl, risk_pts=pos['stop_dist'], year=pos['year']
                    ))
                    in_pos = False; pos = None; continue
                
                if h >= pos['tp1']:
                    pnl = (pos['tp1'] - pos['entry']) * POINT_VAL - COMMISSION - SLIPPAGE
                    trades.append(Trade(
                        date=pos['date'], direction='long', tf=tf_str, entry_mode=entry_mode,
                        entry_px=pos['entry'], stop_px=pos['stop'], tp1_px=pos['tp1'], tp2_px=pos['tp2'],
                        exit_px=pos['tp1'], result='TP1', pnl=pnl, risk_pts=pos['stop_dist'], year=pos['year']
                    ))
                    in_pos = False; pos = None; continue
            else:
                if h >= pos['stop']:
                    pnl = -(pos['stop_dist'] * POINT_VAL) - COMMISSION - SLIPPAGE
                    trades.append(Trade(
                        date=pos['date'], direction='short', tf=tf_str, entry_mode=entry_mode,
                        entry_px=pos['entry'], stop_px=pos['stop'], tp1_px=pos['tp1'], tp2_px=pos['tp2'],
                        exit_px=pos['stop'], result='STOP', pnl=pnl, risk_pts=pos['stop_dist'], year=pos['year']
                    ))
                    in_pos = False; pos = None; continue
                
                if l <= pos['tp1']:
                    pnl = (pos['entry'] - pos['tp1']) * POINT_VAL - COMMISSION - SLIPPAGE
                    trades.append(Trade(
                        date=pos['date'], direction='short', tf=tf_str, entry_mode=entry_mode,
                        entry_px=pos['entry'], stop_px=pos['stop'], tp1_px=pos['tp1'], tp2_px=pos['tp2'],
                        exit_px=pos['tp1'], result='TP1', pnl=pnl, risk_pts=pos['stop_dist'], year=pos['year']
                    ))
                    in_pos = False; pos = None; continue
            
            # EOD exit (bar 15:55)
            if row['datetime'].hour == 15 and row['datetime'].minute >= 55:
                if is_long:
                    pnl = (c - pos['entry']) * POINT_VAL - COMMISSION - SLIPPAGE
                else:
                    pnl = (pos['entry'] - c) * POINT_VAL - COMMISSION - SLIPPAGE
                trades.append(Trade(
                    date=pos['date'], direction=pos['dir'], tf=tf_str, entry_mode=entry_mode,
                    entry_px=pos['entry'], stop_px=pos['stop'], tp1_px=pos['tp1'], tp2_px=pos['tp2'],
                    exit_px=c, result='EOD_EXIT', pnl=pnl, risk_pts=pos['stop_dist'], year=pos['year']
                ))
                in_pos = False; pos = None
    
    return trades

def print_summary(trades: List[Trade], title: str):
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
    
    # Yearly
    yearly = tdf.groupby('year')['pnl'].sum()
    green_years = (yearly > 0).sum()
    
    p(f"\n{'='*75}")
    p(f"  {title}")
    p(f"{'='*75}")
    p(f"  Total Trades:     {n} ({n/8.8:.1f} trades/year, ~{n/(8.8*50):.1f}/week)")
    p(f"  Win Rate:         {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    p(f"  Profit Factor:    {pf:.2f}")
    p(f"  Net Profit (1 NQ): ${net:,.2f} (1 MNQ: ${net/10:,.2f})")
    p(f"  Avg Win / Loss:   +${wins['pnl'].mean():,.2f} / -${abs(losses['pnl'].mean()):,.2f}")
    p(f"  Avg Stop Loss:    {tdf['risk_pts'].mean():.1f} pts (${tdf['risk_pts'].mean()*20:,.0f} NQ / ${tdf['risk_pts'].mean()*2:,.0f} MNQ)")
    p(f"  Green Years:      {green_years}/{len(yearly)}")
    
    p("\n  Year-by-Year Performance:")
    for yr, pnl_val in yearly.items():
        marker = "✅" if pnl_val > 0 else "❌"
        yt = tdf[tdf['year'] == yr]
        y_wr = (yt['pnl'] > 0).sum() / len(yt) * 100
        p(f"    {yr}: {marker} ${pnl_val:>+10,.2f} (1 MNQ: ${pnl_val/10:>+7,.2f}) | {len(yt):>3} trades | {y_wr:.0f}% WR")

# ── RUN COMBINATIONS ──────────────────────────────────────────────────────────

tests = [
    # 5-Minute Tests
    ("1. 5-Minute DIRECT ENTRY + 1H 50 EMA Filter", dict(tf_str='5min', entry_mode='direct', trend_filter='ema50', rr_target=1.5)),
    ("2. 5-Minute 50% RETEST + 1H 50 EMA Filter", dict(tf_str='5min', entry_mode='retest_50', trend_filter='ema50', rr_target=1.5)),
    ("3. 5-Minute DIRECT ENTRY + 1H 20 EMA Filter", dict(tf_str='5min', entry_mode='direct', trend_filter='ema20', rr_target=1.5)),
    
    # 3-Minute Tests
    ("4. 3-Minute DIRECT ENTRY + 1H 50 EMA Filter", dict(tf_str='3min', entry_mode='direct', trend_filter='ema50', rr_target=1.5)),
    ("5. 3-Minute 50% RETEST + 1H 50 EMA Filter", dict(tf_str='3min', entry_mode='retest_50', trend_filter='ema50', rr_target=1.5)),
    
    # 2-Minute Tests
    ("6. 2-Minute DIRECT ENTRY + 1H 50 EMA Filter", dict(tf_str='2min', entry_mode='direct', trend_filter='ema50', rr_target=1.5)),
    ("7. 2-Minute 50% RETEST + 1H 50 EMA Filter", dict(tf_str='2min', entry_mode='retest_50', trend_filter='ema50', rr_target=1.5)),
    
    # 15-Minute Test
    ("8. 15-Minute DIRECT ENTRY + 1H 50 EMA Filter", dict(tf_str='15min', entry_mode='direct', trend_filter='ema50', rr_target=1.5)),
]

for title, kwargs in tests:
    trades = run_nq_absorption_test(**kwargs)
    print_summary(trades, title)

p("\n" + "=" * 80)
p("ALL NQ ABSORPTION TESTS COMPLETE")
p("=" * 80)

