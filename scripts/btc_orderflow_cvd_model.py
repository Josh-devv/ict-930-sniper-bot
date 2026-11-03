"""
BTC Orderflow & CVD Delta Exhaustion / Absorption Model (Fixed Volume Scaling)
=============================================================================
"""

import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

TF_MIN      = 5
VOL_MULT    = 1.4       # Volume Climax threshold (1.4x 20-period SMA)
WICK_RATIO  = 0.30      # Wick absorption ratio >= 30%
BTC_SIZE    = 0.1       # 0.1 BTC (1 MBT contract equivalent)
COMMISSION  = 5.0       # $5 RT commission

DATA_PATH = r'c:\Users\JOSHUA\Desktop\CRTbot\data\crypto\BTCUSD-duka.json'

print("Loading 2-Year BTC data...")
with open(DATA_PATH, 'r') as f:
    raw = json.load(f)

df_raw = pd.DataFrame(raw)
df_raw['dt'] = pd.to_datetime(df_raw['time'], unit='ms', utc=True)
df_raw = df_raw.set_index('dt').drop(columns=['time'])
df_raw.columns = ['open', 'high', 'low', 'close', 'volume']

# Dukascopy BTC volume is in millionths (e.g. 0.0002 BTC). Multiply by 1,000,000 to get clean integer lots
df_raw['volume'] = df_raw['volume'] * 1_000_000
df_raw['volume'] = df_raw['volume'].clip(lower=1)

print(f"Loaded {len(df_raw):,} raw bars | {df_raw.index[0].date()} to {df_raw.index[-1].date()}")

# Resample to 5m
df = df_raw.resample('5min', closed='left', label='left').agg({
    'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'
}).dropna(subset=['open'])

print(f"Resampled to {len(df):,} 5m bars")

# ─── METRICS & FILTERS ───────────────────────────────────────────────────────
hours = df.index.hour
df['active_session'] = ((hours >= 7) & (hours <= 11)) | ((hours >= 13) & (hours <= 18))

# Volume Climax
df['vol_ma'] = df['volume'].rolling(20).mean()
df['vol_climax'] = df['volume'] >= (df['vol_ma'] * VOL_MULT)

# Local 20-period High/Low
df['local_high'] = df['high'].rolling(20).max().shift(1)
df['local_low']  = df['low'].rolling(20).min().shift(1)

# Wicks and Range
df['range'] = (df['high'] - df['low']).clip(lower=1.0)
df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
df['upper_wick_ratio'] = df['upper_wick'] / df['range']
df['lower_wick_ratio'] = df['lower_wick'] / df['range']

# ATR for Stop Loss calculation
tr1 = df['high'] - df['low']
tr2 = (df['high'] - df['close'].shift(1)).abs()
tr3 = (df['low'] - df['close'].shift(1)).abs()
df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['atr'] = df['tr'].rolling(14).mean().bfill()

# Signals
df['bull_absorption'] = (
    df['active_session'] &
    df['vol_climax'] &
    (df['low'] <= df['local_low']) &
    (df['lower_wick_ratio'] >= WICK_RATIO)
)

df['bear_absorption'] = (
    df['active_session'] &
    df['vol_climax'] &
    (df['high'] >= df['local_high']) &
    (df['upper_wick_ratio'] >= WICK_RATIO)
)

print(f"Bull Absorption signals : {df['bull_absorption'].sum()}")
print(f"Bear Absorption signals : {df['bear_absorption'].sum()}")

# ─── BACKTEST SIMULATION ─────────────────────────────────────────────────────
trades = []
in_position = False
current_trade = None

for i in range(25, len(df) - 1):
    row = df.iloc[i]

    if not in_position:
        direction = None
        entry = stop = tp1 = tp2 = None

        if row['bull_absorption']:
            direction = 'long'
            entry = row['low'] + 0.50 * row['lower_wick']
            stop  = row['low'] - 1.0 * row['atr']
            stop_dist = entry - stop
            tp1   = entry + 1.5 * stop_dist
            tp2   = entry + 3.0 * stop_dist

        elif row['bear_absorption']:
            direction = 'short'
            entry = row['high'] - 0.50 * row['upper_wick']
            stop  = row['high'] + 1.0 * row['atr']
            stop_dist = stop - entry
            tp1   = entry - 1.5 * stop_dist
            tp2   = entry - 3.0 * stop_dist

        if direction is not None and stop_dist >= 40:
            next_row = df.iloc[i + 1]
            filled = False
            if direction == 'long' and next_row['low'] <= entry:
                filled = True
            elif direction == 'short' and next_row['high'] >= entry:
                filled = True

            if filled:
                in_position = True
                current_trade = {
                    'entry_time': str(df.index[i + 1]),
                    'date': str(df.index[i + 1].date()),
                    'dir': direction,
                    'entry': entry,
                    'stop': stop,
                    'stop_dist': stop_dist,
                    'tp1': tp1,
                    'tp2': tp2,
                    'tp1_hit': False,
                    'runner_stop': None,
                    'bars_held': 0
                }

    else:
        current_trade['bars_held'] += 1
        h = row['high']
        l = row['low']
        d = current_trade['dir']
        entry = current_trade['entry']
        stop  = current_trade['stop']
        tp1   = current_trade['tp1']
        tp2   = current_trade['tp2']

        tp1_qty = 0.06
        run_qty = 0.04

        if not current_trade['tp1_hit']:
            if d == 'long':
                if l <= stop:
                    loss = (entry - stop) * BTC_SIZE
                    trades.append({'date': current_trade['date'], 'dir': d, 'res': 'STOP', 'pnl': loss - COMMISSION, 'rr': -1.0})
                    in_position = False
                    current_trade = None
                elif h >= tp1:
                    current_trade['tp1_hit'] = True
                    current_trade['tp1_pnl'] = (tp1 - entry) * tp1_qty
                    current_trade['runner_stop'] = entry + 50.0
            else:
                if h >= stop:
                    loss = (entry - stop) * BTC_SIZE
                    trades.append({'date': current_trade['date'], 'dir': d, 'res': 'STOP', 'pnl': loss - COMMISSION, 'rr': -1.0})
                    in_position = False
                    current_trade = None
                elif l <= tp1:
                    current_trade['tp1_hit'] = True
                    current_trade['tp1_pnl'] = (entry - tp1) * tp1_qty
                    current_trade['runner_stop'] = entry - 50.0

        else:
            runner_stop = current_trade['runner_stop']
            tp1_pnl = current_trade['tp1_pnl']

            if d == 'long':
                if l <= runner_stop:
                    run_pnl = (runner_stop - entry) * run_qty
                    trades.append({'date': current_trade['date'], 'dir': d, 'res': 'TP1+BE', 'pnl': tp1_pnl + run_pnl - COMMISSION, 'rr': 1.5})
                    in_position = False
                    current_trade = None
                elif h >= tp2:
                    run_pnl = (tp2 - entry) * run_qty
                    trades.append({'date': current_trade['date'], 'dir': d, 'res': 'TP2_FULL', 'pnl': tp1_pnl + run_pnl - COMMISSION, 'rr': 3.0})
                    in_position = False
                    current_trade = None
            else:
                if h >= runner_stop:
                    run_pnl = (entry - runner_stop) * run_qty
                    trades.append({'date': current_trade['date'], 'dir': d, 'res': 'TP1+BE', 'pnl': tp1_pnl + run_pnl - COMMISSION, 'rr': 1.5})
                    in_position = False
                    current_trade = None
                elif l <= tp2:
                    run_pnl = (entry - tp2) * run_qty
                    trades.append({'date': current_trade['date'], 'dir': d, 'res': 'TP2_FULL', 'pnl': tp1_pnl + run_pnl - COMMISSION, 'rr': 3.0})
                    in_position = False
                    current_trade = None

        if in_position and current_trade['bars_held'] >= 24:
            c = row['close']
            if not current_trade['tp1_hit']:
                pnl = (c - entry)*BTC_SIZE if d == 'long' else (entry - c)*BTC_SIZE
                trades.append({'date': current_trade['date'], 'dir': d, 'res': 'TIME_EXIT', 'pnl': pnl - COMMISSION, 'rr': pnl/current_trade['stop_dist']})
            else:
                run_pnl = (c - entry)*run_qty if d == 'long' else (entry - c)*run_qty
                trades.append({'date': current_trade['date'], 'dir': d, 'res': 'TP1+TIME', 'pnl': current_trade['tp1_pnl'] + run_pnl - COMMISSION, 'rr': 1.5})
            in_position = False
            current_trade = None

# ─── PERFORMANCE REPORT ──────────────────────────────────────────────────────
tdf = pd.DataFrame(trades)

if tdf.empty:
    print("No trades executed.")
else:
    wins   = tdf[tdf['pnl'] > 0]['pnl']
    losses = tdf[tdf['pnl'] <= 0]['pnl']
    gp = wins.sum(); gl = abs(losses.sum())
    pf = gp / gl if gl > 0 else 999
    equity = tdf['pnl'].cumsum()
    max_dd = (equity - equity.cummax()).min()

    tdf['month'] = pd.to_datetime(tdf['date']).dt.to_period('M')
    monthly = tdf.groupby('month')['pnl'].sum()
    pos_months = (monthly > 0).sum()

    print("\n" + "=" * 70)
    print("  BTC ORDERFLOW & CVD DELTA ABSORPTION STRATEGY (V2)")
    print("  London + NY Active Sessions | 0.1 BTC / trade ($0.10/pt)")
    print("  Data: Sep 2024 - Sep 2026 (2 Full Years)")
    print("=" * 70)
    print(f"  Total Trades Taken : {len(tdf):,}  (~{len(tdf)/25:.1f} trades / month)")
    print(f"  Win Rate           : {len(wins)/len(tdf)*100:.1f}% ({len(wins):,}W / {len(losses):,}L)")
    print(f"  Profit Factor      : {pf:.2f}")
    print(f"  Net Profit (0.1BTC): ${tdf['pnl'].sum():,.2f}")
    print(f"  Scaled to 1.0 BTC  : ${tdf['pnl'].sum()*10:,.2f}")
    print(f"  Max Drawdown       : ${max_dd:,.2f}")
    print(f"  Avg Win Trade      : ${wins.mean():,.2f}")
    print(f"  Avg Loss Trade     : ${losses.mean():,.2f}")
    print(f"  Win / Loss Ratio   : {abs(wins.mean()/losses.mean()):.2f} : 1")
    print(f"  Profitable Months  : {pos_months} / {len(monthly)} ({pos_months/len(monthly)*100:.1f}%)")
    print("=" * 70)

    print("\n  Trade Results Breakdown:")
    for res, grp in tdf.groupby('res'):
        print(f"    {res:<12}: {len(grp):>4} trades | Avg ${grp['pnl'].mean():>8.2f} | Total ${grp['pnl'].sum():>10,.2f}")

    print("\n  Direction Breakdown:")
    for d, grp in tdf.groupby('dir'):
        w = (grp['pnl'] > 0).sum()
        print(f"    {d.upper():<5}: {len(grp):>4} trades | {w/len(grp)*100:>5.1f}% WR | Net ${grp['pnl'].sum():>10,.2f}")

    tdf['year'] = pd.to_datetime(tdf['date']).dt.year
    print("\n  Year-by-Year Performance:")
    for yr, grp in tdf.groupby('year'):
        w = (grp['pnl'] > 0).sum()
        gp_y = grp[grp['pnl'] > 0]['pnl'].sum()
        gl_y = abs(grp[grp['pnl'] <= 0]['pnl'].sum())
        pf_y = gp_y / gl_y if gl_y > 0 else 999
        print(f"    {yr}: {len(grp):>4} trades | {w/len(grp)*100:>5.1f}% WR | PF {pf_y:.2f} | Net ${grp['pnl'].sum():>10,.2f}")

    print("=" * 70)
