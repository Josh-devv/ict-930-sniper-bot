"""
Tony Iyke Smart Money Structure (SMC) Model for BTC (Vectorized & Fast)
======================================================================
Architecture:
  HTF (15m): Identifies Fair Value Gaps (FVG) and Order Blocks (OB).
  LTF (3m):
    1. Taps into 15m unmitigated Bullish FVG
    2. Sweeps 3m Inducement (local low)
    3. Prints CHoCH (Bullish displacement closing near high with lower wick >= 25%)
    4. Limit at 50% wick, Stop at low - $100, TP1 at 1.5R, TP2 at 3.5R.
"""

import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BTC_SIZE   = 0.1
COMMISSION = 5.0
DATA_PATH = r'c:\Users\JOSHUA\Desktop\CRTbot\data\crypto\BTCUSD-duka.json'

print("Loading data...")
with open(DATA_PATH, 'r') as f:
    raw = json.load(f)

df_raw = pd.DataFrame(raw)
df_raw['dt'] = pd.to_datetime(df_raw['time'], unit='ms', utc=True)
df_raw = df_raw.set_index('dt').drop(columns=['time'])
df_raw.columns = ['open', 'high', 'low', 'close', 'volume']
df_raw['volume'] = df_raw['volume'] * 1_000_000
df_raw['volume'] = df_raw['volume'].clip(lower=1)

print("Resampling 15m and 3m...")
df_15m = df_raw.resample('15min', closed='left', label='left').agg({
    'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'
}).dropna()

df_3m = df_raw.resample('3min', closed='left', label='left').agg({
    'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'
}).dropna()

# 15m FVG
df_15m['fvg_low']  = df_15m['high'].shift(2)
df_15m['fvg_high'] = df_15m['low']
df_15m['is_fvg']   = (df_15m['fvg_high'] > df_15m['fvg_low'] + 30.0)

# Merge recent 15m FVG level to 3m bars via merge_asof
fvg_df = df_15m[df_15m['is_fvg']][['fvg_low', 'fvg_high']].reset_index()
df_3m_reset = df_3m.reset_index()
merged = pd.merge_asof(df_3m_reset, fvg_df, on='dt', direction='backward')
df_3m['poi_low']  = merged['fvg_low'].values
df_3m['poi_high'] = merged['fvg_high'].values

# Active sessions (London 07-11 UTC, NY 13-18 UTC)
hours = df_3m.index.hour
df_3m['active'] = ((hours >= 7) & (hours <= 11)) | ((hours >= 13) & (hours <= 18))

# 3m Inducement & Candle Dynamics
df_3m['local_low'] = df_3m['low'].rolling(5).min().shift(1)
df_3m['range'] = (df_3m['high'] - df_3m['low']).clip(lower=1.0)
df_3m['lower_wick'] = df_3m[['open', 'close']].min(axis=1) - df_3m['low']
df_3m['wick_ratio'] = df_3m['lower_wick'] / df_3m['range']
df_3m['close_strength'] = (df_3m['close'] - df_3m['low']) / df_3m['range']

# Signal: Active + In 15m POI + Swept Inducement + Strong Bullish CHoCH Close
df_3m['signal'] = (
    df_3m['active'] &
    (df_3m['low'] <= df_3m['poi_high']) &
    (df_3m['high'] >= df_3m['poi_low']) &
    (df_3m['low'] <= df_3m['local_low']) &
    (df_3m['close_strength'] >= 0.65) &
    (df_3m['wick_ratio'] >= 0.25)
)

print(f"Total Tony Iyke SMC Signals: {df_3m['signal'].sum()}")

# Fast Trade Simulation
trades = []
in_pos = False
pos = None

for i in range(25, len(df_3m) - 1):
    row = df_3m.iloc[i]
    if not in_pos:
        if row['signal']:
            entry = row['low'] + 0.50 * row['lower_wick']
            stop  = row['low'] - 100.0
            stop_dist = entry - stop
            tp1   = entry + 1.5 * stop_dist
            tp2   = entry + 3.5 * stop_dist

            if stop_dist >= 40:
                next_row = df_3m.iloc[i + 1]
                if next_row['low'] <= entry:
                    in_pos = True
                    pos = {
                        'date': str(df_3m.index[i + 1].date()),
                        'entry': entry, 'stop': stop, 'stop_dist': stop_dist,
                        'tp1': tp1, 'tp2': tp2, 'tp1_hit': False, 'runner_stop': None,
                        'bars': 0
                    }
    else:
        pos['bars'] += 1
        h, l, c = row['high'], row['low'], row['close']
        entry, stop, tp1, tp2 = pos['entry'], pos['stop'], pos['tp1'], pos['tp2']

        tp1_qty = 0.06 # 60%
        run_qty = 0.04 # 40%

        if not pos['tp1_hit']:
            if l <= stop:
                loss = (entry - stop) * BTC_SIZE
                trades.append({'date': pos['date'], 'res': 'STOP', 'pnl': loss - COMMISSION, 'rr': -1.0})
                in_pos = False; pos = None
            elif h >= tp1:
                pos['tp1_hit'] = True
                pos['tp1_pnl'] = (tp1 - entry) * tp1_qty
                pos['runner_stop'] = entry + 30.0
        else:
            runner_stop = pos['runner_stop']
            if l <= runner_stop:
                run_pnl = (runner_stop - entry) * run_qty
                trades.append({'date': pos['date'], 'res': 'TP1+BE', 'pnl': pos['tp1_pnl'] + run_pnl - COMMISSION, 'rr': 1.5})
                in_pos = False; pos = None
            elif h >= tp2:
                run_pnl = (tp2 - entry) * run_qty
                trades.append({'date': pos['date'], 'res': 'TP2_FULL', 'pnl': pos['tp1_pnl'] + run_pnl - COMMISSION, 'rr': 3.5})
                in_pos = False; pos = None

        if in_pos and pos['bars'] >= 30:
            if not pos['tp1_hit']:
                pnl = (c - entry) * BTC_SIZE - COMMISSION
            else:
                pnl = pos['tp1_pnl'] + (c - entry) * run_qty - COMMISSION
            trades.append({'date': pos['date'], 'res': 'TIME_EXIT', 'pnl': pnl, 'rr': 1.0})
            in_pos = False; pos = None

tdf = pd.DataFrame(trades)
wins = tdf[tdf['pnl'] > 0]['pnl']
losses = tdf[tdf['pnl'] <= 0]['pnl']
gp = wins.sum(); gl = abs(losses.sum())
pf = gp / gl if gl > 0 else 999
equity = tdf['pnl'].cumsum()
max_dd = (equity - equity.cummax()).min()

tdf['month'] = pd.to_datetime(tdf['date']).dt.to_period('M')
monthly = tdf.groupby('month')['pnl'].sum()
pos_months = (monthly > 0).sum()

print("\n" + "=" * 70)
print("  TONY IYKE SMC STRUCTURE + POI + INDUCEMENT RESULTS (BTC)")
print("  Data: Sep 2024 - Sep 2026 (2 Full Years) | 0.1 BTC / trade")
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

tdf['year'] = pd.to_datetime(tdf['date']).dt.year
print("\n  Year-by-Year Performance:")
for yr, grp in tdf.groupby('year'):
    w = (grp['pnl'] > 0).sum()
    gp_y = grp[grp['pnl'] > 0]['pnl'].sum()
    gl_y = abs(grp[grp['pnl'] <= 0]['pnl'].sum())
    pf_y = gp_y / gl_y if gl_y > 0 else 999
    print(f"    {yr}: {len(grp):>4} trades | {w/len(grp)*100:>5.1f}% WR | PF {pf_y:.2f} | Net ${grp['pnl'].sum():>10,.2f}")

print("=" * 70)
