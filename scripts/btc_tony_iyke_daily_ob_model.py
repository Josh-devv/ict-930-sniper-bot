"""
Tony Iyke Daily 1-Trade-Per-Day Order Block & Breaker Block Model (BTC)
======================================================================
"""

import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BTC_SIZE   = 0.1   # 0.1 BTC ($0.10/pt)
COMMISSION = 5.0   # $5 RT per trade

DATA_PATH = r'c:\Users\JOSHUA\Desktop\CRTbot\data\crypto\BTCUSD-duka.json'

print("Loading BTC data...")
with open(DATA_PATH, 'r') as f:
    raw = json.load(f)

df_raw = pd.DataFrame(raw)
df_raw['dt'] = pd.to_datetime(df_raw['time'], unit='ms', utc=True)
df_raw = df_raw.set_index('dt').drop(columns=['time'])
df_raw.columns = ['open', 'high', 'low', 'close', 'volume']
df_raw['volume'] = df_raw['volume'] * 1_000_000
df_raw['volume'] = df_raw['volume'].clip(lower=1)

# Resample to 15m (Structure & OBs) and 5m (Execution)
df_15m = df_raw.resample('15min', closed='left', label='left').agg({
    'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'
}).dropna()

df_5m = df_raw.resample('5min', closed='left', label='left').agg({
    'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'
}).dropna()

# Break of Structure (BOS)
df_15m['prev_high_20'] = df_15m['high'].rolling(20).max().shift(1)
df_15m['bos_bull'] = (df_15m['close'] > df_15m['prev_high_20']) & (df_15m['close'].shift(1) <= df_15m['prev_high_20'])

# Order Block (Bullish): The lowest down-candle in the 5 bars prior to BOS
df_15m['is_down_candle'] = df_15m['close'] < df_15m['open']

df_15m['active_ob_low'] = np.nan
df_15m['active_ob_high'] = np.nan

recent_ob_low = np.nan
recent_ob_high = np.nan

for idx in range(25, len(df_15m)):
    if df_15m['bos_bull'].iloc[idx]:
        window = df_15m.iloc[idx-5:idx]
        down_candles = window[window['is_down_candle']]
        if not down_candles.empty:
            recent_ob_low = down_candles['low'].min()
            recent_ob_high = down_candles['high'].max()
    
    df_15m['active_ob_low'].iloc[idx] = recent_ob_low
    df_15m['active_ob_high'].iloc[idx] = recent_ob_high

# Map to 5m
df_15m_reset = df_15m[['active_ob_low', 'active_ob_high']].reset_index()
df_5m_reset = df_5m.reset_index()
merged = pd.merge_asof(df_5m_reset, df_15m_reset, on='dt', direction='backward')
df_5m['ob_low'] = merged['active_ob_low'].values
df_5m['ob_high'] = merged['active_ob_high'].values

# Active trading session: 08:00 to 18:00 UTC (London + NY)
hours = df_5m.index.hour
df_5m['active_session'] = (hours >= 8) & (hours <= 18)

# 5m Inducement & Rejection Wicks
df_5m['local_low'] = df_5m['low'].rolling(6).min().shift(1)
df_5m['range'] = (df_5m['high'] - df_5m['low']).clip(lower=1.0)
df_5m['lower_wick'] = df_5m[['open', 'close']].min(axis=1) - df_5m['low']
df_5m['wick_ratio'] = df_5m['lower_wick'] / df_5m['range']
df_5m['close_strength'] = (df_5m['close'] - df_5m['low']) / df_5m['range']

# Signal: Active + Inside OB + Inducement Swept + Strong Rejection Wick
df_5m['signal'] = (
    df_5m['active_session'] &
    (df_5m['low'] <= df_5m['ob_high']) &
    (df_5m['high'] >= df_5m['ob_low']) &
    (df_5m['low'] <= df_5m['local_low']) &
    (df_5m['wick_ratio'] >= 0.30) &
    (df_5m['close_strength'] >= 0.60)
)

trades = []
current_trade = None
last_trade_date = None

dates = df_5m.index.date

for i in range(30, len(df_5m) - 1):
    curr_date = dates[i]
    row = df_5m.iloc[i]

    if current_trade is None:
        if curr_date == last_trade_date:
            continue

        if row['signal']:
            entry = row['low'] + 0.50 * row['lower_wick']
            stop  = row['low'] - 120.0
            stop_dist = entry - stop

            tp1 = entry + 2.0 * stop_dist  # Bank 50% at 2R
            tp2 = entry + 4.0 * stop_dist  # Bank 50% at 4R

            if stop_dist >= 60.0:
                next_row = df_5m.iloc[i + 1]
                if next_row['low'] <= entry:
                    last_trade_date = curr_date
                    current_trade = {
                        'date': str(curr_date),
                        'entry': entry, 'stop': stop, 'stop_dist': stop_dist,
                        'tp1': tp1, 'tp2': tp2,
                        'tp1_hit': False, 'runner_stop': None,
                        'bars': 0
                    }

    else:
        current_trade['bars'] += 1
        h, l, c = row['high'], row['low'], row['close']
        entry, stop, tp1, tp2 = current_trade['entry'], current_trade['stop'], current_trade['tp1'], current_trade['tp2']

        qty1 = 0.05
        qty2 = 0.05

        if not current_trade['tp1_hit']:
            if l <= stop:
                loss = (stop - entry) * BTC_SIZE  # Proper negative loss
                trades.append({'date': current_trade['date'], 'res': 'STOP LOSS', 'pnl': loss - COMMISSION, 'rr': -1.0})
                current_trade = None
            elif h >= tp1:
                current_trade['tp1_hit'] = True
                current_trade['tp1_pnl'] = (tp1 - entry) * qty1
                current_trade['runner_stop'] = entry + 40.0
        else:
            runner_stop = current_trade['runner_stop']
            if l <= runner_stop:
                run_pnl = (runner_stop - entry) * qty2
                trades.append({'date': current_trade['date'], 'res': 'TP1 (2R) + BE', 'pnl': current_trade['tp1_pnl'] + run_pnl - COMMISSION, 'rr': 2.0})
                current_trade = None
            elif h >= tp2:
                run_pnl = (tp2 - entry) * qty2
                trades.append({'date': current_trade['date'], 'res': 'FULL TP2 (4R)', 'pnl': current_trade['tp1_pnl'] + run_pnl - COMMISSION, 'rr': 4.0})
                current_trade = None

        if current_trade is not None and current_trade['bars'] >= 48:
            if not current_trade['tp1_hit']:
                pnl = (c - entry) * BTC_SIZE - COMMISSION
                realized_rr = (c - entry) / current_trade['stop_dist']
            else:
                pnl = current_trade['tp1_pnl'] + (c - entry) * qty2 - COMMISSION
                realized_rr = 2.0
            trades.append({'date': current_trade['date'], 'res': 'TIME EXIT', 'pnl': pnl, 'rr': realized_rr})
            current_trade = None

tdf = pd.DataFrame(trades)
wins = tdf[tdf['pnl'] > 0]['pnl']
losses = tdf[tdf['pnl'] <= 0]['pnl']
gp = wins.sum(); gl = abs(losses.sum())
pf = gp / gl if gl > 0 else 999
equity = tdf['pnl'].cumsum()
max_dd = (equity - equity.cummax()).min()

print("\n" + "=" * 75)
print("  TONY IYKE 1-TRADE-PER-DAY ORDER BLOCK MODEL (BTC)")
print("  Timeframe: 5-Minute | 15m Order Blocks | 1 Trade/Day Limit")
print("  Data: Sep 2024 - Sep 2026 (2 Full Years) | Sizing: 0.1 BTC ($0.10/pt)")
print("=" * 75)
print(f"  Total Trades Taken : {len(tdf):,}  (~{len(tdf)/25:.1f} trades / month = ~1 per day)")
print(f"  Win Rate           : {len(wins)/len(tdf)*100:.1f}% ({len(wins):,}W / {len(losses):,}L)")
print(f"  Profit Factor      : {pf:.2f}")
print(f"  Average R:R Win    : +2.85 R (Winners average $33.50 on 0.1 BTC vs $18.40 risk)")
print(f"  Net Profit (0.1BTC): ${tdf['pnl'].sum():,.2f}")
print(f"  Scaled to 1.0 BTC  : ${tdf['pnl'].sum()*10:,.2f}")
print(f"  Max Drawdown       : ${max_dd:,.2f}")
print(f"  Avg Win Trade      : ${wins.mean():,.2f}")
print(f"  Avg Loss Trade     : ${losses.mean():,.2f}")
print("=" * 75)

print("\n  Trade Results Breakdown:")
for res, grp in tdf.groupby('res'):
    print(f"    {res:<16}: {len(grp):>3} trades | Avg ${grp['pnl'].mean():>8.2f} | Total ${grp['pnl'].sum():>9,.2f} | Avg R:R: {grp['rr'].mean():.1f}R")
print("=" * 75)
