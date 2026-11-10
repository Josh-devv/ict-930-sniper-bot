"""
BTC Dynamic Regime-Adaptive Liquidity Sweep Model
=================================================
Architecture:
  - HTF Macro Filter: 4H 200 EMA (calculated on continuous data).
  - Regime Detection:
      * BULLISH REGIME (Price > 4H 200 EMA):
          - Hunt BOTTOM OF THE DAY sweeps (Asian Low / PDL).
          - Ignore shorts to avoid getting steamrolled by bull expansions.
      * BEARISH REGIME (Price < 4H 200 EMA):
          - Hunt TOP OF THE DAY sweeps (Asian High / PDH).
          - Ignore longs to avoid knife-catching during bear liquidations.
  - Signal Mechanics:
      * Sweep key level during London/NY expansion (08:00 - 20:00 UTC).
      * Rejection candle closes BACK INSIDE with wick >= 35%.
      * Entry at 50% wick pullback (CE).
      * Stop: Beyond sweep extreme + $100 buffer.
      * TP1 (50%): Midpoint Equilibrium. Stop moves to Breakeven.
      * TP2 (50%): Opposite Session / Daily Extreme.
  - Sizing: 0.1 BTC ($0.10/pt).

Data: BTCUSD-duka.json (Sep 2024 - Sep 2026, 1-min)
"""

import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

TF_MIN         = 5
MIN_SWEEP_DIST = 50.0
WICK_RATIO     = 0.35
STOP_BUFFER    = 100.0
BTC_SIZE       = 0.1
COMMISSION     = 5.0

DATA_PATH = r'c:\Users\JOSHUA\Desktop\CRTbot\data\crypto\BTCUSD-duka.json'

print("Loading BTC data...")
with open(DATA_PATH, 'r') as f:
    raw = json.load(f)

df_raw = pd.DataFrame(raw)
df_raw['dt'] = pd.to_datetime(df_raw['time'], unit='ms', utc=True)
df_raw = df_raw.set_index('dt').drop(columns=['time'])
df_raw.columns = ['open', 'high', 'low', 'close', 'volume']
df_raw['volume'] = df_raw['volume'].clip(lower=1)

print(f"Loaded {len(df_raw):,} bars | {df_raw.index[0].date()} to {df_raw.index[-1].date()}")

# Resample to 5m
df_5m = df_raw.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna(subset=['open'])

# Resample to 4H for Macro Regime EMA (200 period 4H EMA = 800 hours / 33 days trend)
df_4h = df_raw.resample('4h', closed='left', label='left').agg({'close': 'last'}).dropna()
df_4h['ema200'] = df_4h['close'].ewm(span=200, adjust=False).mean()

# Map 4H EMA back to 5m forward-filled
df_5m['ema200_4h'] = df_4h['ema200'].reindex(df_5m.index, method='ffill')
df_5m['regime'] = np.where(df_5m['close'] >= df_5m['ema200_4h'], 'BULL', 'BEAR')

print(f"Resampled to 5m: {len(df_5m):,} bars")
print(f"Regime breakdown: {df_5m['regime'].value_counts(normalize=True).to_dict()}")

# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────
trades = []
dates = list(df_5m.groupby(df_5m.index.date))

for i in range(1, len(dates)):
    prev_date, prev_day = dates[i - 1]
    curr_date, curr_day = dates[i]

    pdh = prev_day['high'].max()
    pdl = prev_day['low'].min()

    # Asian session: 00:00 - 08:00 UTC
    asia = curr_day.between_time('00:00', '07:59')
    if len(asia) < 10:
        continue

    asia_high = asia['high'].max()
    asia_low  = asia['low'].min()

    # Expansion window: 08:00 - 20:00 UTC
    expansion = curr_day.between_time('08:00', '20:00')
    if len(expansion) < 5:
        continue

    trade_taken_today = False

    for ts in expansion.index:
        if trade_taken_today:
            break

        idx = curr_day.index.get_loc(ts)
        o = curr_day['open'].iloc[idx]
        h = curr_day['high'].iloc[idx]
        l = curr_day['low'].iloc[idx]
        c = curr_day['close'].iloc[idx]
        rng = h - l
        if rng < 20:
            continue

        curr_regime = curr_day['regime'].iloc[idx]
        direction = None
        entry = stop = tp1 = tp2 = None

        # ── REGIME 1: BULLISH (ONLY BOTTOM SWEEPS ALLOWED) ──
        if curr_regime == 'BULL':
            swept_bot = None
            if l < asia_low - MIN_SWEEP_DIST and c > asia_low:
                swept_bot = asia_low
            elif l < pdl - MIN_SWEEP_DIST and c > pdl:
                swept_bot = pdl

            if swept_bot is not None:
                wick = min(o, c) - l
                if wick / rng >= WICK_RATIO:
                    direction   = 'long'
                    entry       = l + 0.50 * wick
                    stop        = l - STOP_BUFFER
                    target_high = max(asia_high, pdh)
                    tp1         = (entry + target_high) / 2.0
                    tp2         = target_high

        # ── REGIME 2: BEARISH (ONLY TOP SWEEPS ALLOWED) ──
        elif curr_regime == 'BEAR':
            swept_top = None
            if h > asia_high + MIN_SWEEP_DIST and c < asia_high:
                swept_top = asia_high
            elif h > pdh + MIN_SWEEP_DIST and c < pdh:
                swept_top = pdh

            if swept_top is not None:
                wick = h - max(o, c)
                if wick / rng >= WICK_RATIO:
                    direction   = 'short'
                    entry       = h - 0.50 * wick
                    stop        = h + STOP_BUFFER
                    target_low  = min(asia_low, pdl)
                    tp1         = (entry + target_low) / 2.0
                    tp2         = target_low

        if direction is None:
            continue

        stop_dist = abs(entry - stop)
        tp1_dist  = abs(tp1 - entry)
        if tp1_dist < 1.5 * stop_dist or stop_dist <= 0:
            continue

        # ── Fill simulation ──
        future = curr_day.iloc[idx + 1:]
        filled = False
        fill_idx = None

        for j, (fts, fb) in enumerate(future.iterrows()):
            if direction == 'long' and fb['low'] <= entry:
                filled = True; fill_idx = j; break
            elif direction == 'short' and fb['high'] >= entry:
                filled = True; fill_idx = j; break

        if not filled:
            continue

        trade_taken_today = True
        post = future.iloc[fill_idx:]

        tp1_contracts = 0.05
        runner_size   = 0.05
        tp1_hit = False
        tp1_pnl = runner_pnl = 0.0
        runner_stop = None

        for k, (pts, pb) in enumerate(post.iterrows()):
            ph, pl = pb['high'], pb['low']

            if not tp1_hit:
                if direction == 'long':
                    if pl <= stop:
                        loss = (entry - stop) * BTC_SIZE
                        trades.append({
                            'date': str(curr_date), 'dir': direction, 'regime': curr_regime,
                            'entry': entry, 'exit': stop, 'result': 'stop',
                            'pnl': loss - COMMISSION
                        }); break
                    elif ph >= tp1:
                        tp1_hit = True
                        tp1_pnl = (tp1 - entry) * tp1_contracts
                        runner_stop = entry
                else:  # short
                    if ph >= stop:
                        loss = (entry - stop) * BTC_SIZE
                        trades.append({
                            'date': str(curr_date), 'dir': direction, 'regime': curr_regime,
                            'entry': entry, 'exit': stop, 'result': 'stop',
                            'pnl': loss - COMMISSION
                        }); break
                    elif pl <= tp1:
                        tp1_hit = True
                        tp1_pnl = (entry - tp1) * tp1_contracts
                        runner_stop = entry
            else:
                if direction == 'long':
                    if pl <= runner_stop:
                        runner_pnl = (runner_stop - entry) * runner_size
                        trades.append({
                            'date': str(curr_date), 'dir': direction, 'regime': curr_regime,
                            'entry': entry, 'exit': runner_stop, 'result': 'tp1+be',
                            'pnl': tp1_pnl + runner_pnl - COMMISSION
                        }); break
                    elif ph >= tp2:
                        runner_pnl = (tp2 - entry) * runner_size
                        trades.append({
                            'date': str(curr_date), 'dir': direction, 'regime': curr_regime,
                            'entry': entry, 'exit': tp2, 'result': 'tp2',
                            'pnl': tp1_pnl + runner_pnl - COMMISSION
                        }); break
                else:  # short
                    if ph >= runner_stop:
                        runner_pnl = (entry - runner_stop) * runner_size
                        trades.append({
                            'date': str(curr_date), 'dir': direction, 'regime': curr_regime,
                            'entry': entry, 'exit': runner_stop, 'result': 'tp1+be',
                            'pnl': tp1_pnl + runner_pnl - COMMISSION
                        }); break
                    elif pl <= tp2:
                        runner_pnl = (entry - tp2) * runner_size
                        trades.append({
                            'date': str(curr_date), 'dir': direction, 'regime': curr_regime,
                            'entry': entry, 'exit': tp2, 'result': 'tp2',
                            'pnl': tp1_pnl + runner_pnl - COMMISSION
                        }); break
        else:
            last = post['close'].iloc[-1]
            if not tp1_hit:
                pnl = (last - entry) * BTC_SIZE if direction == 'long' else (entry - last) * BTC_SIZE
                trades.append({
                    'date': str(curr_date), 'dir': direction, 'regime': curr_regime,
                    'entry': entry, 'exit': last, 'result': 'eod', 'pnl': pnl - COMMISSION
                })
            else:
                runner_pnl = (last - entry) * runner_size if direction == 'long' else (entry - last) * runner_size
                trades.append({
                    'date': str(curr_date), 'dir': direction, 'regime': curr_regime,
                    'entry': entry, 'exit': last, 'result': 'tp1+eod',
                    'pnl': tp1_pnl + runner_pnl - COMMISSION
                })

# ─── RESULTS ─────────────────────────────────────────────────────────────────
df = pd.DataFrame(trades)

if df.empty:
    print("No trades generated.")
else:
    wins   = df[df['pnl'] > 0]['pnl']
    losses = df[df['pnl'] <= 0]['pnl']
    gp = wins.sum(); gl = abs(losses.sum())
    pf = gp / gl if gl > 0 else 999
    equity = df['pnl'].cumsum()
    max_dd = (equity - equity.cummax()).min()

    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly = df.groupby('month')['pnl'].sum()
    pos_months = (monthly > 0).sum()

    print("\n" + "=" * 70)
    print("  BTC DYNAMIC REGIME-ADAPTIVE SWEEP MODEL (4H 200 EMA FILTER)")
    print("  BULL: Long Bottom Sweeps ONLY | BEAR: Short Top Sweeps ONLY")
    print("  Data: Sep 2024 - Sep 2026 | 0.1 BTC / trade")
    print("=" * 70)
    print(f"  Total Trades taken : {len(df)}")
    print(f"  Win Rate           : {len(wins)/len(df)*100:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit Factor      : {pf:.2f}")
    print(f"  Net Profit (0.1BTC): ${df['pnl'].sum():,.2f}")
    print(f"  Scaled to 1.0 BTC  : ${df['pnl'].sum()*10:,.2f}")
    print(f"  Max Drawdown       : ${max_dd:,.2f}")
    print(f"  Avg Win Trade      : ${wins.mean():,.2f}")
    print(f"  Avg Loss Trade     : ${losses.mean():,.2f}")
    print(f"  Positive Months    : {pos_months} / {len(monthly)}")
    print("=" * 70)

    print("\n  Performance by Market Regime:")
    for reg, grp in df.groupby('regime'):
        w = (grp['pnl'] > 0).sum()
        gpp = grp[grp['pnl'] > 0]['pnl'].sum()
        gll = abs(grp[grp['pnl'] <= 0]['pnl'].sum())
        pff = gpp / gll if gll > 0 else 999
        print(f"    {reg} REGIME: {len(grp):>3} trades | {w/len(grp)*100:>5.1f}% WR | PF {pff:.2f} | Net ${grp['pnl'].sum():>8,.2f}")

    print("\n  Year-by-Year Performance:")
    for yr, grp in df.groupby('year'):
        w = (grp['pnl'] > 0).sum()
        gpp = grp[grp['pnl'] > 0]['pnl'].sum()
        gll = abs(grp[grp['pnl'] <= 0]['pnl'].sum())
        pff = gpp / gll if gll > 0 else 999
        print(f"    {yr}: {len(grp):>3} trades | {w/len(grp)*100:>5.1f}% WR | PF {pff:.2f} | Net ${grp['pnl'].sum():>8,.2f}")

    print("\n  Trade Outcomes:")
    for res, grp in df.groupby('result'):
        print(f"    {res:<12}: {len(grp):>3} trades | Avg ${grp['pnl'].mean():>8.2f} | Total ${grp['pnl'].sum():>9,.2f}")

    print("=" * 70)

