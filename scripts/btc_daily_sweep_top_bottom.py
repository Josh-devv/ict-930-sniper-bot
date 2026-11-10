"""
BTC Daily Top/Bottom Liquidity Sweep Model (Judas Swing / Daily Range Reclaim)
=============================================================================
Concept:
  - 24/7 continuous market: Asian session (00:00 - 08:00 UTC) sets the reference range.
  - Previous Day High (PDH) & Previous Day Low (PDL) provide key structural liquidity.
  - London & NY sessions (08:00 - 20:00 UTC) sweep Asian High/PDH or Asian Low/PDL.
  - Rejection: Sweep past level, close BACK INSIDE the swept range with a significant wick (>= 35%).
  - Entry: Limit at the level retest / CE of the rejection wick.
  - Stop: Beyond the sweep extreme (HOD/LOD high/low + buffer).
  - Target:
      TP1 (50% size): Midpoint (Equilibrium) between sweep extreme & opposite extreme.
      TP2 (50% size): Opposite liquidity pool (Asian / Daily opposite extreme).
  - Risk: 0.1 BTC base unit ($0.10/pt).

Data: BTCUSD-duka.json (2 years: Sep 2024 - Sep 2026, 1-min)
"""

import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── PARAMETERS ─────────────────────────────────────────────────────────────
TF_MIN          = 5          # 5m candles for signal clarity
MIN_SWEEP_DIST  = 50.0       # Must sweep by at least $50 past the level
WICK_RATIO      = 0.35       # Rejection wick >= 35% of candle range
STOP_BUFFER     = 100.0      # $100 buffer beyond the sweep extreme
BTC_SIZE        = 0.1        # 0.1 BTC per trade (1 MBT contract equivalent)
COMMISSION      = 5.0        # $5 per RT trade

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
df_5m = df_raw.resample(f'{TF_MIN}min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna(subset=['open'])

print(f"Resampled to {TF_MIN}m: {len(df_5m):,} bars")

# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────
trades = []

dates = list(df_5m.groupby(df_5m.index.date))
print(f"Total days to test: {len(dates)}")

prev_day_high = None
prev_day_low  = None

for i in range(1, len(dates)):
    prev_date, prev_day_bars = dates[i - 1]
    curr_date, curr_day_bars = dates[i]

    # PDH / PDL
    pdh = prev_day_bars['high'].max()
    pdl = prev_day_bars['low'].min()

    # Asian session: 00:00 - 08:00 UTC
    asia_bars = curr_day_bars.between_time('00:00', '07:59')
    if len(asia_bars) < 10:
        continue

    asia_high = asia_bars['high'].max()
    asia_low  = asia_bars['low'].min()

    # Upper key liquidity levels: max(asia_high, pdh) or separate
    # We look for a sweep of the major session high or low during Expansion Window (08:00 - 20:00 UTC)
    sweep_window = curr_day_bars.between_time('08:00', '20:00')
    if len(sweep_window) < 5:
        continue

    # Reference levels for top/bottom detection:
    # Top sweep levels: asia_high or pdh
    # Bottom sweep levels: asia_low or pdl
    trade_taken_today = False

    for ts in sweep_window.index:
        if trade_taken_today:
            break

        idx = curr_day_bars.index.get_loc(ts)
        o = curr_day_bars['open'].iloc[idx]
        h = curr_day_bars['high'].iloc[idx]
        l = curr_day_bars['low'].iloc[idx]
        c = curr_day_bars['close'].iloc[idx]
        rng = h - l
        if rng < 20:
            continue

        direction = None
        entry = stop = tp1 = tp2 = None

        # ── TOP OF THE DAY SWEEP (Short) ──
        # Check sweep of Asia High or PDH
        swept_top_level = None
        if h > asia_high + MIN_SWEEP_DIST and c < asia_high:
            swept_top_level = asia_high
        elif h > pdh + MIN_SWEEP_DIST and c < pdh:
            swept_top_level = pdh

        if swept_top_level is not None:
            wick = h - max(o, c)
            if wick / rng >= WICK_RATIO:
                direction = 'short'
                entry     = h - 0.50 * wick
                stop      = h + STOP_BUFFER
                # Target: Equilibrium (midpoint) & Opposite Low
                target_low = min(asia_low, pdl)
                tp1       = (entry + target_low) / 2.0
                tp2       = target_low

        # ── BOTTOM OF THE DAY SWEEP (Long) ──
        swept_bot_level = None
        if l < asia_low - MIN_SWEEP_DIST and c > asia_low:
            swept_bot_level = asia_low
        elif l < pdl - MIN_SWEEP_DIST and c > pdl:
            swept_bot_level = pdl

        if swept_bot_level is not None and direction is None:
            wick = min(o, c) - l
            if wick / rng >= WICK_RATIO:
                direction = 'long'
                entry     = l + 0.50 * wick
                stop      = l - STOP_BUFFER
                target_high = max(asia_high, pdh)
                tp1       = (entry + target_high) / 2.0
                tp2       = target_high

        if direction is None:
            continue

        # R:R check: must have at least 1.5R to TP1
        stop_dist = abs(entry - stop)
        tp1_dist  = abs(tp1 - entry)
        if tp1_dist < 1.5 * stop_dist or stop_dist <= 0:
            continue

        # ── Check Fill ──
        future = curr_day_bars.iloc[idx + 1:]
        filled = False
        fill_idx = None

        for j, (fts, fb) in enumerate(future.iterrows()):
            if direction == 'short' and fb['high'] >= entry:
                filled = True; fill_idx = j; break
            elif direction == 'long' and fb['low'] <= entry:
                filled = True; fill_idx = j; break

        if not filled:
            continue

        trade_taken_today = True
        post = future.iloc[fill_idx:]

        tp1_contracts = 0.05  # 50%
        runner_size   = 0.05  # 50%
        tp1_hit = False
        tp1_pnl = runner_pnl = 0.0
        runner_stop = None

        for k, (pts, pb) in enumerate(post.iterrows()):
            ph, pl = pb['high'], pb['low']

            if not tp1_hit:
                if direction == 'short':
                    if ph >= stop:
                        loss = (entry - stop) * BTC_SIZE
                        trades.append({
                            'date': str(curr_date), 'dir': direction,
                            'entry': entry, 'exit': stop, 'result': 'stop',
                            'pnl': loss - COMMISSION, 'rr': -1.0
                        }); break
                    elif pl <= tp1:
                        tp1_hit = True
                        tp1_pnl = (entry - tp1) * tp1_contracts
                        runner_stop = entry  # Breakeven
                else:
                    if pl <= stop:
                        loss = (entry - stop) * BTC_SIZE
                        trades.append({
                            'date': str(curr_date), 'dir': direction,
                            'entry': entry, 'exit': stop, 'result': 'stop',
                            'pnl': loss - COMMISSION, 'rr': -1.0
                        }); break
                    elif ph >= tp1:
                        tp1_hit = True
                        tp1_pnl = (tp1 - entry) * tp1_contracts
                        runner_stop = entry  # Breakeven
            else:
                # Runner active
                if direction == 'short':
                    if ph >= runner_stop:
                        runner_pnl = (entry - runner_stop) * runner_size
                        trades.append({
                            'date': str(curr_date), 'dir': direction,
                            'entry': entry, 'exit': runner_stop, 'result': 'tp1+be',
                            'pnl': tp1_pnl + runner_pnl - COMMISSION, 'rr': tp1_dist / stop_dist
                        }); break
                    elif pl <= tp2:
                        runner_pnl = (entry - tp2) * runner_size
                        trades.append({
                            'date': str(curr_date), 'dir': direction,
                            'entry': entry, 'exit': tp2, 'result': 'tp2',
                            'pnl': tp1_pnl + runner_pnl - COMMISSION, 'rr': abs(tp2 - entry) / stop_dist
                        }); break
                else:
                    if pl <= runner_stop:
                        runner_pnl = (runner_stop - entry) * runner_size
                        trades.append({
                            'date': str(curr_date), 'dir': direction,
                            'entry': entry, 'exit': runner_stop, 'result': 'tp1+be',
                            'pnl': tp1_pnl + runner_pnl - COMMISSION, 'rr': tp1_dist / stop_dist
                        }); break
                    elif ph >= tp2:
                        runner_pnl = (tp2 - entry) * runner_size
                        trades.append({
                            'date': str(curr_date), 'dir': direction,
                            'entry': entry, 'exit': tp2, 'result': 'tp2',
                            'pnl': tp1_pnl + runner_pnl - COMMISSION, 'rr': abs(tp2 - entry) / stop_dist
                        }); break
        else:
            # End of day close (23:55)
            last = post['close'].iloc[-1]
            if not tp1_hit:
                pnl = (entry - last) * BTC_SIZE if direction == 'short' else (last - entry) * BTC_SIZE
                trades.append({
                    'date': str(curr_date), 'dir': direction,
                    'entry': entry, 'exit': last, 'result': 'eod',
                    'pnl': pnl - COMMISSION, 'rr': 0.0
                })
            else:
                runner_pnl = (entry - last) * runner_size if direction == 'short' else (last - entry) * runner_size
                trades.append({
                    'date': str(curr_date), 'dir': direction,
                    'entry': entry, 'exit': last, 'result': 'tp1+eod',
                    'pnl': tp1_pnl + runner_pnl - COMMISSION, 'rr': tp1_dist / stop_dist
                })

# ─── RESULTS REPORT ──────────────────────────────────────────────────────────
df = pd.DataFrame(trades)

if df.empty:
    print("No trades generated.")
else:
    wins   = df[df['pnl'] > 0]['pnl']
    losses = df[df['pnl'] <= 0]['pnl']
    gp = wins.sum()
    gl = abs(losses.sum())
    pf = gp / gl if gl > 0 else 999
    equity = df['pnl'].cumsum()
    max_dd = (equity - equity.cummax()).min()

    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly = df.groupby('month')['pnl'].sum()
    pos_months = (monthly > 0).sum()

    print("\n" + "=" * 70)
    print("  BTC DAILY TOP/BOTTOM LIQUIDITY SWEEP (JUDAS REVERSAL)")
    print("  Data: Dukascopy 2-Year (Sep 2024 - Sep 2026) | 0.1 BTC / trade")
    print("=" * 70)
    print(f"  Total Trades taken : {len(df)}")
    print(f"  Win Rate           : {len(wins)/len(df)*100:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit Factor      : {pf:.2f}")
    print(f"  Net Profit         : ${df['pnl'].sum():,.2f}")
    print(f"  Max Drawdown       : ${max_dd:,.2f}")
    print(f"  Avg Win Trade      : ${wins.mean():,.2f}")
    print(f"  Avg Loss Trade     : ${losses.mean():,.2f}")
    print(f"  Win/Loss R:R Ratio : {abs(wins.mean() / losses.mean()):.2f} : 1")
    print(f"  Positive Months    : {pos_months} / {len(monthly)}")
    print("=" * 70)

    print("\n  Trade Outcomes:")
    for res, grp in df.groupby('result'):
        print(f"    {res:<12}: {len(grp):>3} trades | Avg ${grp['pnl'].mean():>8.2f} | Total ${grp['pnl'].sum():>9,.2f}")

    print("\n  Year-by-Year Performance:")
    for yr, grp in df.groupby('year'):
        w = (grp['pnl'] > 0).sum()
        wr = w / len(grp) * 100
        net = grp['pnl'].sum()
        gpl = grp[grp['pnl'] > 0]['pnl'].sum()
        gll = abs(grp[grp['pnl'] <= 0]['pnl'].sum())
        pff = gpl / gll if gll > 0 else 999
        print(f"    {yr}: {len(grp):>3} trades | {wr:>5.1f}% WR | PF {pff:.2f} | Net ${net:>8,.2f}")

    print("\n  Direction Breakdown:")
    for d, grp in df.groupby('dir'):
        w = (grp['pnl'] > 0).sum()
        print(f"    {d.upper():<5}: {len(grp):>3} trades | {w/len(grp)*100:>5.1f}% WR | Net ${grp['pnl'].sum():>8,.2f}")

    print()
    print("  * Sizing: 0.1 BTC ($0.10/pt). Multiply x10 for 1.0 BTC position.")
    print("=" * 70)

