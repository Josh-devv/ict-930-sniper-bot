"""
BTC Daily VWAP Band Reversal Strategy
======================================
Logic:
  - VWAP anchored to 00:00 UTC daily reset (BTC has no session close)
  - Signal window: 13:00-15:00 UTC = NY open (9AM-11AM EST) — highest BTC volume
  - 2.0σ bands, minimum 500pt distance from VWAP
  - Rejection wick >= 35% of candle range touching the band
  - Entry at 50% wick pullback (Consequential Encroachment)
  - TP1: Daily VWAP (mean reversion target)
  - TP2: Opposite 2σ band (extension target)
  - EOD flat: 22:00 UTC (avoid low-liquidity late night)
  - 1 trade per day max
  - Position: 0.1 BTC (= 1 CME Micro BTC / MBT contract)

Data: BTCUSD-duka.json (Sep 2024 - Sep 2026, 1-min, Dukascopy)
"""

import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── STRATEGY PARAMETERS ─────────────────────────────────────────────────────
SIGMA           = 2.0
MIN_BAND_DIST   = 500.0     # min USD distance VWAP->band (BTC is ~15-50x NQ scale)
WICK_RATIO      = 0.35      # wick >= 35% of range
CE_PCT          = 0.50      # entry at 50% wick
STOP_BEYOND     = 150.0     # stop 150 USD beyond candle extreme
TF_MIN          = 5         # 5-minute candles

SIGNAL_START    = '13:00'   # UTC = 9:00 AM EST (NY open)
SIGNAL_END      = '14:59'   # UTC = 10:59 AM EST
EOD_FLAT        = '22:00'   # UTC — close all before late session

BTC_SIZE        = 0.1       # 0.1 BTC per trade (1 MBT equivalent = $0.10/pt)
COMMISSION_USD  = 5.0       # ~$5 round trip for 1 MBT on CME

DATA_PATH = r'c:\Users\JOSHUA\Desktop\CRTbot\data\crypto\BTCUSD-duka.json'

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
print("Loading BTC data...")
with open(DATA_PATH, 'r') as f:
    raw = json.load(f)

df_raw = pd.DataFrame(raw)
df_raw['dt'] = pd.to_datetime(df_raw['time'], unit='ms', utc=True)
df_raw = df_raw.set_index('dt').drop(columns=['time'])
df_raw.columns = ['open', 'high', 'low', 'close', 'volume']
df_raw['volume'] = df_raw['volume'].clip(lower=1)

print(f"Loaded {len(df_raw):,} bars")
print(f"Range : {df_raw.index[0].date()} to {df_raw.index[-1].date()}")
print(f"Price : ${df_raw['close'].min():,.0f} - ${df_raw['close'].max():,.0f}")
print()

# ─── RESAMPLE TO 5-MIN ───────────────────────────────────────────────────────
df_5m = df_raw.resample(f'{TF_MIN}min', closed='left', label='left').agg(
    {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
).dropna(subset=['open'])

print(f"Resampled to {TF_MIN}-min: {len(df_5m):,} bars")

# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────
trades = []

# Group by UTC date (VWAP resets at 00:00 UTC)
dates = df_5m.groupby(df_5m.index.date)
print(f"Trading days: {len(dates)}")
print()

for date, day in dates:
    # Full day for VWAP calculation (00:00 - 22:00 UTC)
    day = day.between_time('00:00', '21:59')
    if len(day) < 10:
        continue

    # ── Build rolling daily VWAP + bands ─────────────────────────────────
    tp = (day['high'] + day['low'] + day['close']) / 3.0
    cum_vol  = day['volume'].cumsum()
    cum_tpv  = (tp * day['volume']).cumsum()
    vwap     = cum_tpv / cum_vol

    cum_tp2v = ((tp**2) * day['volume']).cumsum()
    variance = ((cum_tp2v / cum_vol) - vwap**2).clip(lower=0)
    std_dev  = np.sqrt(variance)
    upper    = vwap + SIGMA * std_dev
    lower    = vwap - SIGMA * std_dev

    # ── Signal scan: NY open window only ─────────────────────────────────
    signal_bars = day.between_time(SIGNAL_START, SIGNAL_END)
    trade_today = False

    for ts in signal_bars.index:
        if trade_today:
            break

        idx = day.index.get_loc(ts)
        o = day['open'].iloc[idx]
        h = day['high'].iloc[idx]
        l = day['low'].iloc[idx]
        c = day['close'].iloc[idx]
        rng = h - l
        if rng < 10:  # ignore tiny candles (< $10 range)
            continue

        vw = vwap.iloc[idx]
        ub = upper.iloc[idx]
        lb = lower.iloc[idx]

        # Need bands to be meaningful — skip early session before VWAP stabilises
        if np.isnan(ub) or np.isnan(lb):
            continue
        if (ub - vw) < MIN_BAND_DIST or (vw - lb) < MIN_BAND_DIST:
            continue

        direction = entry = stop = tp1 = tp2 = None

        # SHORT: high touches upper band, closes below it, wick >= 35%
        if h >= ub and c < ub:
            wick = h - c
            if rng > 0 and wick / rng >= WICK_RATIO:
                direction = 'short'
                entry     = h - CE_PCT * wick
                stop      = h + STOP_BEYOND
                tp1       = vw
                tp2       = lb

        # LONG: low touches lower band, closes above it, wick >= 35%
        elif l <= lb and c > lb:
            wick = c - l
            if rng > 0 and wick / rng >= WICK_RATIO:
                direction = 'long'
                entry     = l + CE_PCT * wick
                stop      = l - STOP_BEYOND
                tp1       = vw
                tp2       = ub

        if direction is None:
            continue

        # Basic R:R filter — TP1 must be at least 1.5x the stop distance
        stop_dist = abs(entry - stop)
        tp1_dist  = abs(tp1 - entry)
        if tp1_dist < 1.5 * stop_dist:
            continue

        # ── Simulate fill + trade execution ──────────────────────────────
        future = day.iloc[idx + 1:]
        future = future[future.index.time <= pd.Timestamp(f'2000-01-01 {EOD_FLAT}').time()]

        filled = False
        fill_j = None
        for j, (fts, fb) in enumerate(future.iterrows()):
            if direction == 'short' and fb['high'] >= entry:
                filled = True; fill_j = j; break
            elif direction == 'long' and fb['low'] <= entry:
                filled = True; fill_j = j; break

        if not filled:
            continue

        trade_today = True
        post = future.iloc[fill_j:]

        tp1_contracts = 0.06   # 60% = 0.06 BTC of the 0.1 BTC position
        runner_size   = 0.04   # 40% runner = 0.04 BTC
        tp1_hit       = False
        tp1_pnl = runner_pnl = 0.0
        runner_stop   = None

        for k, (pts, pb) in enumerate(post.iterrows()):
            ph, pl = pb['high'], pb['low']

            if not tp1_hit:
                if direction == 'short':
                    if ph >= stop:
                        loss = (entry - stop) * BTC_SIZE
                        trades.append({
                            'date': str(date), 'dir': direction,
                            'entry': entry, 'exit': stop,
                            'result': 'stop', 'pnl': loss - COMMISSION_USD
                        }); break
                    elif pl <= tp1:
                        tp1_hit   = True
                        tp1_pnl   = (entry - tp1) * tp1_contracts
                        runner_stop = entry + 100.0  # BE + $100 buffer for runner
                else:
                    if pl <= stop:
                        loss = (entry - stop) * BTC_SIZE
                        trades.append({
                            'date': str(date), 'dir': direction,
                            'entry': entry, 'exit': stop,
                            'result': 'stop', 'pnl': loss - COMMISSION_USD
                        }); break
                    elif ph >= tp1:
                        tp1_hit   = True
                        tp1_pnl   = (tp1 - entry) * tp1_contracts
                        runner_stop = entry - 100.0
            else:
                if direction == 'short':
                    if ph >= runner_stop:
                        runner_pnl = (entry - runner_stop) * runner_size
                        trades.append({
                            'date': str(date), 'dir': direction,
                            'entry': entry, 'exit': runner_stop,
                            'result': 'tp1+be', 'pnl': tp1_pnl + runner_pnl - COMMISSION_USD
                        }); break
                    elif pl <= tp2:
                        runner_pnl = (entry - tp2) * runner_size
                        trades.append({
                            'date': str(date), 'dir': direction,
                            'entry': entry, 'exit': tp2,
                            'result': 'tp2', 'pnl': tp1_pnl + runner_pnl - COMMISSION_USD
                        }); break
                else:
                    if pl <= runner_stop:
                        runner_pnl = (runner_stop - entry) * runner_size
                        trades.append({
                            'date': str(date), 'dir': direction,
                            'entry': entry, 'exit': runner_stop,
                            'result': 'tp1+be', 'pnl': tp1_pnl + runner_pnl - COMMISSION_USD
                        }); break
                    elif ph >= tp2:
                        runner_pnl = (tp2 - entry) * runner_size
                        trades.append({
                            'date': str(date), 'dir': direction,
                            'entry': entry, 'exit': tp2,
                            'result': 'tp2', 'pnl': tp1_pnl + runner_pnl - COMMISSION_USD
                        }); break
        else:
            # EOD flat
            last = post['close'].iloc[-1] if len(post) > 0 else entry
            if not tp1_hit:
                pnl = (entry - last) * BTC_SIZE if direction == 'short' else (last - entry) * BTC_SIZE
                trades.append({
                    'date': str(date), 'dir': direction,
                    'entry': entry, 'exit': last,
                    'result': 'eod', 'pnl': pnl - COMMISSION_USD
                })
            else:
                runner_pnl = (entry - last) * runner_size if direction == 'short' else (last - entry) * runner_size
                trades.append({
                    'date': str(date), 'dir': direction,
                    'entry': entry, 'exit': last,
                    'result': 'tp1+eod', 'pnl': tp1_pnl + runner_pnl - COMMISSION_USD
                })

# ─── RESULTS ─────────────────────────────────────────────────────────────────
df = pd.DataFrame(trades)

if df.empty:
    print("No trades generated — parameters may need adjustment.")
else:
    wins   = df[df['pnl'] > 0]['pnl']
    losses = df[df['pnl'] <= 0]['pnl']
    gp = wins.sum(); gl = abs(losses.sum())
    pf = gp / gl if gl > 0 else 999

    equity   = df['pnl'].cumsum()
    max_dd   = (equity - equity.cummax()).min()

    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly     = df.groupby('month')['pnl'].sum()
    pos_months  = (monthly > 0).sum()

    df['year']  = pd.to_datetime(df['date']).dt.year

    print("=" * 65)
    print("  BTC DAILY VWAP BAND REVERSAL — 5-Min | 0.1 BTC/trade")
    print("  Signal Window: 13:00-15:00 UTC (NY Open)")
    print("=" * 65)
    print(f"  Total Trades     : {len(df)}")
    print(f"  Win Rate         : {len(wins)/len(df)*100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit Factor    : {pf:.2f}")
    print(f"  Net Profit (USD) : ${df['pnl'].sum():,.2f}")
    print(f"  Max Drawdown     : ${max_dd:,.2f}")
    print(f"  Avg Win          : ${wins.mean():,.2f}")
    print(f"  Avg Loss         : ${losses.mean():,.2f}")
    print(f"  Positive Months  : {pos_months}/{len(monthly)}")
    print("=" * 65)

    print("\n  Outcome breakdown:")
    for res, grp in df.groupby('result'):
        print(f"    {res:<12}: {len(grp):>4} trades | avg ${grp['pnl'].mean():>8.2f} | total ${grp['pnl'].sum():>10,.2f}")

    print("\n  Year-by-year:")
    for yr, grp in df.groupby('year'):
        w   = (grp['pnl'] > 0).sum()
        wr  = w / len(grp) * 100
        net = grp['pnl'].sum()
        gl2 = abs(grp[grp['pnl'] <= 0]['pnl'].sum())
        gp2 = grp[grp['pnl'] > 0]['pnl'].sum()
        pf2 = gp2 / gl2 if gl2 > 0 else 999
        print(f"    {yr}: {len(grp):>3} trades | {wr:>5.1f}% WR | PF {pf2:.2f} | Net ${net:>9,.2f}")

    print("\n  Monthly P&L:")
    for mo, pnl in monthly.items():
        bar = '+' * int(max(0, pnl) / 50) if pnl > 0 else '-' * int(abs(min(0, pnl)) / 50)
        print(f"    {mo}  ${pnl:>9,.2f}  {bar}")

    print()
    print(f"  * Position size: 0.1 BTC per trade (~1 CME MBT contract)")
    print(f"  * Scale to 1.0 BTC = multiply all P&L x10")
    print(f"  * Scale to 0.5 BTC = multiply all P&L x5")
    print()

