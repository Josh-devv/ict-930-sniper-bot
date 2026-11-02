"""
8-Year Backtest: ICT 9:30 VWAP Sniper — 1m vs 3m vs 5m
Data: NQ-duka-2018-2026.json (Dukascopy, 1-min, 3.97M bars)
Strategy: SEALED parameters — 5 MNQ, 2σ bands, CE entry, TP1=VWAP, TP2=opposite band
"""

import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── SEALED STRATEGY PARAMETERS ─────────────────────────────────────────────
SIGMA            = 2.0
MIN_BAND_DIST    = 15.0      # min pts from VWAP to band for valid signal
WICK_RATIO       = 0.35      # wick must be >= 35% of candle range
CE_PCT           = 0.50      # entry at 50% wick (Consequential Encroachment)
STOP_BEYOND      = 1.5       # stop 1.5 pts beyond signal candle extreme
TP1_PCT          = 0.60      # 60% closed at TP1 (VWAP)  = 3 of 5 MNQ
TP2_PCT          = 0.40      # 40% runner to opposite band = 2 of 5 MNQ
BE_OFFSET        = 1.0       # runner stop moved to BE+1 after TP1
N_CONTRACTS      = 5
PT_VALUE         = 2.0       # $2/pt per MNQ contract
COMMISSION       = 0.50 * N_CONTRACTS  # $0.50 RT per contract = $2.50/trade

SESSION_OPEN_H   = 9
SESSION_OPEN_M   = 30
SIGNAL_END_H     = 11
SIGNAL_END_M     = 30
EOD_FLAT_H       = 15
EOD_FLAT_M       = 45

DATA_PATH        = r'c:\Users\JOSHUA\Desktop\CRTbot\data\indices\NQ-duka-2018-2026.json'
TIMEFRAMES       = [1, 3, 5]

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
print("Loading data (3.97M bars) — this takes ~30s...")
with open(DATA_PATH, 'r') as f:
    raw = json.load(f)

df_raw = pd.DataFrame(raw)
df_raw['dt'] = pd.to_datetime(df_raw['time'], unit='ms', utc=True)
df_raw = df_raw.set_index('dt').drop(columns=['time'])
df_raw.columns = ['open', 'high', 'low', 'close', 'volume']

# Convert to US/Eastern (handles DST automatically)
df_raw.index = df_raw.index.tz_convert('America/New_York')

# Keep only weekdays with volume
df_raw = df_raw[df_raw.index.dayofweek < 5]
df_raw['volume'] = df_raw['volume'].clip(lower=1)  # floor vol at 1 for VWAP

print(f"Raw bars after weekday filter: {len(df_raw):,}")
print(f"Range: {df_raw.index[0].date()} to {df_raw.index[-1].date()}")
print()

# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────
def resample_to_tf(day_df, tf_min):
    """Resample 1-min day data to tf_min."""
    if tf_min == 1:
        return day_df.copy()
    rule = f'{tf_min}min'
    ohlcv = day_df.resample(rule, closed='left', label='left').agg({
        'open':  'first',
        'high':  'max',
        'low':   'min',
        'close': 'last',
        'volume':'sum'
    }).dropna(subset=['open'])
    return ohlcv


def run_backtest(df_1min, tf_min):
    """
    Run the VWAP Sniper strategy on df_1min data resampled to tf_min.
    Returns list of trade dicts.
    """
    trades = []

    # Group by calendar date
    dates = df_1min.groupby(df_1min.index.date)

    for date, day_1m in dates:
        # Resample to target timeframe
        day = resample_to_tf(day_1m, tf_min)

        # Filter to full session (09:30 – 15:45) for EOD management
        day = day.between_time('09:30', '15:44')
        if len(day) < 3:
            continue

        # ── Build session VWAP + bands ────────────────────────────────────
        tp = (day['high'] + day['low'] + day['close']) / 3.0
        cum_vol  = day['volume'].cumsum()
        cum_tpv  = (tp * day['volume']).cumsum()
        vwap     = cum_tpv / cum_vol

        # Running variance: E[x²] - E[x]²
        cum_tp2v = ((tp ** 2) * day['volume']).cumsum()
        variance = (cum_tp2v / cum_vol) - (vwap ** 2)
        variance = variance.clip(lower=0)
        std_dev  = np.sqrt(variance)

        upper = vwap + SIGMA * std_dev
        lower = vwap - SIGMA * std_dev

        # ── Signal scan (09:30 – 11:30 only) ─────────────────────────────
        signal_window = day.between_time('09:30', '11:29')
        trade_taken_today = False

        bars = list(signal_window.index)

        for i, ts in enumerate(bars):
            if trade_taken_today:
                break

            if ts not in day.index:
                continue

            idx = day.index.get_loc(ts)

            o = day['open'].iloc[idx]
            h = day['high'].iloc[idx]
            l = day['low'].iloc[idx]
            c = day['close'].iloc[idx]
            rng = h - l
            if rng < 0.25:
                continue

            vw   = vwap.iloc[idx]
            ub   = upper.iloc[idx]
            lb   = lower.iloc[idx]

            # Check band distance from VWAP
            ub_dist = ub - vw
            lb_dist = vw - lb

            direction = None
            entry_price = np.nan
            stop_price  = np.nan
            tp1_price   = np.nan
            tp2_price   = np.nan

            # ── SHORT signal: touch upper band, close below band ──────────
            if (h >= ub and c < ub and ub_dist >= MIN_BAND_DIST):
                wick = h - c
                if wick / rng >= WICK_RATIO:
                    direction   = 'short'
                    entry_price = h - CE_PCT * wick        # sell limit
                    stop_price  = h + STOP_BEYOND
                    tp1_price   = vw                        # VWAP at signal
                    tp2_price   = lb                        # opposite band

            # ── LONG signal: touch lower band, close above band ───────────
            elif (l <= lb and c > lb and lb_dist >= MIN_BAND_DIST):
                wick = c - l
                if wick / rng >= WICK_RATIO:
                    direction   = 'long'
                    entry_price = l + CE_PCT * wick         # buy limit
                    stop_price  = l - STOP_BEYOND
                    tp1_price   = vw
                    tp2_price   = ub

            if direction is None:
                continue

            # ── Simulate fill + trade management ──────────────────────────
            # Look at subsequent bars for fill
            future_bars = day.iloc[idx + 1:]
            future_bars = future_bars[future_bars.index.time <= pd.Timestamp(f'{date} {EOD_FLAT_H}:{EOD_FLAT_M}').time()]

            filled = False
            fill_bar_idx = None

            for j, (fts, fbar) in enumerate(future_bars.iterrows()):
                fh = fbar['high']
                fl = fbar['low']
                if direction == 'short' and fh >= entry_price:
                    filled = True
                    fill_bar_idx = j
                    break
                elif direction == 'long' and fl <= entry_price:
                    filled = True
                    fill_bar_idx = j
                    break

            if not filled:
                continue

            trade_taken_today = True

            # Simulate from fill bar onward
            post_fill = future_bars.iloc[fill_bar_idx:]
            tp1_hit = False
            tp1_pnl = 0.0
            runner_pnl = 0.0
            runner_stop = None

            for k, (pts, pbar) in enumerate(post_fill.iterrows()):
                ph = pbar['high']
                pl = pbar['low']

                if not tp1_hit:
                    # Check stop first (intrabar worst-case)
                    if direction == 'short':
                        if ph >= stop_price:
                            # stopped out full size
                            loss = (entry_price - stop_price) * N_CONTRACTS * PT_VALUE
                            trades.append({
                                'date': str(date), 'direction': direction,
                                'entry': entry_price, 'exit': stop_price,
                                'result': 'stop', 'pnl': loss - COMMISSION
                            })
                            break
                        elif pl <= tp1_price:
                            # TP1 hit
                            tp1_hit = True
                            tp1_contracts = round(N_CONTRACTS * TP1_PCT)
                            runner_contracts = N_CONTRACTS - tp1_contracts
                            tp1_pnl = (entry_price - tp1_price) * tp1_contracts * PT_VALUE
                            runner_stop = entry_price + BE_OFFSET  # BE+1 for short runner
                    else:  # long
                        if pl <= stop_price:
                            loss = (entry_price - stop_price) * N_CONTRACTS * PT_VALUE
                            trades.append({
                                'date': str(date), 'direction': direction,
                                'entry': entry_price, 'exit': stop_price,
                                'result': 'stop', 'pnl': loss - COMMISSION
                            })
                            break
                        elif ph >= tp1_price:
                            tp1_hit = True
                            tp1_contracts = round(N_CONTRACTS * TP1_PCT)
                            runner_contracts = N_CONTRACTS - tp1_contracts
                            tp1_pnl = (tp1_price - entry_price) * tp1_contracts * PT_VALUE
                            runner_stop = entry_price - BE_OFFSET  # BE+1 for long runner

                else:
                    # Runner management
                    if direction == 'short':
                        if ph >= runner_stop:
                            # Runner stopped at BE+1
                            runner_pnl = (entry_price - runner_stop) * runner_contracts * PT_VALUE
                            trades.append({
                                'date': str(date), 'direction': direction,
                                'entry': entry_price, 'exit': runner_stop,
                                'result': 'tp1+be', 'pnl': tp1_pnl + runner_pnl - COMMISSION
                            })
                            break
                        elif pl <= tp2_price:
                            # Full TP2
                            runner_pnl = (entry_price - tp2_price) * runner_contracts * PT_VALUE
                            trades.append({
                                'date': str(date), 'direction': direction,
                                'entry': entry_price, 'exit': tp2_price,
                                'result': 'tp2', 'pnl': tp1_pnl + runner_pnl - COMMISSION
                            })
                            break
                    else:  # long
                        if pl <= runner_stop:
                            runner_pnl = (runner_stop - entry_price) * runner_contracts * PT_VALUE
                            trades.append({
                                'date': str(date), 'direction': direction,
                                'entry': entry_price, 'exit': runner_stop,
                                'result': 'tp1+be', 'pnl': tp1_pnl + runner_pnl - COMMISSION
                            })
                            break
                        elif ph >= tp2_price:
                            runner_pnl = (tp2_price - entry_price) * runner_contracts * PT_VALUE
                            trades.append({
                                'date': str(date), 'direction': direction,
                                'entry': entry_price, 'exit': tp2_price,
                                'result': 'tp2', 'pnl': tp1_pnl + runner_pnl - COMMISSION
                            })
                            break
            else:
                # EOD flat — close at last bar's close
                if not tp1_hit:
                    last_close = post_fill['close'].iloc[-1]
                    if direction == 'short':
                        pnl = (entry_price - last_close) * N_CONTRACTS * PT_VALUE
                    else:
                        pnl = (last_close - entry_price) * N_CONTRACTS * PT_VALUE
                    trades.append({
                        'date': str(date), 'direction': direction,
                        'entry': entry_price, 'exit': last_close,
                        'result': 'eod_flat', 'pnl': pnl - COMMISSION
                    })
                else:
                    last_close = post_fill['close'].iloc[-1]
                    if direction == 'short':
                        runner_pnl = (entry_price - last_close) * runner_contracts * PT_VALUE
                    else:
                        runner_pnl = (last_close - entry_price) * runner_contracts * PT_VALUE
                    trades.append({
                        'date': str(date), 'direction': direction,
                        'entry': entry_price, 'exit': last_close,
                        'result': 'tp1+eod', 'pnl': tp1_pnl + runner_pnl - COMMISSION
                    })

    return trades


# ─── STATS CALCULATOR ────────────────────────────────────────────────────────
def calc_stats(trades):
    if not trades:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'wr': 0,
                'pf': 0, 'net': 0, 'max_dd': 0, 'avg_win': 0, 'avg_loss': 0}

    df = pd.DataFrame(trades)
    df['win'] = df['pnl'] > 0
    wins   = df[df['win']]['pnl']
    losses = df[~df['win']]['pnl']

    gross_profit = wins.sum() if len(wins) else 0
    gross_loss   = abs(losses.sum()) if len(losses) else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    # Max drawdown via equity curve
    equity = df['pnl'].cumsum()
    rolling_max = equity.cummax()
    dd = equity - rolling_max
    max_dd = dd.min()

    return {
        'trades':   len(df),
        'wins':     len(wins),
        'losses':   len(losses),
        'wr':       len(wins) / len(df) * 100,
        'pf':       pf,
        'net':      df['pnl'].sum(),
        'max_dd':   max_dd,
        'avg_win':  wins.mean() if len(wins) else 0,
        'avg_loss': losses.mean() if len(losses) else 0,
    }


# ─── YEAR-BY-YEAR RUNNER ─────────────────────────────────────────────────────
years = sorted(df_raw.index.year.unique())
all_results = {}   # {tf: {year: stats}}

for tf in TIMEFRAMES:
    print(f"\nRunning {tf}-minute backtest across {len(years)} years...")
    all_results[tf] = {}
    all_trades_tf = []

    for yr in years:
        yr_data = df_raw[df_raw.index.year == yr]
        if len(yr_data) < 100:
            continue
        trades = run_backtest(yr_data, tf)
        all_results[tf][yr] = calc_stats(trades)
        all_trades_tf.extend(trades)
        print(f"  {yr}: {all_results[tf][yr]['trades']} trades | "
              f"{all_results[tf][yr]['wr']:.1f}% WR | "
              f"PF {all_results[tf][yr]['pf']:.2f} | "
              f"Net ${all_results[tf][yr]['net']:,.0f}")

    all_results[tf]['ALL'] = calc_stats(all_trades_tf)


# ─── PRINT FULL REPORT ───────────────────────────────────────────────────────
print("\n")
print("=" * 100)
print(f"  8-YEAR BACKTEST  |  ICT 9:30 VWAP SNIPER  |  5 MNQ @ $10/pt  |  2018-2026")
print("=" * 100)

header = f"{'Year':<8} | {'1-Min Trades':>12} {'WR':>6} {'PF':>6} {'Net':>10} {'DD':>10} |" \
         f" {'3-Min Trades':>12} {'WR':>6} {'PF':>6} {'Net':>10} {'DD':>10} |" \
         f" {'5-Min Trades':>12} {'WR':>6} {'PF':>6} {'Net':>10} {'DD':>10}"
print(header)
print("-" * 100)

for yr in years:
    row = f"{yr:<8} |"
    for tf in TIMEFRAMES:
        s = all_results[tf].get(yr, {})
        if not s or s.get('trades', 0) == 0:
            row += f" {'—':>12} {'—':>6} {'—':>6} {'—':>10} {'—':>10} |"
        else:
            row += (f" {s['trades']:>12} {s['wr']:>5.1f}% {s['pf']:>6.2f}"
                    f" ${s['net']:>9,.0f} ${s['max_dd']:>9,.0f} |")
    print(row)

print("-" * 100)
# Totals
row = f"{'ALL':<8} |"
for tf in TIMEFRAMES:
    s = all_results[tf]['ALL']
    row += (f" {s['trades']:>12} {s['wr']:>5.1f}% {s['pf']:>6.2f}"
            f" ${s['net']:>9,.0f} ${s['max_dd']:>9,.0f} |")
print(row)
print("=" * 100)

# ─── SUMMARY TABLE ───────────────────────────────────────────────────────────
print("\n")
print("=" * 70)
print("  SUMMARY: 8-YEAR COMBINED (5 MNQ @ $10/pt)")
print("=" * 70)
print(f"  {'Metric':<30} | {'1-Min':>12} | {'3-Min':>12} | {'5-Min':>12}")
print("-" * 70)

metrics = [
    ('Total Trades',    'trades',   '{:,}',    False),
    ('Win Rate',        'wr',       '{:.1f}%',  False),
    ('Wins',            'wins',     '{:,}',    False),
    ('Losses',          'losses',   '{:,}',    False),
    ('Profit Factor',   'pf',       '{:.2f}',  False),
    ('Net Profit',      'net',      '${:,.0f}', False),
    ('Max Drawdown',    'max_dd',   '${:,.0f}', False),
    ('Avg Win',         'avg_win',  '${:,.0f}', False),
    ('Avg Loss',        'avg_loss', '${:,.0f}', False),
]

for label, key, fmt, _ in metrics:
    vals = [all_results[tf]['ALL'].get(key, 0) for tf in TIMEFRAMES]
    row = f"  {label:<30} |"
    for v in vals:
        row += f" {fmt.format(v):>12} |"
    print(row)

print("=" * 70)
print("\nDone.")

