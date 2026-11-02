"""
Walk-Forward Out-of-Sample Test
ICT 9:30 VWAP Sniper — 5-Minute (SEALED)

Split:
  IN-SAMPLE  (IS):  2018–2022  — the "training" period
  OUT-OF-SAMPLE (OOS): 2023–2026  — never used to define rules

If OOS holds up → strategy has genuine edge, zero overfitting.
"""

import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── SEALED PARAMETERS ───────────────────────────────────────────────────────
SIGMA         = 2.0
MIN_BAND_DIST = 15.0
WICK_RATIO    = 0.35
CE_PCT        = 0.50
STOP_BEYOND   = 1.5
N_CONTRACTS   = 5
PT_VALUE      = 2.0
COMMISSION    = 0.50 * N_CONTRACTS
TF_MIN        = 5

DATA_PATH = r'c:\Users\JOSHUA\Desktop\CRTbot\data\indices\NQ-duka-2018-2026.json'

# ─── LOAD ────────────────────────────────────────────────────────────────────
print("Loading data...")
with open(DATA_PATH, 'r') as f:
    raw = json.load(f)

df_raw = pd.DataFrame(raw)
df_raw['dt'] = pd.to_datetime(df_raw['time'], unit='ms', utc=True)
df_raw = df_raw.set_index('dt').drop(columns=['time'])
df_raw.columns = ['open', 'high', 'low', 'close', 'volume']
df_raw.index = df_raw.index.tz_convert('America/New_York')
df_raw = df_raw[df_raw.index.dayofweek < 5]
df_raw['volume'] = df_raw['volume'].clip(lower=1)
print(f"Loaded {len(df_raw):,} bars | {df_raw.index[0].date()} to {df_raw.index[-1].date()}")

# ─── SPLIT ───────────────────────────────────────────────────────────────────
IS_data  = df_raw[df_raw.index.year <= 2022]
OOS_data = df_raw[df_raw.index.year >= 2023]
print(f"\nIN-SAMPLE  (2018-2022): {len(IS_data):,} bars")
print(f"OUT-OF-SAMPLE (2023-2026): {len(OOS_data):,} bars")

# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────
def resample_day(day_df, tf):
    if tf == 1:
        return day_df.copy()
    return day_df.resample(f'{tf}min', closed='left', label='left').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    ).dropna(subset=['open'])

def run_backtest(df_1min, label):
    trades = []
    dates = df_1min.groupby(df_1min.index.date)
    total_days = len(dates)

    for i, (date, day_1m) in enumerate(dates):
        day = resample_day(day_1m, TF_MIN)
        day = day.between_time('09:30', '15:44')
        if len(day) < 3:
            continue

        tp = (day['high'] + day['low'] + day['close']) / 3.0
        cum_vol = day['volume'].cumsum()
        cum_tpv = (tp * day['volume']).cumsum()
        vwap = cum_tpv / cum_vol
        cum_tp2v = ((tp**2) * day['volume']).cumsum()
        variance = ((cum_tp2v / cum_vol) - vwap**2).clip(lower=0)
        std_dev = np.sqrt(variance)
        upper = vwap + SIGMA * std_dev
        lower = vwap - SIGMA * std_dev

        signal_bars = day.between_time('09:30', '11:29')
        trade_today = False

        for ts in signal_bars.index:
            if trade_today:
                break
            idx = day.index.get_loc(ts)
            o,h,l,c = day['open'].iloc[idx], day['high'].iloc[idx], \
                      day['low'].iloc[idx], day['close'].iloc[idx]
            rng = h - l
            if rng < 0.25:
                continue

            vw, ub, lb = vwap.iloc[idx], upper.iloc[idx], lower.iloc[idx]

            direction = entry = stop = tp1 = tp2 = None

            if h >= ub and c < ub and (ub - vw) >= MIN_BAND_DIST:
                wick = h - c
                if wick / rng >= WICK_RATIO:
                    direction = 'short'
                    entry = h - CE_PCT * wick
                    stop  = h + STOP_BEYOND
                    tp1   = vw
                    tp2   = lb

            elif l <= lb and c > lb and (vw - lb) >= MIN_BAND_DIST:
                wick = c - l
                if wick / rng >= WICK_RATIO:
                    direction = 'long'
                    entry = l + CE_PCT * wick
                    stop  = l - STOP_BEYOND
                    tp1   = vw
                    tp2   = ub

            if direction is None:
                continue

            future = day.iloc[idx+1:]
            future = future[future.index.time <= pd.Timestamp('2000-01-01 15:45').time()]

            filled = False
            fill_idx = None
            for j, (fts, fb) in enumerate(future.iterrows()):
                if direction == 'short' and fb['high'] >= entry:
                    filled = True; fill_idx = j; break
                elif direction == 'long' and fb['low'] <= entry:
                    filled = True; fill_idx = j; break

            if not filled:
                continue

            trade_today = True
            post = future.iloc[fill_idx:]
            tp1_hit = False
            tp1_pnl = runner_pnl = 0.0
            runner_stop = None
            tp1_contracts = round(N_CONTRACTS * 0.60)
            runner_contracts = N_CONTRACTS - tp1_contracts

            for k, (pts, pb) in enumerate(post.iterrows()):
                ph, pl = pb['high'], pb['low']
                if not tp1_hit:
                    if direction == 'short':
                        if ph >= stop:
                            pnl = (entry - stop) * N_CONTRACTS * PT_VALUE
                            trades.append({'date':str(date),'result':'stop','pnl':pnl - COMMISSION}); break
                        elif pl <= tp1:
                            tp1_hit = True
                            tp1_pnl = (entry - tp1) * tp1_contracts * PT_VALUE
                            runner_stop = entry + 1.0
                    else:
                        if pl <= stop:
                            pnl = (entry - stop) * N_CONTRACTS * PT_VALUE
                            trades.append({'date':str(date),'result':'stop','pnl':pnl - COMMISSION}); break
                        elif ph >= tp1:
                            tp1_hit = True
                            tp1_pnl = (tp1 - entry) * tp1_contracts * PT_VALUE
                            runner_stop = entry - 1.0
                else:
                    if direction == 'short':
                        if ph >= runner_stop:
                            runner_pnl = (entry - runner_stop) * runner_contracts * PT_VALUE
                            trades.append({'date':str(date),'result':'tp1+be','pnl':tp1_pnl+runner_pnl-COMMISSION}); break
                        elif pl <= tp2:
                            runner_pnl = (entry - tp2) * runner_contracts * PT_VALUE
                            trades.append({'date':str(date),'result':'tp2','pnl':tp1_pnl+runner_pnl-COMMISSION}); break
                    else:
                        if pl <= runner_stop:
                            runner_pnl = (runner_stop - entry) * runner_contracts * PT_VALUE
                            trades.append({'date':str(date),'result':'tp1+be','pnl':tp1_pnl+runner_pnl-COMMISSION}); break
                        elif ph >= tp2:
                            runner_pnl = (tp2 - entry) * runner_contracts * PT_VALUE
                            trades.append({'date':str(date),'result':'tp2','pnl':tp1_pnl+runner_pnl-COMMISSION}); break
            else:
                # EOD flat
                last = post['close'].iloc[-1]
                if not tp1_hit:
                    pnl = (entry - last)*N_CONTRACTS*PT_VALUE if direction=='short' else (last - entry)*N_CONTRACTS*PT_VALUE
                    trades.append({'date':str(date),'result':'eod','pnl':pnl - COMMISSION})
                else:
                    runner_pnl = (entry - last)*runner_contracts*PT_VALUE if direction=='short' else (last - entry)*runner_contracts*PT_VALUE
                    trades.append({'date':str(date),'result':'tp1+eod','pnl':tp1_pnl+runner_pnl-COMMISSION})

    return trades

def stats(trades, label):
    if not trades:
        print(f"{label}: No trades"); return
    df = pd.DataFrame(trades)
    wins   = df[df['pnl'] > 0]['pnl']
    losses = df[df['pnl'] <= 0]['pnl']
    gp = wins.sum(); gl = abs(losses.sum())
    pf = gp / gl if gl > 0 else 999
    equity = df['pnl'].cumsum()
    max_dd = (equity - equity.cummax()).min()

    # Monthly breakdown
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly = df.groupby('month')['pnl'].sum()
    positive_months = (monthly > 0).sum()
    total_months = len(monthly)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades        : {len(df):,}")
    print(f"  Win Rate      : {len(wins)/len(df)*100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit Factor : {pf:.2f}")
    print(f"  Net Profit    : ${df['pnl'].sum():,.0f}")
    print(f"  Max Drawdown  : ${max_dd:,.0f}")
    print(f"  Avg Win       : ${wins.mean():,.0f}")
    print(f"  Avg Loss      : ${losses.mean():,.0f}")
    print(f"  Positive Months: {positive_months}/{total_months}")
    print(f"{'='*60}")

    # Year breakdown
    df['year'] = pd.to_datetime(df['date']).dt.year
    print(f"\n  Year-by-year:")
    for yr, grp in df.groupby('year'):
        w = (grp['pnl'] > 0).sum()
        wr = w / len(grp) * 100
        net = grp['pnl'].sum()
        gl2 = abs(grp[grp['pnl']<=0]['pnl'].sum())
        gp2 = grp[grp['pnl']>0]['pnl'].sum()
        pf2 = gp2/gl2 if gl2 > 0 else 999
        print(f"    {yr}: {len(grp):>3} trades | {wr:>5.1f}% WR | PF {pf2:.2f} | Net ${net:>8,.0f}")

# ─── RUN ─────────────────────────────────────────────────────────────────────
print("\nRunning IN-SAMPLE (2018-2022)...")
is_trades = run_backtest(IS_data, "IS")
stats(is_trades, "IN-SAMPLE (2018-2022) — 5 MNQ @ $10/pt")

print("\nRunning OUT-OF-SAMPLE (2023-2026)...")
oos_trades = run_backtest(OOS_data, "OOS")
stats(oos_trades, "OUT-OF-SAMPLE (2023-2026) — 5 MNQ @ $10/pt  [BLIND TEST]")

# ─── HEAD-TO-HEAD VERDICT ────────────────────────────────────────────────────
def summary_stats(trades):
    df = pd.DataFrame(trades)
    wins = df[df['pnl'] > 0]['pnl']
    losses = df[df['pnl'] <= 0]['pnl']
    gp = wins.sum(); gl = abs(losses.sum())
    return {
        'trades': len(df),
        'wr': len(wins)/len(df)*100,
        'pf': gp/gl if gl > 0 else 999,
        'net': df['pnl'].sum(),
        'dd': (df['pnl'].cumsum() - df['pnl'].cumsum().cummax()).min()
    }

is_s  = summary_stats(is_trades)
oos_s = summary_stats(oos_trades)

print("\n")
print("=" * 65)
print("  WALK-FORWARD VERDICT — Does the edge survive unseen data?")
print("=" * 65)
print(f"  {'Metric':<22} | {'IN-SAMPLE':>14} | {'OUT-OF-SAMPLE':>14}")
print("-" * 65)
print(f"  {'Period':<22} | {'2018-2022':>14} | {'2023-2026':>14}")
print(f"  {'Trades':<22} | {is_s['trades']:>14,} | {oos_s['trades']:>14,}")
print(f"  {'Win Rate':<22} | {is_s['wr']:>13.1f}% | {oos_s['wr']:>13.1f}%")
print(f"  {'Profit Factor':<22} | {is_s['pf']:>14.2f} | {oos_s['pf']:>14.2f}")
print(f"  {'Net Profit':<22} | ${is_s['net']:>13,.0f} | ${oos_s['net']:>13,.0f}")
print(f"  {'Max Drawdown':<22} | ${is_s['dd']:>13,.0f} | ${oos_s['dd']:>13,.0f}")
print("=" * 65)

# Drift check
wr_drift = oos_s['wr'] - is_s['wr']
pf_drift = oos_s['pf'] - is_s['pf']
print(f"\n  Win Rate drift  (OOS - IS): {wr_drift:+.1f}%")
print(f"  PF drift        (OOS - IS): {pf_drift:+.2f}")
if oos_s['wr'] >= 55 and oos_s['pf'] >= 2.0 and oos_s['net'] > 0:
    print("\n  VERDICT: OUT-OF-SAMPLE PASSED.")
    print("  Edge is REAL. Strategy is NOT overfit. Mic drop.")
else:
    print("\n  VERDICT: OOS degraded significantly — review parameters.")
print()

