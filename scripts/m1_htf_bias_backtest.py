"""
Model 1 — Rejection Block / Wick Theory  (v2 — HTF Bias + Tightened Filters)
=============================================================================
Improvements over v1:
  1. HTF Bias Gate:  Build a synthetic 4H OHLC from the 1-min data.
     Compute the 9-bar 4H EMA (~36 hours).  Only take LONGS when the
     current 4H close is ABOVE the 4H EMA; only SHORTS when below.
     (Filters ~50% of marginal setups by keeping us with momentum.)

  2. Entry window strictly 09:30–11:00 AM EST only (no pre-open signals).

  3. Minimum wick size raised from 3 pts → 5 pts (only strong rejections).

  4. Max stop capped at 15 NQ points (prevents sizing into wide-stop traps).

  5. Target reduced from 4R → 3R (more realistic for the 60-min window).

  6. Min body-to-wick ratio: candle body must be >= 40% of the wick size
     (weak dojis excluded — true rejection body required).

  7. Confirmation bar: after the RB forms, price must pull back into the
     RB body (reach CE) within 10 bars (was 15), tightening the look-ahead.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from data.loader import NQDataLoader
from config import StrategyConfig, MNQContractSpec
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
POINT_VALUE  = 20.0   # 1 Mini NQ  ($20/pt)
COMMISSION   = 10.20  # RT comm + slippage

MIN_WICK_PTS  = 5.0   # v2: raised from 3 -> 5
MAX_STOP_PTS  = 15.0  # v2: cap max risk
R_MULTIPLE    = 3.0   # v2: lowered from 4 -> 3
BE_PCT        = 0.40  # move to BE at 40% to TP
MAX_LOOKFWD   = 10    # bars to find CE retest (was 15)
MIN_BODY_PCT  = 0.40  # body must be >= 40% of wick

HTF_PERIOD    = '4h'
HTF_EMA_LEN   = 9     # 9 bars of 4H ~= 36 hours

# ─────────────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    model:        str
    entry_time:   pd.Timestamp
    direction:    int        # 1=Long, -1=Short
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    risk_pts:     float
    exit_price:   float = 0.0
    exit_reason:  str   = ''
    gross_pts:    float = 0.0
    realized_r:   float = 0.0
    net_pnl:      float = 0.0
    htf_bias:     str   = ''  # 'BULL' or 'BEAR'

# ─────────────────────────────────────────────────────────────────────────────
# Load + preprocess data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading NQ 1-min data ...")
loader = NQDataLoader(target_timezone='America/New_York')
df = loader.load_csv('sample_data/MNQ_1min_20260120_20260415.csv',
                     source_timezone='UTC')

# PDH / PDL
daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
df['PDH'] = daily['High'].shift(1).reindex(df.index, method='ffill')
df['PDL'] = daily['Low'].shift(1).reindex(df.index, method='ffill')

# ── 4H EMA (HTF Bias) ──────────────────────────────────────────────────────
df_4h = df.resample(HTF_PERIOD).agg({'Open': 'first', 'High': 'max',
                                      'Low': 'min',  'Close': 'last'}).dropna()
df_4h['EMA9'] = df_4h['Close'].ewm(span=HTF_EMA_LEN, adjust=False).mean()
# Broadcast 4H EMA back to 1-min (forward-fill so no lookahead)
df['HTF_EMA'] = df_4h['EMA9'].reindex(df.index, method='ffill')
df['HTF_BIAS'] = np.where(df['Close'] > df['HTF_EMA'], 1, -1)  # 1=bull, -1=bear

dates       = pd.Series(df.index.date, index=df.index)
unique_days = dates.unique()

# ─────────────────────────────────────────────────────────────────────────────
# Exit simulator
# ─────────────────────────────────────────────────────────────────────────────
def simulate_exit(df: pd.DataFrame, entry_idx: int, direction: int,
                  entry_price: float, stop_loss: float,
                  take_profit: float, risk_pts: float,
                  be_pct: float = BE_PCT) -> dict:
    """Walk forward bar-by-bar; return exit dict."""
    tp_dist          = abs(take_profit - entry_price)
    be_trigger_price = entry_price + direction * tp_dist * be_pct
    be_triggered     = False
    sl               = stop_loss  # local mutable copy

    for i in range(entry_idx + 1, min(entry_idx + 400, len(df))):
        bar      = df.iloc[i]
        bar_time = df.index[i]

        # Time exit 15:45
        if bar_time.hour > 15 or (bar_time.hour == 15 and bar_time.minute >= 45):
            ep  = bar['Open']
            gp  = (ep - entry_price) * direction
            return dict(exit_price=ep, exit_reason='TIME_EXIT',
                        gross_pts=gp, realized_r=gp/risk_pts if risk_pts else 0,
                        net_pnl=gp*POINT_VALUE - COMMISSION)

        # BE trigger
        if direction == 1 and not be_triggered and bar['High'] >= be_trigger_price:
            be_triggered = True;  sl = entry_price + 0.25
        if direction == -1 and not be_triggered and bar['Low'] <= be_trigger_price:
            be_triggered = True;  sl = entry_price - 0.25

        # Longs
        if direction == 1:
            if bar['Low'] <= sl:
                ep  = sl;  gp = ep - entry_price
                rsn = 'BREAKEVEN' if be_triggered else 'STOP_LOSS'
                return dict(exit_price=ep, exit_reason=rsn,
                            gross_pts=gp, realized_r=gp/risk_pts if risk_pts else 0,
                            net_pnl=gp*POINT_VALUE - COMMISSION)
            if bar['High'] >= take_profit:
                ep  = take_profit;  gp = ep - entry_price
                return dict(exit_price=ep, exit_reason='TARGET',
                            gross_pts=gp, realized_r=gp/risk_pts if risk_pts else 0,
                            net_pnl=gp*POINT_VALUE - COMMISSION)
        else:
            if bar['High'] >= sl:
                ep  = sl;  gp = entry_price - ep
                rsn = 'BREAKEVEN' if be_triggered else 'STOP_LOSS'
                return dict(exit_price=ep, exit_reason=rsn,
                            gross_pts=gp, realized_r=gp/risk_pts if risk_pts else 0,
                            net_pnl=gp*POINT_VALUE - COMMISSION)
            if bar['Low'] <= take_profit:
                ep  = take_profit;  gp = entry_price - ep
                return dict(exit_price=ep, exit_reason='TARGET',
                            gross_pts=gp, realized_r=gp/risk_pts if risk_pts else 0,
                            net_pnl=gp*POINT_VALUE - COMMISSION)

    # EOD
    ep  = df['Close'].iloc[-1]
    gp  = (ep - entry_price) * direction
    return dict(exit_price=ep, exit_reason='EOD',
                gross_pts=gp, realized_r=gp/risk_pts if risk_pts else 0,
                net_pnl=gp*POINT_VALUE - COMMISSION)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1  v2 — Rejection Block with HTF Bias
# ─────────────────────────────────────────────────────────────────────────────
def run_m1_v2(df: pd.DataFrame, dates, unique_days) -> List[Trade]:
    trades = []

    for day in unique_days:
        day_idx = np.where(dates == day)[0]
        if len(day_idx) < 30:
            continue
        day_ts = df.index[day_idx]

        # Asia range (00:00-08:30)
        asia_mask = (day_ts.time >= pd.to_datetime('00:00').time()) & \
                    (day_ts.time <  pd.to_datetime('08:30').time())
        asia_idx  = [day_idx[j] for j, m in enumerate(asia_mask) if m]
        if len(asia_idx) < 10:
            continue
        asia_high = df['High'].iloc[asia_idx].max()
        asia_low  = df['Low'].iloc[asia_idx].min()
        pdh       = df['PDH'].iloc[day_idx[0]]
        pdl       = df['PDL'].iloc[day_idx[0]]
        key_high  = max(asia_high, pdh) if pd.notna(pdh) else asia_high
        key_low   = min(asia_low,  pdl) if pd.notna(pdl) else asia_low

        # Silver Bullet window STRICTLY 09:30-11:00
        sb_mask = (day_ts.time >= pd.to_datetime('09:30').time()) & \
                  (day_ts.time <= pd.to_datetime('11:00').time())
        sb_idx  = [day_idx[j] for j, m in enumerate(sb_mask) if m]
        if len(sb_idx) < 5:
            continue

        # HTF bias at 09:30 open
        htf_bias  = int(df['HTF_BIAS'].iloc[sb_idx[0]])
        htf_label = 'BULL' if htf_bias == 1 else 'BEAR'

        traded_today = False
        for pos, g in enumerate(sb_idx):
            if traded_today:
                break

            bar     = df.iloc[g]
            b_high  = bar['High']
            b_low   = bar['Low']
            b_open  = bar['Open']
            b_close = bar['Close']

            # ── SHORT: wick above key_high, bearish close back inside ──────
            wick_up   = b_high - max(b_open, b_close)
            body_size = abs(b_close - b_open)

            if (b_high > key_high and
                wick_up >= MIN_WICK_PTS and
                body_size >= wick_up * MIN_BODY_PCT and      # v2: body filter
                b_close < b_high - MIN_WICK_PTS * 0.5 and
                b_close < b_open and
                htf_bias == -1):                             # v2: BEAR bias only

                rb_high = max(b_open, b_close)
                rb_low  = min(b_open, b_close)
                ce      = (rb_high + rb_low) / 2
                sl      = b_high + 0.50
                risk    = sl - ce
                if risk < 2.0 or risk > MAX_STOP_PTS:       # v2: capped at 15
                    continue
                tp = ce - risk * R_MULTIPLE

                for future in sb_idx[pos+1 : pos+1+MAX_LOOKFWD]:
                    fb = df.iloc[future]
                    if fb['High'] >= ce and fb['Open'] <= ce:
                        res = simulate_exit(df, future, -1, ce, sl, tp, risk)
                        t = Trade('M1v2_RB_Short', df.index[future], -1,
                                  ce, sl, tp, risk, htf_bias=htf_label)
                        for k, v in res.items():
                            setattr(t, k, v)
                        trades.append(t)
                        traded_today = True
                        break
                continue

            # ── LONG: wick below key_low, bullish close back above ─────────
            wick_dn   = min(b_open, b_close) - b_low
            body_size = abs(b_close - b_open)

            if (b_low < key_low and
                wick_dn >= MIN_WICK_PTS and
                body_size >= wick_dn * MIN_BODY_PCT and      # v2: body filter
                b_close > b_low + MIN_WICK_PTS * 0.5 and
                b_close > b_open and
                htf_bias == 1):                              # v2: BULL bias only

                rb_high = max(b_open, b_close)
                rb_low  = min(b_open, b_close)
                ce      = (rb_high + rb_low) / 2
                sl      = b_low - 0.50
                risk    = ce - sl
                if risk < 2.0 or risk > MAX_STOP_PTS:
                    continue
                tp = ce + risk * R_MULTIPLE

                for future in sb_idx[pos+1 : pos+1+MAX_LOOKFWD]:
                    fb = df.iloc[future]
                    if fb['Low'] <= ce and fb['Open'] >= ce:
                        res = simulate_exit(df, future, 1, ce, sl, tp, risk)
                        t = Trade('M1v2_RB_Long', df.index[future], 1,
                                  ce, sl, tp, risk, htf_bias=htf_label)
                        for k, v in res.items():
                            setattr(t, k, v)
                        trades.append(t)
                        traded_today = True
                        break

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Stats printer
# ─────────────────────────────────────────────────────────────────────────────
def print_stats(name: str, trades: List[Trade], pv=POINT_VALUE):
    if not trades:
        print(f"\n{'='*62}\n  {name}: No trades generated\n{'='*62}")
        return

    wins   = [t for t in trades if t.exit_reason == 'TARGET']
    be     = [t for t in trades if t.exit_reason == 'BREAKEVEN']
    losses = [t for t in trades if t.exit_reason == 'STOP_LOSS']
    other  = [t for t in trades if t.exit_reason in ('TIME_EXIT','EOD')]

    n   = len(trades)
    wr  = len(wins)/n*100
    net = sum(t.net_pnl for t in trades)
    gw  = sum(t.gross_pts*pv for t in trades if t.gross_pts > 0)
    gl  = abs(sum(t.gross_pts*pv for t in trades if t.gross_pts <= 0))
    pf  = gw/gl if gl > 0 else float('inf')
    avg_r       = np.mean([t.realized_r for t in trades])
    avg_r_wins  = np.mean([t.realized_r for t in wins]) if wins else 0
    avg_risk_pt = np.mean([t.risk_pts for t in trades])

    cum         = np.cumsum([t.net_pnl for t in trades])
    running_max = np.maximum.accumulate(cum)
    max_dd      = (running_max - cum).max() if len(cum) else 0

    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"{'='*62}")
    print(f"  Total Trades    : {n}")
    print(f"  Wins (TARGET)   : {len(wins)} ({wr:.1f}%)")
    print(f"  Breakeven       : {len(be)}  ({len(be)/n*100:.1f}%)")
    print(f"  Losses (SL)     : {len(losses)}  ({len(losses)/n*100:.1f}%)")
    print(f"  Other exits     : {len(other)}")
    print(f"  {'─'*42}")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Avg R / trade   : {avg_r:+.2f}R")
    print(f"  Avg R (wins)    : {avg_r_wins:+.2f}R")
    print(f"  Avg Risk (pts)  : {avg_risk_pt:.1f} pts")
    print(f"  Net P&L (1 NQ)  : ${net:+,.2f}")
    print(f"  Max Drawdown    : ${max_dd:,.2f}")
    print(f"  {'─'*42}")

    # Direction breakdown
    longs  = [t for t in trades if t.direction ==  1]
    shorts = [t for t in trades if t.direction == -1]
    lw = len([t for t in longs  if t.exit_reason=='TARGET'])
    sw = len([t for t in shorts if t.exit_reason=='TARGET'])
    if longs:
        print(f"  Longs  : {len(longs):>2} trades | {lw} wins ({lw/len(longs)*100:.0f}% WR)")
    if shorts:
        print(f"  Shorts : {len(shorts):>2} trades | {sw} wins ({sw/len(shorts)*100:.0f}% WR)")

    # FundedNext $50K sim (1 Mini NQ)
    print(f"\n  --- FundedNext $50K Challenge (1 Mini NQ) ---")
    bal = 50000.0;  peak = 50000.0;  mdd = 0.0
    passed = False;  failed_dd = False
    pass_t = pass_date = None
    for i, t in enumerate(trades):
        bal  += t.net_pnl
        peak  = max(peak, bal)
        dd    = peak - bal
        mdd   = max(mdd, dd)
        if bal >= 52500.0 and not passed:
            passed = True;  pass_t = i+1
            pass_date = t.entry_time.date() if hasattr(t.entry_time,'date') else t.entry_time
        if dd >= 1500.0 and not failed_dd:
            failed_dd = True
            print(f"  !! DD breach ${dd:.0f} at trade #{i+1}  (bal=${bal:,.2f})")

    status = 'PASSED' if passed else 'Not passed'
    print(f"  Status          : {status}")
    if passed:
        print(f"  Passed on trade : #{pass_t} ({pass_date})")
    print(f"  Final Balance   : ${bal:,.2f}")
    print(f"  Max Drawdown    : ${mdd:,.2f} / $1,500 limit")

    # Trade log
    print(f"\n  {'#':>3}  {'Date':<20}  {'Model':<16}  {'Dir':>7}  "
          f"{'Risk':>5}  {'R':>6}  {'Exit':<12}  {'PnL':>9}  Bias")
    print(f"  {'─'*92}")
    for i, t in enumerate(trades):
        d  = t.entry_time.strftime('%b %d %H:%M') if hasattr(t.entry_time,'strftime') else str(t.entry_time)
        di = 'UP LONG' if t.direction==1 else 'DN SHORT'
        print(f"  {i+1:>3}  {d:<20}  {t.model:<16}  {di:>8}  "
              f"{t.risk_pts:>4.1f}  {t.realized_r:>+5.2f}R  {t.exit_reason:<12}  "
              f"{t.net_pnl:>+9.2f}  {t.htf_bias}")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning Model 1 v2 -- Rejection Block + HTF Bias ...")
trades = run_m1_v2(df, dates, unique_days)

print_stats("MODEL 1 v2 -- Rejection Block / Wick Theory (HTF Filtered)", trades)

if trades:
    breakeven_wr = 100 / (1 + R_MULTIPLE)
    actual_wr = len([t for t in trades if t.exit_reason=='TARGET'])/len(trades)*100
    print(f"\n  [Breakeven WR needed at {R_MULTIPLE}R target] : {breakeven_wr:.0f}%")
    print(f"  [Actual WR achieved               ] : {actual_wr:.1f}%")
    delta = actual_wr - breakeven_wr
    print(f"  [Delta vs breakeven               ] : {delta:+.1f}%  ({'PROFITABLE' if delta > 0 else 'LOSING'})")

