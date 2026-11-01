"""
Model 1 v3 — Long-Only Rejection Block (Sellside Sweep + HTF Bull Bias)
========================================================================
Based on v2 results:
  - Longs: 1/3 wins (33% WR) -> positive expectancy at 3R (+0.99R/trade)
  - Shorts: 0/7 wins (0% WR)  -> scrapped entirely

This version runs LONG-ONLY:
  - HTF 4H EMA9 must be BULL (price above EMA)
  - Price sweeps Asia/PDL sellside liquidity in 9:30-11:00 window
  - Rejection candle body >= 30% of wick (relaxed from 40%)
  - Min wick >= 4 pts (relaxed from 5 to get more samples)
  - Max stop 15 pts
  - Target 3R
  - BE trigger at 40% to TP

Also runs FundedNext $50K simulation with:
  - 1 Mini NQ ($20/pt)
  - 3 MNQ ($2/pt x 3 = $6/pt)
  - 5 MNQ ($2/pt x 5 = $10/pt)
  - Dollar-risk sizing ($150/trade, $200/trade)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from data.loader import NQDataLoader
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
POINT_VALUE  = 20.0   # 1 Mini NQ
COMMISSION   = 10.20  # per RT (comm + slippage)

MIN_WICK_PTS = 4.0    # relaxed: 4 pts min wick
MAX_STOP_PTS = 15.0   # max stop
R_MULTIPLE   = 3.0    # 3R target
BE_PCT       = 0.40   # BE trigger at 40% to TP
MAX_LOOKFWD  = 12     # bars to find CE retest
MIN_BODY_PCT = 0.30   # relaxed: body >= 30% of wick

HTF_PERIOD   = '4h'
HTF_EMA_LEN  = 9

# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_time:  pd.Timestamp
    entry_price: float
    stop_loss:   float
    take_profit: float
    risk_pts:    float
    exit_time:   pd.Timestamp = None
    exit_price:  float = 0.0
    exit_reason: str   = ''
    gross_pts:   float = 0.0
    realized_r:  float = 0.0
    net_pnl_1nq: float = 0.0   # 1 Mini NQ net

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading NQ 1-min data ...")
from data.loader import NQDataLoader
loader = NQDataLoader(target_timezone='America/New_York')
df = loader.load_csv('sample_data/MNQ_1min_20260120_20260415.csv',
                     source_timezone='UTC')

daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
df['PDH'] = daily['High'].shift(1).reindex(df.index, method='ffill')
df['PDL'] = daily['Low'].shift(1).reindex(df.index, method='ffill')

# 4H EMA (HTF Bias) — forward-filled, no lookahead
df_4h = df.resample(HTF_PERIOD).agg({'Open':'first','High':'max',
                                      'Low':'min','Close':'last'}).dropna()
df_4h['EMA9'] = df_4h['Close'].ewm(span=HTF_EMA_LEN, adjust=False).mean()
df['HTF_EMA']  = df_4h['EMA9'].reindex(df.index, method='ffill')
df['HTF_BULL'] = df['Close'] > df['HTF_EMA']

dates       = pd.Series(df.index.date, index=df.index)
unique_days = dates.unique()

# ─────────────────────────────────────────────────────────────────────────────
# Exit simulator (long-only)
# ─────────────────────────────────────────────────────────────────────────────
def simulate_long_exit(df, entry_idx, entry_price, stop_loss, take_profit, risk_pts):
    tp_dist  = take_profit - entry_price
    be_trig  = entry_price + tp_dist * BE_PCT
    be_on    = False
    sl       = stop_loss

    for i in range(entry_idx + 1, min(entry_idx + 400, len(df))):
        bar  = df.iloc[i]
        t    = df.index[i]

        if t.hour > 15 or (t.hour == 15 and t.minute >= 45):
            ep  = bar['Open']
            gp  = ep - entry_price
            return dict(exit_time=t, exit_price=ep, exit_reason='TIME_EXIT',
                        gross_pts=gp, realized_r=gp/risk_pts,
                        net_pnl_1nq=gp*POINT_VALUE - COMMISSION)

        if not be_on and bar['High'] >= be_trig:
            be_on = True;  sl = entry_price + 0.25

        if bar['Low'] <= sl:
            ep  = sl;  gp = ep - entry_price
            rsn = 'BREAKEVEN' if be_on else 'STOP_LOSS'
            return dict(exit_time=t, exit_price=ep, exit_reason=rsn,
                        gross_pts=gp, realized_r=gp/risk_pts,
                        net_pnl_1nq=gp*POINT_VALUE - COMMISSION)

        if bar['High'] >= take_profit:
            ep  = take_profit;  gp = ep - entry_price
            return dict(exit_time=t, exit_price=ep, exit_reason='TARGET',
                        gross_pts=gp, realized_r=gp/risk_pts,
                        net_pnl_1nq=gp*POINT_VALUE - COMMISSION)

    ep  = df['Close'].iloc[-1];  gp = ep - entry_price
    return dict(exit_time=df.index[-1], exit_price=ep, exit_reason='EOD',
                gross_pts=gp, realized_r=gp/risk_pts,
                net_pnl_1nq=gp*POINT_VALUE - COMMISSION)


# ─────────────────────────────────────────────────────────────────────────────
# LONG-ONLY Model 1 v3
# ─────────────────────────────────────────────────────────────────────────────
def run_long_only_rb(df, dates, unique_days) -> List[Trade]:
    trades = []
    for day in unique_days:
        day_idx = np.where(dates == day)[0]
        if len(day_idx) < 30: continue
        day_ts  = df.index[day_idx]

        # Asia range
        asia_m  = (day_ts.time >= pd.to_datetime('00:00').time()) & \
                  (day_ts.time <  pd.to_datetime('08:30').time())
        asia_i  = [day_idx[j] for j,m in enumerate(asia_m) if m]
        if len(asia_i) < 10: continue
        asia_low = df['Low'].iloc[asia_i].min()
        pdl      = df['PDL'].iloc[day_idx[0]]
        key_low  = min(asia_low, pdl) if pd.notna(pdl) else asia_low

        # Silver Bullet 9:30-11:00
        sb_m = (day_ts.time >= pd.to_datetime('09:30').time()) & \
               (day_ts.time <= pd.to_datetime('11:00').time())
        sb_i = [day_idx[j] for j,m in enumerate(sb_m) if m]
        if len(sb_i) < 5: continue

        # HTF must be BULL at session open
        if not df['HTF_BULL'].iloc[sb_i[0]]:
            continue

        traded = False
        for pos, g in enumerate(sb_i):
            if traded: break
            bar    = df.iloc[g]
            b_open = bar['Open']; b_close = bar['Close']
            b_high = bar['High']; b_low   = bar['Low']

            wick_dn   = min(b_open, b_close) - b_low
            body_size = abs(b_close - b_open)

            # LONG setup: wick sweeps below key_low, closes bullish back above
            if (b_low < key_low and
                wick_dn >= MIN_WICK_PTS and
                body_size >= wick_dn * MIN_BODY_PCT and
                b_close > b_low + MIN_WICK_PTS * 0.5 and
                b_close > b_open):

                rb_high = max(b_open, b_close)
                rb_low  = min(b_open, b_close)
                ce      = (rb_high + rb_low) / 2
                sl      = b_low - 0.50
                risk    = ce - sl
                if risk < 2.0 or risk > MAX_STOP_PTS: continue
                tp = ce + risk * R_MULTIPLE

                # Look for CE retest (price pulls back down into body from above)
                for future in sb_i[pos+1 : pos+1+MAX_LOOKFWD]:
                    fb = df.iloc[future]
                    # Entry condition: bar touches CE from above (open >= CE, low <= CE)
                    if fb['Low'] <= ce and fb['Open'] >= ce:
                        res = simulate_long_exit(df, future, ce, sl, tp, risk)
                        t   = Trade(entry_time=df.index[future],
                                    entry_price=ce, stop_loss=sl,
                                    take_profit=tp, risk_pts=risk)
                        for k, v in res.items(): setattr(t, k, v)
                        trades.append(t)
                        traded = True
                        break
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Run backtest
# ─────────────────────────────────────────────────────────────────────────────
print("Running Long-Only Rejection Block v3 (HTF Bull Bias) ...")
trades = run_long_only_rb(df, dates, unique_days)

if not trades:
    print("No trades generated — filters too tight.")
else:
    wins   = [t for t in trades if t.exit_reason == 'TARGET']
    be     = [t for t in trades if t.exit_reason == 'BREAKEVEN']
    losses = [t for t in trades if t.exit_reason == 'STOP_LOSS']
    other  = [t for t in trades if t.exit_reason in ('TIME_EXIT','EOD')]
    n      = len(trades)
    wr     = len(wins)/n*100
    net1   = sum(t.net_pnl_1nq for t in trades)
    gw     = sum(t.gross_pts*POINT_VALUE for t in trades if t.gross_pts > 0)
    gl     = abs(sum(t.gross_pts*POINT_VALUE for t in trades if t.gross_pts <= 0))
    pf     = gw/gl if gl > 0 else float('inf')
    avg_r  = np.mean([t.realized_r for t in trades])
    avg_rp = np.mean([t.risk_pts for t in trades])

    cum    = np.cumsum([t.net_pnl_1nq for t in trades])
    rm     = np.maximum.accumulate(cum)
    max_dd = (rm - cum).max()

    print(f"\n{'='*60}")
    print(f"  LONG-ONLY RB v3 — Sellside Sweep + HTF Bull Filter")
    print(f"{'='*60}")
    print(f"  Total Trades    : {n}")
    print(f"  Wins (3R)       : {len(wins)} ({wr:.1f}%)")
    print(f"  Breakeven       : {len(be)}  ({len(be)/n*100:.1f}%)")
    print(f"  Losses (SL)     : {len(losses)}  ({len(losses)/n*100:.1f}%)")
    print(f"  Time/EOD exits  : {len(other)}")
    print(f"  {'─'*40}")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Avg R / trade   : {avg_r:+.2f}R")
    print(f"  Avg Risk (pts)  : {avg_rp:.1f} pts")
    print(f"  Net P&L (1 NQ)  : ${net1:+,.2f}")
    print(f"  Max Drawdown    : ${max_dd:,.2f}")
    print(f"  {'─'*40}")
    be_wr = 100/(1+R_MULTIPLE)
    print(f"  Breakeven WR    : {be_wr:.0f}%  |  Actual: {wr:.1f}%  |  Delta: {wr-be_wr:+.1f}%")

    # ── Trade log ────────────────────────────────────────────────────────────
    print(f"\n  {'#':>3}  {'Entry Date':<20}  {'Risk':>5}  {'R':>6}  {'Exit':<12}  {'PnL (1NQ)':>10}")
    print(f"  {'─'*65}")
    for i, t in enumerate(trades):
        d = t.entry_time.strftime('%b %d %H:%M') if hasattr(t.entry_time,'strftime') else str(t.entry_time)
        print(f"  {i+1:>3}  {d:<20}  {t.risk_pts:>4.1f}  {t.realized_r:>+5.2f}R  "
              f"{t.exit_reason:<12}  {t.net_pnl_1nq:>+10.2f}")

    # ── FundedNext Challenge Sims ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FUNDEDNEXT $50K CHALLENGE SIMULATIONS")
    print(f"{'='*60}")

    def sim_challenge(trades, label, pnl_per_trade_fn):
        bal = 50000.0; peak = 50000.0; mdd = 0.0
        passed = failed = False
        pass_t = pass_date = None
        for i, t in enumerate(trades):
            pnl  = pnl_per_trade_fn(t)
            bal += pnl
            peak = max(peak, bal)
            dd   = peak - bal
            mdd  = max(mdd, dd)
            if bal >= 52500.0 and not passed:
                passed = True; pass_t = i+1
                pass_date = t.entry_time.date() if hasattr(t.entry_time,'date') else ''
            if dd >= 1500.0 and not failed:
                failed = True
        status = 'PASSED' if passed else ('FAILED' if failed else 'In progress')
        print(f"\n  [{label}]")
        print(f"  Status       : {status}")
        if passed:
            print(f"  Passed trade : #{pass_t} on {pass_date}")
        print(f"  Final Bal    : ${bal:,.2f}  (P&L: ${bal-50000:+,.2f})")
        print(f"  Max DD       : ${mdd:,.2f} / $1,500")

    # 1 Mini NQ
    sim_challenge(trades, "1 Mini NQ  ($20/pt, $10.20 comm+slip)",
                  lambda t: t.net_pnl_1nq)

    # 3 MNQ  ($2/pt × 3 = $6/pt)
    sim_challenge(trades, "3 MNQ  ($2/pt × 3, $6.20 comm/trade)",
                  lambda t: t.gross_pts * 6.0 - 6.20)

    # 5 MNQ
    sim_challenge(trades, "5 MNQ  ($2/pt × 5, $10.20 comm/trade)",
                  lambda t: t.gross_pts * 10.0 - 10.20)

    # Dollar-risk sizing: $150/trade
    def dollar_risk_pnl(t, risk_usd=150.0, pt_val=2.0, comm_per_c=6.20):
        contracts = max(1, int(round(risk_usd / (t.risk_pts * pt_val))))
        return contracts * t.gross_pts * pt_val - contracts * comm_per_c

    sim_challenge(trades, "MNQ Dollar-Risk $150/trade",
                  lambda t: dollar_risk_pnl(t, 150))
    sim_challenge(trades, "MNQ Dollar-Risk $200/trade",
                  lambda t: dollar_risk_pnl(t, 200))

    # ── Per-trade pace to $2,500 ───────────────────────────────────────────
    if net1 > 0:
        avg_per = net1 / n
        trades_needed = int(round(2500.0 / avg_per))
        tpw = n / 15.0   # trades per week estimate (15 weeks of data)
        wks = trades_needed / tpw if tpw > 0 else 999
        print(f"\n  --- Scaling Pace (1 Mini NQ) ---")
        print(f"  Avg net/trade  : ${avg_per:+.2f}")
        print(f"  Trades to +$2,500 target : ~{trades_needed}")
        print(f"  Estimated weeks : ~{wks:.1f} (at {tpw:.1f} trades/wk)")

