"""
12-Month Backtest — External Sweep + 10am Macro Model
======================================================
Runs on the full 252-day synthetic dataset.
Reports: monthly breakdown, full stats, FundedNext simulation,
         per-contract sizing, and equity curve data.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from data.loader import NQDataLoader
from strategy.rules import ICTStrategyLogic, TradeSignal
from execution.simulator import NQIntrabarSimulator
from config import StrategyConfig, MNQContractSpec
import pandas as pd
import numpy as np

# ── Load 12-month data ────────────────────────────────────────────────────────
print("Loading 12-month NQ 1-min data ...")
loader = NQDataLoader(target_timezone='America/New_York')
df = loader.load_csv('sample_data/MNQ_1min_12mo.csv', source_timezone='UTC')

daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
df['PDH'] = daily['High'].shift(1).reindex(df.index, method='ffill')
df['PDL'] = daily['Low'].shift(1).reindex(df.index, method='ffill')

dates       = pd.Series(df.index.date, index=df.index)
unique_days = dates.unique()
print(f"  {len(unique_days)} trading days loaded\n")

# ── Signal generation ─────────────────────────────────────────────────────────
signals = []
for day in unique_days:
    day_indices = np.where(dates == day)[0]
    if len(day_indices) == 0: continue
    day_ts = df.iloc[day_indices].index

    on_mask = (day_ts.time >= pd.to_datetime('00:00:00').time()) & \
              (day_ts.time <= pd.to_datetime('08:30:00').time())
    idx_on  = [day_indices[j] for j, m in enumerate(on_mask) if m]
    if len(idx_on) < 10: continue
    on_high = df['High'].iloc[idx_on].max()
    on_low  = df['Low'].iloc[idx_on].min()
    pdh = df['PDH'].iloc[idx_on[0]]
    pdl = df['PDL'].iloc[idx_on[0]]

    ny_pre_mask = (day_ts.time >= pd.to_datetime('08:30:00').time()) & \
                  (day_ts.time <= pd.to_datetime('10:00:00').time())
    idx_pre = [day_indices[j] for j, m in enumerate(ny_pre_mask) if m]
    if not idx_pre: continue
    pre_high = df['High'].iloc[idx_pre].max()
    pre_low  = df['Low'].iloc[idx_pre].min()

    swept_buyside  = (pre_high > on_high) or (pd.notna(pdh) and pre_high > pdh)
    swept_sellside = (pre_low  < on_low)  or (pd.notna(pdl) and pre_low  < pdl)

    entry_mask = (day_ts.time >= pd.to_datetime('10:00:00').time()) & \
                 (day_ts.time <= pd.to_datetime('10:30:00').time())
    entry_indices = [day_indices[j] for j, m in enumerate(entry_mask) if m]
    if not entry_indices: continue
    open_10am = df['Open'].iloc[entry_indices[0]]

    mark_mask = (day_ts.time >= pd.to_datetime('09:30:00').time()) & \
                (day_ts.time <= pd.to_datetime('09:59:00').time())
    idx_mark  = [day_indices[j] for j, m in enumerate(mark_mask) if m]
    if len(idx_mark) < 5: continue
    df_mark = df.iloc[idx_mark]

    strat_cfg = StrategyConfig(target_r_multiple=2.0, breakeven_trigger_pct=0.5,
                               max_risk_points=40.0)
    logic = ICTStrategyLogic(strategy_config=strat_cfg, contract_spec=MNQContractSpec())
    raw = (logic.engine.detect_order_blocks(df_mark, idx_mark) +
           logic.engine.detect_breaker_blocks(df_mark, idx_mark) +
           logic.engine.detect_rejection_blocks(df_mark, idx_mark))

    valid_pre = []
    for b in raw:
        touched = False
        for g in idx_mark:
            if g > b.formation_idx:
                if b.direction == 1  and df['Low'].iloc[g]  <= b.near_edge: touched = True; break
                if b.direction == -1 and df['High'].iloc[g] >= b.near_edge: touched = True; break
        if not touched: valid_pre.append(b)

    bulls = [b for b in valid_pre if b.direction == 1  and b.near_edge <= open_10am] if swept_sellside else []
    bears = [b for b in valid_pre if b.direction == -1 and b.near_edge >= open_10am] if swept_buyside  else []
    best_long  = logic._select_best_block(bulls, open_10am) if bulls else None
    best_short = logic._select_best_block(bears, open_10am) if bears else None

    for g in entry_indices:
        b_dt   = df.index[g]
        b_open = df['Open'].iloc[g]
        b_low  = df['Low'].iloc[g]
        b_high = df['High'].iloc[g]

        if best_long and b_low <= best_long.near_edge and b_open >= best_long.near_edge:
            lvl = logic.calculate_trade_levels(1, best_long.near_edge, best_long.far_edge)
            if lvl:
                sl, tp, be, rsk = lvl
                signals.append(TradeSignal(g, b_dt, 1, best_long.near_edge,
                                           sl, tp, be, rsk, 'Sweep Long', best_long.block_type))
                break

        if best_short and b_high >= best_short.near_edge and b_open <= best_short.near_edge:
            lvl = logic.calculate_trade_levels(-1, best_short.near_edge, best_short.far_edge)
            if lvl:
                sl, tp, be, rsk = lvl
                signals.append(TradeSignal(g, b_dt, -1, best_short.near_edge,
                                           sl, tp, be, rsk, 'Sweep Short', best_short.block_type))
                break

print(f"Signals generated: {len(signals)}")

# ── Run simulator ─────────────────────────────────────────────────────────────
sim    = NQIntrabarSimulator(strategy_config=StrategyConfig(target_r_multiple=2.0,
                              breakeven_trigger_pct=0.5), contract_spec=MNQContractSpec())
trades = sim.run(df, signals)
print(f"Trades executed  : {len(trades)}\n")

if not trades:
    print("No trades — check data or signal logic.")
    exit()

# ── Overall Stats ─────────────────────────────────────────────────────────────
wins   = [t for t in trades if t.exit_reason == 'TARGET_2R']
be     = [t for t in trades if t.exit_reason == 'BREAKEVEN']
losses = [t for t in trades if t.exit_reason == 'STOP_LOSS']
longs  = [t for t in trades if t.direction ==  1]
shorts = [t for t in trades if t.direction == -1]
lw     = [t for t in longs  if t.exit_reason == 'TARGET_2R']
sw     = [t for t in shorts if t.exit_reason == 'TARGET_2R']
n      = len(trades)

net_mnq  = sum(t.net_pnl for t in trades)          # 1 MNQ
gross_pts= sum(t.gross_points for t in trades)
gw_usd   = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
gl_usd   = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
pf       = gw_usd / gl_usd if gl_usd > 0 else float('inf')
avg_r    = np.mean([t.realized_r for t in trades])
avg_risk = np.mean([t.risk_points for t in trades])
cum_mnq  = pd.Series([t.net_pnl for t in trades]).cumsum()
max_dd   = (cum_mnq.cummax() - cum_mnq).max()

print("=" * 62)
print("  12-MONTH BACKTEST — External Sweep + 10am Macro")
print("=" * 62)
print(f"  Period          : {trades[0].entry_time.date()} to {trades[-1].entry_time.date()}")
print(f"  Trading days    : {len(unique_days)}")
print(f"  Total Trades    : {n}  ({n/len(unique_days)*5:.1f}/week)")
print(f"  Wins (2R)       : {len(wins)} ({len(wins)/n*100:.1f}%)")
print(f"  Breakeven       : {len(be)}  ({len(be)/n*100:.1f}%)")
print(f"  Losses          : {len(losses)}  ({len(losses)/n*100:.1f}%)")
print(f"  {'─'*42}")
print(f"  Profit Factor   : {pf:.2f}")
print(f"  Avg R / trade   : {avg_r:+.2f}R")
print(f"  Avg stop (pts)  : {avg_risk:.1f} pts")
print(f"  Avg stop ($NQ)  : ${avg_risk*20:.0f}  |  ($MNQ) ${avg_risk*2:.0f}")
print(f"  {'─'*42}")
print(f"  Longs  : {len(longs):>3} trades | {len(lw)} wins ({len(lw)/len(longs)*100:.0f}% WR)" if longs else "")
print(f"  Shorts : {len(shorts):>3} trades | {len(sw)} wins ({len(sw)/len(shorts)*100:.0f}% WR)" if shorts else "")
print(f"  {'─'*42}")
print(f"  Net P&L — 1 MNQ  : ${net_mnq:>+10,.2f}   (risk ${avg_risk*2:.0f}/trade)")
print(f"  Net P&L — 1 NQ   : ${net_mnq*10:>+10,.2f}   (risk ${avg_risk*20:.0f}/trade)")
print(f"  Max Drawdown(MNQ): ${max_dd:>10,.2f}")
print(f"  Max Drawdown(NQ) : ${max_dd*10:>10,.2f}")

# ── Monthly breakdown ─────────────────────────────────────────────────────────
print(f"\n  --- Monthly Breakdown (1 MNQ) ---")
print(f"  {'Month':<12}  {'Trades':>6}  {'Wins':>5}  {'WR':>6}  {'Net P&L':>10}  {'Cum P&L':>10}")
print(f"  {'─'*55}")
monthly = {}
for t in trades:
    key = t.entry_time.strftime('%Y-%m')
    monthly.setdefault(key, []).append(t)
cum = 0.0
for mo, mt in sorted(monthly.items()):
    mw   = len([t for t in mt if t.exit_reason == 'TARGET_2R'])
    mnet = sum(t.net_pnl for t in mt)
    cum += mnet
    wr   = mw/len(mt)*100
    label = pd.to_datetime(mo).strftime('%b %Y')
    flag  = ' <-- loss month' if mnet < 0 else ''
    print(f"  {label:<12}  {len(mt):>6}  {mw:>5}  {wr:>5.0f}%  {mnet:>+10,.2f}  {cum:>+10,.2f}{flag}")

# ── FundedNext $50K Challenge ─────────────────────────────────────────────────
print(f"\n  --- FundedNext $50K Challenge Simulations ---")
print(f"  Target: +$2,500 | Max DD: $2,000 | Min days: 5")
print()

for label, pt_val, comm in [
    ("1 MNQ  ($2/pt)",    2.0,  6.20),
    ("3 MNQ  ($6/pt)",    6.0,  6.20*3),
    ("5 MNQ ($10/pt)",   10.0,  6.20*5),
    ("1 NQ  ($20/pt)",   20.0, 10.20),
]:
    bal = 50000.0; peak = 50000.0; mdd = 0.0
    passed = failed = False
    pass_t = pass_date = None
    fail_t = None
    for i, t in enumerate(trades):
        pnl  = t.gross_points * pt_val - comm
        bal += pnl
        peak = max(peak, bal)
        dd   = peak - bal
        mdd  = max(mdd, dd)
        if not passed and bal >= 52500.0:
            passed = True; pass_t = i+1; pass_date = t.entry_time.date()
        if not failed and dd >= 2000.0:
            failed = True; fail_t = i+1

    net_total = bal - 50000
    avg_pt    = net_total / n
    status = 'PASSED' if passed else ('FAILED DD' if failed else 'In progress')
    print(f"  {label:<16} : {status:<12} | "
          f"Net ${net_total:>+8,.0f} | DD ${mdd:>6,.0f} | "
          f"Avg/trade ${avg_pt:>+6.0f}", end='')
    if passed: print(f" | Trade #{pass_t} ({pass_date})", end='')
    print()

# ── Key summary ───────────────────────────────────────────────────────────────
tpw = n / 52.0
avg_nq = net_mnq * 10 / n
trades_to_2500 = int(round(2500 / avg_nq)) if avg_nq > 0 else 9999
wks = trades_to_2500 / tpw
print(f"\n  --- 1 NQ Scaling Summary ---")
print(f"  Trades/week          : {tpw:.1f}")
print(f"  Avg net/trade (1 NQ) : ${avg_nq:+.2f}")
if avg_nq > 0:
    print(f"  Trades to +$2,500    : ~{trades_to_2500}  (~{wks:.0f} weeks / ~{wks/4.3:.1f} months)")
else:
    print(f"  Strategy is NOT profitable on this dataset.")

# ── Save trade log ────────────────────────────────────────────────────────────
rows = []
for t in trades:
    rows.append({
        'date':        t.entry_time.date(),
        'direction':   'LONG' if t.direction==1 else 'SHORT',
        'entry':       t.entry_time,
        'exit':        t.exit_time,
        'risk_pts':    t.risk_points,
        'gross_pts':   t.gross_points,
        'realized_r':  t.realized_r,
        'exit_reason': t.exit_reason,
        'net_mnq':     t.net_pnl,
        'net_nq':      t.gross_points * 20.0 - 10.20,
    })
pd.DataFrame(rows).to_csv('trade_log_12mo.csv', index=False)
print(f"\n  Trade log saved -> trade_log_12mo.csv")

