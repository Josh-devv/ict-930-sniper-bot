"""
External Sweep + 10am Macro Model — HTF Bias Comparison
========================================================
Runs the working External Sweep model in two modes:
  A) BASELINE  — original, no HTF filter (all signals accepted)
  B) HTF GATED — only take longs when 4H EMA9 is bullish,
                 only take shorts when 4H EMA9 is bearish

Side-by-side stats + FundedNext $50K challenge sim for both.
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

# ─────────────────────────────────────────────────────────────────────────────
# Load data + shared preprocessing
# ─────────────────────────────────────────────────────────────────────────────
print("Loading NQ 1-min data ...")
loader = NQDataLoader(target_timezone='America/New_York')
df = loader.load_csv('sample_data/MNQ_1min_20260120_20260415.csv',
                     source_timezone='UTC')

daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
df['PDH'] = daily['High'].shift(1).reindex(df.index, method='ffill')
df['PDL'] = daily['Low'].shift(1).reindex(df.index, method='ffill')

# 4H EMA9 (HTF Bias) — forward-fill, zero lookahead
df_4h = df.resample('4h').agg({'Open':'first','High':'max',
                                'Low':'min','Close':'last'}).dropna()
df_4h['EMA9'] = df_4h['Close'].ewm(span=9, adjust=False).mean()
df['HTF_EMA']  = df_4h['EMA9'].reindex(df.index, method='ffill')
df['HTF_BULL'] = (df['Close'] > df['HTF_EMA']).astype(int)   # 1=bull, 0=bear

dates       = pd.Series(df.index.date, index=df.index)
unique_days = dates.unique()


# ─────────────────────────────────────────────────────────────────────────────
# Signal generator  (htf_gate=False → baseline, True → HTF filtered)
# ─────────────────────────────────────────────────────────────────────────────
def generate_signals(df, dates, unique_days, htf_gate: bool = False):
    signals = []

    for day in unique_days:
        day_indices = np.where(dates == day)[0]
        if len(day_indices) == 0: continue
        day_ts = df.iloc[day_indices].index

        # Overnight/London window (00:00–08:30)
        on_mask = (day_ts.time >= pd.to_datetime('00:00:00').time()) & \
                  (day_ts.time <= pd.to_datetime('08:30:00').time())
        idx_on  = [day_indices[j] for j, m in enumerate(on_mask) if m]
        if len(idx_on) < 10: continue
        on_high = df['High'].iloc[idx_on].max()
        on_low  = df['Low'].iloc[idx_on].min()
        pdh = df['PDH'].iloc[idx_on[0]]
        pdl = df['PDL'].iloc[idx_on[0]]

        # NY pre-market sweep detection (08:30–10:00)
        ny_pre_mask = (day_ts.time >= pd.to_datetime('08:30:00').time()) & \
                      (day_ts.time <= pd.to_datetime('10:00:00').time())
        idx_pre = [day_indices[j] for j, m in enumerate(ny_pre_mask) if m]
        if not idx_pre: continue
        pre_high = df['High'].iloc[idx_pre].max()
        pre_low  = df['Low'].iloc[idx_pre].min()

        swept_buyside  = (pre_high > on_high) or (pd.notna(pdh) and pre_high > pdh)
        swept_sellside = (pre_low  < on_low)  or (pd.notna(pdl) and pre_low  < pdl)

        # Entry window 10:00–10:30
        entry_mask = (day_ts.time >= pd.to_datetime('10:00:00').time()) & \
                     (day_ts.time <= pd.to_datetime('10:30:00').time())
        entry_indices = [day_indices[j] for j, m in enumerate(entry_mask) if m]
        if not entry_indices: continue
        open_10am = df['Open'].iloc[entry_indices[0]]

        # HTF bias at 10:00 open
        htf_bull = bool(df['HTF_BULL'].iloc[entry_indices[0]])

        # Marking window 09:30–09:59
        mark_mask = (day_ts.time >= pd.to_datetime('09:30:00').time()) & \
                    (day_ts.time <= pd.to_datetime('09:59:00').time())
        idx_mark  = [day_indices[j] for j, m in enumerate(mark_mask) if m]
        if len(idx_mark) < 5: continue
        df_mark = df.iloc[idx_mark]

        strat_cfg = StrategyConfig(target_r_multiple=2.0,
                                   breakeven_trigger_pct=0.5,
                                   max_risk_points=40.0)
        logic = ICTStrategyLogic(strategy_config=strat_cfg,
                                  contract_spec=MNQContractSpec())
        raw = (logic.engine.detect_order_blocks(df_mark, idx_mark) +
               logic.engine.detect_breaker_blocks(df_mark, idx_mark) +
               logic.engine.detect_rejection_blocks(df_mark, idx_mark))

        # Filter out pre-touched blocks
        valid_pre = []
        for b in raw:
            touched = False
            for g in idx_mark:
                if g > b.formation_idx:
                    if b.direction == 1 and df['Low'].iloc[g]  <= b.near_edge: touched = True; break
                    if b.direction == -1 and df['High'].iloc[g] >= b.near_edge: touched = True; break
            if not touched:
                valid_pre.append(b)

        # HTF gate: longs only when bull, shorts only when bear (if enabled)
        take_long  = swept_sellside and (not htf_gate or htf_bull)
        take_short = swept_buyside  and (not htf_gate or not htf_bull)

        bulls = [b for b in valid_pre if b.direction == 1  and b.near_edge <= open_10am] if take_long  else []
        bears = [b for b in valid_pre if b.direction == -1 and b.near_edge >= open_10am] if take_short else []

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
                                               sl, tp, be, rsk,
                                               'Sweep Long', best_long.block_type))
                    break

            if best_short and b_high >= best_short.near_edge and b_open <= best_short.near_edge:
                lvl = logic.calculate_trade_levels(-1, best_short.near_edge, best_short.far_edge)
                if lvl:
                    sl, tp, be, rsk = lvl
                    signals.append(TradeSignal(g, b_dt, -1, best_short.near_edge,
                                               sl, tp, be, rsk,
                                               'Sweep Short', best_short.block_type))
                    break

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Stats + FundedNext printer
# ─────────────────────────────────────────────────────────────────────────────
def print_full_stats(label: str, trades):
    if not trades:
        print(f"\n{'='*62}\n  {label}: No trades\n{'='*62}")
        return

    wins   = [t for t in trades if t.exit_reason == 'TARGET_2R']
    be     = [t for t in trades if t.exit_reason == 'BREAKEVEN']
    losses = [t for t in trades if t.exit_reason == 'STOP_LOSS']
    other  = [t for t in trades if t.exit_reason not in ('TARGET_2R','BREAKEVEN','STOP_LOSS')]
    n      = len(trades)
    wr     = len(wins) / n * 100
    net    = sum(t.net_pnl for t in trades)
    gw     = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
    gl     = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
    pf     = gw / gl if gl > 0 else float('inf')
    avg_r  = np.mean([t.realized_r for t in trades])
    cum    = pd.Series([t.net_pnl for t in trades]).cumsum()
    max_dd = (cum.cummax() - cum).max() if not cum.empty else 0

    longs  = [t for t in trades if t.direction ==  1]
    shorts = [t for t in trades if t.direction == -1]
    lw = len([t for t in longs  if t.exit_reason == 'TARGET_2R'])
    sw = len([t for t in shorts if t.exit_reason == 'TARGET_2R'])

    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    print(f"  Total Trades    : {n}")
    print(f"  Wins (2R)       : {len(wins)} ({wr:.1f}%)")
    print(f"  Breakeven       : {len(be)}  ({len(be)/n*100:.1f}%)")
    print(f"  Losses (SL)     : {len(losses)}  ({len(losses)/n*100:.1f}%)")
    print(f"  Other           : {len(other)}")
    print(f"  {'─'*42}")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Avg R / trade   : {avg_r:+.2f}R")
    print(f"  Net P&L (1 MNQ) : ${net:+,.2f}")
    print(f"  Max Drawdown    : ${max_dd:,.2f}")
    if longs:
        print(f"  Longs  : {len(longs):>2} trades | {lw} wins ({lw/len(longs)*100:.0f}% WR)")
    if shorts:
        print(f"  Shorts : {len(shorts):>2} trades | {sw} wins ({sw/len(shorts)*100:.0f}% WR)")

    # FundedNext $50K sim
    print(f"\n  --- FundedNext $50K Sim ---")
    for n_contracts in [3, 4, 5, 6]:
        bal = 50000.0; peak = 50000.0; mdd = 0.0
        passed = failed = False
        pass_t = pass_date = None
        for i, t in enumerate(trades):
            # MNQ: $2/pt, comm $5.20 RT + $1 slippage per contract
            pnl = n_contracts * t.gross_points * 2.0 - n_contracts * 6.20
            bal  += pnl
            peak  = max(peak, bal)
            dd    = peak - bal
            mdd   = max(mdd, dd)
            if bal >= 52500.0 and not passed:
                passed = True; pass_t = i+1
                pass_date = t.exit_time.date() if hasattr(t.exit_time,'date') else ''
            if dd >= 1500.0 and not failed:
                failed = True

        status = 'PASSED' if passed else ('FAILED DD' if failed else 'In progress')
        pt_txt = f" @ trade #{pass_t} ({pass_date})" if passed else ""
        print(f"  {n_contracts} MNQ: ${bal-50000:>+8,.2f} net | DD ${mdd:>6,.2f} | {status}{pt_txt}")

    # Per-trade pace
    avg_net_1mnq = net / n
    if avg_net_1mnq > 0:
        print(f"\n  Avg net/trade (1 MNQ)  : ${avg_net_1mnq:+.2f}")
        tpw = n / 15.0
        for tgt in [2500]:
            nt = int(round(tgt / avg_net_1mnq))
            print(f"  Trades to +${tgt:,} (1 MNQ) : ~{nt}  (~{nt/tpw:.1f} wks at {tpw:.1f}/wk)")

    # Per-trade log (first 20)
    print(f"\n  {'#':>3}  {'Entry':<20}  {'Dir':>7}  {'R':>6}  {'Exit':<12}  {'Net':>8}")
    print(f"  {'─'*62}")
    for i, t in enumerate(trades[:20]):
        d  = t.entry_time.strftime('%b %d %H:%M') if hasattr(t.entry_time,'strftime') else str(t.entry_time)
        di = 'LONG' if t.direction == 1 else 'SHORT'
        print(f"  {i+1:>3}  {d:<20}  {di:>7}  {t.realized_r:>+5.2f}R  {t.exit_reason:<12}  {t.net_pnl:>+8.2f}")
    if len(trades) > 20:
        print(f"  ... (+{len(trades)-20} more)")


# ─────────────────────────────────────────────────────────────────────────────
# Run both versions
# ─────────────────────────────────────────────────────────────────────────────
sim_cfg = StrategyConfig(target_r_multiple=2.0, breakeven_trigger_pct=0.5)
sim     = NQIntrabarSimulator(strategy_config=sim_cfg, contract_spec=MNQContractSpec())

print("\nGenerating signals — BASELINE (no HTF filter) ...")
sigs_base = generate_signals(df, dates, unique_days, htf_gate=False)
trades_base = sim.run(df, sigs_base)
print(f"  {len(sigs_base)} signals -> {len(trades_base)} trades")

print("\nGenerating signals — HTF GATED (4H EMA9 bias) ...")
sigs_htf = generate_signals(df, dates, unique_days, htf_gate=True)
trades_htf = sim.run(df, sigs_htf)
print(f"  {len(sigs_htf)} signals -> {len(trades_htf)} trades")

# ── Print comparison ─────────────────────────────────────────────────────────
print_full_stats("BASELINE — External Sweep + 10am Macro (No HTF Filter)", trades_base)
print_full_stats("HTF GATED — External Sweep + 10am Macro (4H EMA9 Bias)", trades_htf)

# ── Summary diff ─────────────────────────────────────────────────────────────
if trades_base and trades_htf:
    wb = len([t for t in trades_base if t.exit_reason=='TARGET_2R'])
    wh = len([t for t in trades_htf  if t.exit_reason=='TARGET_2R'])
    nb = sum(t.net_pnl for t in trades_base)
    nh = sum(t.net_pnl for t in trades_htf)
    print(f"\n{'='*62}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*62}")
    print(f"  {'Metric':<28}  {'Baseline':>10}  {'HTF Gated':>10}  {'Delta':>8}")
    print(f"  {'─'*58}")
    print(f"  {'Trades':<28}  {len(trades_base):>10}  {len(trades_htf):>10}  {len(trades_htf)-len(trades_base):>+8}")
    print(f"  {'Wins':<28}  {wb:>10}  {wh:>10}  {wh-wb:>+8}")
    wr_b = wb/len(trades_base)*100; wr_h = wh/len(trades_htf)*100 if trades_htf else 0
    print(f"  {'Win Rate':<28}  {wr_b:>9.1f}%  {wr_h:>9.1f}%  {wr_h-wr_b:>+7.1f}%")
    print(f"  {'Net P&L (1 MNQ)':<28}  {nb:>+10.2f}  {nh:>+10.2f}  {nh-nb:>+8.2f}")
    print(f"  {'Trades filtered out':<28}  {'':>10}  {len(trades_base)-len(trades_htf):>10}")

