import sys
sys.path.insert(0, '.')
from data.loader import NQDataLoader
from strategy.rules import ICTStrategyLogic, TradeSignal
from execution.simulator import NQIntrabarSimulator
from config import StrategyConfig, MNQContractSpec
import pandas as pd, numpy as np

loader = NQDataLoader(target_timezone='America/New_York')
df = loader.load_csv('sample_data/MNQ_1min_20260120_20260415.csv', source_timezone='UTC')

strat_cfg = StrategyConfig(target_r_multiple=2.5, breakeven_trigger_pct=0.5)
logic = ICTStrategyLogic(strategy_config=strat_cfg, contract_spec=MNQContractSpec())

dates = pd.Series(df.index.date, index=df.index)
signals = []
for day in dates.unique():
    day_indices = np.where(dates == day)[0]
    if len(day_indices) == 0: continue
    df_day = df.iloc[day_indices]
    day_ts = df_day.index

    mark_mask = (day_ts.time >= pd.to_datetime('09:30:00').time()) & (day_ts.time <= pd.to_datetime('09:59:00').time())
    mark_indices = [day_indices[j] for j, m in enumerate(mark_mask) if m]
    if len(mark_indices) < 5: continue
    df_mark = df.iloc[mark_indices]

    raw = logic.engine.detect_order_blocks(df_mark, mark_indices) + \
          logic.engine.detect_breaker_blocks(df_mark, mark_indices) + \
          logic.engine.detect_rejection_blocks(df_mark, mark_indices)

    valid_pre = []
    for b in raw:
        touched = False
        for g in mark_indices:
            if g > b.formation_idx:
                if b.direction == 1 and df['Low'].iloc[g] <= b.near_edge: touched = True; break
                elif b.direction == -1 and df['High'].iloc[g] >= b.near_edge: touched = True; break
        if not touched: valid_pre.append(b)

    # 10:00 to 10:30 macro entry window
    entry_mask = (day_ts.time >= pd.to_datetime('10:00:00').time()) & (day_ts.time <= pd.to_datetime('10:30:00').time())
    entry_indices = [day_indices[j] for j, m in enumerate(entry_mask) if m]
    if not entry_indices: continue
    open_10am = df['Open'].iloc[entry_indices[0]]

    bulls = [b for b in valid_pre if b.direction == 1 and b.near_edge <= open_10am]
    bears = [b for b in valid_pre if b.direction == -1 and b.near_edge >= open_10am]
    best_long = logic._select_best_block(bulls, open_10am)
    best_short = logic._select_best_block(bears, open_10am)

    long_dead = False
    short_dead = False
    for g in entry_indices:
        b_dt = df.index[g]
        b_open = df['Open'].iloc[g]
        b_high = df['High'].iloc[g]
        b_low = df['Low'].iloc[g]

        if best_long and not long_dead:
            if b_open < best_long.far_edge: long_dead = True
            elif b_low <= best_long.near_edge:
                if b_open >= best_long.near_edge:
                    lvl = logic.calculate_trade_levels(1, best_long.near_edge, best_long.far_edge)
                    if lvl:
                        sl, tp, be, rsk = lvl
                        signals.append(TradeSignal(g, b_dt, 1, best_long.near_edge, sl, tp, be, rsk, 'sig', best_long.block_type))
                        break
                else:
                    if b_low < best_long.far_edge: long_dead = True

        if best_short and not short_dead:
            if b_open > best_short.far_edge: short_dead = True
            elif b_high >= best_short.near_edge:
                if b_open <= best_short.near_edge:
                    lvl = logic.calculate_trade_levels(-1, best_short.near_edge, best_short.far_edge)
                    if lvl:
                        sl, tp, be, rsk = lvl
                        signals.append(TradeSignal(g, b_dt, -1, best_short.near_edge, sl, tp, be, rsk, 'sig', best_short.block_type))
                        break
                else:
                    if b_high > best_short.far_edge: short_dead = True

sim = NQIntrabarSimulator(strategy_config=strat_cfg, contract_spec=MNQContractSpec())
trades = sim.run(df, signals)

print(f"Total Macro 10:00-10:30 Trades: {len(trades)}")
wins = [t for t in trades if t.realized_r > 0]
be_trades = [t for t in trades if t.exit_reason == 'BREAKEVEN']
losses = [t for t in trades if t.exit_reason == 'STOP_LOSS']

print(f"Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
print(f"Breakeven: {len(be_trades)} ({len(be_trades)/len(trades)*100:.1f}%)")
print(f"Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")

gross_w = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
gross_l = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
net_pnl = sum(t.net_pnl for t in trades)
print(f"Gross Profit: ${gross_w:.2f} | Gross Loss: ${gross_l:.2f} | PF: {gross_w/gross_l:.2f} | Net PnL (1 MNQ): ${net_pnl:.2f}")

# Test FundedNext with scaled contracts to hit $2500 target
for risk_per_trade in [75, 100, 120, 150]:
    bal = 50000.0
    peak = 50000.0
    passed = False
    failed = False
    max_dd = 0.0
    pass_trade = None

    for i, t in enumerate(trades):
        contracts = max(1, int(round(risk_per_trade / (t.risk_points * 2.0))))
        t_net = contracts * t.gross_points * 2.0 - contracts * 6.20
        bal += t_net
        if bal > peak:
            peak = bal
        dd = peak - bal
        if dd > max_dd:
            max_dd = dd
        if bal >= 52500.0 and not passed:
            passed = True
            pass_trade = i + 1
        if dd >= 1500.0 and not failed:
            failed = True

    print(f"FundedNext Risk ${risk_per_trade}/trade -> Passed: {passed} (Trade #{pass_trade}), Max DD: ${max_dd:.2f}, Failed DD? {failed}, End Bal: ${bal:.2f}")
