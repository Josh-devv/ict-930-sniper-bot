import sys
sys.path.insert(0, '.')
from data.loader import NQDataLoader
from strategy.rules import ICTStrategyLogic, TradeSignal
from execution.simulator import NQIntrabarSimulator
from config import StrategyConfig, MNQContractSpec
import pandas as pd, numpy as np

loader = NQDataLoader(target_timezone='America/New_York')
df = loader.load_csv('sample_data/MNQ_1min_20260120_20260415.csv', source_timezone='UTC')

# Calculate Prior Day High and Prior Day Low
daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
daily_pdh = daily['High'].shift(1)
daily_pdl = daily['Low'].shift(1)
df['PDH'] = daily_pdh.reindex(df.index, method='ffill')
df['PDL'] = daily_pdl.reindex(df.index, method='ffill')

dates = pd.Series(df.index.date, index=df.index)
unique_days = dates.unique()

signals = []
for day in unique_days:
    day_indices = np.where(dates == day)[0]
    if len(day_indices) == 0: continue
    df_day = df.iloc[day_indices]
    day_ts = df_day.index

    # Overnight / London window (00:00 to 08:30)
    on_mask = (day_ts.time >= pd.to_datetime('00:00:00').time()) & (day_ts.time <= pd.to_datetime('08:30:00').time())
    idx_on = [day_indices[j] for j, m in enumerate(on_mask) if m]
    if len(idx_on) < 10: continue
    on_high = df['High'].iloc[idx_on].max()
    on_low = df['Low'].iloc[idx_on].min()
    pdh = df['PDH'].iloc[idx_on[0]]
    pdl = df['PDL'].iloc[idx_on[0]]

    # Did NY morning (08:30 - 10:00) sweep external liquidity?
    ny_pre_mask = (day_ts.time >= pd.to_datetime('08:30:00').time()) & (day_ts.time <= pd.to_datetime('10:00:00').time())
    idx_pre = [day_indices[j] for j, m in enumerate(ny_pre_mask) if m]
    if not idx_pre: continue
    pre_high = df['High'].iloc[idx_pre].max()
    pre_low = df['Low'].iloc[idx_pre].min()

    swept_buyside = (pre_high > on_high) or (pd.notna(pdh) and pre_high > pdh)
    swept_sellside = (pre_low < on_low) or (pd.notna(pdl) and pre_low < pdl)

    # Entry in 10:00 - 10:30 macro window
    entry_mask = (day_ts.time >= pd.to_datetime('10:00:00').time()) & (day_ts.time <= pd.to_datetime('10:30:00').time())
    entry_indices = [day_indices[j] for j, m in enumerate(entry_mask) if m]
    if not entry_indices: continue
    open_10am = df['Open'].iloc[entry_indices[0]]

    # Detect blocks in 09:30-09:59
    mark_mask = (day_ts.time >= pd.to_datetime('09:30:00').time()) & (day_ts.time <= pd.to_datetime('09:59:00').time())
    idx_mark = [day_indices[j] for j, m in enumerate(mark_mask) if m]
    if len(idx_mark) < 5: continue
    df_mark = df.iloc[idx_mark]

    strat_cfg = StrategyConfig(target_r_multiple=2.0, breakeven_trigger_pct=0.5, max_risk_points=40.0)
    logic = ICTStrategyLogic(strategy_config=strat_cfg, contract_spec=MNQContractSpec())
    raw = logic.engine.detect_order_blocks(df_mark, idx_mark) + \
          logic.engine.detect_breaker_blocks(df_mark, idx_mark) + \
          logic.engine.detect_rejection_blocks(df_mark, idx_mark)

    valid_pre = []
    for b in raw:
        touched = False
        for g in idx_mark:
            if g > b.formation_idx:
                if b.direction == 1 and df['Low'].iloc[g] <= b.near_edge: touched = True; break
                elif b.direction == -1 and df['High'].iloc[g] >= b.near_edge: touched = True; break
        if not touched: valid_pre.append(b)

    bulls = [b for b in valid_pre if b.direction == 1 and b.near_edge <= open_10am] if swept_sellside else []
    bears = [b for b in valid_pre if b.direction == -1 and b.near_edge >= open_10am] if swept_buyside else []

    best_long = logic._select_best_block(bulls, open_10am) if bulls else None
    best_short = logic._select_best_block(bears, open_10am) if bears else None

    for g in entry_indices:
        b_dt = df.index[g]
        b_open = df['Open'].iloc[g]
        b_low = df['Low'].iloc[g]
        b_high = df['High'].iloc[g]

        if best_long and b_low <= best_long.near_edge and b_open >= best_long.near_edge:
            lvl = logic.calculate_trade_levels(1, best_long.near_edge, best_long.far_edge)
            if lvl:
                sl, tp, be, rsk = lvl
                signals.append(TradeSignal(g, b_dt, 1, best_long.near_edge, sl, tp, be, rsk, 'External Sweep Long', best_long.block_type))
                break

        if best_short and b_high >= best_short.near_edge and b_open <= best_short.near_edge:
            lvl = logic.calculate_trade_levels(-1, best_short.near_edge, best_short.far_edge)
            if lvl:
                sl, tp, be, rsk = lvl
                signals.append(TradeSignal(g, b_dt, -1, best_short.near_edge, sl, tp, be, rsk, 'External Sweep Short', best_short.block_type))
                break

sim = NQIntrabarSimulator(strategy_config=StrategyConfig(target_r_multiple=2.0, breakeven_trigger_pct=0.5), contract_spec=MNQContractSpec())
trades = sim.run(df, signals)
wins = [t for t in trades if t.realized_r > 0]
be_trades = [t for t in trades if t.exit_reason == 'BREAKEVEN']
losses = [t for t in trades if t.exit_reason == 'STOP_LOSS']

wr = len(wins)/len(trades)*100 if trades else 0
gw = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
gl = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
pf = gw/gl if gl > 0 else 0
net = sum(t.net_pnl for t in trades)

cum = pd.Series([t.net_pnl for t in trades]).cumsum()
max_dd = (cum.cummax() - cum).max() if not cum.empty else 0

print("=" * 60)
print("  EXTERNAL LIQUIDITY SWEEP + 10:00 MACRO STRATEGY")
print("=" * 60)
print(f"Total Trades         : {len(trades)}")
print(f"Winning Trades       : {len(wins)} ({wr:.1f}%)")
print(f"Breakeven Trades     : {len(be_trades)} ({len(be_trades)/len(trades)*100:.1f}%)")
print(f"Losing Trades        : {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
print(f"Gross Profit         : ${gw:.2f}")
print(f"Gross Loss           : ${gl:.2f}")
print(f"Profit Factor        : {pf:.2f}")
print(f"Net Realized P&L     : ${net:.2f}")
print(f"Max Drawdown (1 MNQ) : ${max_dd:.2f}")

print("\n--- FUNDEDNEXT $50K SIMULATION ---")
for contracts in [3, 4, 5, 6]:
    c_net = net * contracts
    c_dd = max_dd * contracts
    trades_needed = int(round(2500.0 / (net * contracts / len(trades))))
    weeks = trades_needed / (len(trades) / 15)
    print(f"{contracts} MNQ: Net/trade=${net*contracts/len(trades):.2f} | Max DD: ${c_dd:.2f} | Trades to $2,500: {trades_needed} (~{weeks:.1f} weeks / ~{weeks/4.33:.1f} mos)")
