import sys
sys.path.insert(0, '.')
from data.loader import NQDataLoader
from strategy.rules import ICTStrategyLogic, TradeSignal
from execution.simulator import NQIntrabarSimulator
from config import StrategyConfig, MNQContractSpec
import pandas as pd, numpy as np

loader = NQDataLoader(target_timezone='America/New_York')
df = loader.load_csv('sample_data/MNQ_1min_20260120_20260415.csv', source_timezone='UTC')

daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
df['PDH'] = daily['High'].shift(1).reindex(df.index, method='ffill')
df['PDL'] = daily['Low'].shift(1).reindex(df.index, method='ffill')

dates = pd.Series(df.index.date, index=df.index)
unique_days = dates.unique()
signals = []

for day in unique_days:
    day_indices = np.where(dates == day)[0]
    if len(day_indices) == 0: continue
    df_day = df.iloc[day_indices]
    day_ts = df_day.index

    on_mask = (day_ts.time >= pd.to_datetime('00:00:00').time()) & (day_ts.time <= pd.to_datetime('08:30:00').time())
    idx_on = [day_indices[j] for j, m in enumerate(on_mask) if m]
    if len(idx_on) < 10: continue
    on_high = df['High'].iloc[idx_on].max()
    on_low = df['Low'].iloc[idx_on].min()
    pdh = df['PDH'].iloc[idx_on[0]]
    pdl = df['PDL'].iloc[idx_on[0]]

    ny_pre_mask = (day_ts.time >= pd.to_datetime('08:30:00').time()) & (day_ts.time <= pd.to_datetime('10:00:00').time())
    idx_pre = [day_indices[j] for j, m in enumerate(ny_pre_mask) if m]
    if not idx_pre: continue
    pre_high = df['High'].iloc[idx_pre].max()
    pre_low = df['Low'].iloc[idx_pre].min()

    swept_buyside = (pre_high > on_high) or (pd.notna(pdh) and pre_high > pdh)
    swept_sellside = (pre_low < on_low) or (pd.notna(pdl) and pre_low < pdl)

    entry_mask = (day_ts.time >= pd.to_datetime('10:00:00').time()) & (day_ts.time <= pd.to_datetime('10:30:00').time())
    entry_indices = [day_indices[j] for j, m in enumerate(entry_mask) if m]
    if not entry_indices: continue
    open_10am = df['Open'].iloc[entry_indices[0]]

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
                signals.append(TradeSignal(g, b_dt, 1, best_long.near_edge, sl, tp, be, rsk, 'Sweep Long', best_long.block_type))
                break

        if best_short and b_high >= best_short.near_edge and b_open <= best_short.near_edge:
            lvl = logic.calculate_trade_levels(-1, best_short.near_edge, best_short.far_edge)
            if lvl:
                sl, tp, be, rsk = lvl
                signals.append(TradeSignal(g, b_dt, -1, best_short.near_edge, sl, tp, be, rsk, 'Sweep Short', best_short.block_type))
                break

sim = NQIntrabarSimulator(strategy_config=StrategyConfig(target_r_multiple=2.0, breakeven_trigger_pct=0.5), contract_spec=MNQContractSpec())
trades = sim.run(df, signals)

print(f"Total trades: {len(trades)}")

for risk_amt in [150, 200, 250]:
    bal = 50000.0
    peak = 50000.0
    passed = False
    failed = False
    pass_trade = None
    pass_date = None
    max_dd = 0.0

    trade_records = []
    for i, t in enumerate(trades):
        contracts = max(1, int(round(risk_amt / (t.risk_points * 2.0))))
        net_t = contracts * (t.gross_points * 2.0) - contracts * 6.20
        bal += net_t
        if bal > peak:
            peak = bal
        dd = peak - bal
        if dd > max_dd:
            max_dd = dd

        if bal >= 52500.0 and not passed:
            passed = True
            pass_trade = i + 1
            pass_date = t.exit_time
        if dd >= 1500.0 and not failed:
            failed = True

        trade_records.append({'net': net_t, 'bal': bal, 'dd': dd})

    print(f"Risk ${risk_amt}/trade -> Final Balance: ${bal:.2f} | Net Profit: +${bal - 50000:.2f} | Max Drawdown: ${max_dd:.2f} | Target Reached ($2,500)? {passed} | Hit $1,500 DD? {failed}")
    
    # Calculate how many trades to reach $2,500
    avg_net_per_trade = (bal - 50000) / len(trades)
    trades_to_target = int(round(2500.0 / avg_net_per_trade)) if avg_net_per_trade > 0 else 999
    # Sizing pace
    trades_per_week = len(trades) / 15
    weeks_to_target = trades_to_target / trades_per_week
    months_to_target = weeks_to_target / 4.33
    print(f"   --> Average Net Profit per trade: +${avg_net_per_trade:.2f}")
    print(f"   --> Total Trades to reach +$2,500: ~{trades_to_target} trades")
    print(f"   --> Estimated Time: ~{weeks_to_target:.1f} weeks (~{months_to_target:.1f} months)\n")
