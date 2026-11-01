import sys
sys.path.insert(0, '.')
from data.loader import NQDataLoader
from strategy.rules import ICTStrategyLogic, TradeSignal
from execution.simulator import NQIntrabarSimulator
from config import StrategyConfig, MNQContractSpec, NQContractSpec
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

# Fixed MNQ Sizing
print("\n=== FIXED MNQ SIZING (NO COMMISSION INFLATION) ===")
for num_c in [4, 5, 6, 8]:
    bal = 50000.0
    peak = 50000.0
    max_dd = 0.0
    passed = False
    pass_trade = None
    pass_date = None

    for i, t in enumerate(trades):
        # 1 contract commission = 5.20, slippage = 1.0 (0.25 tick * 2.0 * 2) = 1.0
        t_net = num_c * (t.gross_points * 2.0) - num_c * 6.20
        bal += t_net
        if bal > peak:
            peak = bal
        dd = peak - bal
        if dd > max_dd:
            max_dd = dd
        if bal >= 52500.0 and not passed:
            passed = True
            pass_trade = i + 1
            pass_date = t.exit_time

    print(f"{num_c} MNQ: Net PnL=+${bal-50000:.2f} | Max DD=${max_dd:.2f} | Hit $2,500? {passed} (Trade #{pass_trade} on {pass_date})")

# 1 Mini NQ Contract ($20/pt, commission = $5.20 round turn, slippage = 1 tick = $5.00)
print("\n=== 1 FULL MINI NQ CONTRACT ($20/POINT) ===")
bal_nq = 50000.0
peak_nq = 50000.0
max_dd_nq = 0.0
passed_nq = False
pass_trade_nq = None
pass_date_nq = None

for i, t in enumerate(trades):
    t_net = 1 * (t.gross_points * 20.0) - 10.20 # $5.20 comm + $5.00 slippage
    bal_nq += t_net
    if bal_nq > peak_nq:
        peak_nq = bal_nq
    dd = peak_nq - bal_nq
    if dd > max_dd_nq:
        max_dd_nq = dd
    if bal_nq >= 52500.0 and not passed_nq:
        passed_nq = True
        pass_trade_nq = i + 1
        pass_date_nq = t.exit_time

print(f"1 Mini NQ: Net PnL=+${bal_nq-50000:.2f} | Max DD=${max_dd_nq:.2f} | Hit $2,500? {passed_nq} (Trade #{pass_trade_nq} on {pass_date_nq})")
