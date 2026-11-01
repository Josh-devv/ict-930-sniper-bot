import sys
sys.path.insert(0, '.')
from data.loader import NQDataLoader
from strategy.rules import ICTStrategyLogic, TradeSignal
from execution.simulator import NQIntrabarSimulator
from config import StrategyConfig, MNQContractSpec
import pandas as pd, numpy as np

print("Loading data for Dual AM + PM Session Backtest...")
loader = NQDataLoader(target_timezone='America/New_York')
df = loader.load_csv('sample_data/MNQ_1min_20260120_20260415.csv', source_timezone='UTC')

strat_cfg = StrategyConfig(
    target_r_multiple=2.5,
    breakeven_trigger_pct=0.50,
    max_risk_points=40.0,
    time_exit="16:00"
)
logic = ICTStrategyLogic(strategy_config=strat_cfg, contract_spec=MNQContractSpec())

dates = pd.Series(df.index.date, index=df.index)
unique_days = dates.unique()

# Session Definitions:
# AM Session: Mark 09:30-09:59, Bias 10:00 Open, Entry Window 10:00-10:30
# PM Session: Mark 13:30-13:59, Bias 14:00 Open, Entry Window 14:00-14:30
sessions = [
    {
        "name": "AM",
        "mark_start": "09:30:00",
        "mark_end": "09:59:00",
        "bias_time": "10:00:00",
        "entry_end": "10:30:00",
    },
    {
        "name": "PM",
        "mark_start": "13:30:00",
        "mark_end": "13:59:00",
        "bias_time": "14:00:00",
        "entry_end": "14:30:00",
    }
]

signals = []

for day in unique_days:
    day_indices = np.where(dates == day)[0]
    if len(day_indices) == 0:
        continue
    df_day = df.iloc[day_indices]
    day_ts = df_day.index

    for sess in sessions:
        # Mark blocks in session window
        mark_mask = (day_ts.time >= pd.to_datetime(sess["mark_start"]).time()) & \
                    (day_ts.time <= pd.to_datetime(sess["mark_end"]).time())
        mark_indices = [day_indices[j] for j, m in enumerate(mark_mask) if m]
        if len(mark_indices) < 5:
            continue
        df_mark = df.iloc[mark_indices]

        raw = logic.engine.detect_order_blocks(df_mark, mark_indices) + \
              logic.engine.detect_breaker_blocks(df_mark, mark_indices) + \
              logic.engine.detect_rejection_blocks(df_mark, mark_indices)

        valid_pre = []
        for b in raw:
            touched = False
            for g in mark_indices:
                if g > b.formation_idx:
                    if b.direction == 1 and df['Low'].iloc[g] <= b.near_edge:
                        touched = True; break
                    elif b.direction == -1 and df['High'].iloc[g] >= b.near_edge:
                        touched = True; break
            if not touched:
                valid_pre.append(b)

        # Entry window for session
        entry_mask = (day_ts.time >= pd.to_datetime(sess["bias_time"]).time()) & \
                     (day_ts.time <= pd.to_datetime(sess["entry_end"]).time())
        entry_indices = [day_indices[j] for j, m in enumerate(entry_mask) if m]
        if not entry_indices:
            continue
        open_bias_price = df['Open'].iloc[entry_indices[0]]

        bulls = [b for b in valid_pre if b.direction == 1 and b.near_edge <= open_bias_price]
        bears = [b for b in valid_pre if b.direction == -1 and b.near_edge >= open_bias_price]
        best_long = logic._select_best_block(bulls, open_bias_price)
        best_short = logic._select_best_block(bears, open_bias_price)

        long_dead = False
        short_dead = False
        traded_sess = False

        for g in entry_indices:
            if traded_sess:
                break
            b_dt = df.index[g]
            b_open = df['Open'].iloc[g]
            b_high = df['High'].iloc[g]
            b_low = df['Low'].iloc[g]

            if best_long and not long_dead:
                if b_open < best_long.far_edge:
                    long_dead = True
                elif b_low <= best_long.near_edge:
                    if b_open >= best_long.near_edge:
                        lvl = logic.calculate_trade_levels(1, best_long.near_edge, best_long.far_edge)
                        if lvl:
                            sl, tp, be, rsk = lvl
                            signals.append(TradeSignal(g, b_dt, 1, best_long.near_edge, sl, tp, be, rsk, f"{sess['name']} Long", best_long.block_type))
                            traded_sess = True
                            break
                    else:
                        if b_low < best_long.far_edge:
                            long_dead = True

            if best_short and not short_dead:
                if b_open > best_short.far_edge:
                    short_dead = True
                elif b_high >= best_short.near_edge:
                    if b_open <= best_short.near_edge:
                        lvl = logic.calculate_trade_levels(-1, best_short.near_edge, best_short.far_edge)
                        if lvl:
                            sl, tp, be, rsk = lvl
                            signals.append(TradeSignal(g, b_dt, -1, best_short.near_edge, sl, tp, be, rsk, f"{sess['name']} Short", best_short.block_type))
                            traded_sess = True
                            break
                    else:
                        if b_high > best_short.far_edge:
                            short_dead = True

print(f"\nTotal signals generated across AM + PM: {len(signals)}")
am_sigs = [s for s in signals if "AM" in s.reason]
pm_sigs = [s for s in signals if "PM" in s.reason]
print(f"AM Signals: {len(am_sigs)} | PM Signals: {len(pm_sigs)}")

sim = NQIntrabarSimulator(strategy_config=strat_cfg, contract_spec=MNQContractSpec())
trades = sim.run(df, signals)

print(f"\n====================================================")
print(f"        DUAL AM + PM SESSION PERFORMANCE REPORT     ")
print(f"====================================================")
wins = [t for t in trades if t.realized_r > 0]
losses = [t for t in trades if t.realized_r < 0]
be_trades = [t for t in trades if t.exit_reason == 'BREAKEVEN']
win_rate = len(wins) / len(trades) * 100 if trades else 0

gross_profit = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
gross_loss = abs(sum(t.gross_pnl for t in trades if t.gross_pnl < 0))
pf = gross_profit / gross_loss if gross_loss > 0 else 99.0
net_pnl = sum(t.net_pnl for t in trades)
fees = sum(t.fees for t in trades)
slippage = sum(t.slippage for t in trades)

cum_pnl = pd.Series([t.net_pnl for t in trades]).cumsum()
max_dd = (cum_pnl.cummax() - cum_pnl).max()

print(f"Total Trades           : {len(trades)}")
print(f"Winning Trades         : {len(wins)} ({win_rate:.2f}%)")
print(f"Losing Trades          : {len(losses)}")
print(f"Breakeven Trades       : {len(be_trades)}")
print(f"Profit Factor          : {pf:.2f}")
print(f"Net Realized P&L (1 MNQ): ${net_pnl:.2f}")
print(f"Gross Profit           : ${gross_profit:.2f}")
print(f"Gross Loss             : ${gross_loss:.2f}")
print(f"Max Drawdown (1 MNQ)   : ${max_dd:.2f}")
print(f"Average Realized R     : {pd.Series([t.realized_r for t in trades]).mean():.2f} R")

# FundedNext Evaluation
print(f"\n--- FundedNext $50k Account Feasibility (Target: $2,500, Max DD: $1,500) ---")
for contracts in [3, 4, 5]:
    c_net_per_trade = net_pnl * contracts / len(trades)
    trades_needed = 2500.0 / c_net_per_trade if c_net_per_trade > 0 else 999
    trades_per_week = len(trades) / (74 / 5)
    weeks = trades_needed / trades_per_week if trades_per_week > 0 else 999
    c_max_dd = max_dd * contracts
    dd_status = "SAFE (<$1500)" if c_max_dd < 1500 else "EXCEEDS ($1500)"
    print(f"{contracts} MNQ: Net/trade=${c_net_per_trade:.2f} | Trades needed={trades_needed:.0f} | Time: ~{weeks:.1f} weeks (~{weeks/4.33:.1f} mos) | Max DD: ${c_max_dd:.2f} [{dd_status}]")

# Export trade log for review
export_df = pd.DataFrame([{
    'trade_id': t.trade_id,
    'direction': 'LONG' if t.direction == 1 else 'SHORT',
    'entry_time': t.entry_time,
    'exit_time': t.exit_time,
    'entry_price': t.entry_price,
    'exit_price': t.exit_price,
    'risk_points': t.risk_points,
    'gross_points': t.gross_points,
    'gross_pnl': t.gross_pnl,
    'fees': t.fees,
    'slippage': t.slippage,
    'net_pnl': t.net_pnl,
    'realized_r': t.realized_r,
    'exit_reason': t.exit_reason
} for t in trades])
export_df.to_csv('trade_log_am_pm.csv', index=False)
print("\nExported dual session trades to trade_log_am_pm.csv")
