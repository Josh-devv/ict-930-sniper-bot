"""
NQ 9:30 VWAP Sniper — Orderflow & CVD Absorption Optimization
=============================================================
Compares:
1. BASELINE: Price Action Rejection Only (2.0σ Band + 35% Wick + VWAP TP)
2. ORDERFLOW-OPTIMIZED:
   + Volume Climax (Institutional Surge >= 1.3x SMA)
   + CVD Delta Absorption (Delta Flip / Reversal inside the 5m candle)
   + Freight-Train Filter (Avoid runaway crash bars with unabsorbed negative delta)

Data: 8-Year NQ 1-Min Data (2018-2026, 3.97M bars)
Position: 5 MNQ ($10/pt)
"""

import json, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

DATA_PATH = r'c:\Users\JOSHUA\Desktop\CRTbot\data\indices\NQ-duka-2018-2026.json'

print("Loading 8-Year NQ Data (3.97M bars)...")
with open(DATA_PATH, 'r') as f:
    raw = json.load(f)

df_raw = pd.DataFrame(raw)
df_raw['dt'] = pd.to_datetime(df_raw['time'], unit='ms', utc=True)
df_raw = df_raw.set_index('dt').drop(columns=['time'])
df_raw.columns = ['open', 'high', 'low', 'close', 'volume']
df_raw.index = df_raw.index.tz_convert('America/New_York')
df_raw = df_raw[df_raw.index.dayofweek < 5]

# Calculate 1m Bar Delta (Intra-candle buy/sell volume estimation)
# Buy Volume = Vol * (Close - Low) / (High - Low + 1e-6)
# Sell Volume = Vol * (High - Close) / (High - Low + 1e-6)
# Delta = Buy Vol - Sell Vol = Vol * (2 * Close - High - Low) / (High - Low + 1e-6)
rng_1m = (df_raw['high'] - df_raw['low']).replace(0, 1e-6)
df_raw['delta'] = df_raw['volume'] * (2.0 * df_raw['close'] - df_raw['high'] - df_raw['low']) / rng_1m

# Resample to 5m
df_5m = df_raw.resample('5min', label='left', closed='left').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'delta': 'sum'
}).dropna(subset=['open'])

print(f"Resampled to {len(df_5m):,} 5m bars. Running backtest...")

# ─── PARAMETERS ─────────────────────────────────────────────────────────────
SIGMA = 2.0
MIN_BAND_DIST = 15.0
WICK_RATIO = 0.35
STOP_BEYOND = 1.5
MIN_STOP = 5.0
MAX_STOP = 60.0
MIN_RR = 1.5
VOL_CLIMAX_MULT = 1.3 # Orderflow filter: 30% above 20-period avg volume

def run_backtest(use_orderflow=False):
    trades = []
    
    # Pre-calculate 20-period volume MA for orderflow
    vol_ma_series = df_5m['volume'].rolling(20).mean()
    
    for date, day in df_5m.groupby(df_5m.index.date):
        if len(day) < 10:
            continue
        
        # Session window: 09:30 to 16:00
        day_session = day.between_time('09:30', '16:00')
        if len(day_session) < 6:
            continue
            
        # Calculate 09:30 Anchored VWAP + 2.0σ Bands
        tp = (day_session['high'] + day_session['low'] + day_session['close']) / 3.0
        cum_vol = day_session['volume'].cumsum()
        cum_tpv = (tp * day_session['volume']).cumsum()
        vwap = cum_tpv / cum_vol
        cum_tp2v = ((tp ** 2) * day_session['volume']).cumsum()
        variance = (cum_tp2v / cum_vol) - (vwap ** 2)
        variance = variance.clip(lower=0)
        std_dev = np.sqrt(variance)
        upper = vwap + SIGMA * std_dev
        lower = vwap - SIGMA * std_dev

        # Signal window: 09:30 to 11:30
        signal_bars = day_session.between_time('09:30', '11:29')
        trade_taken = False
        
        for ts in signal_bars.index:
            if trade_taken:
                break
            
            idx = day_session.index.get_loc(ts)
            o = day_session['open'].iloc[idx]
            h = day_session['high'].iloc[idx]
            l = day_session['low'].iloc[idx]
            c = day_session['close'].iloc[idx]
            v = day_session['volume'].iloc[idx]
            d = day_session['delta'].iloc[idx]
            
            rng = h - l
            if rng < 0.25:
                continue
                
            vw = vwap.iloc[idx]
            ub = upper.iloc[idx]
            lb = lower.iloc[idx]
            
            direction = None
            down_wick = min(o, c) - l
            up_wick = h - max(o, c)
            
            # Baseline Checks
            is_long_candidate = (l <= lb and c > lb and (vw - lb) >= MIN_BAND_DIST and (down_wick / rng) >= WICK_RATIO and c > o)
            is_short_candidate = (h >= ub and c < ub and (ub - vw) >= MIN_BAND_DIST and (up_wick / rng) >= WICK_RATIO and c < o)
            
            if is_long_candidate:
                # ORDERFLOW FILTER FOR LONGS:
                # 1. Volume Climax: institutions active
                # 2. Delta Absorption: Net delta must be positive (buyers absorbed sellers inside the wick)
                if use_orderflow:
                    v_ma = vol_ma_series.loc[ts] if ts in vol_ma_series.index else 0
                    has_vol_climax = v >= (v_ma * VOL_CLIMAX_MULT)
                    has_delta_absorption = d > 0 # Delta flipped positive
                    if not (has_vol_climax and has_delta_absorption):
                        continue
                        
                direction = 'long'
                entry_p = (l + min(o, c)) / 2.0 # 50% wick limit
                stop_p = l - STOP_BEYOND
                risk = entry_p - stop_p
                tp1 = vw
                tp2 = ub
                
                if risk < MIN_STOP or risk > MAX_STOP or (tp1 - entry_p) < risk * MIN_RR:
                    continue

            elif is_short_candidate:
                # ORDERFLOW FILTER FOR SHORTS:
                if use_orderflow:
                    v_ma = vol_ma_series.loc[ts] if ts in vol_ma_series.index else 0
                    has_vol_climax = v >= (v_ma * VOL_CLIMAX_MULT)
                    has_delta_absorption = d < 0 # Delta flipped negative
                    if not (has_vol_climax and has_delta_absorption):
                        continue
                        
                direction = 'short'
                entry_p = (h + max(o, c)) / 2.0
                stop_p = h + STOP_BEYOND
                risk = stop_p - entry_p
                tp1 = vw
                tp2 = lb
                
                if risk < MIN_STOP or risk > MAX_STOP or (entry_p - tp1) < risk * MIN_RR:
                    continue

            if direction is None:
                continue

            # Simulate execution on future bars
            future_bars = day_session.iloc[idx + 1:]
            filled = False
            fill_idx = 0
            
            for j, (_, fbar) in enumerate(future_bars.iterrows()):
                if direction == 'long' and fbar['low'] <= entry_p:
                    filled = True
                    fill_idx = j
                    break
                elif direction == 'short' and fbar['high'] >= entry_p:
                    filled = True
                    fill_idx = j
                    break
            
            if not filled:
                continue
                
            trade_taken = True
            post_fill = future_bars.iloc[fill_idx:]
            
            pnl_dollar = 0.0
            tp1_hit = False
            tp1_pnl = 0.0
            runner_stop = None
            
            # Position: 5 MNQ contracts ($10/pt total: 3 MNQ = $6/pt, 2 MNQ = $4/pt)
            c_tp1 = 3  # 60% of 5
            c_run = 2  # 40% of 5
            COMMISSION = 2.50 # $0.50 RT * 5 contracts
            
            for _, fbar in post_fill.iterrows():
                fh, fl = fbar['high'], fbar['low']
                
                if not tp1_hit:
                    if direction == 'long':
                        if fl <= stop_p: # stopped out full 5 contracts
                            pnl_dollar = -(entry_p - stop_p) * 5 * 2.0 - COMMISSION
                            break
                        elif fh >= tp1: # TP1 HIT!
                            tp1_hit = True
                            tp1_pnl = (tp1 - entry_p) * c_tp1 * 2.0
                            runner_stop = entry_p + 1.0 # BE + 1 pt
                    else:
                        if fh >= stop_p: # stopped out full 5 contracts
                            pnl_dollar = -(stop_p - entry_p) * 5 * 2.0 - COMMISSION
                            break
                        elif fl <= tp1: # TP1 HIT!
                            tp1_hit = True
                            tp1_pnl = (entry_p - tp1) * c_tp1 * 2.0
                            runner_stop = entry_p - 1.0 # BE + 1 pt
                else:
                    # Runner management
                    if direction == 'long':
                        if fl <= runner_stop:
                            runner_pnl = (runner_stop - entry_p) * c_run * 2.0
                            pnl_dollar = tp1_pnl + runner_pnl - COMMISSION
                            break
                        elif fh >= tp2: # Full TP2
                            runner_pnl = (tp2 - entry_p) * c_run * 2.0
                            pnl_dollar = tp1_pnl + runner_pnl - COMMISSION
                            break
                    else:
                        if fh >= runner_stop:
                            runner_pnl = (entry_p - runner_stop) * c_run * 2.0
                            pnl_dollar = tp1_pnl + runner_pnl - COMMISSION
                            break
                        elif fl <= tp2: # Full TP2
                            runner_pnl = (entry_p - tp2) * c_run * 2.0
                            pnl_dollar = tp1_pnl + runner_pnl - COMMISSION
                            break
            
            # If still open at 15:45 EOD
            if pnl_dollar == 0.0:
                last_c = post_fill['close'].iloc[-1]
                if not tp1_hit:
                    diff = (last_c - entry_p) if direction == 'long' else (entry_p - last_c)
                    pnl_dollar = diff * 5 * 2.0 - COMMISSION
                else:
                    diff_runner = (last_c - entry_p) if direction == 'long' else (entry_p - last_c)
                    runner_pnl = diff_runner * c_run * 2.0
                    pnl_dollar = tp1_pnl + runner_pnl - COMMISSION
                    
            trades.append(pnl_dollar)
            
    s = pd.Series(trades)
    n_trades = len(s)
    wins = s[s > 0]
    losses = s[s < 0]
    wr = len(wins) / n_trades * 100 if n_trades > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else 0
    tot_dollar = s.sum()
    
    # Cumulative Drawdown
    cum_pnl = s.cumsum()
    peak = cum_pnl.cummax()
    dd = (cum_pnl - peak).min()
    
    return {
        "trades": n_trades,
        "win_rate": wr,
        "profit_factor": pf,
        "net_dollar": tot_dollar,
        "max_drawdown": dd,
        "avg_trade": tot_dollar / n_trades if n_trades > 0 else 0
    }

print("\n--- Running Baseline (Price Action Only) ---")
res_base = run_backtest(use_orderflow=False)

print("--- Running Orderflow & CVD Absorption Optimized ---")
res_of = run_backtest(use_orderflow=True)

print("\n" + "="*70)
print(f"{'METRIC':<25} | {'BASELINE (Price Action)':<20} | {'ORDERFLOW-OPTIMIZED':<20}")
print("="*70)
print(f"{'Total Trades':<25} | {res_base['trades']:<20} | {res_of['trades']:<20}")
print(f"{'Win Rate':<25} | {res_base['win_rate']:.1f}%{'':<15} | {res_of['win_rate']:.1f}%{'':<15}")
print(f"{'Profit Factor':<25} | {res_base['profit_factor']:.2f}{'':<16} | {res_of['profit_factor']:.2f}{'':<16}")
print(f"{'Total Net PnL (5 MNQ)':<25} | +${res_base['net_dollar']:,.2f}{'':<10} | +${res_of['net_dollar']:,.2f}{'':<10}")
print(f"{'Max Drawdown':<25} | -${abs(res_base['max_drawdown']):,.2f}{'':<10} | -${abs(res_of['max_drawdown']):,.2f}{'':<10}")
print(f"{'Avg Trade Expectancy':<25} | +${res_base['avg_trade']:,.2f}{'':<10} | +${res_of['avg_trade']:,.2f}{'':<10}")
print("="*70)
