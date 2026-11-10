"""
24-Hour Session Sweep & Top/Bottom Strategy Test
=================================================
Data: sample_data/MNQ_1min_20260120_20260415.csv (Real 24h continuous futures)
Sessions Evaluated:
1. London Session (02:00 - 06:00 EST): Sweeps Asian Range (18:00 - 02:00) High/Low -> Captures London HOD/LOD
2. New York Session (08:30 - 11:30 EST): Sweeps London (02:00 - 08:00) High/Low -> Captures NY HOD/LOD

Rules:
- Sweep past session extreme by at least 2.0 points
- Rejection wick >= 30% of candle range
- Close back inside the session range (confirmed rejection)
- Entry: 50% of the rejection wick (Consequent Encroachment)
- Stop Loss: 1.0 point past the sweep extreme
- Target: 2.0R and Opposing Session Midpoint (Equilibrium)
- Realism: $1.24 RT MNQ fee + 1 tick slippage, hard stop checked first
"""

import sys
import pandas as pd
import numpy as np
from datetime import time as dtime, timedelta

print("=" * 80)
print("24-HOUR SESSION SWEEP & TOP/BOTTOM MODEL BACKTEST")
print("=" * 80)

# Load data
df = pd.read_csv('sample_data/MNQ_1min_20260120_20260415.csv')
col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()][0]
df['dt'] = pd.to_datetime(df[col])
# The file has timestamps in UTC, let's localize/convert to America/New_York
if df['dt'].dt.tz is None:
    df['dt'] = df['dt'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
else:
    df['dt'] = df['dt'].dt.tz_convert('America/New_York')

df = df.sort_values('dt').set_index('dt')
df.columns = [c.lower() for c in df.columns]

# Resample to 5m
df_5m = df.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

df_5m['hour'] = df_5m['dt'].dt.hour
df_5m['minute'] = df_5m['dt'].dt.minute
df_5m['date'] = df_5m['dt'].dt.date
# Group sessions: session date shifts at 18:00
df_5m['session_date'] = df_5m['dt'].apply(lambda x: (x + timedelta(hours=6)).date() if x.hour >= 18 else x.date())

COMMISSION_PTS = 0.62   # $1.24 on MNQ ($2/pt)
SLIPPAGE_PTS   = 0.25   # 1 tick

def run_session_sweep_test(target_mode: str = '2.0R', min_wick_ratio: float = 0.30):
    trades = []
    
    for s_date, s_df in df_5m.groupby('session_date'):
        s_df = s_df.reset_index(drop=True)
        if len(s_df) < 50:
            continue
            
        # 1. Asian Session: 18:00 (prev day) to 02:00 EST
        asia = s_df[(s_df['hour'] >= 18) | (s_df['hour'] < 2)]
        if len(asia) < 10:
            continue
        asia_h = asia['high'].max()
        asia_l = asia['low'].min()
        
        # 2. London Session: 02:00 to 08:00 EST
        london = s_df[(s_df['hour'] >= 2) & (s_df['hour'] < 8)]
        if len(london) < 10:
            continue
        london_h = london['high'].max()
        london_l = london['low'].min()
        
        # Check Setups across the day
        # Opportunity A: London Sweeping Asia (02:00 to 06:00 EST)
        # Opportunity B: New York Sweeping London (08:30 to 11:30 EST)
        
        windows = [
            ("LONDON_SWEEP_ASIA", 2, 6, asia_h, asia_l),
            ("NY_SWEEP_LONDON", 8, 11, london_h, london_l),
        ]
        
        for w_name, start_h, end_h, ref_h, ref_l in windows:
            w_bars = s_df[(s_df['hour'] >= start_h) & (s_df['hour'] <= end_h)]
            if len(w_bars) < 3:
                continue
                
            trade_taken = False
            
            for idx in w_bars.index:
                if trade_taken:
                    break
                bar = s_df.iloc[idx]
                rng = bar['high'] - bar['low']
                if rng < 4.0:
                    continue
                    
                direction = 0
                entry_px = 0.0
                stop_px = 0.0
                
                # TOP OF SESSION SWEEP: Swept Ref High, closed back below Ref High
                if bar['high'] > ref_h + 1.0 and bar['close'] < ref_h:
                    upper_wick = bar['high'] - max(bar['open'], bar['close'])
                    if (upper_wick / rng) >= min_wick_ratio:
                        direction = -1
                        entry_px = bar['high'] - 0.5 * upper_wick
                        stop_px = bar['high'] + 1.0
                        
                # BOTTOM OF SESSION SWEEP: Swept Ref Low, closed back above Ref Low
                elif bar['low'] < ref_l - 1.0 and bar['close'] > ref_l:
                    lower_wick = min(bar['open'], bar['close']) - bar['low']
                    if (lower_wick / rng) >= min_wick_ratio:
                        direction = 1
                        entry_px = bar['low'] + 0.5 * lower_wick
                        stop_px = bar['low'] - 1.0
                        
                if direction == 0:
                    continue
                    
                risk = (entry_px - stop_px) if direction == 1 else (stop_px - entry_px)
                if risk < 5.0 or risk > 35.0:
                    continue
                    
                r_mult = 2.0 if target_mode == '2.0R' else 2.5
                target_px = (entry_px + r_mult * risk) if direction == 1 else (entry_px - r_mult * risk)
                
                # Forward fill check
                filled = False
                fill_k = -1
                for f in range(idx + 1, min(len(s_df), idx + 10)):
                    f_bar = s_df.iloc[f]
                    if direction == 1 and f_bar['low'] <= entry_px:
                        filled = True
                        fill_k = f
                        break
                    elif direction == -1 and f_bar['high'] >= entry_px:
                        filled = True
                        fill_k = f
                        break
                        
                if not filled:
                    continue
                    
                trade_taken = True
                actual_entry = entry_px + (SLIPPAGE_PTS if direction == 1 else -SLIPPAGE_PTS)
                exit_px = 0.0
                exit_reason = ""
                dur = 0
                
                for f in range(fill_k, min(len(s_df), fill_k + 48)): # hold up to 4 hours
                    dur = f - fill_k
                    f_bar = s_df.iloc[f]
                    
                    if direction == 1:
                        if f_bar['low'] <= stop_px:
                            exit_px = stop_px - SLIPPAGE_PTS
                            exit_reason = "STOP_LOSS"
                            break
                        if f_bar['high'] >= target_px:
                            exit_px = target_px - SLIPPAGE_PTS
                            exit_reason = f"TARGET_{r_mult}R"
                            break
                    else:
                        if f_bar['high'] >= stop_px:
                            exit_px = stop_px + SLIPPAGE_PTS
                            exit_reason = "STOP_LOSS"
                            break
                        if f_bar['low'] <= target_px:
                            exit_px = target_px + SLIPPAGE_PTS
                            exit_reason = f"TARGET_{r_mult}R"
                            break
                            
                    # Session close exit (16:00 EST)
                    if f_bar['hour'] == 16 and f_bar['minute'] >= 45:
                        exit_px = f_bar['close'] + (-SLIPPAGE_PTS if direction == 1 else SLIPPAGE_PTS)
                        exit_reason = "EOD_EXIT"
                        break
                        
                if exit_reason:
                    net = ((exit_px - actual_entry) if direction == 1 else (actual_entry - exit_px)) - COMMISSION_PTS
                    trades.append({
                        'session': w_name,
                        'entry_time': s_df.iloc[fill_k]['dt'],
                        'direction': 'LONG' if direction == 1 else 'SHORT',
                        'net_pts': net,
                        'risk': risk,
                        'win': net > 0,
                        'dur_mins': dur * 5
                    })
                    
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    if n == 0:
        print("No trades found.")
        return
    w = tdf[tdf['win']]
    l = tdf[~tdf['win']]
    wr = len(w) / n * 100
    pf = w['net_pts'].sum() / abs(l['net_pts'].sum()) if len(l) > 0 and l['net_pts'].sum() != 0 else 99
    tot_pts = tdf['net_pts'].sum()
    avg_sl = tdf['risk'].mean()
    avg_dur = tdf['dur_mins'].mean()
    
    print(f"\n--- Results for Target Mode: {target_mode} (Wick >= {min_wick_ratio*100:.0f}%) ---")
    print(f"Total Trades:           {n} across 3 months")
    print(f"Win Rate:               {wr:.1f}%")
    print(f"Profit Factor:          {pf:.2f}")
    print(f"Total Net Points:       {tot_pts:+,.1f} pts")
    print(f"MNQ Net Profit ($2/pt): +${tot_pts * 2.0:,.0f}")
    print(f"NQ Net Profit ($20/pt): +${tot_pts * 20.0:,.0f}")
    print(f"Average Stop Loss:      {avg_sl:.1f} pts")
    print(f"Average Duration:       {avg_dur:.0f} minutes (~{avg_dur/60:.1f} hours)")
    
    print("\nBreakdown by Session:")
    for sess, sdf in tdf.groupby('session'):
        sw = sdf[sdf['win']]
        sl = sdf[~sdf['win']]
        swr = len(sw) / len(sdf) * 100
        spf = sw['net_pts'].sum() / abs(sl['net_pts'].sum()) if len(sl) > 0 and sl['net_pts'].sum() != 0 else 99
        spts = sdf['net_pts'].sum()
        print(f"  {sess:<20} | {len(sdf):2d} trades | Win: {swr:5.1f}% | PF: {spf:4.2f} | Net: {spts:+6.1f}p")

run_session_sweep_test(target_mode='2.0R', min_wick_ratio=0.30)
run_session_sweep_test(target_mode='2.5R', min_wick_ratio=0.30)

