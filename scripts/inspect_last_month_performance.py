"""
Inspect Last Month Performance: High-Frequency Tony Iyke SMC (2-Min & 3-Min)
=============================================================================
Detailed breakdown of October 2026 (and September / recent months)
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List

# Import simulation logic from test_high_frequency_tony_iyke
from test_high_frequency_tony_iyke import run_hf_tony_iyke, df_1m

print("Data range in NQ dataset:")
print(f"Start: {df_1m.index[0]}  |  End: {df_1m.index[-1]}")

# Run 2-Minute Engine
print("\nRunning 2-Minute Engine...")
trades_2m = run_hf_tony_iyke(timeframe='2min', htf_mode='1h', entry_mode='limit_50', target_r=1.5)
df_trades_2m = pd.DataFrame([t.__dict__ for t in trades_2m])
df_trades_2m['month'] = df_trades_2m['entry_time'].dt.to_period('M')

# Run 3-Minute Engine
print("Running 3-Minute Engine...")
trades_3m = run_hf_tony_iyke(timeframe='3min', htf_mode='1h', entry_mode='limit_50', target_r=1.5)
df_trades_3m = pd.DataFrame([t.__dict__ for t in trades_3m])
df_trades_3m['month'] = df_trades_3m['entry_time'].dt.to_period('M')

def print_month_detail(df_tr: pd.DataFrame, label: str):
    print("\n" + "=" * 80)
    print(f"MONTHLY BREAKDOWN: {label}")
    print("=" * 80)
    
    unique_months = sorted(df_tr['month'].unique())
    recent_months = unique_months[-4:] # Last 4 months
    
    for m in recent_months:
        sub = df_tr[df_tr['month'] == m]
        n = len(sub)
        wins = sub[sub['net_pts'] > 0]
        losses = sub[sub['net_pts'] <= 0]
        wr = len(wins) / n * 100.0 if n > 0 else 0.0
        gains = wins['net_pts'].sum()
        loss_val = abs(losses['net_pts'].sum())
        pf = gains / loss_val if loss_val > 0 else 99.0
        net_pts = sub['net_pts'].sum()
        net_usd = net_pts * 20.0
        avg_sl = sub['risk_pts'].mean() if n > 0 else 0.0
        
        print(f"Month: {m} | Trades: {n:>2} | WR: {wr:>5.1f}% ({len(wins)}W/{len(losses)}L) | PF: {pf:>4.2f} | Net: {net_pts:>+7.1f} pts (${net_usd:>+9.2f} NQ / ${net_usd/10:>+7.2f} MNQ) | Avg SL: {avg_sl:.1f}p")

    # Detailed trade log of the very last complete month
    last_m = recent_months[-1]
    print(f"\n--- DETAILED TRADE LOG FOR LAST MONTH ({last_m}) [{label}] ---")
    sub_last = df_tr[df_tr['month'] == last_m].sort_values('entry_time')
    
    print(f"{'Date/Time':<17} | {'Dir':<4} | {'Entry':<8} | {'Stop':<8} | {'Risk':<5} | {'Exit':<8} | {'Reason':<6} | {'Net Pts':<8} | {'Net USD (NQ)'}")
    print("-" * 85)
    cum_pts = 0.0
    for _, row in sub_last.iterrows():
        t_str = row['entry_time'].strftime('%Y-%m-%d %H:%M')
        dir_str = "BUY" if row['direction'] == 1 else "SELL"
        pts = row['net_pts']
        cum_pts += pts
        usd = pts * 20.0
        print(f"{t_str:<17} | {dir_str:<4} | {row['entry_px']:<8.2f} | {row['stop_px']:<8.2f} | {row['risk_pts']:<5.1f} | {row['exit_px']:<8.2f} | {row['exit_reason']:<6} | {pts:>+7.1f}p | ${usd:>+8.2f} (Cum: ${cum_pts*20:>+8.2f})")

print_month_detail(df_trades_2m, "2-MINUTE SMC (Daily Operator)")
print_month_detail(df_trades_3m, "3-MINUTE SMC (Active Morning/PM)")

