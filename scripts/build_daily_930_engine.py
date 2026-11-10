"""
The Daily 09:30-10:30 NQ System: Finding a Setup that Fires Almost Every Single Day
===================================================================================
Total trading days in 8.8 years: ~2,218 days (~252 days/year)
Target: 150 - 220+ trades/year (~3 to 5 trades per week, nearly every day!)

Testing Models:
1. DAILY ORB RETEST (09:30-09:40 Opening Range Breakout Retest on 3m/5m)
2. DAILY 09:30 OPENING DRIVE RETEST (Enter 50% pullback of first 5m/3m bar every day)
3. DAILY 09:30-10:30 3-MINUTE FVG CONTINUATION (Enter first 3m FVG after 09:30 open)
4. DAILY VWAP BOUNCE / CROSS AT OPEN (First retest of 09:30 Anchored VWAP)

Strict Realism:
- $5.20 RT NQ / $1.24 MNQ fees
- 1 tick slippage
- Hard stop loss checked FIRST on every bar
- Realistic limit fills
"""

import sys
import pandas as pd
import numpy as np

# Load 1m
df_1m = pd.read_csv('sample_data/NQ_1min_RTH_2018_2026.csv')
df_1m.columns = [c.lower() for c in df_1m.columns]
df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
df_1m = df_1m[df_1m['high'] > df_1m['low']].reset_index(drop=True)
df_1m = df_1m.set_index('datetime').sort_index()

# Resample 3m and 5m
df_3m = df_1m.resample('3min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

df_5m = df_1m.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

COMMISSION_PTS = 0.26
SLIPPAGE_PTS   = 0.25

# ─── MODEL 1: DAILY 09:30-09:35 CANDLE 1 50% RETEST (Every Day) ──────────────
def test_daily_candle1_retest(df_tf: pd.DataFrame, tf_name: str, target_r: float = 1.25):
    trades = []
    
    for date, day_df in df_tf.groupby(df_tf['datetime'].dt.date):
        day_session = day_df[(day_df['datetime'].dt.hour >= 9) & (day_df['datetime'].dt.hour <= 15)].reset_index(drop=True)
        if len(day_session) < 10:
            continue
            
        c0 = day_session.iloc[0]
        if c0['datetime'].hour != 9 or c0['datetime'].minute != 30:
            continue
            
        rng = c0['high'] - c0['low']
        if rng < 4.0:
            continue
            
        # Direction purely determined by Candle 1 close vs open
        is_bull = c0['close'] >= c0['open']
        direction = 1 if is_bull else -1
        
        # 50% retracement entry
        entry_px = (c0['high'] + c0['low']) / 2.0
        stop_px = (c0['low'] - 1.0) if is_bull else (c0['high'] + 1.0)
        risk = (entry_px - stop_px) if is_bull else (stop_px - entry_px)
        
        if risk < 5.0 or risk > 35.0:
            continue
            
        target_px = (entry_px + target_r * risk) if is_bull else (entry_px - target_r * risk)
        
        # Walk forward through morning (until 11:30)
        filled = False
        fill_idx = -1
        for k in range(1, min(len(day_session), 15)):
            bar = day_session.iloc[k]
            if is_bull and bar['low'] <= entry_px:
                filled = True
                fill_idx = k
                break
            elif not is_bull and bar['high'] >= entry_px:
                filled = True
                fill_idx = k
                break
                
        if not filled:
            continue
            
        actual_entry = entry_px + (SLIPPAGE_PTS if is_bull else -SLIPPAGE_PTS)
        exit_px = 0.0
        exit_reason = ""
        
        for k in range(fill_idx, min(len(day_session), fill_idx + 40)):
            bar = day_session.iloc[k]
            if is_bull:
                if bar['low'] <= stop_px:
                    exit_px = stop_px - SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if bar['high'] >= target_px:
                    exit_px = target_px - SLIPPAGE_PTS
                    exit_reason = "TARGET"
                    break
            else:
                if bar['high'] >= stop_px:
                    exit_px = stop_px + SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if bar['low'] <= target_px:
                    exit_px = target_px + SLIPPAGE_PTS
                    exit_reason = "TARGET"
                    break
                    
            if bar['datetime'].hour == 15 and bar['datetime'].minute >= 50:
                exit_px = bar['close'] + (-SLIPPAGE_PTS if is_bull else SLIPPAGE_PTS)
                exit_reason = "EOD"
                break
                
        if exit_reason:
            net = ((exit_px - actual_entry) if is_bull else (actual_entry - exit_px)) - COMMISSION_PTS
            trades.append({
                'year': c0['datetime'].year,
                'net_pts': net,
                'risk': risk,
                'win': net > 0
            })
            
    return trades


# ─── MODEL 2: DAILY 09:35 - 09:40 OPENING RANGE BREAKOUT & RETEST ────────────
def test_daily_orb_retest(target_r: float = 1.25):
    trades = []
    
    for date, day_df in df_3m.groupby(df_3m['datetime'].dt.date):
        day_session = day_df[(day_df['datetime'].dt.hour >= 9) & (day_df['datetime'].dt.hour <= 15)].reset_index(drop=True)
        if len(day_session) < 15:
            continue
            
        # First two 3m bars: 09:30 and 09:33 (09:30 - 09:36 Opening Range)
        or_bars = day_session.iloc[:2]
        if or_bars.iloc[0]['datetime'].hour != 9 or or_bars.iloc[0]['datetime'].minute != 30:
            continue
            
        orh = or_bars['high'].max()
        orl = or_bars['low'].min()
        or_rng = orh - orl
        
        if or_rng < 5.0 or or_rng > 60.0:
            continue
            
        # Look for breakout between 09:36 and 10:15
        bo_idx = -1
        is_bull = False
        for k in range(2, min(len(day_session), 15)):
            bar = day_session.iloc[k]
            if bar['close'] > orh:
                bo_idx = k
                is_bull = True
                break
            elif bar['close'] < orl:
                bo_idx = k
                is_bull = False
                break
                
        if bo_idx == -1:
            continue
            
        bo_bar = day_session.iloc[bo_idx]
        entry_px = orh if is_bull else orl
        stop_px = (bo_bar['low'] - 1.0) if is_bull else (bo_bar['high'] + 1.0)
        risk = (entry_px - stop_px) if is_bull else (stop_px - entry_px)
        
        if risk < 5.0 or risk > 30.0:
            continue
            
        target_px = (entry_px + target_r * risk) if is_bull else (entry_px - target_r * risk)
        
        # Fill on retest
        filled = False
        fill_idx = -1
        for k in range(bo_idx + 1, min(len(day_session), bo_idx + 12)):
            bar = day_session.iloc[k]
            if is_bull and bar['low'] <= entry_px:
                filled = True
                fill_idx = k
                break
            elif not is_bull and bar['high'] >= entry_px:
                filled = True
                fill_idx = k
                break
                
        if not filled:
            continue
            
        actual_entry = entry_px + (SLIPPAGE_PTS if is_bull else -SLIPPAGE_PTS)
        exit_px = 0.0
        exit_reason = ""
        
        for k in range(fill_idx, min(len(day_session), fill_idx + 35)):
            bar = day_session.iloc[k]
            if is_bull:
                if bar['low'] <= stop_px:
                    exit_px = stop_px - SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if bar['high'] >= target_px:
                    exit_px = target_px - SLIPPAGE_PTS
                    exit_reason = "TARGET"
                    break
            else:
                if bar['high'] >= stop_px:
                    exit_px = stop_px + SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if bar['low'] <= target_px:
                    exit_px = target_px + SLIPPAGE_PTS
                    exit_reason = "TARGET"
                    break
                    
            if bar['datetime'].hour == 15 and bar['datetime'].minute >= 50:
                exit_px = bar['close'] + (-SLIPPAGE_PTS if is_bull else SLIPPAGE_PTS)
                exit_reason = "EOD"
                break
                
        if exit_reason:
            net = ((exit_px - actual_entry) if is_bull else (actual_entry - exit_px)) - COMMISSION_PTS
            trades.append({
                'year': bo_bar['datetime'].year,
                'net_pts': net,
                'risk': risk,
                'win': net > 0
            })
            
    return trades


# ─── MODEL 3: DAILY 09:30-10:15 3-MINUTE FVG SWEEP CONTINUATION ───────────────
def test_daily_3m_fvg(target_r: float = 1.5):
    """
    Every day between 09:30 and 10:15:
    Find the first 3-minute Fair Value Gap (FVG) created by displacement.
    Buy/Sell the 50% retest of the FVG.
    Stop: 1 pt behind the displacement candle swing.
    """
    trades = []
    
    for date, day_df in df_3m.groupby(df_3m['datetime'].dt.date):
        day_session = day_df[(day_df['datetime'].dt.hour >= 9) & (day_df['datetime'].dt.hour <= 15)].reset_index(drop=True)
        if len(day_session) < 15:
            continue
            
        c0 = day_session.iloc[0]
        if c0['datetime'].hour != 9 or c0['datetime'].minute != 30:
            continue
            
        fvg_found = False
        fvg_entry = 0.0
        fvg_stop = 0.0
        is_bull = False
        found_idx = -1
        
        # Look through 09:33 to 10:15 (bars 1 to 15)
        for k in range(2, min(len(day_session), 16)):
            b2 = day_session.iloc[k-2]
            b1 = day_session.iloc[k-1]
            b0 = day_session.iloc[k]
            
            # Bullish FVG: Low of b0 > High of b2
            if b0['low'] > b2['high'] and (b1['close'] > b1['open']):
                gap_size = b0['low'] - b2['high']
                if gap_size >= 2.0:
                    fvg_entry = (b0['low'] + b2['high']) / 2.0
                    fvg_stop = b1['low'] - 1.0
                    is_bull = True
                    fvg_found = True
                    found_idx = k
                    break
                    
            # Bearish FVG: High of b0 < Low of b2
            elif b0['high'] < b2['low'] and (b1['close'] < b1['open']):
                gap_size = b2['low'] - b0['high']
                if gap_size >= 2.0:
                    fvg_entry = (b2['low'] + b0['high']) / 2.0
                    fvg_stop = b1['high'] + 1.0
                    is_bull = False
                    fvg_found = True
                    found_idx = k
                    break
                    
        if not fvg_found:
            continue
            
        risk = (fvg_entry - fvg_stop) if is_bull else (fvg_stop - fvg_entry)
        if risk < 5.0 or risk > 30.0:
            continue
            
        target_px = (fvg_entry + target_r * risk) if is_bull else (fvg_entry - target_r * risk)
        
        # Fill on retest
        filled = False
        fill_idx = -1
        for k in range(found_idx + 1, min(len(day_session), found_idx + 12)):
            bar = day_session.iloc[k]
            if is_bull and bar['low'] <= fvg_entry:
                filled = True
                fill_idx = k
                break
            elif not is_bull and bar['high'] >= fvg_entry:
                filled = True
                fill_idx = k
                break
                
        if not filled:
            continue
            
        actual_entry = fvg_entry + (SLIPPAGE_PTS if is_bull else -SLIPPAGE_PTS)
        exit_px = 0.0
        exit_reason = ""
        
        for k in range(fill_idx, min(len(day_session), fill_idx + 35)):
            bar = day_session.iloc[k]
            if is_bull:
                if bar['low'] <= fvg_stop:
                    exit_px = fvg_stop - SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if bar['high'] >= target_px:
                    exit_px = target_px - SLIPPAGE_PTS
                    exit_reason = "TARGET"
                    break
            else:
                if bar['high'] >= fvg_stop:
                    exit_px = fvg_stop + SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if bar['low'] <= target_px:
                    exit_px = target_px + SLIPPAGE_PTS
                    exit_reason = "TARGET"
                    break
                    
            if bar['datetime'].hour == 15 and bar['datetime'].minute >= 50:
                exit_px = bar['close'] + (-SLIPPAGE_PTS if is_bull else SLIPPAGE_PTS)
                exit_reason = "EOD"
                break
                
        if exit_reason:
            net = ((exit_px - actual_entry) if is_bull else (actual_entry - exit_px)) - COMMISSION_PTS
            trades.append({
                'year': c0['datetime'].year,
                'net_pts': net,
                'risk': risk,
                'win': net > 0
            })
            
    return trades


# ─── RUN AND PRINT RESULTS ───────────────────────────────────────────────────
print("\n" + "=" * 105)
print(f"| {'EVERYDAY 09:30-10:30 STRATEGY':<35} | {'TRADES':>6} | {'TR/YR':>6} | {'TR/WK':>5} | {'WIN%':>6} | {'PF':>5} | {'AVG SL':>6} | {'NET PTS':>8} | {'NQ NET $':>10} |")
print("=" * 105)

test_configs = [
    ("1. Daily 5m Candle 1 Retest (1.25R)", test_daily_candle1_retest(df_5m, "5m", 1.25)),
    ("2. Daily 5m Candle 1 Retest (1.5R)",  test_daily_candle1_retest(df_5m, "5m", 1.50)),
    ("3. Daily 3m Candle 1 Retest (1.25R)", test_daily_candle1_retest(df_3m, "3m", 1.25)),
    ("4. Daily 3m ORB Retest (1.25R)",      test_daily_orb_retest(1.25)),
    ("5. Daily 3m ORB Retest (1.5R)",       test_daily_orb_retest(1.50)),
    ("6. Daily 3m FVG Retest (1.25R)",      test_daily_3m_fvg(1.25)),
    ("7. Daily 3m FVG Retest (1.5R)",       test_daily_3m_fvg(1.50)),
]

for name, t_list in test_configs:
    tdf = pd.DataFrame(t_list)
    n = len(tdf)
    if n == 0:
        continue
    w = tdf[tdf['win']]
    l = tdf[~tdf['win']]
    wr = len(w) / n * 100
    pf = w['net_pts'].sum() / abs(l['net_pts'].sum()) if len(l) > 0 and l['net_pts'].sum() != 0 else 99
    tot = tdf['net_pts'].sum()
    avg_sl = tdf['risk'].mean()
    tr_yr = n / 8.8
    tr_wk = tr_yr / 52.0
    
    print(f"| {name:<35} | {n:>6} | {tr_yr:>6.1f} | {tr_wk:>5.1f} | {wr:>5.1f}% | {pf:>5.2f} | {avg_sl:>5.1f}p | {tot:>+7.1f}p | +${tot*20:>9,.0f} |")

print("=" * 105)

