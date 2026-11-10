"""
09:30 Open with 1-Hour HTF Alignment & VWAP Expansion
=====================================================
Testing 09:30 Open Strategies with Macro Directional Filter:
1. 09:30 Candle 1 Retest with 1H Trend Alignment
2. 09:45 ORB Retest with 1H Trend Alignment
3. 09:30 Opening Drive + FVG Fill in Direction of HTF
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

# 5m and 1h
df_5m = df_1m.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

df_1h = df_1m.resample('1h', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_1h['ema20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
df_1h['htf_trend'] = np.where(df_1h['close'] > df_1h['ema20'], 1, -1)

df_5m['hour_ts'] = df_5m['datetime'].dt.floor('1h')
df_5m['htf_trend'] = df_5m['hour_ts'].map(df_1h['htf_trend']).fillna(0)

COMMISSION_PTS = 0.26
SLIPPAGE_PTS   = 0.25

def test_930_trend_model(target_r: float = 1.5, require_htf: bool = True):
    trades = []
    
    for date, day_df in df_5m.groupby(df_5m['datetime'].dt.date):
        day_session = day_df[(day_df['datetime'].dt.hour >= 9) & (day_df['datetime'].dt.hour <= 15)].reset_index(drop=True)
        if len(day_session) < 10:
            continue
            
        c0 = day_session.iloc[0]
        if c0['datetime'].hour != 9 or c0['datetime'].minute != 30:
            continue
            
        rng = c0['high'] - c0['low']
        if rng < 5.0 or rng > 65.0:
            continue
            
        body = abs(c0['close'] - c0['open'])
        if (body / rng) < 0.45:
            continue
            
        is_bull = c0['close'] > c0['open']
        direction = 1 if is_bull else -1
        
        # Check HTF trend
        if require_htf:
            if is_bull and c0['htf_trend'] != 1:
                continue
            if not is_bull and c0['htf_trend'] != -1:
                continue
                
        # Entry at 50% Mean Threshold of 09:30 candle
        entry_px = (c0['high'] + c0['low']) / 2.0
        stop_px = (c0['low'] - 1.0) if is_bull else (c0['high'] + 1.0)
        risk = (entry_px - stop_px) if is_bull else (stop_px - entry_px)
        
        if risk < 6.0 or risk > 35.0:
            continue
            
        target_px = (entry_px + target_r * risk) if is_bull else (entry_px - target_r * risk)
        
        filled = False
        fill_idx = -1
        for k in range(1, min(len(day_session), 12)):
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
                    exit_reason = f"TARGET_{target_r}R"
                    break
            else:
                if bar['high'] >= stop_px:
                    exit_px = stop_px + SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if bar['low'] <= target_px:
                    exit_px = target_px + SLIPPAGE_PTS
                    exit_reason = f"TARGET_{target_r}R"
                    break
                    
            if bar['datetime'].hour == 15 and bar['datetime'].minute >= 50:
                exit_px = bar['close'] + (-SLIPPAGE_PTS if is_bull else SLIPPAGE_PTS)
                exit_reason = "EOD_EXIT"
                break
                
        if exit_reason:
            net = ((exit_px - actual_entry) if is_bull else (actual_entry - exit_px)) - COMMISSION_PTS
            trades.append({
                'year': c0['datetime'].year,
                'net_pts': net,
                'risk': risk,
                'win': net > 0
            })
            
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    w = tdf[tdf['win']]
    l = tdf[~tdf['win']]
    wr = len(w) / n * 100
    pf = w['net_pts'].sum() / abs(l['net_pts'].sum())
    tot_pts = tdf['net_pts'].sum()
    avg_sl = tdf['risk'].mean()
    
    print(f"Trades: {n} ({n/8.8:.1f}/yr, {n/8.8/52:.1f}/wk) | Win: {wr:.1f}% | PF: {pf:.2f} | Avg SL: {avg_sl:.1f}p | Net Pts: {tot_pts:+,.1f}p | NQ $: +${tot_pts*20:,.0f}")
    
    print("Year by year:")
    for yr, ydf in tdf.groupby('year'):
        yw = ydf[ydf['win']]
        yl = ydf[~ydf['win']]
        y_wr = len(yw) / len(ydf) * 100
        y_pf = yw['net_pts'].sum() / abs(yl['net_pts'].sum()) if len(yl) > 0 and yl['net_pts'].sum() != 0 else 99
        y_pts = ydf['net_pts'].sum()
        print(f"  {yr}: {len(ydf):2d} trades | Win: {y_wr:5.1f}% | PF: {y_pf:4.2f} | Net: {y_pts:+6.1f}p | NQ: +${y_pts*20:>7,.0f}")

print("\n--- 09:30 CANDLE 1 RETEST WITH 1H TREND ALIGNMENT (1.5R) ---")
test_930_trend_model(target_r=1.5, require_htf=True)

print("\n--- 09:30 CANDLE 1 RETEST WITH 1H TREND ALIGNMENT (1.25R) ---")
test_930_trend_model(target_r=1.25, require_htf=True)

