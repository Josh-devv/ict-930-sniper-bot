"""
Powell Trades 10 AM AMD Silver Bullet: Diagnosis & Optimization
==============================================================
Tests Powell's 10:00 AM AMD Strategy:
A: Accumulation (00:00 - 09:29)
M: Manipulation (09:30 - 10:00)
D: Distribution (10:00 - 11:00 Silver Bullet)

Compares:
1. Raw Powell M3 (Blind Reversal targeting 3.5R)
2. Optimized Powell 10 AM:
   + 1-Hour HTF Trend Alignment (Only take distribution WITH the macro trend)
   + Orderflow Delta Absorption on the manipulation sweep
   + Realistic Target (1.5R and 2.0R)
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

# 1-min Delta proxy
rng_1m = (df_1m['high'] - df_1m['low']).replace(0, 1e-6)
df_1m['delta'] = df_1m['volume'] * (2.0 * df_1m['close'] - df_1m['high'] - df_1m['low']) / rng_1m

# Resample 5m and 1h
df_5m = df_1m.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'delta': 'sum'
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

def run_powell_10am(use_htf: bool = False, use_orderflow: bool = False, target_r: float = 2.0):
    trades = []
    
    for date, day_df in df_5m.groupby(df_5m['datetime'].dt.date):
        day_session = day_df[(day_df['datetime'].dt.hour >= 9) & (day_df['datetime'].dt.hour <= 15)].reset_index(drop=True)
        if len(day_session) < 15:
            continue
            
        # Accumulation / Opening baseline: 09:30 candle
        c0 = day_session.iloc[0]
        if c0['datetime'].hour != 9 or c0['datetime'].minute != 30:
            continue
            
        # Manipulation phase: 09:30 to 10:00 (first 6 bars)
        manip_bars = day_session.iloc[:6]
        m_high = manip_bars['high'].max()
        m_low = manip_bars['low'].min()
        m_rng = m_high - m_low
        
        if m_rng < 8.0:
            continue
            
        # Distribution phase: 10:00 to 11:00 (bars 6 to 18)
        # Look for 10:00+ displacement
        for k in range(6, min(len(day_session), 18)):
            bar = day_session.iloc[k]
            prev = day_session.iloc[k-1]
            body = abs(bar['close'] - bar['open'])
            
            # Bullish Distribution: sweeps manipulation low, then displaces upward
            is_bull = (prev['low'] <= m_low) and (bar['close'] > bar['open']) and (body >= 6.0)
            # Bearish Distribution: sweeps manipulation high, then displaces downward
            is_bear = (prev['high'] >= m_high) and (bar['close'] < bar['open']) and (body >= 6.0)
            
            if not (is_bull or is_bear):
                continue
                
            direction = 1 if is_bull else -1
            
            # HTF filter
            if use_htf:
                if is_bull and bar['htf_trend'] != 1:
                    continue
                if is_bear and bar['htf_trend'] != -1:
                    continue
                    
            # Orderflow absorption on the sweep bar (prev)
            if use_orderflow:
                prev_rng = prev['high'] - prev['low']
                if prev_rng > 0:
                    if is_bull:
                        wick = min(prev['open'], prev['close']) - prev['low']
                        if (wick / prev_rng < 0.25) and (prev['delta'] <= 0):
                            continue
                    else:
                        wick = prev['high'] - max(prev['open'], prev['close'])
                        if (wick / prev_rng < 0.25) and (prev['delta'] >= 0):
                            continue
                            
            # Entry at 50% Mean Threshold of displacement / OB
            entry_px = (bar['high'] + bar['low']) / 2.0
            stop_px = (m_low - 1.0) if is_bull else (m_high + 1.0)
            risk = (entry_px - stop_px) if is_bull else (stop_px - entry_px)
            
            if risk < 6.0 or risk > 35.0:
                continue
                
            target_px = (entry_px + target_r * risk) if is_bull else (entry_px - target_r * risk)
            
            # Walk forward for fill
            filled = False
            fill_idx = -1
            for f in range(k + 1, min(len(day_session), k + 8)):
                f_bar = day_session.iloc[f]
                if is_bull and f_bar['low'] <= entry_px:
                    filled = True
                    fill_idx = f
                    break
                elif not is_bull and f_bar['high'] >= entry_px:
                    filled = True
                    fill_idx = f
                    break
                    
            if not filled:
                continue
                
            actual_entry = entry_px + (SLIPPAGE_PTS if is_bull else -SLIPPAGE_PTS)
            exit_px = 0.0
            exit_reason = ""
            
            for f in range(fill_idx, min(len(day_session), fill_idx + 35)):
                f_bar = day_session.iloc[f]
                if is_bull:
                    if f_bar['low'] <= stop_px:
                        exit_px = stop_px - SLIPPAGE_PTS
                        exit_reason = "STOP_LOSS"
                        break
                    if f_bar['high'] >= target_px:
                        exit_px = target_px - SLIPPAGE_PTS
                        exit_reason = f"TARGET_{target_r}R"
                        break
                else:
                    if f_bar['high'] >= stop_px:
                        exit_px = stop_px + SLIPPAGE_PTS
                        exit_reason = "STOP_LOSS"
                        break
                    if f_bar['low'] <= target_px:
                        exit_px = target_px + SLIPPAGE_PTS
                        exit_reason = f"TARGET_{target_r}R"
                        break
                        
                if f_bar['datetime'].hour == 15 and f_bar['datetime'].minute >= 50:
                    exit_px = f_bar['close'] + (-SLIPPAGE_PTS if is_bull else SLIPPAGE_PTS)
                    exit_reason = "EOD_EXIT"
                    break
                    
            if exit_reason:
                net = ((exit_px - actual_entry) if is_bull else (actual_entry - exit_px)) - COMMISSION_PTS
                trades.append({
                    'year': bar['datetime'].year,
                    'net_pts': net,
                    'win': net > 0
                })
                break
                
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    if n == 0:
        return 0, 0, 0, 0
    w = tdf[tdf['win']]
    l = tdf[~tdf['win']]
    wr = len(w) / n * 100
    pf = w['net_pts'].sum() / abs(l['net_pts'].sum()) if len(l) > 0 and l['net_pts'].sum() != 0 else 99
    tot = tdf['net_pts'].sum()
    return n, wr, pf, tot

print("\n" + "=" * 80)
print("POWELL 10 AM AMD SILVER BULLET: RAW VS OPTIMIZED")
print("=" * 80)

for name, htf, of, trg in [
    ("1. Raw Powell 10 AM AMD (No HTF, 3.5R)", False, False, 3.5),
    ("2. Raw Powell 10 AM AMD (No HTF, 2.0R)", False, False, 2.0),
    ("3. Powell 10 AM + 1H Trend (2.0R)", True, False, 2.0),
    ("4. Powell 10 AM + 1H Trend (1.5R)", True, False, 1.5),
    ("5. Powell 10 AM + 1H Trend + Orderflow (1.5R)", True, True, 1.5),
]:
    n, wr, pf, tot = run_powell_10am(use_htf=htf, use_orderflow=of, target_r=trg)
    print(f"{name:<45} | Trades: {n:4d} | Win: {wr:5.1f}% | PF: {pf:4.2f} | Net: {tot:+7.1f}p | NQ $: +${tot*20:>8,.0f}")

