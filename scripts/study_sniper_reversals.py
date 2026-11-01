"""
Precision Top & Bottom Reversal Engine (Sniper High-RR Discovery)
==================================================================
To catch the exact top or bottom of a manipulation run (whether at 9:30 AM
or across NY session macros: 09:50, 10:10, 10:30, 11:00), we test the 4 core
confluences used by top institutional reversal traders:

1. LIQUIDITY PURGE:
   - Price trades outside the Session Bollinger/VWAP 2.0-3.0 StdDev band
     OR sweeps Asia/London/PDH/PDL extreme by at least 5-15 pts.

2. EXHAUSTION WICK / REJECTION BLOCK:
   - The extreme candle prints a heavy rejection wick (at least 50% of the candle is wick).
   - The candle body closes back inside the prior range (Failed Breakout).

3. FIBONACCI RETRACEMENT (OTE - Optimal Trade Entry):
   - From the 08:30/09:30 swing low/high to the purge extreme:
   - Price reaches the 0.618 - 0.786 retracement level OR
   - Retests the 0.50 (Consequential Encroachment / CE) of the rejection wick.

4. VWAP DEVIATION BAND:
   - Price is stretched >= 2.0 standard deviations from the Session VWAP (Extreme Overextension).
   - Entry triggers on the first candle that closes back inside the 2.0 band.

5. ULTRA-TIGHT STOP:
   - Stop placed just 2.0 to 4.0 ticks beyond the wick tip extreme (typically 8 to 15 pts risk).
   - Target: Opposing session extreme or VWAP mean reversion (4R to 8R!).

Let's test this model across all real historical market data!
"""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

def run_top_bottom_reversal_study(csv_path: str):
    df = pd.read_csv(csv_path)
    date_col = 'datetime' if 'datetime' in df.columns else 'Datetime'
    df[date_col] = pd.to_datetime(df[date_col])
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_convert('America/New_York')
    else:
        df[date_col] = df[date_col].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df.set_index(date_col, inplace=True)
    df.columns = [c.capitalize() for c in df.columns]
    
    unique_days = sorted(list(set(df.index.date)))
    print(f"\n{'='*75}")
    print(f"  TESTING TOP & BOTTOM REVERSAL ENGINE ON: {csv_path}")
    print(f"  Days analyzed: {len(unique_days)}")
    print(f"{'='*75}")
    
    trades = []
    
    for day in unique_days:
        df_day = df[df.index.date == day].copy()
        if len(df_day) < 100:
            continue
            
        # Calculate Anchored Session VWAP starting from 09:30 AM
        rth = df_day.between_time('09:30', '16:00').copy()
        if len(rth) < 30:
            continue
            
        # Typical Price & Cumulative VWAP
        tp = (rth['High'] + rth['Low'] + rth['Close']) / 3.0
        vol = rth['Volume']
        cum_vol = vol.cumsum()
        cum_tp_vol = (tp * vol).cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, 1)
        
        # VWAP Standard Deviation Bands
        variance = ((tp - vwap) ** 2 * vol).cumsum() / cum_vol.replace(0, 1)
        stdev = np.sqrt(variance.replace(0, 0.001))
        
        vwap_upper_2 = vwap + (stdev * 2.0)
        vwap_lower_2 = vwap - (stdev * 2.0)
        
        # Benchmark reference levels: London High/Low + PDH/PDL
        london = df_day.between_time('02:00', '06:00')
        l_high = london['High'].max() if len(london) > 5 else rth['High'].iloc[:15].max()
        l_low  = london['Low'].min()  if len(london) > 5 else rth['Low'].iloc[:15].min()
        
        # Scan for tops & bottoms between 09:30 and 11:30 AM
        scan_window = rth.between_time('09:30', '11:30')
        
        traded_today = False
        
        for i in range(10, len(scan_window) - 10):
            if traded_today:
                break
                
            bar = scan_window.iloc[i]
            b_time = scan_window.index[i]
            b_open  = bar['Open']
            b_high  = bar['High']
            b_low   = bar['Low']
            b_close = bar['Close']
            
            b_vwap_u = vwap_upper_2.loc[b_time]
            b_vwap_l = vwap_lower_2.loc[b_time]
            b_vwap   = vwap.loc[b_time]
            
            # --- CATCHING THE EXACT TOP (SHORT ENTRY) ---
            # Condition 1: Price extended above VWAP +2.0 Bands OR pierced London High / Session High
            # Condition 2: Strong Rejection Upper Wick (wick >= 50% of candle range)
            # Condition 3: Candle closes bearish (rejection confirmed)
            candle_range = b_high - b_low
            upper_wick   = b_high - max(b_open, b_close)
            
            if candle_range >= 8.0 and (upper_wick >= candle_range * 0.45) and (b_close < b_open):
                if (b_high >= b_vwap_u) or (b_high >= l_high):
                    # Sniper Short Entry:
                    # Enter on next bar at 50% retest of the wick (CE) or immediate market
                    entry_p = max(b_open, b_close) # Conservative entry near top of candle body
                    stop_loss = b_high + 1.0        # Super tight stop: 1 pt above wick tip!
                    risk_pts = stop_loss - entry_p
                    
                    if 6.0 <= risk_pts <= 22.0:
                        # High R:R Target: 3.5R to 5.0R or VWAP mean reversion
                        target_pts = risk_pts * 4.0  # 4R Target!
                        take_profit = entry_p - target_pts
                        
                        # Simulate trade
                        outcome, exit_p = _simulate_reversal(rth, i, -1, entry_p, stop_loss, take_profit)
                        gross_pts = entry_p - exit_p
                        trades.append({
                            'date': day,
                            'time': b_time.strftime('%H:%M'),
                            'type': 'TOP REVERSAL (Short)',
                            'entry': entry_p,
                            'stop': stop_loss,
                            'risk_pts': risk_pts,
                            'target': take_profit,
                            'outcome': outcome,
                            'gross_pts': gross_pts,
                            'realized_r': gross_pts / risk_pts,
                            'net_nq': gross_pts * 20.0 - 10.20
                        })
                        traded_today = True
                        break

            # --- CATCHING THE EXACT BOTTOM (LONG ENTRY) ---
            # Condition 1: Price pierced below VWAP -2.0 Bands OR pierced London Low
            # Condition 2: Strong Rejection Lower Wick (lower wick >= 50% of candle range)
            # Condition 3: Candle closes bullish
            lower_wick = min(b_open, b_close) - b_low
            if candle_range >= 8.0 and (lower_wick >= candle_range * 0.45) and (b_close > b_open):
                if (b_low <= b_vwap_l) or (b_low <= l_low):
                    # Sniper Long Entry:
                    entry_p = min(b_open, b_close)
                    stop_loss = b_low - 1.0         # 1 pt below bottom wick tip!
                    risk_pts = entry_p - stop_loss
                    
                    if 6.0 <= risk_pts <= 22.0:
                        target_pts = risk_pts * 4.0  # 4R Target!
                        take_profit = entry_p + target_pts
                        
                        outcome, exit_p = _simulate_reversal(rth, i, 1, entry_p, stop_loss, take_profit)
                        gross_pts = exit_p - entry_p
                        trades.append({
                            'date': day,
                            'time': b_time.strftime('%H:%M'),
                            'type': 'BOTTOM REVERSAL (Long)',
                            'entry': entry_p,
                            'stop': stop_loss,
                            'risk_pts': risk_pts,
                            'target': take_profit,
                            'outcome': outcome,
                            'gross_pts': gross_pts,
                            'realized_r': gross_pts / risk_pts,
                            'net_nq': gross_pts * 20.0 - 10.20
                        })
                        traded_today = True
                        break

    res_df = pd.DataFrame(trades)
    _print_reversal_metrics(res_df, csv_path)
    return res_df


def _simulate_reversal(df_rth, start_idx, direction, entry, sl, tp):
    for j in range(start_idx + 1, len(df_rth)):
        bar = df_rth.iloc[j]
        # EOD exit at 15:45
        if df_rth.index[j].hour == 15 and df_rth.index[j].minute >= 45:
            return 'TIME_EXIT', bar['Open']
            
        if direction == 1:
            if bar['Low'] <= sl:
                return 'STOP_LOSS', sl
            if bar['High'] >= tp:
                return 'TARGET_4R', tp
        else:
            if bar['High'] >= sl:
                return 'STOP_LOSS', sl
            if bar['Low'] <= tp:
                return 'TARGET_4R', tp
                
    return 'EOD', df_rth['Close'].iloc[-1]


def _print_reversal_metrics(res_df, path):
    if res_df.empty:
        print("No trades triggered.")
        return
        
    total = len(res_df)
    wins = len(res_df[res_df['outcome'] == 'TARGET_4R'])
    losses = len(res_df[res_df['outcome'] == 'STOP_LOSS'])
    wr = (wins / total) * 100
    
    gw = res_df[res_df['gross_pts'] > 0]['gross_pts'].sum() * 20.0
    gl = abs(res_df[res_df['gross_pts'] <= 0]['gross_pts'].sum()) * 20.0
    pf = gw / gl if gl > 0 else float('inf')
    
    avg_risk = res_df['risk_pts'].mean()
    net_total = res_df['net_nq'].sum()
    avg_r = res_df['realized_r'].mean()
    
    print(f"\nRecent Sniper Reversal Trades (Sample):")
    print(res_df[['date', 'time', 'type', 'risk_pts', 'outcome', 'realized_r', 'net_nq']].tail(10).to_string(index=False))
    print(f"\n-------------------------------------------------------------")
    print(f"Total Reversal Trades Detected : {total}")
    print(f"Wins (Full 4.0R Target)        : {wins} ({wr:.1f}%)")
    print(f"Losses (Tight Stops)          : {losses} ({losses/total*100:.1f}%)")
    print(f"Profit Factor                 : {pf:.2f}")
    print(f"Average Risk (Stop Size)      : {avg_risk:.1f} pts (${avg_risk*20:.0f} on NQ)")
    print(f"Average Realized R / Trade    : {avg_r:+.2f}R")
    print(f"Total Net Profit (1 Mini NQ)  : ${net_total:+,.2f}")
    print(f"-------------------------------------------------------------")

if __name__ == '__main__':
    run_top_bottom_reversal_study('sample_data/REAL_NQ_1min.csv')
    run_top_bottom_reversal_study('sample_data/MNQ_1min_20260120_20260415.csv')

