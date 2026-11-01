"""
Study Fibonacci Retracement (OTE) and Market Structure Shift (MSS)
==================================================================
Why did catching the blind wick fail on the trending week?
Because in a strong trend, price pierces the VWAP band and KEEPS GOING.
Institutions DO NOT blindly fade the wick.
Instead, they wait for TWO confirmations:
1. THE RUN: Price sweeps liquidity / VWAP 2.0 Band.
2. THE SHIFT (MSS): A 1-minute candle breaks internal structure (closes below prior candle low).
3. THE FIB RETRACEMENT (OTE): Price pulls back into the 0.618 - 0.786 Fibonacci retracement
   of that displacement impulse.
4. ENTRY: Enter at the 0.618 Fib, Stop above the swing high.
Let's measure this exact Fibonacci OTE mechanism!
"""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

def run_fib_ote_reversal_study(csv_path: str):
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
    print(f"  TESTING FIBONACCI OTE + STRUCTURE SHIFT (MSS) ON: {csv_path}")
    print(f"{'='*75}")
    
    trades = []
    
    for day in unique_days:
        df_day = df[df.index.date == day].copy()
        rth = df_day.between_time('09:30', '16:00')
        if len(rth) < 30:
            continue
            
        # London benchmarks
        london = df_day.between_time('02:00', '06:00')
        l_high = london['High'].max() if len(london) > 5 else rth['High'].iloc[:15].max()
        l_low  = london['Low'].min()  if len(london) > 5 else rth['Low'].iloc[:15].min()
        
        # Look between 09:30 and 11:30 AM
        scan = rth.between_time('09:30', '11:30')
        traded_today = False
        
        for i in range(15, len(scan) - 15):
            if traded_today:
                break
                
            bar = scan.iloc[i]
            prev_bars = scan.iloc[i-10:i]
            
            # --- FIB TOP REVERSAL (SHORT) ---
            # 1. Did price reach a manipulation peak above London High / Session High?
            peak_h = scan['High'].iloc[:i+1].max()
            if bar['High'] == peak_h and peak_h > l_high:
                # 2. Look for displacement down (MSS: breaks below recent 3-bar low)
                recent_low = prev_bars['Low'].iloc[-3:].min()
                # Check next 5 bars for displacement break
                for j in range(i+1, min(i+6, len(scan))):
                    d_bar = scan.iloc[j]
                    if d_bar['Close'] < recent_low: # MSS Confirmed!
                        impulse_high = peak_h
                        impulse_low  = d_bar['Low']
                        impulse_range = impulse_high - impulse_low
                        
                        if 12.0 <= impulse_range <= 50.0:
                            # 3. Calculate 61.8% Fib Retracement
                            fib_618 = impulse_low + (impulse_range * 0.618)
                            fib_786 = impulse_low + (impulse_range * 0.786)
                            
                            # Check next 8 bars for retest of 61.8% Fib
                            for k in range(j+1, min(j+10, len(scan))):
                                f_bar = scan.iloc[k]
                                if f_bar['High'] >= fib_618 and f_bar['Open'] <= fib_786:
                                    # Sniper Entry at 61.8% Fib!
                                    entry_p = fib_618
                                    stop_loss = impulse_high + 1.5 # tight stop above peak
                                    risk_pts = stop_loss - entry_p
                                    
                                    if 6.0 <= risk_pts <= 25.0:
                                        target_pts = risk_pts * 3.0 # 3R Target
                                        take_profit = entry_p - target_pts
                                        
                                        # Simulate trade
                                        outcome, exit_p = _sim_fib_trade(rth, k, -1, entry_p, stop_loss, take_profit)
                                        gross_pts = entry_p - exit_p
                                        trades.append({
                                            'date': day,
                                            'time': scan.index[k].strftime('%H:%M'),
                                            'direction': 'SHORT (Fib 61.8% Top)',
                                            'risk_pts': risk_pts,
                                            'outcome': outcome,
                                            'realized_r': gross_pts / risk_pts,
                                            'net_nq': gross_pts * 20.0 - 10.20
                                        })
                                        traded_today = True
                                        break
                            if traded_today: break
                            
            # --- FIB BOTTOM REVERSAL (LONG) ---
            trough_l = scan['Low'].iloc[:i+1].min()
            if bar['Low'] == trough_l and trough_l < l_low:
                recent_high = prev_bars['High'].iloc[-3:].max()
                for j in range(i+1, min(i+6, len(scan))):
                    d_bar = scan.iloc[j]
                    if d_bar['Close'] > recent_high: # MSS Bullish Confirmed!
                        impulse_high = d_bar['High']
                        impulse_low  = trough_l
                        impulse_range = impulse_high - impulse_low
                        
                        if 12.0 <= impulse_range <= 50.0:
                            fib_618 = impulse_high - (impulse_range * 0.618)
                            fib_786 = impulse_high - (impulse_range * 0.786)
                            
                            for k in range(j+1, min(j+10, len(scan))):
                                f_bar = scan.iloc[k]
                                if f_bar['Low'] <= fib_618 and f_bar['Open'] >= fib_786:
                                    entry_p = fib_618
                                    stop_loss = impulse_low - 1.5
                                    risk_pts = entry_p - stop_loss
                                    
                                    if 6.0 <= risk_pts <= 25.0:
                                        target_pts = risk_pts * 3.0
                                        take_profit = entry_p + target_pts
                                        
                                        outcome, exit_p = _sim_fib_trade(rth, k, 1, entry_p, stop_loss, take_profit)
                                        gross_pts = exit_p - entry_p
                                        trades.append({
                                            'date': day,
                                            'time': scan.index[k].strftime('%H:%M'),
                                            'direction': 'LONG (Fib 61.8% Bottom)',
                                            'risk_pts': risk_pts,
                                            'outcome': outcome,
                                            'realized_r': gross_pts / risk_pts,
                                            'net_nq': gross_pts * 20.0 - 10.20
                                        })
                                        traded_today = True
                                        break
                            if traded_today: break

    res_df = pd.DataFrame(trades)
    _print_fib_summary(res_df, csv_path)
    return res_df

def _sim_fib_trade(rth, idx, direction, entry, sl, tp):
    for j in range(idx + 1, len(rth)):
        bar = rth.iloc[j]
        if rth.index[j].hour == 15 and rth.index[j].minute >= 45:
            return 'TIME_EXIT', bar['Open']
        if direction == 1:
            if bar['Low'] <= sl: return 'STOP_LOSS', sl
            if bar['High'] >= tp: return 'TARGET_3R', tp
        else:
            if bar['High'] >= sl: return 'STOP_LOSS', sl
            if bar['Low'] <= tp: return 'TARGET_3R', tp
    return 'EOD', rth['Close'].iloc[-1]

def _print_fib_summary(res_df, path):
    if res_df.empty:
        print("No Fib OTE setups triggered.")
        return
    total = len(res_df)
    wins = len(res_df[res_df['outcome'] == 'TARGET_3R'])
    wr = (wins / total) * 100
    avg_risk = res_df['risk_pts'].mean()
    net = res_df['net_nq'].sum()
    print(res_df[['date', 'time', 'direction', 'risk_pts', 'outcome', 'realized_r', 'net_nq']])
    print(f"\nTotal Fib Trades : {total}")
    print(f"Wins (3.0R Target): {wins} ({wr:.1f}%)")
    print(f"Average Stop Size : {avg_risk:.1f} pts (${avg_risk*20:.0f} on NQ)")
    print(f"Total Net P&L     : ${net:+,.2f}")

if __name__ == '__main__':
    run_fib_ote_reversal_study('sample_data/REAL_NQ_1min.csv')
    run_fib_ote_reversal_study('sample_data/MNQ_1min_20260120_20260415.csv')

