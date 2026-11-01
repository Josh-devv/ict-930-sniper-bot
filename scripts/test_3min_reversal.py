"""
Multi-Timeframe Test: 3-Minute Chart Reversal Performance
=========================================================
Does the VWAP 2.2 Band + 50% Wick CE Reversal work on the 3-minute chart?
Let's resample our 1-minute data into 3-minute candles and run the exact same backtest!
"""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

def run_3min_test(csv_path: str):
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
    trades = []
    
    for day in unique_days:
        df_day = df[df.index.date == day].copy()
        rth_1m = df_day.between_time('09:30', '16:00').copy()
        if len(rth_1m) < 30:
            continue
            
        # Resample to 3-MINUTE Candles!
        rth_3m = rth_1m.resample('3min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # Calculate Anchored Session VWAP on 3-minute bars
        tp = (rth_3m['High'] + rth_3m['Low'] + rth_3m['Close']) / 3.0
        vol = rth_3m['Volume']
        cum_vol = vol.cumsum()
        cum_tp_vol = (tp * vol).cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, 1)
        variance = ((tp - vwap) ** 2 * vol).cumsum() / cum_vol.replace(0, 1)
        stdev = np.sqrt(variance.replace(0, 0.001))
        
        vwap_u2 = vwap + (stdev * 2.2)
        vwap_l2 = vwap - (stdev * 2.2)
        
        scan_3m = rth_3m.between_time('09:35', '11:30')
        traded_today = False
        
        for i in range(2, len(scan_3m) - 2):
            if traded_today: break
            bar = scan_3m.iloc[i]
            b_time = scan_3m.index[i]
            
            # --- 3-MIN TOP REVERSAL ---
            if bar['High'] >= vwap_u2.loc[b_time]:
                c_range = bar['High'] - bar['Low']
                upper_wick = bar['High'] - max(bar['Open'], bar['Close'])
                # On 3m, wick >= 40% of range and body bearish
                if c_range >= 10.0 and upper_wick >= c_range * 0.40 and bar['Close'] < bar['Open']:
                    entry_p = (bar['High'] + max(bar['Open'], bar['Close'])) / 2.0
                    stop_loss = bar['High'] + 2.0
                    risk_pts = stop_loss - entry_p
                    take_profit = vwap.loc[b_time]
                    
                    if 8.0 <= risk_pts <= 35.0 and (entry_p - take_profit) >= risk_pts * 1.8:
                        outcome, exit_p = _sim_3m_trade(rth_3m, i, -1, entry_p, stop_loss, take_profit)
                        gross = entry_p - exit_p
                        trades.append({
                            'date': day, 'time': b_time.strftime('%H:%M'),
                            'type': '3M SHORT (Top Reversal)',
                            'risk_pts': risk_pts, 'outcome': outcome,
                            'realized_r': gross / risk_pts,
                            'net_nq': gross * 20.0 - 10.20
                        })
                        traded_today = True
                        break
                        
            # --- 3-MIN BOTTOM REVERSAL ---
            if bar['Low'] <= vwap_l2.loc[b_time]:
                c_range = bar['High'] - bar['Low']
                lower_wick = min(bar['Open'], bar['Close']) - bar['Low']
                if c_range >= 10.0 and lower_wick >= c_range * 0.40 and bar['Close'] > bar['Open']:
                    entry_p = (bar['Low'] + min(bar['Open'], bar['Close'])) / 2.0
                    stop_loss = bar['Low'] - 2.0
                    risk_pts = entry_p - stop_loss
                    take_profit = vwap.loc[b_time]
                    
                    if 8.0 <= risk_pts <= 35.0 and (take_profit - entry_p) >= risk_pts * 1.8:
                        outcome, exit_p = _sim_3m_trade(rth_3m, i, 1, entry_p, stop_loss, take_profit)
                        gross = exit_p - entry_p
                        trades.append({
                            'date': day, 'time': b_time.strftime('%H:%M'),
                            'type': '3M LONG (Bottom Reversal)',
                            'risk_pts': risk_pts, 'outcome': outcome,
                            'realized_r': gross / risk_pts,
                            'net_nq': gross * 20.0 - 10.20
                        })
                        traded_today = True
                        break

    res_df = pd.DataFrame(trades)
    print(f"\n{'='*70}")
    print(f"  3-MINUTE CHART BACKTEST RESULTS: {csv_path}")
    print(f"{'='*70}")
    if res_df.empty:
        print("No trades triggered on 3m.")
        return
        
    total = len(res_df)
    wins = len(res_df[res_df['outcome'] == 'TARGET'])
    wr = (wins / total) * 100
    net = res_df['net_nq'].sum()
    pf = (res_df[res_df['net_nq'] > 0]['net_nq'].sum() / abs(res_df[res_df['net_nq'] <= 0]['net_nq'].sum()))
    print(res_df[['date', 'time', 'type', 'risk_pts', 'outcome', 'realized_r', 'net_nq']].tail(10))
    print(f"\nTotal 3-Min Trades : {total}")
    print(f"Win Rate           : {wins} ({wr:.1f}%)")
    print(f"Profit Factor      : {pf:.2f}")
    print(f"Net Profit (1 NQ)  : ${net:+,.2f}")
    print(f"Avg Realized R     : {res_df['realized_r'].mean():+.2f}R")

def _sim_3m_trade(rth_3m, idx, direction, entry, sl, tp):
    for j in range(idx + 1, len(rth_3m)):
        bar = rth_3m.iloc[j]
        if rth_3m.index[j].hour == 15 and rth_3m.index[j].minute >= 45:
            return 'TIME_EXIT', bar['Open']
        if direction == 1:
            if bar['Low'] <= sl: return 'STOP_LOSS', sl
            if bar['High'] >= tp: return 'TARGET', tp
        else:
            if bar['High'] >= sl: return 'STOP_LOSS', sl
            if bar['Low'] <= tp: return 'TARGET', tp
    return 'EOD', rth_3m['Close'].iloc[-1]

if __name__ == '__main__':
    run_3min_test('sample_data/REAL_NQ_1min.csv')
    run_3min_test('sample_data/MNQ_1min_20260120_20260415.csv')

