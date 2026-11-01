"""
Automated Predictive Feature Extraction for 10:00 AM Direction
===============================================================
We study every single trading day and extract:
1. Overnight / Asia range & direction (00:00 - 08:30)
2. London session expansion / trend (02:00 - 06:00)
3. 08:30 premarket news / swing
4. 09:30 - 10:00 cash open characteristics:
   - Swept London High? (Buyside Sweep)
   - Swept London Low?  (Sellside Sweep)
   - Swept PDH / PDL?
   - Open vs Close of 9:30 candle (Body direction & magnitude)
   - Upper Wick ratio vs Lower Wick ratio
   - Displacement speed (bars to high vs bars to low)
   - Relative volume
   - True Day Open (midnight) position (above or below TDO)
5. Target: What did 10:00 - 11:00 AM actually do?
   - Net move in pts
   - Bullish Expansion (> +20 pts) vs Bearish Expansion (< -20 pts) vs Chop
"""

import sys
import pandas as pd
import numpy as np

def build_predictive_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    date_col = 'datetime' if 'datetime' in df.columns else 'Datetime'
    df[date_col] = pd.to_datetime(df[date_col])
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_convert('America/New_York')
    else:
        df[date_col] = df[date_col].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        
    df.set_index(date_col, inplace=True)
    df.columns = [c.capitalize() for c in df.columns]
    
    # Calculate daily PDH and PDL
    daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    df['PDH'] = daily['High'].shift(1).reindex(df.index, method='ffill')
    df['PDL'] = daily['Low'].shift(1).reindex(df.index, method='ffill')
    
    unique_days = sorted(list(set(df.index.date)))
    samples = []
    
    for day in unique_days:
        df_day = df[df.index.date == day]
        if len(df_day) < 60:
            continue
            
        # 1. Midnight Open (True Day Open)
        midnight = df_day.between_time('00:00', '00:05')
        tdo = midnight['Open'].iloc[0] if not midnight.empty else df_day['Open'].iloc[0]
        
        # 2. London Session (02:00 - 06:00)
        london = df_day.between_time('02:00', '06:00')
        if len(london) < 10:
            continue
        l_high = london['High'].max()
        l_low  = london['Low'].min()
        l_open = london['Open'].iloc[0]
        l_close= london['Close'].iloc[-1]
        l_trend = 1 if l_close > l_open else -1
        
        pdh = df_day['PDH'].iloc[0]
        pdl = df_day['PDL'].iloc[0]
        key_h = max(l_high, pdh) if pd.notna(pdh) else l_high
        key_l = min(l_low, pdl)   if pd.notna(pdl) else l_low
        
        # 3. 09:30 - 09:59 Open Window
        open_930 = df_day.between_time('09:30', '09:59')
        if len(open_930) < 15:
            continue
            
        o_930 = open_930['Open'].iloc[0]
        h_930 = open_930['High'].max()
        l_930 = open_930['Low'].min()
        c_930 = open_930['Close'].iloc[-1]
        
        # Sweeps
        swept_high = 1 if h_930 > key_h else 0
        swept_low  = 1 if l_930 < key_l else 0
        
        # Position relative to TDO
        above_tdo = 1 if c_930 > tdo else 0
        
        # Candle structure of 9:30-10:00
        body = c_930 - o_930
        upper_wick = h_930 - max(o_930, c_930)
        lower_wick = min(o_930, c_930) - l_930
        range_930 = h_930 - l_930
        
        # 4. Target: 10:00 - 11:00 AM (The Expansion Window)
        rth_10am = df_day.between_time('10:00', '11:00')
        if len(rth_10am) < 15:
            continue
            
        o_10am = rth_10am['Open'].iloc[0]
        c_11am = rth_10am['Close'].iloc[-1]
        h_10am = rth_10am['High'].max()
        l_10am = rth_10am['Low'].min()
        
        net_move_10am = c_11am - o_10am
        max_bull_run  = h_10am - o_10am
        max_bear_run  = o_10am - l_10am
        
        # Binary target: 1 = Bullish Expansion, -1 = Bearish Expansion, 0 = Chop
        if max_bull_run >= 25.0 and max_bull_run > max_bear_run * 1.3:
            outcome = 1
        elif max_bear_run >= 25.0 and max_bear_run > max_bull_run * 1.3:
            outcome = -1
        else:
            outcome = 0
            
        samples.append({
            'date': day,
            'l_trend': l_trend,
            'above_tdo': above_tdo,
            'swept_high': swept_high,
            'swept_low': swept_low,
            'body_930': body,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'range_930': range_930,
            'net_move_10am': net_move_10am,
            'max_bull_run': max_bull_run,
            'max_bear_run': max_bear_run,
            'outcome': outcome
        })
        
    res_df = pd.DataFrame(samples)
    print(f"Dataset Built: {len(res_df)} trading days processed.")
    return res_df

if __name__ == '__main__':
    df_samples = build_predictive_dataset('sample_data/MNQ_1min_20260120_20260415.csv')
    print("\nTarget Distribution:")
    print(df_samples['outcome'].value_counts())
