"""
Empirical Rhythm Analysis across Multi-Month Dataset
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np

def analyze_rhythm(csv_path: str):
    df = pd.read_csv(csv_path)
    date_col = 'datetime' if 'datetime' in df.columns else 'Datetime'
    df[date_col] = pd.to_datetime(df[date_col])
    # Ensure Eastern timezone
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_convert('America/New_York')
    else:
        df[date_col] = df[date_col].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        
    df.set_index(date_col, inplace=True)
    df.columns = [c.capitalize() for c in df.columns]
    
    unique_days = sorted(list(set(df.index.date)))
    print(f"\n{'='*70}")
    print(f"  ANALYZING 09:30 - 10:00 RHYTHM ON: {csv_path}")
    print(f"  Days analyzed: {len(unique_days)}")
    print(f"{'='*70}")
    
    reversals = 0
    continuations = 0
    chops = 0
    
    reversal_points = []
    continuation_points = []
    
    for day in unique_days:
        df_day = df[df.index.date == day]
        leg_930 = df_day.between_time('09:30', '09:59')
        leg_1000 = df_day.between_time('10:00', '10:30')
        
        if len(leg_930) < 15 or len(leg_1000) < 15:
            continue
            
        o_930 = leg_930['Open'].iloc[0]
        c_959 = leg_930['Close'].iloc[-1]
        move_930 = c_959 - o_930
        
        o_1000 = leg_1000['Open'].iloc[0]
        c_1030 = leg_1000['Close'].iloc[-1]
        move_1000 = c_1030 - o_1000
        
        if abs(move_930) < 10.0:
            chops += 1
        elif (move_930 > 0 and move_1000 < -5.0) or (move_930 < 0 and move_1000 > 5.0):
            reversals += 1
            reversal_points.append(abs(move_1000))
        elif (move_930 > 0 and move_1000 > 5.0) or (move_930 < 0 and move_1000 < -5.0):
            continuations += 1
            continuation_points.append(abs(move_1000))
        else:
            chops += 1
            
    total_valid = reversals + continuations + chops
    print(f"Total Trading Days Analyzed       : {total_valid}")
    print(f"10:00 AM Reversals (Judas Fade)   : {reversals} ({reversals/total_valid*100:.1f}%)")
    print(f"10:00 AM Continuations (Trend Run): {continuations} ({continuations/total_valid*100:.1f}%)")
    print(f"Chop / Indecision (<10 pts)       : {chops} ({chops/total_valid*100:.1f}%)")
    if reversal_points:
        print(f"Avg Move on Reversals             : {np.mean(reversal_points):.1f} pts")
    if continuation_points:
        print(f"Avg Move on Continuations         : {np.mean(continuation_points):.1f} pts")
    print(f"-------------------------------------------------------------")

if __name__ == '__main__':
    analyze_rhythm('sample_data/REAL_NQ_1min.csv')
    analyze_rhythm('sample_data/MNQ_1min_20260120_20260415.csv')
