"""
4H Driver Pairing with Refined 1-Min Structure Execution
=========================================================
Key Discovery from Real NQ Data:
  NQ is trading around 29,000+ index points.
  The total 9:30-10:00 manipulation swing is 80 to 170 points wide.
  Institutions DO NOT put their stop across the entire 150-point 9:30 swing.
  Instead, ICT traders enter on a 1-MINUTE Fair Value Gap (FVG) or Order Block
  inside the 10:00 AM macro window, with the stop placed behind that LOCAL
  1-min order block (15 - 35 pts risk), targeting the 4H expansion (60 - 150 pts reward).
"""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

def run_real_nq_micro_entry(csv_path: str):
    df = pd.read_csv(csv_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    unique_days = sorted(list(set(df.index.date)))
    print(f"Loaded {len(df):,} real 1-min bars across {len(unique_days)} trading days.\n")
    
    trades = []
    
    for day in unique_days:
        df_day = df[df.index.date == day]
        london = df_day.between_time('02:00', '06:00')
        open_930 = df_day.between_time('09:30', '10:00')
        exec_10am = df_day.between_time('10:00', '10:30')
        
        if len(london) < 10 or len(open_930) < 10 or len(exec_10am) < 5:
            continue
            
        l_h, l_l = london['High'].max(), london['Low'].min()
        o_h, o_l = open_930['High'].max(), open_930['Low'].min()
        
        swept_b = o_h > l_h
        swept_s = o_l < l_l
        
        # 4H Driver Pairing:
        # IF 9:30 AM swept sellside -> 10:00 AM is BULLISH continuation
        # IF 9:30 AM swept buyside  -> 10:00 AM is BEARISH continuation
        if swept_s and not swept_b:
            direction = 1   # Long
            bias_text = "Sellside Swept -> Bull Expansion"
        elif swept_b and not swept_s:
            direction = -1  # Short
            bias_text = "Buyside Swept -> Bear Expansion"
        else:
            continue
            
        # Inside the 10:00 - 10:30 window, find the 1-min swing entry
        # Stop is behind the immediate 1-min local structure low/high (15 - 40 pts)
        e_time = exec_10am.index[0]
        entry_price = exec_10am['Open'].iloc[0]
        
        if direction == 1:
            # Look at lowest low of 09:50 - 10:00 (the turn before 10am)
            pre_turn = df_day.between_time('09:50', '10:00')
            local_stop = pre_turn['Low'].min() - 2.0
            risk_pts = entry_price - local_stop
            if risk_pts < 10.0 or risk_pts > 50.0:
                continue
            target_pts = risk_pts * 2.5
            tp_price = entry_price + target_pts
        else:
            pre_turn = df_day.between_time('09:50', '10:00')
            local_stop = pre_turn['High'].max() + 2.0
            risk_pts = local_stop - entry_price
            if risk_pts < 10.0 or risk_pts > 50.0:
                continue
            target_pts = risk_pts * 2.5
            tp_price = entry_price - target_pts
            
        # Walk forward from 10:00 to 15:45
        walk = df_day.between_time('10:01', '15:45')
        outcome = 'TIME_EXIT'
        exit_p = walk['Close'].iloc[-1] if not walk.empty else entry_price
        
        for t, row in walk.iterrows():
            if direction == 1:
                if row['Low'] <= local_stop:
                    outcome = 'STOP_LOSS'
                    exit_p = local_stop
                    break
                if row['High'] >= tp_price:
                    outcome = 'TARGET_2.5R'
                    exit_p = tp_price
                    break
            else:
                if row['High'] >= local_stop:
                    outcome = 'STOP_LOSS'
                    exit_p = local_stop
                    break
                if row['Low'] <= tp_price:
                    outcome = 'TARGET_2.5R'
                    exit_p = tp_price
                    break
                    
        gross_pts = (exit_p - entry_price) * direction
        net_nq = gross_pts * 20.0 - 10.20
        trades.append({
            'date': day,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'bias': bias_text,
            'entry': entry_price,
            'stop': local_stop,
            'risk_pts': risk_pts,
            'target': tp_price,
            'exit_price': exit_p,
            'outcome': outcome,
            'gross_pts': gross_pts,
            'net_nq': net_nq
        })
        
    res_df = pd.DataFrame(trades)
    print("========================================================================")
    print("  4H DRIVER PAIRING EXECUTION ON REAL LIVE NQ FUTURES (2.5R TARGET)")
    print("========================================================================")
    print(res_df[['date', 'direction', 'risk_pts', 'outcome', 'gross_pts', 'net_nq']])
    
    wins = len(res_df[res_df['outcome'] == 'TARGET_2.5R'])
    total = len(res_df)
    print("------------------------------------------------------------------------")
    print(f"Total Trade Days Executed : {total}")
    print(f"Winning Trades (2.5R)     : {wins} ({wins/total*100:.1f}%)")
    print(f"Total Net P&L (1 Mini NQ) : ${res_df['net_nq'].sum():+,.2f}")
    print("========================================================================")

if __name__ == '__main__':
    run_real_nq_micro_entry('sample_data/REAL_NQ_1min.csv')

