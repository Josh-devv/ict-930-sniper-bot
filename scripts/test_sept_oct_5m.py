import pandas as pd
import numpy as np
from datetime import time as dtime

df = pd.read_csv('sample_data/NQ_5min_60days.csv')
df['dt'] = pd.to_datetime(df['dt'])
df['date'] = pd.to_datetime(df['date']).dt.date
df['time'] = df['dt'].dt.time

windows = [
    ('September 2026', '2026-09-01', '2026-09-30'),
    ('October 2026',   '2026-10-01', '2026-10-31'),
    ('Combined (Sep + Oct 2026)', '2026-09-01', '2026-10-31')
]

for month_name, s_d, e_d in windows:
    m_df = df[(df['date'] >= pd.to_datetime(s_d).date()) & (df['date'] <= pd.to_datetime(e_d).date())].copy()
    
    VWAP_MULT = 2.0
    MIN_BAND_PT = 15.0
    WICK_PCT = 0.35
    MIN_STOP = 5.0
    MAX_STOP = 60.0
    MIN_RR = 1.5
    STOP_BUF = 1.5
    trades = []
    
    for day, day_df in m_df.groupby('date'):
        day_df = day_df.reset_index(drop=True)
        cum_vol = 0.0
        cum_pv = 0.0
        cum_pv2 = 0.0
        traded_today = False
        active_trade = None
        
        for _, row in day_df.iterrows():
            t = row['time']
            o, h, l, c, v = row['open'], row['high'], row['low'], row['close'], max(row['volume'], 1.0)
            tp_bar = (h + l + c) / 3.0
            
            cum_vol += v
            cum_pv  += tp_bar * v
            cum_pv2 += tp_bar * tp_bar * v
            vwap = cum_pv / cum_vol
            stdev = np.sqrt(max(0.0, cum_pv2 / cum_vol - vwap**2))
            upper = vwap + VWAP_MULT * stdev
            lower = vwap - VWAP_MULT * stdev
            
            if t >= dtime(15, 45) and active_trade:
                ep = active_trade['entry_p']
                side = active_trade['side']
                pts = (ep - c) if side == 'short' else (c - ep)
                trades.append({**active_trade, 'pts': pts, 'result': 'WIN' if pts > 0 else 'LOSS'})
                active_trade = None
                continue
                
            if active_trade:
                ep = active_trade['entry_p']
                sl = active_trade['stop']
                tp = active_trade['target']
                side = active_trade['side']
                if side == 'short':
                    if l <= tp:
                        trades.append({**active_trade, 'pts': ep - tp, 'result': 'WIN'})
                        active_trade = None
                    elif h >= sl:
                        trades.append({**active_trade, 'pts': ep - sl, 'result': 'LOSS'})
                        active_trade = None
                else:
                    if h >= tp:
                        trades.append({**active_trade, 'pts': tp - ep, 'result': 'WIN'})
                        active_trade = None
                    elif l <= sl:
                        trades.append({**active_trade, 'pts': sl - ep, 'result': 'LOSS'})
                        active_trade = None
                continue
                
            if traded_today or stdev < 0.5:
                continue
            if not (dtime(9, 30) <= t <= dtime(11, 30)):
                continue
            
            crange = h - l
            up_w = h - max(o, c)
            dn_w = min(o, c) - l
            
            if (upper - vwap) >= MIN_BAND_PT and h >= upper and crange >= 6.0 and up_w >= crange * WICK_PCT and c < o:
                ep = (h + max(o, c)) / 2.0
                sl = h + STOP_BUF
                rk = min(max(sl - ep, 1.0), MAX_STOP)
                tp_p = vwap
                rw = ep - tp_p
                if rk >= MIN_STOP and rw >= rk * MIN_RR:
                    active_trade = {
                        'date': str(day), 'time': str(t)[:5], 'side': 'short',
                        'entry_p': ep, 'stop': sl, 'target': tp_p,
                        'risk_pts': rk, 'reward_pts': rw, 'rr': rw/rk
                    }
                    traded_today = True
            elif (vwap - lower) >= MIN_BAND_PT and l <= lower and crange >= 6.0 and dn_w >= crange * WICK_PCT and c > o:
                ep = (l + min(o, c)) / 2.0
                sl = l - STOP_BUF
                rk = min(max(ep - sl, 1.0), MAX_STOP)
                tp_p = vwap
                rw = tp_p - ep
                if rk >= MIN_STOP and rw >= rk * MIN_RR:
                    active_trade = {
                        'date': str(day), 'time': str(t)[:5], 'side': 'long',
                        'entry_p': ep, 'stop': sl, 'target': tp_p,
                        'risk_pts': rk, 'reward_pts': rw, 'rr': rw/rk
                    }
                    traded_today = True

    tdf = pd.DataFrame(trades)
    if len(tdf) > 0:
        wins = tdf[tdf['result'] == 'WIN']
        losses = tdf[tdf['result'] == 'LOSS']
        wr = len(wins) / len(tdf) * 100
        pnl_5mnq = (tdf['pts'] * 10.0 - 6.20).sum()
        pnl_1nq  = (tdf['pts'] * 20.0 - 10.20).sum()
        cum = (tdf['pts'] * 10.0 - 6.20).cumsum()
        dd_5mnq = (cum - cum.cummax()).min()
        avg_stop = tdf['risk_pts'].mean()
        avg_r = wins['rr'].mean() if len(wins) > 0 else 0
        print(f"\n========================================================")
        print(f"  {month_name.upper()}")
        print(f"========================================================")
        print(f"  Total Trades        : {len(tdf)}")
        print(f"  Win Rate            : {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
        print(f"  Avg Stop Loss       : {avg_stop:.1f} pts (~${avg_stop*10:.2f} on 5 MNQ)")
        print(f"  Avg Winner R:R      : {avg_r:.2f}R")
        print(f"  Net Profit (5 MNQ)  : +${pnl_5mnq:.2f}")
        print(f"  Max Drawdown (5 MNQ): -${abs(dd_5mnq):.2f}")
        print(f"  Net Profit (1 NQ)   : +${pnl_1nq:.2f}")
        print(f"  Max Drawdown (1 NQ) : -${abs(dd_5mnq)*2:.2f}")
    else:
        print(f"\n{month_name}: 0 trades")

