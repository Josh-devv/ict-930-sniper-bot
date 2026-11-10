"""
HEAD-TO-HEAD: 1-MINUTE vs 3-MINUTE vs 5-MINUTE TIMEFRAME COMPARISON
Dataset: CME MNQ Futures (Feb 1 - Mar 31, 2026) — 100% unseen out-of-sample
Position: 5 MNQ contracts ($10/pt total)
Evaluates: Trade frequency, win rate, avg stop size, dollar risk, profit factor, net profit, max DD
"""

import pandas as pd
import numpy as np
from datetime import time as dtime
import warnings
warnings.filterwarnings("ignore")

MNQ_PV = 10.0   # 5 MNQ = $10/pt
COMM   = 6.20   # $1.24 * 5 contracts round-turn

def load_data():
    df = pd.read_csv("sample_data/MNQ_1min_20260120_20260415.csv")
    df['dt'] = pd.to_datetime(df['datetime'])
    if df['dt'].dt.tz is None:
        df['dt'] = df['dt'].dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
    df = df.sort_values('dt').set_index('dt')
    for col in ['open','high','low','close','volume']:
        found = [c for c in df.columns if c.lower() == col]
        if found:
            df[col] = df[found[0]].astype(float)
    return df[['open','high','low','close','volume']]

def run_tf(df_1m, tf_mins, start, end):
    rule = f"{tf_mins}min"
    resampled = df_1m.resample(rule, closed='left', label='left').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna().reset_index()

    resampled['date']   = resampled['dt'].dt.date
    resampled['time']   = resampled['dt'].dt.time
    resampled['hour']   = resampled['dt'].dt.hour
    resampled['minute'] = resampled['dt'].dt.minute

    df = resampled[
        (resampled['hour'] >= 9) &
        ~((resampled['hour'] == 9) & (resampled['minute'] < 30)) &
        (resampled['hour'] < 16)
    ].copy()
    s, e = pd.to_datetime(start).date(), pd.to_datetime(end).date()
    df = df[(df['date'] >= s) & (df['date'] <= e)].reset_index(drop=True)

    VWAP_MULT   = 2.0
    MIN_BAND_PT = 15.0
    WICK_PCT    = 0.35
    MIN_STOP    = 5.0
    MAX_STOP    = 60.0
    MIN_RR      = 1.5
    STOP_BUF    = 1.5

    trades = []

    for day, ddf in df.groupby('date'):
        ddf = ddf.reset_index(drop=True)
        cv = cp = cp2 = 0.0
        traded = False
        trade = None

        for _, r in ddf.iterrows():
            t = r['time']
            o,h,l,c,v = r['open'],r['high'],r['low'],r['close'],max(r['volume'],1.0)
            tp = (h+l+c)/3.0
            cv += v; cp += tp*v; cp2 += tp*tp*v
            vw = cp/cv
            sd = np.sqrt(max(0.0, cp2/cv - vw*vw))
            ub = vw + VWAP_MULT*sd
            lb = vw - VWAP_MULT*sd

            # EOD
            if t >= dtime(15,45) and trade:
                pts = (trade['ep']-c) if trade['side']=='short' else (c-trade['ep'])
                trade['exit']=c; trade['pts']=pts
                trade['pnl']=pts*MNQ_PV-COMM
                trade['result']='WIN' if pts>0 else 'LOSS'
                trades.append(trade); trade=None; continue

            # Exit management
            if trade:
                ep,sl,tp1 = trade['ep'],trade['sl'],trade['tp1']
                side = trade['side']
                if side=='short':
                    if l<=tp1:
                        pts=ep-tp1; trade.update({'exit':tp1,'pts':pts,'pnl':pts*MNQ_PV-COMM,'result':'WIN'}); trades.append(trade); trade=None
                    elif h>=sl:
                        pts=ep-sl; trade.update({'exit':sl,'pts':pts,'pnl':pts*MNQ_PV-COMM,'result':'LOSS'}); trades.append(trade); trade=None
                else:
                    if h>=tp1:
                        pts=tp1-ep; trade.update({'exit':tp1,'pts':pts,'pnl':pts*MNQ_PV-COMM,'result':'WIN'}); trades.append(trade); trade=None
                    elif l<=sl:
                        pts=sl-ep; trade.update({'exit':sl,'pts':pts,'pnl':pts*MNQ_PV-COMM,'result':'LOSS'}); trades.append(trade); trade=None
                continue

            # Signal scan
            if traded or sd<0.5: continue
            if not (dtime(9,30) <= t <= dtime(11,30)): continue

            cr=h-l; uw=h-max(o,c); dw=min(o,c)-l

            if (ub-vw)>=MIN_BAND_PT and h>=ub and cr>=6.0 and uw>=cr*WICK_PCT and c<o:
                ep=(h+max(o,c))/2.0; sl=h+STOP_BUF
                rk=min(max(sl-ep,1.0),MAX_STOP); rw=ep-vw
                if rk>=MIN_STOP and rw>=rk*MIN_RR:
                    trade={'date':str(day),'side':'short','ep':ep,'sl':sl,'tp1':vw,'risk_pts':rk,'reward_pts':rw}
                    traded=True

            elif (vw-lb)>=MIN_BAND_PT and l<=lb and cr>=6.0 and dw>=cr*WICK_PCT and c>o:
                ep=(l+min(o,c))/2.0; sl=l-STOP_BUF
                rk=min(max(ep-sl,1.0),MAX_STOP); rw=vw-ep
                if rk>=MIN_STOP and rw>=rk*MIN_RR:
                    trade={'date':str(day),'side':'long','ep':ep,'sl':sl,'tp1':vw,'risk_pts':rk,'reward_pts':rw}
                    traded=True

    return pd.DataFrame(trades)

def stats(tdf):
    if len(tdf)==0: return {}
    wins=tdf[tdf['result']=='WIN']; losses=tdf[tdf['result']=='LOSS']
    gp=wins['pnl'].sum(); gl=abs(losses['pnl'].sum()) if len(losses)>0 else 1.0
    eq=tdf['pnl'].cumsum(); dd=(eq-eq.cummax()).min()
    return {
        'trades': len(tdf),
        'wins': len(wins),
        'losses': len(losses),
        'wr': len(wins)/len(tdf)*100,
        'avg_stop': tdf['risk_pts'].mean(),
        'avg_risk_$': tdf['risk_pts'].mean()*MNQ_PV,
        'avg_rr': wins['reward_pts'].mean()/wins['risk_pts'].mean() if len(wins)>0 else 0,
        'pf': gp/gl if gl>0 else 999.0,
        'net_pnl': tdf['pnl'].sum(),
        'max_dd': dd
    }

# ── Run ────────────────────────────────────────────────────────────────────────
print("Loading CME data (Feb–Mar 2026)...")
df = load_data()

periods = [
    ("February 2026", "2026-02-01", "2026-02-28"),
    ("March 2026",    "2026-03-01", "2026-03-31"),
    ("Feb + Mar 2026","2026-02-01", "2026-03-31")
]

for period_name, s, e in periods:
    t1m = run_tf(df, 1, s, e)
    t3m = run_tf(df, 3, s, e)
    t5m = run_tf(df, 5, s, e)

    s1=stats(t1m); s3=stats(t3m); s5=stats(t5m)

    print()
    print("=" * 90)
    print(f"  {period_name.upper()} — 1-MINUTE vs 3-MINUTE vs 5-MINUTE (5 MNQ @ $10/pt)")
    print("=" * 90)
    print(f"  {'Metric':<30} | {'1-Minute':>18} | {'3-Minute':>18} | {'5-Minute':>18}")
    print("-" * 90)
    print(f"  {'Total Trades':<30} | {s1.get('trades',0):>18} | {s3.get('trades',0):>18} | {s5.get('trades',0):>18}")
    print(f"  {'Win Rate':<30} | {s1.get('wr',0):>17.1f}% | {s3.get('wr',0):>17.1f}% | {s5.get('wr',0):>17.1f}%")
    print(f"  {'Wins / Losses':<30} | {str(s1.get('wins',0))+'W / '+str(s1.get('losses',0))+'L':>18} | {str(s3.get('wins',0))+'W / '+str(s3.get('losses',0))+'L':>18} | {str(s5.get('wins',0))+'W / '+str(s5.get('losses',0))+'L':>18}")
    print(f"  {'Avg Stop Loss (pts)':<30} | {s1.get('avg_stop',0):>17.1f}  | {s3.get('avg_stop',0):>17.1f}  | {s5.get('avg_stop',0):>17.1f} ")
    print(f"  {'Avg Dollar Risk (5 MNQ)':<30} | ${s1.get('avg_risk_$',0):>16.2f} | ${s3.get('avg_risk_$',0):>16.2f} | ${s5.get('avg_risk_$',0):>16.2f}")
    print(f"  {'Avg Winner R:R':<30} | {s1.get('avg_rr',0):>17.2f}R | {s3.get('avg_rr',0):>17.2f}R | {s5.get('avg_rr',0):>17.2f}R")
    print(f"  {'Profit Factor':<30} | {s1.get('pf',0):>18.2f} | {s3.get('pf',0):>18.2f} | {s5.get('pf',0):>18.2f}")
    print("-" * 90)
    print(f"  {'NET PROFIT (5 MNQ)':<30} | +${s1.get('net_pnl',0):>15.2f} | +${s3.get('net_pnl',0):>15.2f} | +${s5.get('net_pnl',0):>15.2f}")
    print(f"  {'MAX DRAWDOWN (5 MNQ)':<30} | -${abs(s1.get('max_dd',0)):>15.2f} | -${abs(s3.get('max_dd',0)):>15.2f} | -${abs(s5.get('max_dd',0)):>15.2f}")
    print()

