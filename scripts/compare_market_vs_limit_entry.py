import json, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

DATA_PATH = r'c:\Users\JOSHUA\Desktop\CRTbot\data\indices\NQ-duka-2018-2026.json'

print("Loading recent data...")
with open(DATA_PATH, 'r') as f:
    raw = json.load(f)

df_raw = pd.DataFrame(raw)
df_raw['dt'] = pd.to_datetime(df_raw['time'], unit='ms', utc=True)
df_raw = df_raw.set_index('dt').drop(columns=['time'])
df_raw.columns = ['open', 'high', 'low', 'close', 'volume']
df_raw.index = df_raw.index.tz_convert('America/New_York')
df_raw = df_raw[df_raw.index.dayofweek < 5]

# Test on 2023 - 2026 (last 3+ years)
df_raw = df_raw.loc['2023-01-01':]

# Resample to 5m
df_5m = df_raw.resample('5min', label='left', closed='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()

SIGMA = 2.0
MIN_BAND_DIST = 15.0
WICK_RATIO = 0.35
STOP_BEYOND = 1.5

def test_entry_mode(use_direct_market_entry=False):
    trades = []
    for date, day in df_5m.groupby(df_5m.index.date):
        if len(day) < 10:
            continue
        day_session = day.between_time('09:30', '16:00')
        if len(day_session) < 5:
            continue
            
        tp = (day['high'] + day['low'] + day['close']) / 3.0
        cum_vol = day['volume'].cumsum()
        cum_tpv = (tp * day['volume']).cumsum()
        vwap = cum_tpv / cum_vol
        cum_tp2v = ((tp ** 2) * day['volume']).cumsum()
        variance = (cum_tp2v / cum_vol) - (vwap ** 2)
        variance = variance.clip(lower=0)
        std_dev = np.sqrt(variance)
        upper = vwap + SIGMA * std_dev
        lower = vwap - SIGMA * std_dev

        signal_window = day.between_time('09:30', '11:29')
        trade_taken = False
        
        for ts in signal_window.index:
            if trade_taken:
                break
            idx = day.index.get_loc(ts)
            o, h, l, c = day['open'].iloc[idx], day['high'].iloc[idx], day['low'].iloc[idx], day['close'].iloc[idx]
            rng = h - l
            if rng < 0.25:
                continue
            vw, ub, lb = vwap.iloc[idx], upper.iloc[idx], lower.iloc[idx]
            
            direction = None
            if h >= ub and c < ub and (ub - vw) >= MIN_BAND_DIST:
                wick = h - c
                if wick / rng >= WICK_RATIO:
                    direction = 'short'
                    entry_p = c if use_direct_market_entry else (h - 0.50 * wick)
                    stop_p = h + STOP_BEYOND
                    tp1 = vw
                    tp2 = lb
            elif l <= lb and c > lb and (vw - lb) >= MIN_BAND_DIST:
                wick = c - l
                if wick / rng >= WICK_RATIO:
                    direction = 'long'
                    entry_p = c if use_direct_market_entry else (l + 0.50 * wick)
                    stop_p = l - STOP_BEYOND
                    tp1 = vw
                    tp2 = ub
                    
            if direction is None:
                continue
                
            future_bars = day.iloc[idx + 1:]
            if len(future_bars) == 0:
                continue
                
            # If limit entry, check fill
            fill_idx = 0
            if not use_direct_market_entry:
                filled = False
                for j, (_, fbar) in enumerate(future_bars.iterrows()):
                    if direction == 'short' and fbar['high'] >= entry_p:
                        filled = True
                        fill_idx = j
                        break
                    elif direction == 'long' and fbar['low'] <= entry_p:
                        filled = True
                        fill_idx = j
                        break
                if not filled:
                    continue
                    
            trade_taken = True
            
            # Simulate trade outcome
            post_fill = future_bars.iloc[fill_idx:]
            pnl_pts = 0
            tp1_hit = False
            hit_stop = False
            
            for _, fbar in post_fill.iterrows():
                fh, fl = fbar['high'], fbar['low']
                if direction == 'long':
                    if fl <= stop_p:
                        pnl_pts = -(entry_p - stop_p)
                        hit_stop = True
                        break
                    if not tp1_hit and fh >= tp1:
                        tp1_hit = True
                        stop_p = entry_p + 1.0 # BE+1
                    if tp1_hit and fh >= tp2:
                        # 60% at TP1, 40% at TP2
                        pnl_pts = 0.60 * (tp1 - entry_p) + 0.40 * (tp2 - entry_p)
                        break
                else:
                    if fh >= stop_p:
                        pnl_pts = -(stop_p - entry_p)
                        hit_stop = True
                        break
                    if not tp1_hit and fl <= tp1:
                        tp1_hit = True
                        stop_p = entry_p - 1.0 # BE+1
                    if tp1_hit and fl <= tp2:
                        pnl_pts = 0.60 * (entry_p - tp1) + 0.40 * (entry_p - tp2)
                        break
            
            if not hit_stop and pnl_pts == 0:
                # EOD close
                last_c = post_fill['close'].iloc[-1]
                if tp1_hit:
                    pnl_pts = 0.60 * (abs(tp1 - entry_p)) + 0.40 * (last_c - entry_p if direction=='long' else entry_p - last_c)
                else:
                    pnl_pts = (last_c - entry_p) if direction == 'long' else (entry_p - last_c)
                    
            trades.append(pnl_pts)
            
    df_res = pd.Series(trades)
    win_rate = (df_res > 0).mean() * 100
    total_pts = df_res.sum()
    total_trades = len(df_res)
    # 5 MNQ contracts = $10/pt
    total_dollar = total_pts * 10.0
    return total_trades, win_rate, total_pts, total_dollar

t_limit, wr_limit, pts_limit, d_limit = test_entry_mode(use_direct_market_entry=False)
t_market, wr_market, pts_market, d_market = test_entry_mode(use_direct_market_entry=True)

print("="*60)
print(f"50% WICK LIMIT ENTRY:   {t_limit} trades | {wr_limit:.1f}% WR | +{pts_limit:,.1f} pts | +${d_limit:,.2f}")
print(f"DIRECT MARKET AT CLOSE: {t_market} trades | {wr_market:.1f}% WR | +{pts_market:,.1f} pts | +${d_market:,.2f}")
print("="*60)

