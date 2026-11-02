"""
August 2026 Day-Swing Test (Aug 24 – Aug 31)
"""
import yfinance as yf
import pandas as pd
from datetime import timedelta, time as dtime

POINT_VALUE = 20.0
COMMISSION  = 10.20

t = yf.Ticker('NQ=F')
df = t.history(period='60d', interval='5m', prepost=True).reset_index()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0].lower() for c in df.columns]
else:
    df.columns = [c.lower() for c in df.columns]

dt_col = [c for c in df.columns if 'date' in c or 'time' in c or 'index' in c][0]
df['dt'] = pd.to_datetime(df[dt_col]).dt.tz_convert('America/New_York')
df['session_date'] = df['dt'].apply(lambda x: (x + timedelta(hours=6)).date() if x.hour >= 18 else x.date())
df['time'] = df['dt'].dt.time
df['hour'] = df['dt'].dt.hour
df['minute'] = df['dt'].dt.minute

aug_df = df[df['dt'].dt.month == 8].copy()

trades = []
for sess_date, s_df in aug_df.groupby("session_date"):
    s_df = s_df.reset_index(drop=True)
    if len(s_df) < 25: continue

    london = s_df[(s_df["time"] >= dtime(2, 0)) & (s_df["time"] <= dtime(5, 0))]
    if london.empty: continue
    london_h = london["high"].max()
    london_l = london["low"].min()

    active_trade = None
    traded_today = False

    for i in range(len(s_df)):
        row = s_df.iloc[i]
        t = row["time"]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        if active_trade:
            ep = active_trade["entry_p"]
            sl = active_trade["stop"]
            tp = active_trade["target"]
            side = active_trade["side"]
            risk_pts = active_trade["risk_pts"]
            bars_held = i - active_trade["entry_bar"]

            if not active_trade["be_active"]:
                if side == "short" and (ep - l) >= risk_pts * 1.5:
                    active_trade["stop"] = ep - 1.0
                    active_trade["be_active"] = True
                    sl = active_trade["stop"]
                elif side == "long" and (h - ep) >= risk_pts * 1.5:
                    active_trade["stop"] = ep + 1.0
                    active_trade["be_active"] = True
                    sl = active_trade["stop"]

            is_time_exit = bars_held >= 48 or (row["hour"] == 16 and row["minute"] >= 0)

            if side == "short":
                if l <= tp:
                    pts = ep - tp
                    trades.append({**active_trade, "exit_price": tp, "exit_type": "TARGET",
                                   "pnl_pts": pts, "pnl_usd": pts * POINT_VALUE - COMMISSION, "hold_mins": bars_held * 5})
                    active_trade = None
                elif h >= sl:
                    pts = ep - sl
                    ex_type = "BE_EXIT" if active_trade["be_active"] else "STOP"
                    trades.append({**active_trade, "exit_price": sl, "exit_type": ex_type,
                                   "pnl_pts": pts, "pnl_usd": pts * POINT_VALUE - COMMISSION, "hold_mins": bars_held * 5})
                    active_trade = None
                elif is_time_exit:
                    pts = ep - c
                    trades.append({**active_trade, "exit_price": c, "exit_type": "TIME_EXIT",
                                   "pnl_pts": pts, "pnl_usd": pts * POINT_VALUE - COMMISSION, "hold_mins": bars_held * 5})
                    active_trade = None
            else:
                if h >= tp:
                    pts = tp - ep
                    trades.append({**active_trade, "exit_price": tp, "exit_type": "TARGET",
                                   "pnl_pts": pts, "pnl_usd": pts * POINT_VALUE - COMMISSION, "hold_mins": bars_held * 5})
                    active_trade = None
                elif l <= sl:
                    pts = sl - ep
                    ex_type = "BE_EXIT" if active_trade["be_active"] else "STOP"
                    trades.append({**active_trade, "exit_price": sl, "exit_type": ex_type,
                                   "pnl_pts": pts, "pnl_usd": pts * POINT_VALUE - COMMISSION, "hold_mins": bars_held * 5})
                    active_trade = None
                elif is_time_exit:
                    pts = c - ep
                    trades.append({**active_trade, "exit_price": c, "exit_type": "TIME_EXIT",
                                   "pnl_pts": pts, "pnl_usd": pts * POINT_VALUE - COMMISSION, "hold_mins": bars_held * 5})
                    active_trade = None
            continue

        if traded_today: continue
        if not (dtime(9, 30) <= t <= dtime(11, 30)): continue

        if h > london_h and c < london_h and (h - c) >= 12.0:
            ep  = (h + c) / 2.0
            sl  = h + 3.0
            rk  = sl - ep
            tp  = min(london_l, ep - rk * 3.5)
            tp  = max(tp, ep - 250.0)
            rw  = ep - tp
            if 15.0 <= rk <= 45.0 and rw >= rk * 2.0:
                active_trade = {
                    "date": str(sess_date), "time": str(t)[:5], "side": "short",
                    "entry_p": ep, "stop": sl, "target": tp, "risk_pts": rk,
                    "reward_pts": rw, "rr": rw / rk, "entry_bar": i, "be_active": False
                }
                traded_today = True

        elif l < london_l and c > london_l and (c - l) >= 12.0:
            ep  = (l + c) / 2.0
            sl  = l - 3.0
            rk  = ep - sl
            tp  = max(london_h, ep + rk * 3.5)
            tp  = min(tp, ep + 250.0)
            rw  = tp - ep
            if 15.0 <= rk <= 45.0 and rw >= rk * 2.0:
                active_trade = {
                    "date": str(sess_date), "time": str(t)[:5], "side": "long",
                    "entry_p": ep, "stop": sl, "target": tp, "risk_pts": rk,
                    "reward_pts": rw, "rr": rw / rk, "entry_bar": i, "be_active": False
                }
                traded_today = True

tdf = pd.DataFrame(trades)
print(f"Total August trades: {len(tdf)}")
if not tdf.empty:
    tdf['hold_time'] = tdf['hold_mins'].apply(lambda m: f'{m//60}h {m%60}m')
    print(tdf[['date', 'time', 'side', 'entry_p', 'pnl_pts', 'hold_time', 'exit_type', 'pnl_usd']].to_string(index=False))
    print(f"Net Profit: ${tdf['pnl_usd'].sum():,.2f}")

