"""
September 2026 Day-Swing Test (London Liquidity Sweep + Breakeven Rule)
Fetches full 24-hour Globex CME NQ=F data for September 2026
Runs the exact Day-Swing strategy and outputs the complete trade log
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
import warnings
warnings.filterwarnings("ignore")

POINT_VALUE = 20.0    # 1 Full NQ Contract ($20 / pt)
COMMISSION  = 10.20   # Round trip per contract

print("Fetching full 24-hour Globex CME NQ=F 5-minute data (last 60 days)...")
ticker = yf.Ticker("NQ=F")
df_60d = ticker.history(period="60d", interval="5m", prepost=True)

if df_60d.empty:
    print("Error fetching data from yfinance.")
    exit()

df_60d = df_60d.reset_index()
if isinstance(df_60d.columns, pd.MultiIndex):
    df_60d.columns = [c[0].lower() for c in df_60d.columns]
else:
    df_60d.columns = [c.lower() for c in df_60d.columns]

dt_col = [c for c in df_60d.columns if "date" in c or "time" in c or "index" in c][0]
df_60d["dt"] = pd.to_datetime(df_60d[dt_col]).dt.tz_convert("America/New_York")
df_60d = df_60d.sort_values("dt").reset_index(drop=True)

df_60d["session_date"] = df_60d["dt"].apply(lambda x: (x + timedelta(hours=6)).date() if x.hour >= 18 else x.date())
df_60d["time"]   = df_60d["dt"].dt.time
df_60d["hour"]   = df_60d["dt"].dt.hour
df_60d["minute"] = df_60d["dt"].dt.minute

# Filter specifically to September 2026
sep_df = df_60d[(df_60d["dt"].dt.year == 2026) & (df_60d["dt"].dt.month == 9)].copy()
print(f"Loaded {len(sep_df):,} bars across {sep_df['session_date'].nunique()} trading days in September 2026.\n")

trades = []

for sess_date, s_df in sep_df.groupby("session_date"):
    s_df = s_df.reset_index(drop=True)
    if len(s_df) < 25:
        continue

    # London Session (02:00 to 05:00 EST)
    london = s_df[(s_df["time"] >= dtime(2, 0)) & (s_df["time"] <= dtime(5, 0))]
    if london.empty:
        continue
    london_h = london["high"].max()
    london_l = london["low"].min()

    active_trade = None
    traded_today = False

    for i in range(len(s_df)):
        row = s_df.iloc[i]
        t = row["time"]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        # Trade Management (Holding 1 to 4 hours)
        if active_trade:
            ep        = active_trade["entry_p"]
            sl        = active_trade["stop"]
            tp        = active_trade["target"]
            side      = active_trade["side"]
            risk_pts  = active_trade["risk_pts"]
            bars_held = i - active_trade["entry_bar"]

            # Breakeven adjustment (+1.5R triggers stop to entry + 1 pt)
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
            else: # long
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

        if traded_today:
            continue
        if not (dtime(9, 30) <= t <= dtime(11, 30)):
            continue

        # Short HOD Sweep
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

        # Long LOD Sweep
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

print("=" * 65)
print("  RESULTS: SEPTEMBER 2026 DAY-SWING MODEL (WITH BREAKEVEN)")
print("=" * 65)

if tdf.empty:
    print("  No trades triggered in September 2026.")
else:
    wins     = tdf[tdf["pnl_usd"] > 0]
    losses   = tdf[tdf["pnl_usd"] <= 0]
    be_exits = tdf[tdf["exit_type"] == "BE_EXIT"]
    total    = len(tdf)

    win_rate = len(wins) / total * 100
    net_pnl  = tdf["pnl_usd"].sum()
    gw       = wins["pnl_usd"].sum() if len(wins) else 0
    gl       = abs(losses["pnl_usd"].sum()) if len(losses) else 1
    pf       = gw / gl

    avg_win  = wins["pnl_usd"].mean() if len(wins) else 0
    avg_loss = abs(losses["pnl_usd"].mean()) if len(losses) else 0
    avg_win_pts = wins["pnl_pts"].mean() if len(wins) else 0

    tdf["cum"] = tdf["pnl_usd"].cumsum()
    peak       = tdf["cum"].cummax()
    max_dd     = (tdf["cum"] - peak).min()

    print(f"  Total Trades Taken   : {total} trades")
    print(f"  Profitable Exits     : {len(wins)} ({win_rate:.1f}%)")
    print(f"  Breakeven Exits      : {len(be_exits)}")
    print(f"  Stop Outs            : {len(losses)} ({(len(losses)/total)*100:.1f}%)")
    print("-" * 65)
    print(f"  Win Rate             : {win_rate:.1f}%")
    print(f"  Profit Factor        : {pf:.2f}")
    print(f"  Average Winning Trade: +{avg_win_pts:.1f} pts (+${avg_win:,.2f})")
    print(f"  Average Losing Trade : -${avg_loss:,.2f}")
    print("-" * 65)
    print(f"  Net P&L (1 NQ)       : +${net_pnl:,.2f}")
    print(f"  Max Drawdown         : -${abs(max_dd):,.2f}")
    print("=" * 65)

    print("\nTRADE LOG FOR SEPTEMBER 2026:")
    cols = ["date", "time", "side", "entry_p", "pnl_pts", "hold_mins", "exit_type", "pnl_usd"]
    tdf["hold_time"] = tdf["hold_mins"].apply(lambda m: f"{m//60}h {m%60}m")
    print(tdf[["date", "time", "side", "entry_p", "pnl_pts", "hold_time", "exit_type", "pnl_usd"]].to_string(index=False))

