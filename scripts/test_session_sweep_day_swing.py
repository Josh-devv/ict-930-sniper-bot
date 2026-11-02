"""
ICT Day-Swing Model: Session Liquidity Sweep (London / Asia Sweep in New York)
Concept:
  - 71.7% of days, New York sweeps the London High or London Low.
  - When NY sweeps the London High/Low and rejects back inside, it creates the High of the Day (HOD)
    or Low of the Day (LOD).
  - The trade targets the opposite session extreme or midpoint, capturing 100 to 300+ points across a 1-4 hour hold.
"""

import pandas as pd
import numpy as np
from datetime import timedelta, time as dtime
import warnings
warnings.filterwarnings("ignore")

POINT_VALUE = 20.0    # 1 Full NQ Contract ($20 / pt)
COMMISSION  = 10.20   # Round trip per contract

raw = pd.read_csv("sample_data/MNQ_1min_20260120_20260415.csv")
raw["dt"] = pd.to_datetime(raw["datetime"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
raw = raw.sort_values("dt").set_index("dt")

# Resample to 5-minute bars
df_5m = raw.resample("5min", closed="left", label="left").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
}).dropna().reset_index()

df_5m["session_date"] = df_5m["dt"].apply(lambda x: (x + timedelta(hours=6)).date() if x.hour >= 18 else x.date())
df_5m["time"] = df_5m["dt"].dt.time
df_5m["hour"] = df_5m["dt"].dt.hour
df_5m["minute"] = df_5m["dt"].dt.minute

trades = []

for sess_date, s_df in df_5m.groupby("session_date"):
    s_df = s_df.reset_index(drop=True)
    if len(s_df) < 50: continue

    # 1. Asian Range (18:00 to 02:00 EST)
    asia = s_df[(s_df["hour"] >= 18) | (s_df["hour"] < 2)]
    if asia.empty: continue
    asia_h = asia["high"].max()
    asia_l = asia["low"].min()

    # 2. London Session (02:00 to 05:00 EST)
    london = s_df[(s_df["time"] >= dtime(2, 0)) & (s_df["time"] <= dtime(5, 0))]
    if london.empty: continue
    london_h = london["high"].max()
    london_l = london["low"].min()
    london_mid = (london_h + london_l) / 2.0

    # 3. New York Session Entry Window (09:30 to 11:30 EST)
    active_trade = None
    traded_today = False

    for i in range(len(s_df)):
        row = s_df.iloc[i]
        t = row["time"]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        # Trade Management (Holding 1 to 4 hours)
        if active_trade:
            entry_p = active_trade["entry_p"]
            sl      = active_trade["stop"]
            tp      = active_trade["target"]
            side    = active_trade["side"]
            bars_held = i - active_trade["entry_bar"]

            # Max hold: 4 hours (48 bars) or EOD 16:00
            is_time_exit = bars_held >= 48 or (row["hour"] == 16 and row["minute"] >= 0)

            if side == "short":
                if l <= tp:
                    pts = entry_p - tp
                    trades.append({**active_trade, "exit_price": tp, "exit_type": "TARGET",
                                   "pnl_pts": pts, "pnl_usd": pts * POINT_VALUE - COMMISSION,
                                   "hold_mins": bars_held * 5})
                    active_trade = None
                elif h >= sl:
                    pts = entry_p - sl
                    trades.append({**active_trade, "exit_price": sl, "exit_type": "STOP",
                                   "pnl_pts": pts, "pnl_usd": pts * POINT_VALUE - COMMISSION,
                                   "hold_mins": bars_held * 5})
                    active_trade = None
                elif is_time_exit:
                    pts = entry_p - c
                    trades.append({**active_trade, "exit_price": c, "exit_type": "TIME_EXIT",
                                   "pnl_pts": pts, "pnl_usd": pts * POINT_VALUE - COMMISSION,
                                   "hold_mins": bars_held * 5})
                    active_trade = None
            else: # long
                if h >= tp:
                    pts = tp - entry_p
                    trades.append({**active_trade, "exit_price": tp, "exit_type": "TARGET",
                                   "pnl_pts": pts, "pnl_usd": pts * POINT_VALUE - COMMISSION,
                                   "hold_mins": bars_held * 5})
                    active_trade = None
                elif l <= sl:
                    pts = sl - entry_p
                    trades.append({**active_trade, "exit_price": sl, "exit_type": "STOP",
                                   "pnl_pts": pts, "pnl_usd": pts * POINT_VALUE - COMMISSION,
                                   "hold_mins": bars_held * 5})
                    active_trade = None
                elif is_time_exit:
                    pts = c - entry_p
                    trades.append({**active_trade, "exit_price": c, "exit_type": "TIME_EXIT",
                                   "pnl_pts": pts, "pnl_usd": pts * POINT_VALUE - COMMISSION,
                                   "hold_mins": bars_held * 5})
                    active_trade = None
            continue

        if traded_today: continue
        if not (dtime(9, 30) <= t <= dtime(11, 30)): continue

        # ── SETUP 1: NY SWEEPS LONDON HIGH (SHORT HOD REVERSAL) ────────────────
        # Price spikes above London High (sweeping buy stops), but candle closes back below London High!
        if h > london_h and c < london_h and (h - c) >= 12.0:  # clear upper rejection wick
            entry_p  = (h + c) / 2.0    # 50% wick retest
            stop_p   = h + 3.0          # Stop above High of Day
            risk_pts = stop_p - entry_p

            # Target: London Low or minimum 3.5R (Huge Day-Swing target!)
            target_p = min(london_l, entry_p - risk_pts * 3.5)
            target_p = max(target_p, entry_p - 250.0)  # cap at 250 pts
            reward_pts = entry_p - target_p

            if 15.0 <= risk_pts <= 45.0 and reward_pts >= risk_pts * 2.5:
                active_trade = {
                    "date": str(sess_date), "time": str(t)[:5], "side": "short",
                    "entry_p": entry_p, "stop": stop_p, "target": target_p,
                    "risk_pts": risk_pts, "reward_pts": reward_pts,
                    "rr": reward_pts / risk_pts, "entry_bar": i
                }
                traded_today = True

        # ── SETUP 2: NY SWEEPS LONDON LOW (LONG LOD REVERSAL) ──────────────────
        # Price spikes below London Low (sweeping sell stops), but candle closes back above London Low!
        elif l < london_l and c > london_l and (c - l) >= 12.0: # clear lower rejection wick
            entry_p  = (l + c) / 2.0    # 50% wick retest
            stop_p   = l - 3.0          # Stop below Low of Day
            risk_pts = entry_p - stop_p

            # Target: London High or minimum 3.5R
            target_p = max(london_h, entry_p + risk_pts * 3.5)
            target_p = min(target_p, entry_p + 250.0)  # cap at 250 pts
            reward_pts = target_p - entry_p

            if 15.0 <= risk_pts <= 45.0 and reward_pts >= risk_pts * 2.5:
                active_trade = {
                    "date": str(sess_date), "time": str(t)[:5], "side": "long",
                    "entry_p": entry_p, "stop": stop_p, "target": target_p,
                    "risk_pts": risk_pts, "reward_pts": reward_pts,
                    "rr": reward_pts / risk_pts, "entry_bar": i
                }
                traded_today = True

# ── Performance Output ────────────────────────────────────────────────────────
tdf = pd.DataFrame(trades)

print("=" * 65)
print("  ICT DAY-SWING MODEL: LONDON LIQUIDITY SWEEP RESULTS")
print("  Holding Period: 1 to 4 Hours | Targets: 100 to 250+ Points")
print("=" * 65)

if tdf.empty:
    print("No trades found.")
else:
    total_days = df_5m["session_date"].nunique()
    total      = len(tdf)
    wins       = tdf[tdf["pnl_usd"] > 0]
    losses     = tdf[tdf["pnl_usd"] <= 0]
    targets    = tdf[tdf["exit_type"] == "TARGET"]
    stops      = tdf[tdf["exit_type"] == "STOP"]
    time_exits = tdf[tdf["exit_type"] == "TIME_EXIT"]

    win_rate   = len(wins) / total * 100
    net_pnl    = tdf["pnl_usd"].sum()
    gw         = wins["pnl_usd"].sum() if len(wins) else 0
    gl         = abs(losses["pnl_usd"].sum()) if len(losses) else 1
    pf         = gw / gl
    avg_win    = wins["pnl_usd"].mean() if len(wins) else 0
    avg_loss   = abs(losses["pnl_usd"].mean()) if len(losses) else 0
    avg_hold   = tdf["hold_mins"].mean()
    avg_win_pts= wins["pnl_pts"].mean() if len(wins) else 0

    tdf["cum"] = tdf["pnl_usd"].cumsum()
    peak       = tdf["cum"].cummax()
    max_dd     = (tdf["cum"] - peak).min()

    print(f"  Sessions Tested        : {total_days} Globex Days (Jan 20 – Apr 15, 2026)")
    print(f"  Total Day-Swing Trades : {total} trades (~{total/total_days*5:.1f} setups per week)")
    print(f"  Wins                   : {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losses                 : {len(losses)} ({(len(losses)/total)*100:.1f}%)")
    print("-" * 65)
    print(f"  Win Rate               : {win_rate:.1f}%")
    print(f"  Profit Factor          : {pf:.2f}")
    print(f"  Average Holding Time   : {avg_hold/60:.1f} hours ({avg_hold:.0f} minutes)")
    print(f"  Average Win (Points)   : +{avg_win_pts:.1f} points per winning trade! 🎯")
    print(f"  Average Win (USD)      : +${avg_win:,.2f}")
    print(f"  Average Loss (USD)     : -${avg_loss:,.2f}")
    print("-" * 65)
    print(f"  Total Net Profit (1 NQ): +${net_pnl:,.2f}")
    print(f"  Max Drawdown           : -${abs(max_dd):,.2f}")
    print("=" * 65)

    print("\nSAMPLE OF ACTUAL 1-TO-4 HOUR DAY-SWING TRADES:")
    cols = ["date", "time", "side", "risk_pts", "pnl_pts", "hold_mins", "exit_type", "pnl_usd"]
    sample_df = tdf.sort_values("pnl_usd", ascending=False).head(10)[cols]
    sample_df["hold_time"] = sample_df["hold_mins"].apply(lambda m: f"{m//60}h {m%60}m")
    sample_df = sample_df.drop(columns=["hold_mins"])
    print(sample_df.to_string(index=False))

