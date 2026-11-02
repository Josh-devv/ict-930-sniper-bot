"""
1-to-4 Hour Day Swing Strategy Backtest
Model: Liquidity Sweep (PDH/PDL/Asia/London) + Daily 2.5σ VWAP Exhaustion + MSS Displacement
Target: 150 to 300+ points across 1-4 hour holding windows
Data: Real CME Globex 24-hour 1-min data (Jan 20 – Apr 15, 2026, 84,000+ bars)
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dtime, timedelta
import warnings
warnings.filterwarnings("ignore")

POINT_VALUE = 20.0    # 1 Full NQ Contract ($20 / pt)
COMMISSION  = 10.20   # Round trip per contract

print("=" * 65)
print("  ICT 1-TO-4 HOUR DAY SWING STRATEGY STUDY")
print("  Model: Top & Bottom Liquidity Sweep + Daily 2.5σ VWAP + MSS")
print("=" * 65)

# ── Load 24-hour Globex Data ─────────────────────────────────────────────────
print("Loading CME 24-hour historical Globex data...")
raw = pd.read_csv("sample_data/MNQ_1min_20260120_20260415.csv")
raw["dt"] = pd.to_datetime(raw["datetime"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
raw = raw.sort_values("dt").set_index("dt")

# Resample to 5-minute bars (ideal for capturing 1-4 hour day swings)
df_5m = raw.resample("5min", closed="left", label="left").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum"
}).dropna().reset_index()

df_5m["date"]   = df_5m["dt"].dt.date
df_5m["time"]   = df_5m["dt"].dt.time
df_5m["hour"]   = df_5m["dt"].dt.hour
df_5m["minute"] = df_5m["dt"].dt.minute

# Globex Trading Day begins at 18:00 EST (yesterday evening) and ends at 17:00 EST
# Assign trading day id
df_5m["session_date"] = df_5m["dt"].apply(
    lambda x: (x + timedelta(hours=6)).date() if x.hour >= 18 else x.date()
)

print(f"Loaded {len(df_5m):,} 5-minute bars across {df_5m['session_date'].nunique()} Globex trading sessions.\n")

# ── Strategy Engine ──────────────────────────────────────────────────────────
# Sessions in EST:
# Asia Range:   18:00 (prev day) to 02:00
# London Window: 02:30 to 05:00
# NY Window:     09:30 to 11:30
# EOD Exit:      16:15 EST

trades = []

# Group by Globex session date
for sess_date, s_df in df_5m.groupby("session_date"):
    s_df = s_df.reset_index(drop=True)
    if len(s_df) < 50:
        continue

    # 1. Calculate Daily Anchored VWAP (resets at 18:00 EST)
    s_df["tp"] = (s_df["high"] + s_df["low"] + s_df["close"]) / 3.0
    s_df["vol"] = s_df["volume"].clip(lower=1.0)
    s_df["cum_vol"] = s_df["vol"].cumsum()
    s_df["cum_pv"]  = (s_df["tp"] * s_df["vol"]).cumsum()
    s_df["cum_pv2"] = (s_df["tp"]**2 * s_df["vol"]).cumsum()
    s_df["vwap"]    = s_df["cum_pv"] / s_df["cum_vol"]
    s_df["var"]     = (s_df["cum_pv2"] / s_df["cum_vol"] - s_df["vwap"]**2).clip(lower=0.0)
    s_df["stdev"]   = np.sqrt(s_df["var"])
    s_df["upper_25"]= s_df["vwap"] + 2.5 * s_df["stdev"]
    s_df["lower_25"]= s_df["vwap"] - 2.5 * s_df["stdev"]

    # 2. Extract Asian Session High & Low (18:00 to 02:00 EST)
    asia_df = s_df[(s_df["hour"] >= 18) | (s_df["hour"] < 2)]
    if asia_df.empty:
        continue
    asia_h = asia_df["high"].max()
    asia_l = asia_df["low"].min()

    # Track trade state
    active_trade = None
    traded_in_session = False

    # Scan through trading windows (London: 02:30-05:00 OR New York: 09:30-11:30)
    for i in range(5, len(s_df)):
        row = s_df.iloc[i]
        t = row["time"]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        vwap = row["vwap"]
        up_25 = row["upper_25"]
        dn_25 = row["lower_25"]

        # Check Active Trade Exit Conditions
        if active_trade:
            entry_p = active_trade["entry_p"]
            sl      = active_trade["stop"]
            tp      = active_trade["target"]
            side    = active_trade["side"]
            bars_held = i - active_trade["entry_bar"]

            # Max hold time: 4 hours (48 five-minute bars) or EOD 16:15 EST
            is_time_exit = bars_held >= 48 or (row["hour"] == 16 and row["minute"] >= 15)

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

        if traded_in_session:
            continue

        # Check if we are inside either the London window (02:30-05:00) or NY window (09:30-11:30)
        is_london = dtime(2, 30) <= t <= dtime(5, 0)
        is_ny     = dtime(9, 30) <= t <= dtime(11, 30)
        if not (is_london or is_ny):
            continue

        # Recent 3-bar swing levels for Market Structure Shift (MSS)
        recent_low  = s_df.iloc[i-4:i]["low"].min()
        recent_high = s_df.iloc[i-4:i]["high"].max()

        # ── SETUP 1: TOP OF DAY (SHORT SWING) ──────────────────────────────────
        # Price swept Asia High OR recent session high AND reached outer 2.5σ band
        swept_top = (h >= asia_h or h >= up_25) and h >= s_df.iloc[i-5:i]["high"].max()
        # Market structure shift: Candle displays strong displacement closing below recent swing low
        mss_short = c < recent_low and (o - c) >= 10.0  # clean bearish displacement

        if swept_top and mss_short:
            swing_high = s_df.iloc[i-3:i+1]["high"].max()
            entry_p    = (o + c) / 2.0   # FVG / 50% displacement retest
            stop_p     = swing_high + 3.0
            risk_pts   = stop_p - entry_p

            # Target: Asian Session Low or opposite band (Huge Day-Swing Target: 150-250+ pts)
            target_p   = max(asia_l, entry_p - risk_pts * 4.0)
            reward_pts = entry_p - target_p

            if 15.0 <= risk_pts <= 45.0 and reward_pts >= risk_pts * 2.5:
                active_trade = {
                    "session": "London" if is_london else "New York",
                    "date": str(sess_date), "time": str(t)[:5],
                    "side": "short", "entry_p": entry_p, "stop": stop_p,
                    "target": target_p, "risk_pts": risk_pts,
                    "reward_pts": reward_pts, "rr": reward_pts / risk_pts,
                    "entry_bar": i
                }
                traded_in_session = True

        # ── SETUP 2: BOTTOM OF DAY (LONG SWING) ─────────────────────────────────
        # Price swept Asia Low OR reached outer -2.5σ band
        swept_bot = (l <= asia_l or l <= dn_25) and l <= s_df.iloc[i-5:i]["low"].min()
        # Market structure shift: Candle displays strong displacement closing above recent swing high
        mss_long  = c > recent_high and (c - o) >= 10.0  # clean bullish displacement

        if swept_bot and mss_long:
            swing_low = s_df.iloc[i-3:i+1]["low"].min()
            entry_p   = (o + c) / 2.0   # FVG / 50% displacement retest
            stop_p    = swing_low - 3.0
            risk_pts  = entry_p - stop_p

            # Target: Asian Session High or opposite band (Huge Day-Swing Target: 150-250+ pts)
            target_p   = min(asia_h, entry_p + risk_pts * 4.0)
            reward_pts = target_p - entry_p

            if 15.0 <= risk_pts <= 45.0 and reward_pts >= risk_pts * 2.5:
                active_trade = {
                    "session": "London" if is_london else "New York",
                    "date": str(sess_date), "time": str(t)[:5],
                    "side": "long", "entry_p": entry_p, "stop": stop_p,
                    "target": target_p, "risk_pts": risk_pts,
                    "reward_pts": reward_pts, "rr": reward_pts / risk_pts,
                    "entry_bar": i
                }
                traded_in_session = True

# ── Results & Analysis ────────────────────────────────────────────────────────
tdf = pd.DataFrame(trades)

print("=" * 65)
print("  DAY-SWING RESULTS (1-TO-4 HOUR HOLD TIME)")
print("=" * 65)

if tdf.empty:
    print("No trades triggered.")
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
    avg_pts    = wins["pnl_pts"].mean() if len(wins) else 0

    tdf["cum"] = tdf["pnl_usd"].cumsum()
    peak       = tdf["cum"].cummax()
    max_dd     = (tdf["cum"] - peak).min()

    print(f"  Test Period            : Jan 20, 2026 – Apr 15, 2026 ({total_days} Sessions)")
    print(f"  Total Day-Swing Trades : {total} trades (~{total/total_days*5:.1f} per week)")
    print(f"  Full Target Reached    : {len(targets)} ({len(targets)/total*100:.1f}%)")
    print(f"  Time-based Profit Exits: {len(time_exits[time_exits['pnl_usd']>0])}")
    print(f"  Stop Outs              : {len(stops)} ({len(stops)/total*100:.1f}%)")
    print("-" * 65)
    print(f"  Win Rate               : {win_rate:.1f}%")
    print(f"  Profit Factor          : {pf:.2f}")
    print(f"  Average Holding Time   : {avg_hold/60:.1f} hours ({avg_hold:.0f} mins)")
    print(f"  Average Winning Move   : +{avg_pts:.1f} points per trade!")
    print(f"  Average Win (USD)      : +${avg_win:,.2f}")
    print(f"  Average Loss (USD)     : -${avg_loss:,.2f}")
    print("-" * 65)
    print(f"  Total Net Profit (1 NQ): +${net_pnl:,.2f}")
    print(f"  Max Drawdown           : -${abs(max_dd):,.2f}")
    print("=" * 65)

    print("\nSAMPLE OF 1-TO-4 HOUR TRADES CAPTURING HIGH/LOW OF DAY:")
    cols = ["date", "session", "time", "side", "risk_pts", "pnl_pts", "hold_mins", "exit_type", "pnl_usd"]
    sample_df = tdf.sort_values("pnl_usd", ascending=False).head(8)[cols]
    sample_df["hold_time"] = sample_df["hold_mins"].apply(lambda m: f"{m//60}h {m%60}m")
    sample_df = sample_df.drop(columns=["hold_mins"])
    print(sample_df.to_string(index=False))

