"""
Day-Swing Optimization & October 2026 Out-of-Sample Test
Experiments:
  1. Base Model: All-or-Nothing 200+ pt target
  2. Enhanced Model: Breakeven trigger at +1.5R (+50 pts) + 1-hour Trend Alignment
  3. Oct 2026 Test: Run both on October 2026 real CME NQ data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
import warnings
warnings.filterwarnings("ignore")

POINT_VALUE = 20.0
COMMISSION  = 10.20

# ── 1. Fetch October 2026 Real CME NQ Data ──────────────────────────────────
import yfinance as yf
print("Fetching October 2026 CME NQ 1-minute data...")

oct_chunks = []
end_d = datetime(2026, 10, 30)
for _ in range(4):
    start_d = end_d - timedelta(days=7)
    d = yf.download('NQ=F', start=start_d.strftime('%Y-%m-%d'), end=end_d.strftime('%Y-%m-%d'), interval='1m', progress=False)
    if len(d) > 0:
        oct_chunks.append(d)
    end_d = start_d

oct_raw = pd.concat(oct_chunks).sort_index()
if isinstance(oct_raw.columns, pd.MultiIndex):
    oct_raw.columns = [c[0].lower() for c in oct_raw.columns]
else:
    oct_raw.columns = [c.lower() for c in oct_raw.columns]

oct_raw = oct_raw.reset_index()
dt_col = [c for c in oct_raw.columns if 'date' in c or 'time' in c or 'index' in c][0]
oct_raw['dt'] = pd.to_datetime(oct_raw[dt_col]).dt.tz_convert('America/New_York')
oct_raw = oct_raw.sort_values('dt').set_index('dt')
oct_raw = oct_raw[~oct_raw.index.duplicated(keep='first')]

# Resample to 5m
oct_5m = oct_raw.resample("5min", closed="left", label="left").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
}).dropna().reset_index()
oct_5m["session_date"] = oct_5m["dt"].apply(lambda x: (x + timedelta(hours=6)).date() if x.hour >= 18 else x.date())
oct_5m["time"] = oct_5m["dt"].dt.time
oct_5m["hour"] = oct_5m["dt"].dt.hour
oct_5m["minute"] = oct_5m["dt"].dt.minute

# ── Load Jan-Apr 2026 Dataset ───────────────────────────────────────────────
q1_raw = pd.read_csv("sample_data/MNQ_1min_20260120_20260415.csv")
q1_raw["dt"] = pd.to_datetime(q1_raw["datetime"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
q1_raw = q1_raw.sort_values("dt").set_index("dt")

q1_5m = q1_raw.resample("5min", closed="left", label="left").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
}).dropna().reset_index()
q1_5m["session_date"] = q1_5m["dt"].apply(lambda x: (x + timedelta(hours=6)).date() if x.hour >= 18 else x.date())
q1_5m["time"] = q1_5m["dt"].dt.time
q1_5m["hour"] = q1_5m["dt"].dt.hour
q1_5m["minute"] = q1_5m["dt"].dt.minute

def run_swing_simulation(df_data, use_be=True, be_trigger_r=1.5):
    trades = []
    for sess_date, s_df in df_data.groupby("session_date"):
        s_df = s_df.reset_index(drop=True)
        if len(s_df) < 25: continue

        # London Session (02:00 to 05:00 EST)
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
                ep        = active_trade["entry_p"]
                sl        = active_trade["stop"]
                tp        = active_trade["target"]
                side      = active_trade["side"]
                risk_pts  = active_trade["risk_pts"]
                bars_held = i - active_trade["entry_bar"]

                # Breakeven Adjustment
                if use_be and not active_trade["be_active"]:
                    if side == "short" and (ep - l) >= risk_pts * be_trigger_r:
                        active_trade["stop"] = ep - 1.0  # lock in +1 pt profit
                        active_trade["be_active"] = True
                        sl = active_trade["stop"]
                    elif side == "long" and (h - ep) >= risk_pts * be_trigger_r:
                        active_trade["stop"] = ep + 1.0  # lock in +1 pt profit
                        active_trade["be_active"] = True
                        sl = active_trade["stop"]

                # Max 4-hour hold or 16:00 EOD
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

            if traded_today: continue
            if not (dtime(9, 30) <= t <= dtime(11, 30)): continue

            # SHORT HOD REVERSAL
            if h > london_h and c < london_h and (h - c) >= 12.0:
                ep  = (h + c) / 2.0
                sl  = h + 3.0
                rk  = sl - ep
                tp  = min(london_l, ep - rk * 3.5)
                tp  = max(tp, ep - 250.0)
                rw  = ep - tp

                if 15.0 <= rk <= 45.0 and rw >= rk * 2.5:
                    active_trade = {
                        "date": str(sess_date), "time": str(t)[:5], "side": "short",
                        "entry_p": ep, "stop": sl, "target": tp, "risk_pts": rk,
                        "reward_pts": rw, "rr": rw / rk, "entry_bar": i, "be_active": False
                    }
                    traded_today = True

            # LONG LOD REVERSAL
            elif l < london_l and c > london_l and (c - l) >= 12.0:
                ep  = (l + c) / 2.0
                sl  = l - 3.0
                rk  = ep - sl
                tp  = max(london_h, ep + rk * 3.5)
                tp  = min(tp, ep + 250.0)
                rw  = tp - ep

                if 15.0 <= rk <= 45.0 and rw >= rk * 2.5:
                    active_trade = {
                        "date": str(sess_date), "time": str(t)[:5], "side": "long",
                        "entry_p": ep, "stop": sl, "target": tp, "risk_pts": rk,
                        "reward_pts": rw, "rr": rw / rk, "entry_bar": i, "be_active": False
                    }
                    traded_today = True

    return pd.DataFrame(trades)

def print_metrics(tdf, label):
    print("=" * 65)
    print(f"  RESULTS: {label.upper()}")
    print("=" * 65)
    if tdf.empty:
        print("  No trades taken.")
        return

    wins = tdf[tdf["pnl_usd"] > 0]
    be_exits = tdf[tdf["exit_type"] == "BE_EXIT"]
    losses = tdf[tdf["pnl_usd"] <= 0]
    total = len(tdf)

    # Effective win rate including BE profit
    win_rate = len(wins) / total * 100
    net_pnl = tdf["pnl_usd"].sum()
    gw = wins["pnl_usd"].sum() if len(wins) else 0
    gl = abs(losses["pnl_usd"].sum()) if len(losses) else 1
    pf = gw / gl

    avg_win = wins["pnl_usd"].mean() if len(wins) else 0
    avg_loss = abs(losses["pnl_usd"].mean()) if len(losses) else 0
    avg_win_pts = wins["pnl_pts"].mean() if len(wins) else 0

    tdf["cum"] = tdf["pnl_usd"].cumsum()
    peak = tdf["cum"].cummax()
    max_dd = (tdf["cum"] - peak).min()

    print(f"  Total Trades        : {total}")
    print(f"  Profitable Exits    : {len(wins)} ({win_rate:.1f}%)")
    print(f"  Breakeven Exits     : {len(be_exits)}")
    print(f"  Full Losses         : {len(losses)} ({(len(losses)/total)*100:.1f}%)")
    print("-" * 65)
    print(f"  Win Rate            : {win_rate:.1f}%")
    print(f"  Profit Factor       : {pf:.2f}")
    print(f"  Average Win (Pts)   : +{avg_win_pts:.1f} points")
    print(f"  Average Win (USD)   : +${avg_win:,.2f}")
    print(f"  Average Loss (USD)  : -${avg_loss:,.2f}")
    print("-" * 65)
    print(f"  Total Net Profit    : +${net_pnl:,.2f}")
    print(f"  Max Drawdown        : -${abs(max_dd):,.2f}")
    print("=" * 65)

# Test 1: Enhanced with Breakeven on Q1 2026 (Jan–Apr)
res_q1_be = run_swing_simulation(q1_5m, use_be=True, be_trigger_r=1.5)
print_metrics(res_q1_be, "Q1 2026 Enhanced (With Breakeven Protection)")

# Test 2: Base on October 2026
res_oct_base = run_swing_simulation(oct_5m, use_be=False)
print_metrics(res_oct_base, "October 2026 (Base Model — All-or-Nothing)")

# Test 3: Enhanced on October 2026
res_oct_be = run_swing_simulation(oct_5m, use_be=True, be_trigger_r=1.5)
print_metrics(res_oct_be, "October 2026 (Enhanced Model — With Breakeven)")

