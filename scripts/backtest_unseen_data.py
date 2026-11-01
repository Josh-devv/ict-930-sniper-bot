"""
OUT-OF-SAMPLE BACKTEST: Unseen 3-Month Real Historical Data
Data Period: January 20, 2026 - April 15, 2026 (~60 trading days)
Timeframes Tested: 3-Minute (User's preferred) and 5-Minute
Strategy: Anchored 09:30 VWAP ± 2.0σ Band Rejection + 50% Wick CE Retest
"""
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
import warnings
warnings.filterwarnings("ignore")

# ── Load 84,000+ Real 1-Minute Bars ──────────────────────────────────────────
raw_df = pd.read_csv("sample_data/MNQ_1min_20260120_20260415.csv")
raw_df["dt"] = pd.to_datetime(raw_df["datetime"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
raw_df = raw_df.sort_values("dt").set_index("dt")

print("=" * 65)
print("  ICT 9:30 SNIPER — UNSEEN OUT-OF-SAMPLE DATA BACKTEST")
print("  Dataset: Real CME Futures (Jan 20, 2026 to Apr 15, 2026)")
print("=" * 65)

POINT_VALUE = 20.0    # 1 Full NQ Contract ($20 / point)
COMMISSION  = 10.20   # Round trip per contract

def run_backtest_for_tf(timeframe_mins=3):
    # Resample 1m bars to chosen timeframe (3m or 5m)
    rule = f"{timeframe_mins}min"
    resampled = raw_df.resample(rule, closed="left", label="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()

    resampled["date"]   = resampled["dt"].dt.date
    resampled["hour"]   = resampled["dt"].dt.hour
    resampled["minute"] = resampled["dt"].dt.minute
    resampled["time"]   = resampled["dt"].dt.time

    # Filter to regular session hours (09:30 to 16:00 ET)
    df = resampled[(resampled["hour"] >= 9) & ~((resampled["hour"] == 9) & (resampled["minute"] < 30))].copy()
    df = df[df["hour"] < 16].copy()
    df = df.reset_index(drop=True)

    VWAP_MULT   = 2.0
    MIN_BAND_PT = 15.0
    WICK_PCT    = 0.35
    MIN_STOP    = 5.0
    MAX_STOP    = 60.0
    MIN_RR      = 1.5
    STOP_BUF    = 1.5

    ENTRY_START = dtime(9, 30)
    ENTRY_END   = dtime(11, 30)
    EOD_TIME    = dtime(15, 45)

    trades = []

    for day, day_df in df.groupby("date"):
        day_df = day_df.reset_index(drop=True)
        cum_vol = 0.0; cum_pv = 0.0; cum_pv2 = 0.0
        vwap = None; upper = None; lower = None; stdev = 0.0
        traded_today = False
        active_trade = None

        for _, row in day_df.iterrows():
            t = row["time"]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            v = max(row["volume"], 1.0)
            tp_bar = (h + l + c) / 3.0

            # Session-Anchored VWAP from 09:30
            cum_vol += v
            cum_pv  += tp_bar * v
            cum_pv2 += tp_bar * tp_bar * v
            vwap    = cum_pv / cum_vol
            var_    = max(0.0, cum_pv2 / cum_vol - vwap**2)
            stdev   = np.sqrt(var_)
            upper   = vwap + VWAP_MULT * stdev
            lower   = vwap - VWAP_MULT * stdev

            # EOD Flat at 15:45
            if t >= EOD_TIME and active_trade:
                ep = active_trade["entry_p"]; side = active_trade["side"]
                pnl_pts = (ep - c) if side == "short" else (c - ep)
                trades.append({**active_trade, "exit_price": c, "exit_type": "EOD",
                               "pnl_pts": pnl_pts, "pnl_usd": pnl_pts * POINT_VALUE - COMMISSION})
                active_trade = None
                continue

            # Trade Management
            if active_trade:
                ep = active_trade["entry_p"]
                sl = active_trade["stop"]
                tp = active_trade["target"]
                side = active_trade["side"]

                if side == "short":
                    if l <= tp:
                        pnl_pts = ep - tp
                        trades.append({**active_trade, "exit_price": tp, "exit_type": "TP",
                                       "pnl_pts": pnl_pts, "pnl_usd": pnl_pts * POINT_VALUE - COMMISSION})
                        active_trade = None
                    elif h >= sl:
                        pnl_pts = ep - sl
                        trades.append({**active_trade, "exit_price": sl, "exit_type": "SL",
                                       "pnl_pts": pnl_pts, "pnl_usd": pnl_pts * POINT_VALUE - COMMISSION})
                        active_trade = None
                else: # long
                    if h >= tp:
                        pnl_pts = tp - ep
                        trades.append({**active_trade, "exit_price": tp, "exit_type": "TP",
                                       "pnl_pts": pnl_pts, "pnl_usd": pnl_pts * POINT_VALUE - COMMISSION})
                        active_trade = None
                    elif l <= sl:
                        pnl_pts = sl - ep
                        trades.append({**active_trade, "exit_price": sl, "exit_type": "SL",
                                       "pnl_pts": pnl_pts, "pnl_usd": pnl_pts * POINT_VALUE - COMMISSION})
                        active_trade = None
                continue

            # Signal Detection
            if traded_today or stdev < 0.5:
                continue
            if not (ENTRY_START <= t <= ENTRY_END):
                continue

            crange = h - l
            up_w   = h - max(o, c)
            dn_w   = min(o, c) - l

            # SHORT Setup
            if (upper - vwap) >= MIN_BAND_PT and h >= upper and crange >= 6.0 and up_w >= crange * WICK_PCT and c < o:
                ep  = (h + max(o, c)) / 2.0
                sl  = h + STOP_BUF
                rk  = min(max(sl - ep, 1.0), MAX_STOP)
                tgt = vwap
                rw  = ep - tgt
                rr  = rw / rk if rk > 0 else 0

                if rk >= MIN_STOP and rw >= rk * MIN_RR:
                    active_trade = {"date": str(day), "time": str(t)[:5], "side": "short",
                                    "entry_p": ep, "stop": sl, "target": tgt,
                                    "risk_pts": rk, "reward_pts": rw, "rr": rr}
                    traded_today = True

            # LONG Setup
            elif (vwap - lower) >= MIN_BAND_PT and l <= lower and crange >= 6.0 and dn_w >= crange * WICK_PCT and c > o:
                ep  = (l + min(o, c)) / 2.0
                sl  = l - STOP_BUF
                rk  = min(max(ep - sl, 1.0), MAX_STOP)
                tgt = vwap
                rw  = tgt - ep
                rr  = rw / rk if rk > 0 else 0

                if rk >= MIN_STOP and rw >= rk * MIN_RR:
                    active_trade = {"date": str(day), "time": str(t)[:5], "side": "long",
                                    "entry_p": ep, "stop": sl, "target": tgt,
                                    "risk_pts": rk, "reward_pts": rw, "rr": rr}
                    traded_today = True

    return pd.DataFrame(trades), df["date"].nunique()

def summarize(tdf, total_days, tf_name):
    print(f"\n{'='*25} RESULTS: {tf_name.upper()} TIMEFRAME {'='*25}")
    if tdf.empty:
        print("No trades found.")
        return

    wins   = tdf[tdf["exit_type"] == "TP"]
    losses = tdf[tdf["exit_type"] == "SL"]
    eods   = tdf[tdf["exit_type"] == "EOD"]

    total    = len(tdf)
    win_cnt  = len(wins)
    loss_cnt = len(losses)
    wr       = (win_cnt / total) * 100
    net_pnl  = tdf["pnl_usd"].sum()
    gw       = wins["pnl_usd"].sum() if win_cnt else 0
    gl       = abs(losses["pnl_usd"].sum()) if loss_cnt else 1
    pf       = gw / gl if gl else float("inf")
    avg_rr   = tdf["rr"].mean()
    avg_stop = tdf["risk_pts"].mean()
    avg_win  = wins["pnl_usd"].mean() if win_cnt else 0
    avg_loss = losses["pnl_usd"].mean() if loss_cnt else 0

    tdf["cum"] = tdf["pnl_usd"].cumsum()
    peak = tdf["cum"].cummax()
    max_dd = (tdf["cum"] - peak).min()

    print(f"  Test Period         : Jan 20, 2026 – Apr 15, 2026 ({total_days} Trading Days)")
    print(f"  Total Trades        : {total} trades  (~{total / total_days * 5:.1f} per week)")
    print(f"  Wins (TP Target)    : {win_cnt} ({wr:.1f}%)")
    print(f"  Losses (Stop Loss)  : {loss_cnt} ({(loss_cnt / total) * 100:.1f}%)")
    print(f"  EOD Exits           : {len(eods)}")
    print("-" * 65)
    print(f"  Win Rate            : {wr:.1f}%")
    print(f"  Profit Factor       : {pf:.2f}")
    print(f"  Avg Risk:Reward     : {avg_rr:.2f}R")
    print(f"  Avg Stop Loss Size  : {avg_stop:.1f} pts (~${avg_stop * POINT_VALUE:,.0f})")
    print(f"  Avg Winning Trade   : +${avg_win:,.2f}")
    print(f"  Avg Losing Trade    : -${abs(avg_loss):,.2f}")
    print("-" * 65)
    print(f"  Net Profit (1 NQ)   : +${net_pnl:,.2f}")
    print(f"  Max Drawdown        : -${abs(max_dd):,.2f}")
    print("=" * 65)

    print(f"\n  🎯 FUNDEDNEXT / TOPSTEP $50K EVALUATION VERDICT ({tf_name}):")
    print(f"     Target: +$2,500 | Max DD Limit: -$2,000")
    print(f"     Observed Profit  : +${net_pnl:,.2f}")
    print(f"     Observed Max DD  : -${abs(max_dd):,.2f}")

    if net_pnl >= 2500 and abs(max_dd) < 2000:
        weeks = 2500 / (net_pnl / total_days * 5)
        print(f"     STATUS           : ✅ PASSES CHALLENGE EASILY in ~{weeks:.1f} weeks!")
    elif net_pnl > 0 and abs(max_dd) < 2000:
        print(f"     STATUS           : ✅ PROFITABLE & SAFE (Passes with more time)")
    else:
        print(f"     STATUS           : ⚠️ Needs adjustments")

# Run both 3-minute and 5-minute tests
tdf_3m, days_3m = run_backtest_for_tf(timeframe_mins=3)
summarize(tdf_3m, days_3m, "3-Minute (User Preferred)")

tdf_5m, days_5m = run_backtest_for_tf(timeframe_mins=5)
summarize(tdf_5m, days_5m, "5-Minute")

