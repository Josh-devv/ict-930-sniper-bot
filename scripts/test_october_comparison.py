"""
OCTOBER HEAD-TO-HEAD COMPARISON:
Mode A: 1 Full NQ ($20/pt)
  - NO Break-Even
  - Take Profit: TP1 ONLY (Anchored VWAP), NO TP2/Runner
  - Commission: $10.20 round-turn

Mode B: 5 Micro MNQ ($10/pt total)
  - Scale-Out: 60% (3 MNQ) closed at TP1 (Anchored VWAP)
  - Break-Even: Stop moved to Entry + 1.0 pt lock immediately on TP1 fill
  - Runner: 40% (2 MNQ) held for TP2 (Opposite 2.0σ Band) or EOD 15:45
  - Commission: $6.20 round-turn ($1.24 * 5)

Tested on:
  1. October 2025 (from sample_data/MNQ_1min_12mo.csv)
  2. October 2026 (from real CME Futures data)
"""

import pandas as pd
import numpy as np
from datetime import time as dtime
import warnings
warnings.filterwarnings("ignore")

def run_simulation(df, timeframe_str="5m"):
    VWAP_MULT   = 2.0
    MIN_BAND_PT = 15.0
    WICK_PCT    = 0.35
    MIN_STOP    = 5.0
    MAX_STOP    = 60.0
    MIN_RR      = 1.5
    STOP_BUF    = 1.5

    trades_a = []  # 1 NQ (No BE, TP1 only)
    trades_b = []  # 5 MNQ (3 MNQ at TP1 + BE on 2 MNQ + TP2 runner)

    for day, day_df in df.groupby("date"):
        day_df = day_df.reset_index(drop=True)
        cum_vol = 0.0; cum_pv = 0.0; cum_pv2 = 0.0
        traded_today = False

        # Active trade state
        # signal info: dir, ep, sl, tp1, tp2
        active = None
        # State for Mode A
        done_a = False
        # State for Mode B
        done_b = False
        tp1_hit_b = False
        pnl_pts_b_tp1 = 0.0

        for _, row in day_df.iterrows():
            t = row["time"]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            v = max(row["volume"], 1.0)
            tp_bar = (h + l + c) / 3.0

            cum_vol += v
            cum_pv  += tp_bar * v
            cum_pv2 += tp_bar * tp_bar * v
            vwap = cum_pv / cum_vol
            stdev = np.sqrt(max(0.0, cum_pv2 / cum_vol - vwap**2))
            upper = vwap + VWAP_MULT * stdev
            lower = vwap - VWAP_MULT * stdev

            # Process active trade
            if active is not None:
                side = active["side"]
                ep   = active["ep"]
                sl   = active["sl"]
                tp1  = active["tp1"]
                tp2  = active["tp2"]

                # ── EVALUATE MODE A (1 NQ: All at TP1, Hard SL, NO BE) ──────
                if not done_a:
                    if side == "short":
                        if l <= tp1:
                            pts = ep - tp1
                            trades_a.append({
                                "date": day, "time": str(t)[:5], "side": side,
                                "entry_p": ep, "exit_p": tp1, "pts": pts,
                                "pnl": pts * 20.0 - 10.20, "result": "WIN"
                            })
                            done_a = True
                        elif h >= sl:
                            pts = ep - sl
                            trades_a.append({
                                "date": day, "time": str(t)[:5], "side": side,
                                "entry_p": ep, "exit_p": sl, "pts": pts,
                                "pnl": pts * 20.0 - 10.20, "result": "LOSS"
                            })
                            done_a = True
                        elif t >= dtime(15, 45):
                            pts = ep - c
                            trades_a.append({
                                "date": day, "time": str(t)[:5], "side": side,
                                "entry_p": ep, "exit_p": c, "pts": pts,
                                "pnl": pts * 20.0 - 10.20, "result": "WIN" if pts > 0 else "LOSS"
                            })
                            done_a = True
                    else: # Long
                        if h >= tp1:
                            pts = tp1 - ep
                            trades_a.append({
                                "date": day, "time": str(t)[:5], "side": side,
                                "entry_p": ep, "exit_p": tp1, "pts": pts,
                                "pnl": pts * 20.0 - 10.20, "result": "WIN"
                            })
                            done_a = True
                        elif l <= sl:
                            pts = sl - ep
                            trades_a.append({
                                "date": day, "time": str(t)[:5], "side": side,
                                "entry_p": ep, "exit_p": sl, "pts": pts,
                                "pnl": pts * 20.0 - 10.20, "result": "LOSS"
                            })
                            done_a = True
                        elif t >= dtime(15, 45):
                            pts = c - ep
                            trades_a.append({
                                "date": day, "time": str(t)[:5], "side": side,
                                "entry_p": ep, "exit_p": c, "pts": pts,
                                "pnl": pts * 20.0 - 10.20, "result": "WIN" if pts > 0 else "LOSS"
                            })
                            done_a = True

                # ── EVALUATE MODE B (5 MNQ: 3 MNQ @ TP1, BE Lock, 2 MNQ @ TP2) 
                if not done_b:
                    if side == "short":
                        # Check if TP1 has not yet been hit
                        if not tp1_hit_b:
                            if l <= tp1:
                                tp1_hit_b = True
                                pnl_pts_b_tp1 = ep - tp1  # 3 contracts banked at TP1
                                # Now check if in the same bar or next bar it also hits TP2 or BE
                                if l <= tp2:
                                    pts2 = ep - tp2
                                    total_pnl = (pnl_pts_b_tp1 * 3 * 2.0) + (pts2 * 2 * 2.0) - 6.20
                                    trades_b.append({
                                        "date": day, "time": str(t)[:5], "side": side,
                                        "entry_p": ep, "pts_tp1": pnl_pts_b_tp1, "pts_tp2": pts2,
                                        "pnl": total_pnl, "result": "WIN", "runner_status": "TP2 HIT"
                                    })
                                    done_b = True
                            elif h >= sl:
                                # Full initial stop hit
                                pts = ep - sl
                                total_pnl = pts * 5 * 2.0 - 6.20
                                trades_b.append({
                                    "date": day, "time": str(t)[:5], "side": side,
                                    "entry_p": ep, "pts_tp1": pts, "pts_tp2": pts,
                                    "pnl": total_pnl, "result": "LOSS", "runner_status": "STOPPED"
                                })
                                done_b = True
                            elif t >= dtime(15, 45):
                                pts = ep - c
                                total_pnl = pts * 5 * 2.0 - 6.20
                                trades_b.append({
                                    "date": day, "time": str(t)[:5], "side": side,
                                    "entry_p": ep, "pts_tp1": pts, "pts_tp2": pts,
                                    "pnl": total_pnl, "result": "WIN" if pts > 0 else "LOSS", "runner_status": "EOD"
                                })
                                done_b = True
                        else:
                            # TP1 was already hit, stop is now at Breakeven (ep - 1.0 pt)
                            be_stop = ep - 1.0
                            if l <= tp2:
                                pts2 = ep - tp2
                                total_pnl = (pnl_pts_b_tp1 * 3 * 2.0) + (pts2 * 2 * 2.0) - 6.20
                                trades_b.append({
                                    "date": day, "time": str(t)[:5], "side": side,
                                    "entry_p": ep, "pts_tp1": pnl_pts_b_tp1, "pts_tp2": pts2,
                                    "pnl": total_pnl, "result": "WIN", "runner_status": "TP2 HIT"
                                })
                                done_b = True
                            elif h >= be_stop:
                                pts2 = 1.0  # BE + 1 pt lock
                                total_pnl = (pnl_pts_b_tp1 * 3 * 2.0) + (pts2 * 2 * 2.0) - 6.20
                                trades_b.append({
                                    "date": day, "time": str(t)[:5], "side": side,
                                    "entry_p": ep, "pts_tp1": pnl_pts_b_tp1, "pts_tp2": pts2,
                                    "pnl": total_pnl, "result": "WIN", "runner_status": "BE LOCK"
                                })
                                done_b = True
                            elif t >= dtime(15, 45):
                                pts2 = ep - c
                                total_pnl = (pnl_pts_b_tp1 * 3 * 2.0) + (pts2 * 2 * 2.0) - 6.20
                                trades_b.append({
                                    "date": day, "time": str(t)[:5], "side": side,
                                    "entry_p": ep, "pts_tp1": pnl_pts_b_tp1, "pts_tp2": pts2,
                                    "pnl": total_pnl, "result": "WIN", "runner_status": "EOD"
                                })
                                done_b = True
                    else: # Long
                        if not tp1_hit_b:
                            if h >= tp1:
                                tp1_hit_b = True
                                pnl_pts_b_tp1 = tp1 - ep
                                if h >= tp2:
                                    pts2 = tp2 - ep
                                    total_pnl = (pnl_pts_b_tp1 * 3 * 2.0) + (pts2 * 2 * 2.0) - 6.20
                                    trades_b.append({
                                        "date": day, "time": str(t)[:5], "side": side,
                                        "entry_p": ep, "pts_tp1": pnl_pts_b_tp1, "pts_tp2": pts2,
                                        "pnl": total_pnl, "result": "WIN", "runner_status": "TP2 HIT"
                                    })
                                    done_b = True
                            elif l <= sl:
                                pts = sl - ep
                                total_pnl = pts * 5 * 2.0 - 6.20
                                trades_b.append({
                                    "date": day, "time": str(t)[:5], "side": side,
                                    "entry_p": ep, "pts_tp1": pts, "pts_tp2": pts,
                                    "pnl": total_pnl, "result": "LOSS", "runner_status": "STOPPED"
                                })
                                done_b = True
                            elif t >= dtime(15, 45):
                                pts = c - ep
                                total_pnl = pts * 5 * 2.0 - 6.20
                                trades_b.append({
                                    "date": day, "time": str(t)[:5], "side": side,
                                    "entry_p": ep, "pts_tp1": pts, "pts_tp2": pts,
                                    "pnl": total_pnl, "result": "WIN" if pts > 0 else "LOSS", "runner_status": "EOD"
                                })
                                done_b = True
                        else:
                            be_stop = ep + 1.0
                            if h >= tp2:
                                pts2 = tp2 - ep
                                total_pnl = (pnl_pts_b_tp1 * 3 * 2.0) + (pts2 * 2 * 2.0) - 6.20
                                trades_b.append({
                                    "date": day, "time": str(t)[:5], "side": side,
                                    "entry_p": ep, "pts_tp1": pnl_pts_b_tp1, "pts_tp2": pts2,
                                    "pnl": total_pnl, "result": "WIN", "runner_status": "TP2 HIT"
                                })
                                done_b = True
                            elif l <= be_stop:
                                pts2 = 1.0
                                total_pnl = (pnl_pts_b_tp1 * 3 * 2.0) + (pts2 * 2 * 2.0) - 6.20
                                trades_b.append({
                                    "date": day, "time": str(t)[:5], "side": side,
                                    "entry_p": ep, "pts_tp1": pnl_pts_b_tp1, "pts_tp2": pts2,
                                    "pnl": total_pnl, "result": "WIN", "runner_status": "BE LOCK"
                                })
                                done_b = True
                            elif t >= dtime(15, 45):
                                pts2 = c - ep
                                total_pnl = (pnl_pts_b_tp1 * 3 * 2.0) + (pts2 * 2 * 2.0) - 6.20
                                trades_b.append({
                                    "date": day, "time": str(t)[:5], "side": side,
                                    "entry_p": ep, "pts_tp1": pnl_pts_b_tp1, "pts_tp2": pts2,
                                    "pnl": total_pnl, "result": "WIN", "runner_status": "EOD"
                                })
                                done_b = True

                if done_a and done_b:
                    active = None
                continue

            # Signal scanning (One per day)
            if traded_today or stdev < 0.5:
                continue
            if not (dtime(9, 30) <= t <= dtime(11, 30)):
                continue

            crange = h - l
            up_w   = h - max(o, c)
            dn_w   = min(o, c) - l

            # SHORT Setup
            if (upper - vwap) >= MIN_BAND_PT and h >= upper and crange >= 6.0 and up_w >= crange * WICK_PCT and c < o:
                ep  = (h + max(o, c)) / 2.0
                sl  = h + STOP_BUF
                rk  = min(max(sl - ep, 1.0), MAX_STOP)
                tp1 = vwap
                rw1 = ep - tp1
                tp2 = lower  # Opposite 2.0σ Band

                if rk >= MIN_STOP and rw1 >= rk * MIN_RR:
                    active = {
                        "date": str(day), "time": str(t)[:5], "side": "short",
                        "ep": ep, "sl": sl, "tp1": tp1, "tp2": tp2,
                        "risk_pts": rk, "reward_pts": rw1
                    }
                    traded_today = True
                    done_a = False
                    done_b = False
                    tp1_hit_b = False

            # LONG Setup
            elif (vwap - lower) >= MIN_BAND_PT and l <= lower and crange >= 6.0 and dn_w >= crange * WICK_PCT and c > o:
                ep  = (l + min(o, c)) / 2.0
                sl  = l - STOP_BUF
                rk  = min(max(ep - sl, 1.0), MAX_STOP)
                tp1 = vwap
                rw1 = tp1 - ep
                tp2 = upper  # Opposite 2.0σ Band

                if rk >= MIN_STOP and rw1 >= rk * MIN_RR:
                    active = {
                        "date": str(day), "time": str(t)[:5], "side": "long",
                        "ep": ep, "sl": sl, "tp1": tp1, "tp2": tp2,
                        "risk_pts": rk, "reward_pts": rw1
                    }
                    traded_today = True
                    done_a = False
                    done_b = False
                    tp1_hit_b = False

    return pd.DataFrame(trades_a), pd.DataFrame(trades_b)

def print_comparison(title, tdf_a, tdf_b):
    print("=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)

    # Mode A metrics
    cnt_a  = len(tdf_a)
    wins_a = tdf_a[tdf_a["result"] == "WIN"] if cnt_a else []
    wr_a   = len(wins_a) / cnt_a * 100 if cnt_a else 0.0
    pnl_a  = tdf_a["pnl"].sum() if cnt_a else 0.0
    eq_a   = tdf_a["pnl"].cumsum() if cnt_a else pd.Series([0])
    dd_a   = (eq_a - eq_a.cummax()).min() if cnt_a else 0.0
    gp_a   = wins_a["pnl"].sum() if len(wins_a) else 0.0
    gl_a   = abs(tdf_a[tdf_a["result"] == "LOSS"]["pnl"].sum()) if cnt_a else 1.0
    pf_a   = gp_a / gl_a if gl_a > 0 else 999.0

    # Mode B metrics
    cnt_b  = len(tdf_b)
    wins_b = tdf_b[tdf_b["result"] == "WIN"] if cnt_b else []
    wr_b   = len(wins_b) / cnt_b * 100 if cnt_b else 0.0
    pnl_b  = tdf_b["pnl"].sum() if cnt_b else 0.0
    eq_b   = tdf_b["pnl"].cumsum() if cnt_b else pd.Series([0])
    dd_b   = (eq_b - eq_b.cummax()).min() if cnt_b else 0.0
    gp_b   = wins_b["pnl"].sum() if len(wins_b) else 0.0
    gl_b   = abs(tdf_b[tdf_b["result"] == "LOSS"]["pnl"].sum()) if cnt_b else 1.0
    pf_b   = gp_b / gl_b if gl_b > 0 else 999.0

    print(f"{'Performance Metric':<28} | {'1 Full NQ (NO BE, TP1 Only)':<28} | {'5 MNQ (TP1 60% + BE + Runner)':<28}")
    print("-" * 88)
    print(f"{'Position Sizing / Leverage':<28} | {'1 NQ ($20/pt)':<28} | {'5 MNQ ($10/pt total)':<28}")
    print(f"{'Total Trades':<28} | {cnt_a:<28} | {cnt_b:<28}")
    print(f"{'Win Rate':<28} | {wr_a:<26.1f}% | {wr_b:<26.1f}%")
    print(f"{'Profit Factor (PF)':<28} | {pf_a:<28.2f} | {pf_b:<28.2f}")
    print("-" * 88)
    print(f"{'NET PROFIT (USD)':<28} | +${pnl_a:<26.2f} | +${pnl_b:<26.2f}")
    print(f"{'MAX DRAWDOWN (USD)':<28} | -${abs(dd_a):<26.2f} | -${abs(dd_b):<26.2f}")
    print("=" * 88)
    if cnt_b > 0 and "runner_status" in tdf_b.columns:
        print("  Runner Status Breakdown (5 MNQ):")
        for status, count in tdf_b["runner_status"].value_counts().items():
            print(f"    - {status}: {count} trades")
    print()

def main():
    # ── Test 1: October 2025 ─────────────────────────────────────────────────
    print("\nLoading October 2025 data...")
    df_12m = pd.read_csv("sample_data/MNQ_1min_12mo.csv")
    df_12m['dt'] = pd.to_datetime(df_12m['Datetime']).dt.tz_convert('America/New_York')
    df_12m['date'] = df_12m['dt'].dt.date
    df_12m['time'] = df_12m['dt'].dt.time
    df_12m['hour'] = df_12m['dt'].dt.hour
    df_12m['minute'] = df_12m['dt'].dt.minute

    # Filter to Oct 2025 RTH
    oct25 = df_12m[(df_12m['dt'].dt.year == 2025) & (df_12m['dt'].dt.month == 10)].copy()
    oct25_rth = oct25[(oct25["hour"] >= 9) & ~((oct25["hour"] == 9) & (oct25["minute"] < 30)) & (oct25["hour"] < 16)].copy()
    
    # Resample 1m to 5m
    oct25_rth = oct25_rth.set_index("dt")
    oct25_5m = oct25_rth.resample("5min", closed="left", label="left").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna().reset_index()
    oct25_5m.columns = [c.lower() for c in oct25_5m.columns]
    oct25_5m['date'] = oct25_5m['dt'].dt.date
    oct25_5m['time'] = oct25_5m['dt'].dt.time

    t_a_25, t_b_25 = run_simulation(oct25_5m)
    print_comparison("OCTOBER 2025 TEST RESULTS", t_a_25, t_b_25)

    # ── Test 2: October 2026 (Real CME Live Feed) ───────────────────────────
    print("Loading October 2026 data...")
    df_60d = pd.read_csv("sample_data/NQ_5min_60days.csv")
    df_60d['dt'] = pd.to_datetime(df_60d['dt'])
    df_60d['date'] = pd.to_datetime(df_60d['date']).dt.date
    df_60d['time'] = df_60d['dt'].dt.time

    oct26 = df_60d[(df_60d['date'] >= pd.to_datetime("2026-10-01").date()) & 
                   (df_60d['date'] <= pd.to_datetime("2026-10-31").date())].copy()

    t_a_26, t_b_26 = run_simulation(oct26)
    print_comparison("OCTOBER 2026 TEST RESULTS", t_a_26, t_b_26)

if __name__ == "__main__":
    main()

