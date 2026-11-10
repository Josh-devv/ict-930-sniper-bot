"""
2-MONTH UNSEEN OUT-OF-SAMPLE TEST: 3-MINUTE vs 5-MINUTE
Dataset: CME MNQ/NQ Futures Data (Feb 1, 2026 to Mar 31, 2026)
Both 3M and 5M tested using the exact ICT 9:30 Sniper logic.
Outputs full breakdown for both 5 MNQ ($10/pt) and 1 Full NQ ($20/pt).
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
import warnings
warnings.filterwarnings("ignore")

POINT_VAL_NQ   = 20.0
POINT_VAL_MNQ  = 10.0   # 5 MNQ contracts = $10/pt
COMM_NQ        = 10.20  # Round turn 1 NQ
COMM_MNQ       = 6.20   # Round turn 5 MNQ ($1.24 * 5)

def load_data(filepath):
    df = pd.read_csv(filepath)
    dt_col = 'datetime' if 'datetime' in df.columns else 'Datetime'
    df['dt'] = pd.to_datetime(df[dt_col])
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    df = df.sort_values('dt').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        found = [c for c in df.columns if c.lower() == col]
        if found:
            df[col] = df[found[0]].astype(float)
    return df[['dt', 'open', 'high', 'low', 'close', 'volume']].set_index('dt')

def run_backtest(df_1m, start_date, end_date, tf_mins=3):
    rule = f"{tf_mins}min"
    resampled = df_1m.resample(rule, closed="left", label="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()

    resampled["date"]   = resampled["dt"].dt.date
    resampled["time"]   = resampled["dt"].dt.time
    resampled["hour"]   = resampled["dt"].dt.hour
    resampled["minute"] = resampled["dt"].dt.minute

    # Filter to regular session 09:30 to 16:00
    df = resampled[(resampled["hour"] >= 9) & ~((resampled["hour"] == 9) & (resampled["minute"] < 30))].copy()
    df = df[df["hour"] < 16].copy()
    
    s_dt = pd.to_datetime(start_date).date()
    e_dt = pd.to_datetime(end_date).date()
    df = df[(df["date"] >= s_dt) & (df["date"] <= e_dt)].copy()
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
        traded_today = False
        active_trade = None

        for _, row in day_df.iterrows():
            t = row["time"]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            v = max(row["volume"], 1.0)
            tp_bar = (h + l + c) / 3.0

            cum_vol += v
            cum_pv  += tp_bar * v
            cum_pv2 += tp_bar * tp_bar * v
            vwap    = cum_pv / cum_vol
            var_    = max(0.0, cum_pv2 / cum_vol - vwap**2)
            stdev   = np.sqrt(var_)
            upper   = vwap + VWAP_MULT * stdev
            lower   = vwap - VWAP_MULT * stdev

            # EOD exit at 15:45
            if t >= EOD_TIME and active_trade:
                ep = active_trade["entry_p"]
                side = active_trade["side"]
                pts = (ep - c) if side == "short" else (c - ep)
                trades.append({
                    **active_trade, "exit_p": c, "exit_type": "EOD", "pts": pts,
                    "pnl_mnq": pts * POINT_VAL_MNQ - COMM_MNQ,
                    "pnl_nq": pts * POINT_VAL_NQ - COMM_NQ,
                    "result": "WIN" if pts > 0 else "LOSS"
                })
                active_trade = None
                continue

            # Active trade evaluation
            if active_trade:
                ep = active_trade["entry_p"]
                sl = active_trade["stop"]
                tp = active_trade["target"]
                side = active_trade["side"]

                if side == "short":
                    if l <= tp:
                        pts = ep - tp
                        trades.append({
                            **active_trade, "exit_p": tp, "exit_type": "TP", "pts": pts,
                            "pnl_mnq": pts * POINT_VAL_MNQ - COMM_MNQ,
                            "pnl_nq": pts * POINT_VAL_NQ - COMM_NQ,
                            "result": "WIN"
                        })
                        active_trade = None
                    elif h >= sl:
                        pts = ep - sl
                        trades.append({
                            **active_trade, "exit_p": sl, "exit_type": "SL", "pts": pts,
                            "pnl_mnq": pts * POINT_VAL_MNQ - COMM_MNQ,
                            "pnl_nq": pts * POINT_VAL_NQ - COMM_NQ,
                            "result": "LOSS"
                        })
                        active_trade = None
                else: # long
                    if h >= tp:
                        pts = tp - ep
                        trades.append({
                            **active_trade, "exit_p": tp, "exit_type": "TP", "pts": pts,
                            "pnl_mnq": pts * POINT_VAL_MNQ - COMM_MNQ,
                            "pnl_nq": pts * POINT_VAL_NQ - COMM_NQ,
                            "result": "WIN"
                        })
                        active_trade = None
                    elif l <= sl:
                        pts = sl - ep
                        trades.append({
                            **active_trade, "exit_p": sl, "exit_type": "SL", "pts": pts,
                            "pnl_mnq": pts * POINT_VAL_MNQ - COMM_MNQ,
                            "pnl_nq": pts * POINT_VAL_NQ - COMM_NQ,
                            "result": "LOSS"
                        })
                        active_trade = None
                continue

            # Signal scanning
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
                    active_trade = {
                        "date": str(day), "time": str(t)[:5], "side": "short",
                        "entry_p": ep, "stop": sl, "target": tgt,
                        "risk_pts": rk, "reward_pts": rw, "rr": rr
                    }
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
                    active_trade = {
                        "date": str(day), "time": str(t)[:5], "side": "long",
                        "entry_p": ep, "stop": sl, "target": tgt,
                        "risk_pts": rk, "reward_pts": rw, "rr": rr
                    }
                    traded_today = True

    return pd.DataFrame(trades)

def get_stats(tdf, pnl_col="pnl_mnq"):
    if len(tdf) == 0:
        return {"count": 0, "wins": 0, "losses": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "dd": 0.0, "avg_stop": 0.0, "avg_r": 0.0}

    wins = tdf[tdf["result"] == "WIN"]
    losses = tdf[tdf["result"] == "LOSS"]

    gp = wins[pnl_col].sum()
    gl = abs(losses[pnl_col].sum())
    pf = gp / gl if gl > 0 else 999.0

    eq = tdf[pnl_col].cumsum()
    dd = (eq - eq.cummax()).min()

    return {
        "count": len(tdf),
        "wins": len(wins),
        "losses": len(losses),
        "wr": len(wins) / len(tdf) * 100.0,
        "pf": pf,
        "pnl": tdf[pnl_col].sum(),
        "dd": dd,
        "avg_stop": tdf["risk_pts"].mean(),
        "avg_r": wins["rr"].mean() if len(wins) > 0 else 0.0
    }

def main():
    data_file = "sample_data/MNQ_1min_20260120_20260415.csv"
    df_1m = load_data(data_file)

    periods = [
        ("Month 1: February 2026", "2026-02-01", "2026-02-28"),
        ("Month 2: March 2026",    "2026-03-01", "2026-03-31"),
        ("2-Month Combined (Feb + Mar 2026)", "2026-02-01", "2026-03-31")
    ]

    for title, s, e in periods:
        print("=" * 80)
        print(f" {title.upper()} ({s} to {e})")
        print("=" * 80)

        t3 = run_backtest(df_1m, s, e, tf_mins=3)
        t5 = run_backtest(df_1m, s, e, tf_mins=5)

        s3_mnq = get_stats(t3, "pnl_mnq")
        s5_mnq = get_stats(t5, "pnl_mnq")
        s3_nq  = get_stats(t3, "pnl_nq")
        s5_nq  = get_stats(t5, "pnl_nq")

        print(f"{'Performance Metric':<28} | {'3-Minute Timeframe':<22} | {'5-Minute Timeframe':<22}")
        print("-" * 80)
        print(f"{'Total Trades':<28} | {s3_mnq['count']:<22} | {s5_mnq['count']:<22}")
        print(f"{'Win Rate':<28} | {s3_mnq['wr']:<20.1f}% | {s5_mnq['wr']:<20.1f}%")
        print(f"{'Wins / Losses':<28} | {str(s3_mnq['wins'])+'W / '+str(s3_mnq['losses'])+'L':<22} | {str(s5_mnq['wins'])+'W / '+str(s5_mnq['losses'])+'L':<22}")
        print(f"{'Average Stop Loss Size':<28} | {s3_mnq['avg_stop']:<18.1f} pts | {s5_mnq['avg_stop']:<18.1f} pts")
        print(f"{'Avg Risk on 5 MNQ ($10/pt)':<28} | ${s3_mnq['avg_stop']*10:<18.2f} | ${s5_mnq['avg_stop']*10:<18.2f}")
        print(f"{'Avg Winner R:R':<28} | {s3_mnq['avg_r']:<19.2f}R | {s5_mnq['avg_r']:<19.2f}R")
        print(f"{'Profit Factor':<28} | {s3_mnq['pf']:<22.2f} | {s5_mnq['pf']:<22.2f}")
        print("-" * 80)
        print(f"{'NET PROFIT (5 MNQ)':<28} | +${s3_mnq['pnl']:<20.2f} | +${s5_mnq['pnl']:<20.2f}")
        print(f"{'MAX DRAWDOWN (5 MNQ)':<28} | -${abs(s3_mnq['dd']):<20.2f} | -${abs(s5_mnq['dd']):<20.2f}")
        print("-" * 80)
        print(f"{'NET PROFIT (1 Full NQ)':<28} | +${s3_nq['pnl']:<20.2f} | +${s5_nq['pnl']:<20.2f}")
        print(f"{'MAX DRAWDOWN (1 Full NQ)':<28} | -${abs(s3_nq['dd']):<20.2f} | -${abs(s5_nq['dd']):<20.2f}")
        print("\n")

if __name__ == "__main__":
    main()

