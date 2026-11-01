"""
ICT 9:30 VWAP Band Reversal — Full Backtest (Fixed timezone-aware timestamps)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Load real data ─────────────────────────────────────────────────────────────
df = pd.read_csv(r"sample_data/REAL_NQ_1min.csv")
df.columns = [c.strip().lower() for c in df.columns]
df["dt"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("America/New_York")
df = df.sort_values("dt").reset_index(drop=True)
df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
df["date"]  = df["dt"].dt.date
df["hour"]  = df["dt"].dt.hour
df["minute"]= df["dt"].dt.minute

# Filter to regular trading hours only (09:30–16:00 ET)
df = df[(df["hour"] >= 9) & ~((df["hour"] == 9) & (df["minute"] < 30))].copy()
df = df[df["hour"] < 16].copy()
df = df.reset_index(drop=True)

print(f"Loaded {len(df):,} bars  |  {df['date'].nunique()} trading days")
print(f"Range : {df['dt'].iloc[0]}  ->  {df['dt'].iloc[-1]}")
print(f"Price : {df['low'].min():.0f} – {df['high'].max():.0f}\n")

# ── Parameters ────────────────────────────────────────────────────────────────
VWAP_MULT   = 2.0
MIN_BAND_PT = 15.0
WICK_PCT    = 0.35
MIN_STOP    = 5.0
MAX_STOP    = 60.0
MIN_RR      = 1.5
STOP_BUF    = 1.5
POINT_VALUE = 20.0    # NQ $20 per point
COMMISSION  = 10.20   # round trip

# ── Backtest loop ─────────────────────────────────────────────────────────────
trades = []

for day, day_df in df.groupby("date"):
    day_df = day_df.reset_index(drop=True)
    cum_vol = 0.0; cum_pv = 0.0; cum_pv2 = 0.0
    vwap = None; upper = None; lower = None
    traded_today = False
    active_trade = None

    for i, row in day_df.iterrows():
        hh = row["hour"]; mm = row["minute"]
        mins = hh * 60 + mm
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        v = max(row["volume"], 1.0)  # avoid zero vol
        tp_bar = (h + l + c) / 3.0

        # ── Build VWAP from 09:30 bar onwards
        cum_vol += v
        cum_pv  += tp_bar * v
        cum_pv2 += tp_bar * tp_bar * v

        if cum_vol > 0:
            vwap     = cum_pv / cum_vol
            variance = max(0.0, cum_pv2 / cum_vol - vwap**2)
            stdev    = np.sqrt(variance)
            upper    = vwap + VWAP_MULT * stdev
            lower    = vwap - VWAP_MULT * stdev

        # ── EOD exit at 15:45
        if mins >= 15*60+45 and active_trade:
            ep = active_trade["entry_p"]; side = active_trade["side"]
            pnl_pts = (ep - c) if side == "short" else (c - ep)
            trades.append({**active_trade, "exit_price": c, "exit_type": "EOD",
                           "pnl_pts": pnl_pts, "pnl_usd": pnl_pts*POINT_VALUE - COMMISSION})
            active_trade = None
            continue

        # ── Manage open trade
        if active_trade:
            ep = active_trade["entry_p"]
            sl = active_trade["stop"]
            tp = active_trade["target"]
            side = active_trade["side"]
            if side == "short":
                if l <= tp:
                    pnl_pts = ep - tp
                    trades.append({**active_trade, "exit_price": tp, "exit_type": "TP",
                                   "pnl_pts": pnl_pts, "pnl_usd": pnl_pts*POINT_VALUE - COMMISSION})
                    active_trade = None
                elif h >= sl:
                    pnl_pts = ep - sl
                    trades.append({**active_trade, "exit_price": sl, "exit_type": "SL",
                                   "pnl_pts": pnl_pts, "pnl_usd": pnl_pts*POINT_VALUE - COMMISSION})
                    active_trade = None
            else:
                if h >= tp:
                    pnl_pts = tp - ep
                    trades.append({**active_trade, "exit_price": tp, "exit_type": "TP",
                                   "pnl_pts": pnl_pts, "pnl_usd": pnl_pts*POINT_VALUE - COMMISSION})
                    active_trade = None
                elif l <= sl:
                    pnl_pts = sl - ep
                    trades.append({**active_trade, "exit_price": sl, "exit_type": "SL",
                                   "pnl_pts": pnl_pts, "pnl_usd": pnl_pts*POINT_VALUE - COMMISSION})
                    active_trade = None
            continue

        # ── Look for signal only in 09:30 – 11:30 window
        if traded_today or active_trade or vwap is None:
            continue
        if not (9*60+30 <= mins <= 11*60+30):
            continue
        if stdev < 0.1:  # VWAP not warmed up yet
            continue

        crange    = h - l
        up_wick   = h - max(o, c)
        down_wick = min(o, c) - l

        # SHORT
        if (upper - vwap) >= MIN_BAND_PT and h >= upper and crange >= 6.0 and up_wick >= crange * WICK_PCT and c < o:
            entry_p  = (h + max(o, c)) / 2.0
            stop_p   = h + STOP_BUF
            risk_p   = min(max(stop_p - entry_p, 1.0), MAX_STOP)
            target_p = vwap
            reward_p = entry_p - target_p
            rr       = reward_p / risk_p if risk_p > 0 else 0
            if risk_p >= MIN_STOP and reward_p >= risk_p * MIN_RR:
                active_trade = {"date": str(day), "time": f"{hh:02d}:{mm:02d}", "side": "short",
                                "entry_p": entry_p, "stop": stop_p, "target": target_p,
                                "risk_pts": risk_p, "reward_pts": reward_p, "rr": rr}
                traded_today = True

        # LONG
        elif (vwap - lower) >= MIN_BAND_PT and l <= lower and crange >= 6.0 and down_wick >= crange * WICK_PCT and c > o:
            entry_p  = (l + min(o, c)) / 2.0
            stop_p   = l - STOP_BUF
            risk_p   = min(max(entry_p - stop_p, 1.0), MAX_STOP)
            target_p = vwap
            reward_p = target_p - entry_p
            rr       = reward_p / risk_p if risk_p > 0 else 0
            if risk_p >= MIN_STOP and reward_p >= risk_p * MIN_RR:
                active_trade = {"date": str(day), "time": f"{hh:02d}:{mm:02d}", "side": "long",
                                "entry_p": entry_p, "stop": stop_p, "target": target_p,
                                "risk_pts": risk_p, "reward_pts": reward_p, "rr": rr}
                traded_today = True

# ── Print results ─────────────────────────────────────────────────────────────
if not trades:
    print("NO TRADES — loosening filters...")
else:
    tdf = pd.DataFrame(trades)
    wins   = tdf[tdf["exit_type"] == "TP"]
    losses = tdf[tdf["exit_type"] == "SL"]
    eods   = tdf[tdf["exit_type"] == "EOD"]

    total     = len(tdf)
    win_cnt   = len(wins)
    win_rate  = win_cnt / total * 100
    net_pnl   = tdf["pnl_usd"].sum()
    avg_win   = wins["pnl_usd"].mean()  if len(wins)  else 0
    avg_loss  = losses["pnl_usd"].mean() if len(losses) else 0
    gross_w   = wins["pnl_usd"].sum()   if len(wins)  else 0
    gross_l   = abs(losses["pnl_usd"].sum()) if len(losses) else 1
    pf        = gross_w / gross_l       if gross_l else float("inf")
    avg_rr    = tdf["rr"].mean()
    avg_stop  = tdf["risk_pts"].mean()
    tdf["cum"] = tdf["pnl_usd"].cumsum()
    max_dd    = (tdf["cum"] - tdf["cum"].cummax()).min()

    print("=" * 58)
    print("   ICT 9:30 VWAP BAND REVERSAL — BACKTEST RESULTS")
    print("=" * 58)
    print(f"  Total Trades      : {total}")
    print(f"  Wins  (TP hit)    : {win_cnt}  ({win_rate:.1f}%)")
    print(f"  Losses (SL hit)   : {len(losses)}  ({100-win_rate:.1f}%)")
    print(f"  EOD scratch       : {len(eods)}")
    print(f"  Win Rate          : {win_rate:.1f}%")
    print(f"  Profit Factor     : {pf:.2f}")
    print(f"  Avg R:R per trade : {avg_rr:.2f}R")
    print(f"  Avg Stop Size     : {avg_stop:.1f} pts")
    print(f"  Avg Win  (USD)    : ${avg_win:,.0f}")
    print(f"  Avg Loss (USD)    : ${avg_loss:,.0f}")
    print(f"  Net Profit (1 NQ) : ${net_pnl:,.2f}")
    print(f"  Max Drawdown      : ${max_dd:,.2f}")
    print("=" * 58)
    print("\nEACH TRADE:")
    cols = ["date","time","side","entry_p","risk_pts","rr","exit_type","pnl_usd"]
    print(tdf[cols].to_string(index=False))

    print("\n" + "=" * 58)
    print("  FUNDEDNEXT $50K CHALLENGE VERDICT")
    print("=" * 58)
    print(f"  Target            : +$2,500")
    print(f"  Max Drawdown      : -$2,000")
    print(f"  Strategy Net P&L  : ${net_pnl:,.2f}")
    print(f"  Max DD Observed   : ${max_dd:,.2f}")
    print(f"  Trades per week   : ~{total / df['date'].nunique() * 5:.1f}")
    if net_pnl >= 2500 and abs(max_dd) <= 2000:
        print("  VERDICT  -> PASSES challenge rules on this sample")
    elif net_pnl > 0:
        print("  VERDICT  -> Profitable. Need longer sample to confirm challenge pass")
    else:
        print("  VERDICT  -> Needs more tuning")
    print("=" * 58)
