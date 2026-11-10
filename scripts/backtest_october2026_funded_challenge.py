"""
October 2026 — $50k Prop Firm Challenge Simulation
====================================================
Uses REAL Dukascopy 1m NQ data for October 2026.
Runs the SEALED strategy (5m VWAP 2.0sigma, 35% wick, 20pt adaptive entry, 1.5R filter).
Scores the month against FundedNext AND TopStep $50k challenge rules.
Reports day-by-day P&L, what would have been on top (best day, worst day, streaks).
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

# ─── Strategy Parameters (Sealed) ────────────────────────────────────────────
SIGMA        = 2.0
MIN_BAND_PT  = 15.0   # NQ1! futures
WICK_MIN     = 0.35
STOP_BUFFER  = 1.5
MIN_STOP     = 4.0
MAX_STOP     = 60.0
MIN_RR       = 1.5
ADAPT_THRESH = 20.0
MIN_CRANGE   = 6.0

# ─── MNQ Contract Sizing (5 MNQ, $2/pt per contract) ─────────────────────────
# Split: 3 MNQ to TP1 (VWAP), 2 MNQ runner to TP2 (opposite band)
MNQ_TP1 = 3
MNQ_TP2 = 2
TICK_VAL = 2.0   # $2 per point per MNQ

# ─── $50k Prop Firm Rules ─────────────────────────────────────────────────────
ACCOUNT_SIZE   = 50_000.0

# FundedNext Evaluation $50k
FN_PROFIT_TGT  = 3_000.0   # 6% profit target
FN_DAILY_LOSS  = 2_500.0   # 5% max daily loss (from start-of-day balance)
FN_MAX_DD      = 2_500.0   # 5% max overall trailing drawdown

# TopStep $50k
TS_PROFIT_TGT  = 3_000.0   # 6% profit target
TS_DAILY_LOSS  = 2_000.0   # flat $2,000 daily loss limit
TS_MAX_DD      = 2_000.0   # trailing drawdown from peak equity

DATA_PATH = r"c:\Users\JOSHUA\Desktop\CRTbot\data\indices\NQ-duka-2018-2026.json"


def load_october_2026(path):
    print("Loading Dukascopy 1m NQ data...")
    with open(path, "r") as f:
        raw = json.load(f)
    bars = raw if isinstance(raw, list) else raw.get("data", raw.get("bars", []))
    df = pd.DataFrame(bars)

    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    ts_col = next((c for c in df.columns if "time" in c or "ts" in c or "date" in c), df.columns[0])
    df = df.rename(columns={ts_col: "ts"})

    # Parse timestamps
    if df["ts"].dtype in [np.float64, np.int64]:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    else:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

    df["ny_time"] = df["ts"].dt.tz_convert(NY_TZ)

    # October 2026 only
    df = df[(df["ny_time"].dt.year == 2026) & (df["ny_time"].dt.month == 10)].copy()
    df = df.reset_index(drop=True)
    print(f"  October 2026 bars loaded: {len(df):,}")
    return df


def compute_5m_vwap_bands(df_1m_day):
    """Resample 1m to 5m, compute session VWAP anchored at 9:30 AM."""
    df = df_1m_day.set_index("ny_time")
    df_5m = df.resample("5min", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna().reset_index()

    # Anchor at 9:30
    anchor = df_5m[(df_5m["ny_time"].dt.hour == 9) & (df_5m["ny_time"].dt.minute == 30)].index
    if len(anchor) == 0:
        return None
    df_5m = df_5m.loc[anchor[0]:].copy()

    tp = (df_5m["high"] + df_5m["low"] + df_5m["close"]) / 3.0
    vol = df_5m["volume"].replace(0, 1.0)
    cum_vol  = vol.cumsum()
    cum_pv   = (tp * vol).cumsum()
    cum_pv2  = (tp * tp * vol).cumsum()
    vwap     = cum_pv / cum_vol
    variance = np.maximum(0.0, cum_pv2 / cum_vol - vwap**2)
    stdev    = np.sqrt(variance)

    df_5m["vwap"]    = vwap.values
    df_5m["upper_b"] = (vwap + SIGMA * stdev).values
    df_5m["lower_b"] = (vwap - SIGMA * stdev).values
    return df_5m


def resample_3m(df_1m_day):
    df = df_1m_day.set_index("ny_time")
    df_3m = df.resample("3min", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna().reset_index()
    # Only 9:30 onwards
    df_3m = df_3m[
        (df_3m["ny_time"].dt.hour > 9) |
        ((df_3m["ny_time"].dt.hour == 9) & (df_3m["ny_time"].dt.minute >= 30))
    ]
    # Only up to 11:30
    df_3m = df_3m[
        (df_3m["ny_time"].dt.hour < 11) |
        ((df_3m["ny_time"].dt.hour == 11) & (df_3m["ny_time"].dt.minute <= 30))
    ]
    return df_3m.reset_index(drop=True)


def find_signal(df_3m, vwap_v, lower_v, upper_v):
    """Check 3m candles for a band sweep + wick rejection signal."""
    for _, c in df_3m.iterrows():
        crange    = c["high"] - c["low"]
        down_wick = min(c["open"], c["close"]) - c["low"]
        up_wick   = c["high"] - max(c["open"], c["close"])

        if crange < MIN_CRANGE:
            continue

        # LONG
        bw_lower = vwap_v - lower_v
        if (bw_lower >= MIN_BAND_PT and c["low"] <= lower_v and
                c["close"] > lower_v and c["close"] > c["open"] and
                down_wick >= crange * WICK_MIN):
            use_market = crange <= ADAPT_THRESH
            entry = c["close"] if use_market else (c["low"] + 0.50 * down_wick)
            stop  = c["low"] - STOP_BUFFER
            risk  = entry - stop
            if MIN_STOP <= risk <= MAX_STOP and (vwap_v - entry) >= risk * MIN_RR:
                return {
                    "direction": "LONG",
                    "bar_time":  c["ny_time"],
                    "entry":     entry,
                    "stop":      stop,
                    "tp1":       vwap_v,
                    "tp2":       upper_v,
                    "risk":      risk,
                    "market":    use_market,
                }

        # SHORT
        bw_upper = upper_v - vwap_v
        if (bw_upper >= MIN_BAND_PT and c["high"] >= upper_v and
                c["close"] < upper_v and c["close"] < c["open"] and
                up_wick >= crange * WICK_MIN):
            use_market = crange <= ADAPT_THRESH
            entry = c["close"] if use_market else (c["high"] - 0.50 * up_wick)
            stop  = c["high"] + STOP_BUFFER
            risk  = stop - entry
            if MIN_STOP <= risk <= MAX_STOP and (entry - vwap_v) >= risk * MIN_RR:
                return {
                    "direction": "SHORT",
                    "bar_time":  c["ny_time"],
                    "entry":     entry,
                    "stop":      stop,
                    "tp1":       vwap_v,
                    "tp2":       lower_v,
                    "risk":      risk,
                    "market":    use_market,
                }
    return None


def simulate_pnl(sig, df_1m_day):
    """
    Simulate the trade outcome using 1m data after the signal bar.
    Returns (tp1_hit, tp2_hit, sl_hit, pnl_dollars).
    Scale: 3 MNQ to TP1, 2 MNQ runner to TP2.
    """
    entry    = sig["entry"]
    stop     = sig["stop"]
    tp1      = sig["tp1"]
    tp2      = sig["tp2"]
    d        = sig["direction"]
    sig_time = sig["bar_time"]

    after = df_1m_day[df_1m_day["ny_time"] > sig_time].copy()
    after = after[after["ny_time"].dt.hour <= 16]  # close at 4 PM NY

    tp1_hit = tp2_hit = sl_hit = False
    remaining_mnq = MNQ_TP1 + MNQ_TP2  # 5 total
    pnl = 0.0

    for _, bar in after.iterrows():
        if sl_hit:
            break

        if d == "LONG":
            # Check TP1
            if not tp1_hit and bar["high"] >= tp1:
                pts = tp1 - entry
                pnl += pts * MNQ_TP1 * TICK_VAL
                tp1_hit = True
                remaining_mnq = MNQ_TP2
                # Move stop to breakeven after TP1
                stop = entry + 1.0

            # Check TP2 (runner)
            if tp1_hit and bar["high"] >= tp2:
                pts = tp2 - entry
                pnl += pts * MNQ_TP2 * TICK_VAL
                tp2_hit = True
                break

            # Check stop
            if bar["low"] <= stop:
                if tp1_hit:
                    pts = stop - entry  # BE stop, should be ~0 or +1
                    pnl += pts * MNQ_TP2 * TICK_VAL
                else:
                    pts = stop - entry  # negative
                    pnl += pts * (MNQ_TP1 + MNQ_TP2) * TICK_VAL
                sl_hit = True

        else:  # SHORT
            if not tp1_hit and bar["low"] <= tp1:
                pts = entry - tp1
                pnl += pts * MNQ_TP1 * TICK_VAL
                tp1_hit = True
                remaining_mnq = MNQ_TP2
                stop = entry - 1.0  # BE

            if tp1_hit and bar["low"] <= tp2:
                pts = entry - tp2
                pnl += pts * MNQ_TP2 * TICK_VAL
                tp2_hit = True
                break

            if bar["high"] >= stop:
                if tp1_hit:
                    pts = entry - stop
                    pnl += pts * MNQ_TP2 * TICK_VAL
                else:
                    pts = entry - stop
                    pnl += pts * (MNQ_TP1 + MNQ_TP2) * TICK_VAL
                sl_hit = True

    # If market closed without hitting any target, close at last price
    if not tp1_hit and not sl_hit and len(after) > 0:
        last_price = after.iloc[-1]["close"]
        if d == "LONG":
            pts = last_price - entry
        else:
            pts = entry - last_price
        pnl += pts * (MNQ_TP1 + MNQ_TP2) * TICK_VAL
        sl_hit = pts < 0

    return tp1_hit, tp2_hit, sl_hit, round(pnl, 2)


def run_october_simulation():
    df = load_october_2026(DATA_PATH)
    if len(df) == 0:
        print("ERROR: No October 2026 data found in the file.")
        return

    trading_days = sorted(df["ny_time"].dt.date.unique())
    # Filter weekdays
    trading_days = [d for d in trading_days if datetime(d.year, d.month, d.day).weekday() < 5]
    print(f"  Trading days in October 2026: {len(trading_days)}")

    trades = []
    daily_results = []

    for day in trading_days:
        df_day = df[df["ny_time"].dt.date == day].copy()
        df_5m  = compute_5m_vwap_bands(df_day)
        if df_5m is None or len(df_5m) < 2:
            continue
        df_3m  = resample_3m(df_day)

        # Get VWAP levels (use the bar right before the first signal window)
        last_5m  = df_5m.iloc[-1]
        vwap_v   = float(last_5m["vwap"])
        lower_v  = float(last_5m["lower_b"])
        upper_v  = float(last_5m["upper_b"])

        # Use mid-session 5m values (more stable than very last bar)
        mid = df_5m.iloc[min(3, len(df_5m)-1)]
        vwap_v  = float(mid["vwap"])
        lower_v = float(mid["lower_b"])
        upper_v = float(mid["upper_b"])

        sig = find_signal(df_3m, vwap_v, lower_v, upper_v)
        if sig is None:
            daily_results.append({"date": day, "pnl": 0.0, "outcome": "NO SIGNAL"})
            continue

        tp1_hit, tp2_hit, sl_hit, pnl = simulate_pnl(sig, df_day)

        if tp2_hit:
            outcome = "FULL WIN (TP1 + TP2)"
        elif tp1_hit:
            outcome = "PARTIAL WIN (TP1 only, runner BE)"
        elif sl_hit:
            outcome = "LOSS (SL Hit)"
        else:
            outcome = "FLAT (time exit)"

        entry_type = "MARKET" if sig["market"] else "LIMIT"

        trades.append({
            "date":       day,
            "direction":  sig["direction"],
            "entry_type": entry_type,
            "entry":      sig["entry"],
            "stop":       sig["stop"],
            "tp1":        sig["tp1"],
            "tp2":        sig["tp2"],
            "risk_pts":   sig["risk"],
            "outcome":    outcome,
            "pnl":        pnl,
        })
        daily_results.append({"date": day, "pnl": pnl, "outcome": outcome})

    # ─── RESULTS ──────────────────────────────────────────────────────────────
    df_trades = pd.DataFrame(trades)
    df_daily  = pd.DataFrame(daily_results)

    total_pnl  = df_daily["pnl"].sum()
    wins       = df_trades[df_trades["pnl"] > 0]
    losses     = df_trades[df_trades["pnl"] <= 0]
    win_rate   = len(wins) / len(df_trades) * 100 if len(df_trades) > 0 else 0
    gross_win  = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    pf         = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Equity curve and drawdown
    equity     = ACCOUNT_SIZE + df_daily["pnl"].cumsum()
    peak_eq    = equity.cummax()
    drawdown   = peak_eq - equity
    max_dd     = drawdown.max()
    max_daily_loss = df_daily["pnl"].min()

    # Best and worst days
    best_day  = df_daily.loc[df_daily["pnl"].idxmax()]
    worst_day = df_daily.loc[df_daily["pnl"].idxmin()]

    # Win/loss streaks
    pnl_signs   = [1 if p > 0 else -1 if p < 0 else 0 for p in df_daily["pnl"] if p != 0]
    max_wstreak = max_lstreak = cur_w = cur_l = 0
    for s in pnl_signs:
        if s == 1:
            cur_w += 1; cur_l = 0
        elif s == -1:
            cur_l += 1; cur_w = 0
        max_wstreak = max(max_wstreak, cur_w)
        max_lstreak = max(max_lstreak, cur_l)

    # ─── FundedNext Challenge Check ───────────────────────────────────────────
    fn_pass = (total_pnl >= FN_PROFIT_TGT and
               max_dd    <= FN_MAX_DD and
               abs(max_daily_loss) <= FN_DAILY_LOSS)

    ts_pass = (total_pnl >= TS_PROFIT_TGT and
               max_dd    <= TS_MAX_DD and
               abs(max_daily_loss) <= TS_DAILY_LOSS)

    # --- PRINT REPORT ---
    print("\n" + "="*65)
    print("  OCTOBER 2026 - NQ 9:30 VWAP SNIPER BACKTEST")
    print("  (Real Dukascopy 1m data | 5m Zone + 3m Trigger | Sealed Strategy)")
    print("="*65)

    print("\n" + "-"*65)
    print("  MONTH SUMMARY")
    print("-"*65)
    print(f"  Trading Days      : {len(trading_days)}")
    print(f"  Signals Generated : {len(df_trades)}")
    print(f"  No-Signal Days    : {len(trading_days) - len(df_trades)}")
    print(f"  Win Rate          : {win_rate:.1f}%")
    print(f"  Profit Factor     : {pf:.2f}")
    print(f"  Gross Profit      : ${gross_win:,.2f}")
    print(f"  Gross Loss        : ${gross_loss:,.2f}")
    print(f"  NET P&L (October) : ${total_pnl:,.2f}")
    print(f"  Peak Equity       : ${(ACCOUNT_SIZE + total_pnl):,.2f}")

    print("\n" + "-"*65)
    print("  RISK METRICS")
    print("-"*65)
    print(f"  Max Drawdown      : ${max_dd:,.2f}")
    print(f"  Worst Single Day  : ${max_daily_loss:,.2f}  ({worst_day['date']})")
    print(f"  Best Single Day   : ${df_daily['pnl'].max():,.2f}  ({best_day['date']})")
    print(f"  Max Win Streak    : {max_wstreak} days")
    print(f"  Max Loss Streak   : {max_lstreak} days")

    print("\n" + "-"*65)
    print("  PROP FIRM CHALLENGE VERDICT")
    print("-"*65)
    fn_icon = "PASS" if fn_pass else "FAIL"
    ts_icon = "PASS" if ts_pass else "FAIL"
    print(f"\n  FundedNext $50k Challenge: [{fn_icon}]")
    print(f"    Profit Target $3,000  : ${total_pnl:,.2f}  {'OK' if total_pnl >= FN_PROFIT_TGT else 'NOT MET'}")
    print(f"    Max Daily Loss $2,500 : ${abs(max_daily_loss):,.2f}  {'OK' if abs(max_daily_loss) <= FN_DAILY_LOSS else 'VIOLATED'}")
    print(f"    Max Drawdown  $2,500  : ${max_dd:,.2f}  {'OK' if max_dd <= FN_MAX_DD else 'VIOLATED'}")

    print(f"\n  TopStep $50k Challenge: [{ts_icon}]")
    print(f"    Profit Target $3,000  : ${total_pnl:,.2f}  {'OK' if total_pnl >= TS_PROFIT_TGT else 'NOT MET'}")
    print(f"    Max Daily Loss $2,000 : ${abs(max_daily_loss):,.2f}  {'OK' if abs(max_daily_loss) <= TS_DAILY_LOSS else 'VIOLATED'}")
    print(f"    Max Drawdown  $2,000  : ${max_dd:,.2f}  {'OK' if max_dd <= TS_MAX_DD else 'VIOLATED'}")

    print("\n" + "-"*65)
    print("  DAY-BY-DAY BREAKDOWN")
    print("-"*65)
    running_pnl = 0.0
    for _, row in df_daily.iterrows():
        running_pnl += row["pnl"]
        marker = ""
        if row["pnl"] > 500:
            marker = " <-- BIG WIN"
        elif row["pnl"] < -500:
            marker = " <-- BIG LOSS"
        pnl_str = f"+${row['pnl']:,.0f}" if row['pnl'] >= 0 else f"-${abs(row['pnl']):,.0f}"
        eq_str  = f"${ACCOUNT_SIZE + running_pnl:,.0f}"
        print(f"  {row['date']}  {pnl_str:>10}   Equity: {eq_str:>10}   {row['outcome']}{marker}")

    print("\n" + "-"*65)
    print("  TRADE LOG")
    print("-"*65)
    for _, t in df_trades.iterrows():
        entry_str = f"${t['entry']:,.2f}"
        pnl_str   = f"+${t['pnl']:,.0f}" if t["pnl"] >= 0 else f"-${abs(t['pnl']):,.0f}"
        print(f"  {t['date']}  {t['direction']:5}  {t['entry_type']:8}  "
              f"Entry:{entry_str}  Risk:{t['risk_pts']:.1f}pts  "
              f"{pnl_str:>8}  {t['outcome']}")

    print("\n" + "="*65)


if __name__ == "__main__":
    run_october_simulation()

