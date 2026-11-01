"""
ICT 9:30 VWAP Band Reversal — Full Backtest on 60-day 5min NQ data
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("sample_data/NQ_5min_60days.csv")
df["dt"]     = pd.to_datetime(df["dt"], utc=True).dt.tz_convert("America/New_York")
df["hour"]   = df["dt"].dt.hour
df["minute"] = df["dt"].dt.minute
df["date"]   = df["dt"].dt.date
df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1.0)
df = df.sort_values("dt").reset_index(drop=True)

print(f"Loaded {len(df):,} bars  |  {df['date'].nunique()} trading days")
print(f"Range : {df['dt'].iloc[0]}  ->  {df['dt'].iloc[-1]}")
print(f"Price : {df['low'].min():.0f} – {df['high'].max():.0f}\n")

# ── Parameters ────────────────────────────────────────────────────────────────
VWAP_MULT   = 2.0
MIN_BAND_PT = 15.0
WICK_PCT    = 0.35
MIN_STOP    = 5.0
MAX_STOP    = 70.0
MIN_RR      = 1.5
STOP_BUF    = 2.0
POINT_VALUE = 20.0
COMMISSION  = 10.20   # round trip 1 NQ

# ── Backtest ──────────────────────────────────────────────────────────────────
trades = []

for day, day_df in df.groupby("date"):
    day_df = day_df.reset_index(drop=True)
    cum_vol=0.0; cum_pv=0.0; cum_pv2=0.0
    vwap=None; upper=None; lower=None; stdev=0.0
    traded_today=False; active_trade=None

    for _, row in day_df.iterrows():
        hh=row["hour"]; mm=row["minute"]
        mins = hh*60+mm
        o=row["open"]; h=row["high"]; l=row["low"]; c=row["close"]
        v=max(row["volume"],1.0)
        tp_bar=(h+l+c)/3.0

        # Accumulate VWAP
        cum_vol+=v; cum_pv+=tp_bar*v; cum_pv2+=tp_bar*tp_bar*v
        vwap    = cum_pv/cum_vol
        var_    = max(0.0, cum_pv2/cum_vol - vwap**2)
        stdev   = np.sqrt(var_)
        upper   = vwap + VWAP_MULT*stdev
        lower   = vwap - VWAP_MULT*stdev

        # EOD exit
        if mins >= 15*60+45 and active_trade:
            ep=active_trade["entry_p"]; side=active_trade["side"]
            pnl_pts=(ep-c) if side=="short" else (c-ep)
            trades.append({**active_trade,"exit_price":c,"exit_type":"EOD",
                           "pnl_pts":pnl_pts,"pnl_usd":pnl_pts*POINT_VALUE-COMMISSION})
            active_trade=None; continue

        # Manage open trade
        if active_trade:
            ep=active_trade["entry_p"]; sl=active_trade["stop"]; tp=active_trade["target"]
            side=active_trade["side"]
            if side=="short":
                if l<=tp:
                    pnl_pts=ep-tp
                    trades.append({**active_trade,"exit_price":tp,"exit_type":"TP",
                                   "pnl_pts":pnl_pts,"pnl_usd":pnl_pts*POINT_VALUE-COMMISSION})
                    active_trade=None
                elif h>=sl:
                    pnl_pts=ep-sl
                    trades.append({**active_trade,"exit_price":sl,"exit_type":"SL",
                                   "pnl_pts":pnl_pts,"pnl_usd":pnl_pts*POINT_VALUE-COMMISSION})
                    active_trade=None
            else:
                if h>=tp:
                    pnl_pts=tp-ep
                    trades.append({**active_trade,"exit_price":tp,"exit_type":"TP",
                                   "pnl_pts":pnl_pts,"pnl_usd":pnl_pts*POINT_VALUE-COMMISSION})
                    active_trade=None
                elif l<=sl:
                    pnl_pts=sl-ep
                    trades.append({**active_trade,"exit_price":sl,"exit_type":"SL",
                                   "pnl_pts":pnl_pts,"pnl_usd":pnl_pts*POINT_VALUE-COMMISSION})
                    active_trade=None
            continue

        if traded_today or stdev<0.5: continue
        if not (9*60+30 <= mins <= 11*60+30): continue

        crange=h-l; up_w=h-max(o,c); dn_w=min(o,c)-l

        # SHORT
        if (upper-vwap)>=MIN_BAND_PT and h>=upper and crange>=8.0 and up_w>=crange*WICK_PCT and c<o:
            ep=( h+max(o,c))/2.0; sl=h+STOP_BUF
            rk=min(max(sl-ep,1.0),MAX_STOP); tgt=vwap; rw=ep-tgt; rr=rw/rk if rk>0 else 0
            if rk>=MIN_STOP and rw>=rk*MIN_RR:
                active_trade={"date":str(day),"time":f"{hh:02d}:{mm:02d}","side":"short",
                              "entry_p":ep,"stop":sl,"target":tgt,"risk_pts":rk,"rr":rr}
                traded_today=True

        # LONG
        elif (vwap-lower)>=MIN_BAND_PT and l<=lower and crange>=8.0 and dn_w>=crange*WICK_PCT and c>o:
            ep=(l+min(o,c))/2.0; sl=l-STOP_BUF
            rk=min(max(ep-sl,1.0),MAX_STOP); tgt=vwap; rw=tgt-ep; rr=rw/rk if rk>0 else 0
            if rk>=MIN_STOP and rw>=rk*MIN_RR:
                active_trade={"date":str(day),"time":f"{hh:02d}:{mm:02d}","side":"long",
                              "entry_p":ep,"stop":sl,"target":tgt,"risk_pts":rk,"rr":rr}
                traded_today=True

# ── Results ───────────────────────────────────────────────────────────────────
if not trades:
    print("NO TRADES — data may be missing or filters too tight.")
else:
    tdf=pd.DataFrame(trades)
    wins=tdf[tdf["exit_type"]=="TP"]
    losses=tdf[tdf["exit_type"]=="SL"]
    eods=tdf[tdf["exit_type"]=="EOD"]
    total=len(tdf); wc=len(wins); lc=len(losses)
    wr=wc/total*100
    net=tdf["pnl_usd"].sum()
    gw=wins["pnl_usd"].sum() if wc else 0
    gl=abs(losses["pnl_usd"].sum()) if lc else 1
    pf=gw/gl if gl else float("inf")
    avg_rr=tdf["rr"].mean()
    avg_stop=tdf["risk_pts"].mean()
    avg_win=wins["pnl_usd"].mean() if wc else 0
    avg_loss=losses["pnl_usd"].mean() if lc else 0
    tdf["cum"]=tdf["pnl_usd"].cumsum()
    mdd=(tdf["cum"]-tdf["cum"].cummax()).min()
    days=df["date"].nunique()

    print("=" * 62)
    print("    ICT 9:30 VWAP BAND REVERSAL — 60-DAY BACKTEST RESULTS")
    print("=" * 62)
    print(f"  Trading Days        : {days}")
    print(f"  Total Trades        : {total}")
    print(f"  Wins  (TP)          : {wc}  ({wr:.1f}%)")
    print(f"  Losses (SL)         : {lc}  ({100-wr-len(eods)/total*100:.1f}%)")
    print(f"  EOD scratch         : {len(eods)}")
    print(f"  Trades / week       : {total/days*5:.1f}")
    print("-" * 62)
    print(f"  Win Rate            : {wr:.1f}%")
    print(f"  Profit Factor       : {pf:.2f}")
    print(f"  Avg R:R per trade   : {avg_rr:.2f}R")
    print(f"  Avg Stop Size       : {avg_stop:.1f} pts")
    print(f"  Avg Win  (USD)      : ${avg_win:,.0f}")
    print(f"  Avg Loss (USD)      : ${avg_loss:,.0f}")
    print(f"  Net Profit (1 NQ)   : ${net:,.2f}")
    print(f"  Max Drawdown        : ${mdd:,.2f}")
    print("=" * 62)
    print("\n  TRADE-BY-TRADE LOG:")
    print(f"  {'Date':<12} {'Time':<6} {'Side':<6} {'Entry':>9} {'Stop_pts':>8} {'R:R':>5} {'Exit':<5} {'P&L':>9}")
    print("  " + "-"*60)
    for _, r in tdf.iterrows():
        print(f"  {r['date']:<12} {r['time']:<6} {r['side']:<6} {r['entry_p']:>9.2f} {r['risk_pts']:>8.1f} {r['rr']:>5.2f} {r['exit_type']:<5} ${r['pnl_usd']:>8,.0f}")

    print("\n" + "=" * 62)
    print("  FUNDEDNEXT $50K CHALLENGE ASSESSMENT")
    print("=" * 62)
    weeks_to_target = 2500 / (net/days*5) if net > 0 else float("inf")
    print(f"  Challenge Target    : +$2,500  (8% of $50k)")
    print(f"  Max Allowed DD      : -$2,000  (4% of $50k)")
    print(f"  Per-week profit     : ${net/days*5:,.0f}")
    print(f"  Est. weeks to pass  : {weeks_to_target:.1f} weeks")
    print(f"  Max DD Observed     : ${mdd:,.2f}")
    dd_ok = abs(mdd) <= 2000
    profitable = net > 0
    if profitable and dd_ok and wr >= 30:
        print(f"\n  VERDICT  ->  LOOKS GOOD for the challenge")
        print(f"               Win rate {wr:.0f}% with {avg_rr:.1f}R avg = math works")
        print(f"               Stay disciplined: 1 trade/day, tight stops")
    elif profitable and not dd_ok:
        print(f"\n  VERDICT  ->  Profitable but drawdown exceeds limit")
        print(f"               Consider reducing to MNQ (1/10 size) first")
    else:
        print(f"\n  VERDICT  ->  Needs more tuning or larger data sample")
    print("=" * 62)

