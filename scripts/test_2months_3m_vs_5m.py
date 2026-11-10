"""
2-MONTH UNSEEN OUT-OF-SAMPLE TEST: 3-MINUTE vs 5-MINUTE
Dataset: CME MNQ 1-minute historical data (Jan 20 - Apr 15, 2026)
Testing 2 Full Consecutive Months:
  - Month 1: February 2026 (20 Trading Days)
  - Month 2: March 2026 (22 Trading Days)

Evaluates:
  - 3M vs 5M Trade Frequency
  - Win Rate %
  - Average Stop Loss Size (points)
  - Profit Factor (PF)
  - Net Profit on 5 MNQ ($10/pt) vs 1 NQ ($20/pt)
  - Max Trailing Drawdown
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
import warnings
warnings.filterwarnings("ignore")

MNQ_PT_VAL    = 2.0     # $2 per pt per contract
COMMISSION_MNQ = 1.24    # Tradovate/NinjaTrader round-trip per MNQ contract

def load_data(filepath):
    df = pd.read_csv(filepath)
    dt_col = 'datetime' if 'datetime' in df.columns else 'Datetime'
    df['dt'] = pd.to_datetime(df[dt_col])
    
    # If tz-aware, convert to America/New_York
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_convert('America/New_York').dt.tz_localize(None)
        
    df = df.sort_values('dt').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        found = [c for c in df.columns if c.lower() == col]
        if found:
            df[col] = df[found[0]].astype(float)
    return df[['dt', 'open', 'high', 'low', 'close', 'volume']].set_index('dt')

def backtest_window(df_1m, start_date, end_date, tf_mins=3, mnq_contracts=5):
    # Resample 1m to tf_mins
    rule = f"{tf_mins}min"
    resampled = df_1m.resample(rule, closed="left", label="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()

    resampled["date"]   = resampled["dt"].dt.date
    resampled["hour"]   = resampled["dt"].dt.hour
    resampled["minute"] = resampled["dt"].dt.minute

    # Filter to RTH 09:30 to 16:00
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
        cum_vol = 0.0
        cum_pv  = 0.0
        cum_pv2 = 0.0

        traded_today = False
        in_trade     = False
        pending_order = None
        current_trade = None

        for i, bar in day_df.iterrows():
            bar_time = bar["dt"].time()
            tp = (bar["high"] + bar["low"] + bar["close"]) / 3.0
            vol = bar["volume"]

            cum_vol += vol
            cum_pv  += tp * vol
            cum_pv2 += tp * tp * vol

            if cum_vol == 0:
                continue

            vwap = cum_pv / cum_vol
            variance = max(0.0, cum_pv2 / cum_vol - vwap * vwap)
            stdev = np.sqrt(variance)
            upper_band = vwap + VWAP_MULT * stdev
            lower_band = vwap - VWAP_MULT * stdev

            # Handle active trade exit
            if in_trade:
                t = current_trade
                is_win  = False
                is_loss = False
                exit_price = None

                if t["dir"] == "SHORT":
                    # Check stop
                    if bar["high"] >= t["stop_p"]:
                        is_loss = True
                        exit_price = t["stop_p"]
                    # Check target
                    elif bar["low"] <= t["target_p"]:
                        is_win = True
                        exit_price = t["target_p"]
                    elif bar_time >= EOD_TIME:
                        exit_price = bar["close"]
                        is_win = exit_price < t["entry_p"]
                        is_loss = not is_win
                else:  # LONG
                    # Check stop
                    if bar["low"] <= t["stop_p"]:
                        is_loss = True
                        exit_price = t["stop_p"]
                    # Check target
                    elif bar["high"] >= t["target_p"]:
                        is_win = True
                        exit_price = t["target_p"]
                    elif bar_time >= EOD_TIME:
                        exit_price = bar["close"]
                        is_win = exit_price > t["entry_p"]
                        is_loss = not is_win

                if is_win or is_loss:
                    pts = (t["entry_p"] - exit_price) if t["dir"] == "SHORT" else (exit_price - t["entry_p"])
                    # PnL for mnq_contracts
                    gross_pnl = pts * MNQ_PT_VAL * mnq_contracts
                    comm = COMMISSION_MNQ * mnq_contracts
                    net_pnl = gross_pnl - comm

                    t["exit_dt"]    = bar["dt"]
                    t["exit_p"]     = exit_price
                    t["pts"]        = pts
                    t["r_mult"]     = pts / t["risk_pts"]
                    t["net_pnl"]    = net_pnl
                    t["result"]     = "WIN" if pts > 0 else "LOSS"
                    trades.append(t)

                    in_trade = False
                    current_trade = None
                continue

            # Check pending limit order fill
            if pending_order is not None:
                p = pending_order
                filled = False
                if p["dir"] == "SHORT" and bar["high"] >= p["entry_p"]:
                    filled = True
                elif p["dir"] == "LONG" and bar["low"] <= p["entry_p"]:
                    filled = True

                if filled:
                    in_trade = True
                    current_trade = p
                    current_trade["fill_dt"] = bar["dt"]
                    pending_order = None
                else:
                    # Cancel if out of entry window
                    if bar_time > ENTRY_END:
                        pending_order = None
                continue

            # Signal scanning (One trade per day)
            if not traded_today and ENTRY_START <= bar_time <= ENTRY_END:
                crange    = bar["high"] - bar["low"]
                up_wick   = bar["high"] - max(bar["open"], bar["close"])
                down_wick = min(bar["open"], bar["close"]) - bar["low"]

                # SHORT Setup
                if (upper_band - vwap) >= MIN_BAND_PT and bar["high"] >= upper_band:
                    if crange >= 6.0 and up_wick >= crange * WICK_PCT and bar["close"] < bar["open"]:
                        ep = (bar["high"] + max(bar["open"], bar["close"])) / 2.0
                        sl = bar["high"] + STOP_BUF
                        risk = min(max(sl - ep, 1.0), MAX_STOP)
                        tp_p = vwap
                        reward = ep - tp_p

                        if risk >= MIN_STOP and reward >= risk * MIN_RR:
                            pending_order = {
                                "date": day, "dir": "SHORT", "signal_dt": bar["dt"],
                                "entry_p": ep, "stop_p": sl, "target_p": tp_p,
                                "risk_pts": risk, "reward_pts": reward, "rr": reward/risk
                            }
                            traded_today = True

                # LONG Setup
                elif (vwap - lower_band) >= MIN_BAND_PT and bar["low"] <= lower_band:
                    if crange >= 6.0 and down_wick >= crange * WICK_PCT and bar["close"] > bar["open"]:
                        ep = (bar["low"] + min(bar["open"], bar["close"])) / 2.0
                        sl = bar["low"] - STOP_BUF
                        risk = min(max(ep - sl, 1.0), MAX_STOP)
                        tp_p = vwap
                        reward = tp_p - ep

                        if risk >= MIN_STOP and reward >= risk * MIN_RR:
                            pending_order = {
                                "date": day, "dir": "LONG", "signal_dt": bar["dt"],
                                "entry_p": ep, "stop_p": sl, "target_p": tp_p,
                                "risk_pts": risk, "reward_pts": reward, "rr": reward/risk
                            }
                            traded_today = True

    return pd.DataFrame(trades)

def compute_metrics(trades_df):
    if len(trades_df) == 0:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "net_pnl": 0.0, "profit_factor": 0.0, "avg_stop": 0.0,
            "avg_reward": 0.0, "avg_r": 0.0, "max_dd": 0.0
        }

    wins = trades_df[trades_df["result"] == "WIN"]
    losses = trades_df[trades_df["result"] == "LOSS"]

    gross_profit = wins["net_pnl"].sum()
    gross_loss = abs(losses["net_pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.nan

    # Drawdown calculation
    equity = trades_df["net_pnl"].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = dd.min()

    return {
        "trades": len(trades_df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades_df) * 100.0,
        "net_pnl": trades_df["net_pnl"].sum(),
        "profit_factor": pf,
        "avg_stop": trades_df["risk_pts"].mean(),
        "avg_reward": trades_df["reward_pts"].mean(),
        "avg_r": wins["r_mult"].mean() if len(wins) > 0 else 0.0,
        "max_dd": max_dd
    }

def main():
    print("=" * 80)
    print("  HEAD-TO-HEAD BACKTEST: 3-MINUTE vs 5-MINUTE (UNSEEN 2-MONTH OUT-OF-SAMPLE)")
    print("  Position Sizing: 5 MNQ Contracts ($10/pt) | Tradovate/TopStep All-In Comm.")
    print("=" * 80)

    data_file = "sample_data/MNQ_1min_20260120_20260415.csv"
    df_1m = load_data(data_file)

    windows = [
        ("Month 1 (February 2026)", "2026-02-01", "2026-02-28"),
        ("Month 2 (March 2026)",    "2026-03-01", "2026-03-31"),
        ("2-Month Combined Total",   "2026-02-01", "2026-03-31")
    ]

    for title, s_date, e_date in windows:
        print(f"\n>>> TESTING PERIOD: {title} ({s_date} to {e_date})")
        print("-" * 80)
        
        t_3m = backtest_window(df_1m, s_date, e_date, tf_mins=3, mnq_contracts=5)
        m_3m = compute_metrics(t_3m)

        t_5m = backtest_window(df_1m, s_date, e_date, tf_mins=5, mnq_contracts=5)
        m_5m = compute_metrics(t_5m)

        print(f"{'Metric':<25} | {'3-Minute Timeframe':<22} | {'5-Minute Timeframe':<22}")
        print("-" * 80)
        print(f"{'Total Trades':<25} | {m_3m['trades']:<22} | {m_5m['trades']:<22}")
        print(f"{'Win Rate':<25} | {m_3m['win_rate']:<20.1f}% | {m_5m['win_rate']:<20.1f}%")
        print(f"{'Wins / Losses':<25} | {str(m_3m['wins']) + 'W / ' + str(m_3m['losses']) + 'L':<22} | {str(m_5m['wins']) + 'W / ' + str(m_5m['losses']) + 'L':<22}")
        print(f"{'Average Stop Loss':<25} | {m_3m['avg_stop']:<18.2f} pts | {m_5m['avg_stop']:<18.2f} pts")
        print(f"{'Average Dollar Risk (5 MNQ)':<25} | ${m_3m['avg_stop']*10:<18.2f} | ${m_5m['avg_stop']*10:<18.2f}")
        print(f"{'Avg Winner R:R':<25} | {m_3m['avg_r']:<19.2f}R | {m_5m['avg_r']:<19.2f}R")
        print(f"{'Profit Factor':<25} | {m_3m['profit_factor']:<22.2f} | {m_5m['profit_factor']:<22.2f}")
        print(f"{'Net Profit (5 MNQ)':<25} | +${m_3m['net_pnl']:<20.2f} | +${m_5m['net_pnl']:<20.2f}")
        print(f"{'Net Profit (1 Full NQ)':<25} | +${m_3m['net_pnl']*2:<20.2f} | +${m_5m['net_pnl']*2:<20.2f}")
        print(f"{'Max Drawdown (5 MNQ)':<25} | -${abs(m_3m['max_dd']):<20.2f} | -${abs(m_5m['max_dd']):<20.2f}")

if __name__ == "__main__":
    main()

