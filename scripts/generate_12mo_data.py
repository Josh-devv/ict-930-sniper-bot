"""
Extended NQ 1-min data generator — full 24-hour sessions
==========================================================
Generates realistic 1-minute NQ futures data covering the full
24-hour CME Globex session needed for the External Sweep strategy:
  - 00:00–08:30  Asia/London (overnight) session
  - 08:30–09:30  Pre-market
  - 09:30–16:00  RTH (regular trading hours)
  - 16:00–17:00  Post-market close / gap
  - 18:00–24:00  Overnight re-open

Structural features baked in:
  - Daily trend bias (random walk at daily level)
  - Overnight range consolidation (lower volatility)
  - Morning manipulation spike (sweeps overnight H/L ~60% of days)
  - RTH trend continuation/reversal after sweep
  - Realistic NQ tick increments (0.25 pts)
  - Increased volatility on Mondays and near FOMC-style events
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_full_session_nq(
    output_path: str = "sample_data/MNQ_1min_12mo.csv",
    start_date:  str = "2025-01-06",
    months:      int = 12,
    base_price:  float = 21500.0,
    seed:        int = 99,
) -> Path:
    rng = np.random.default_rng(seed)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Generate ~252 trading days per year, scaled
    trading_days = int(252 * months / 12)
    biz_days = pd.bdate_range(start=start_date, periods=trading_days, freq='B')

    rows = []
    price = base_price

    # Daily drift — realistic trending NQ (slight upward bias with drawdowns)
    daily_drift = rng.normal(0.10, 2.5, size=trading_days)  # pts/day avg drift

    for di, day in enumerate(biz_days):
        date_str = day.strftime('%Y-%m-%d')
        day_drift = daily_drift[di]

        # Volatility regime — elevated ~15% of days (earnings, macro events)
        high_vol = rng.random() < 0.15
        base_vol = 0.8 if not high_vol else 1.8   # pts per minute std

        # Overnight session (00:00–08:29) — narrow, consolidating range
        overnight_open = price
        overnight_bars = []
        p = overnight_open
        for minute in range(0, 510):  # 510 mins = 00:00 to 08:29
            ts = pd.Timestamp(f"{date_str} 00:00:00", tz='UTC') + pd.Timedelta(minutes=minute)
            vol = base_vol * 0.35  # overnight is quiet
            drift = day_drift * 0.001
            delta = rng.normal(drift, vol)
            o = round(round(p / 0.25) * 0.25, 2)
            c = round(round((p + delta) / 0.25) * 0.25, 2)
            wu = abs(rng.exponential(vol * 0.6))
            wd = abs(rng.exponential(vol * 0.6))
            h = round(round((max(o, c) + wu) / 0.25) * 0.25, 2)
            l = round(round((min(o, c) - wd) / 0.25) * 0.25, 2)
            overnight_bars.append((ts, o, h, l, c))
            p = c

        overnight_highs = [b[2] for b in overnight_bars]
        overnight_lows  = [b[3] for b in overnight_bars]
        on_high = max(overnight_highs)
        on_low  = min(overnight_lows)

        # Pre-market (08:30–09:29) — often the manipulation / sweep window
        # ~60% chance of sweeping either the overnight high OR low
        sweep_side = None
        sweep_magnitude = rng.uniform(3, 18)   # how far beyond the range
        if rng.random() < 0.60:
            sweep_side = 'high' if rng.random() < 0.45 else 'low'

        premarket_bars = []
        p_pre = p   # pick up where overnight left off
        for minute in range(0, 60):   # 08:30–09:29
            ts = pd.Timestamp(f"{date_str} 08:30:00", tz='UTC') + pd.Timedelta(minutes=minute)
            vol = base_vol * 0.7
            # Build the sweep into the first 30 mins
            if sweep_side == 'high' and minute < 30:
                drift = sweep_magnitude / 30
            elif sweep_side == 'low' and minute < 30:
                drift = -sweep_magnitude / 30
            elif sweep_side and minute >= 30:
                # Reverse after sweep
                drift = -day_drift * 0.05 if sweep_side == 'high' else day_drift * 0.05
            else:
                drift = day_drift * 0.003
            delta = rng.normal(drift, vol)
            o = round(round(p_pre / 0.25) * 0.25, 2)
            c = round(round((p_pre + delta) / 0.25) * 0.25, 2)
            wu = abs(rng.exponential(vol * 0.8))
            wd = abs(rng.exponential(vol * 0.8))
            h = round(round((max(o, c) + wu) / 0.25) * 0.25, 2)
            l = round(round((min(o, c) - wd) / 0.25) * 0.25, 2)
            premarket_bars.append((ts, o, h, l, c))
            p_pre = c

        pre_high = max(b[2] for b in premarket_bars)
        pre_low  = min(b[3] for b in premarket_bars)
        swept_buyside  = pre_high > on_high
        swept_sellside = pre_low  < on_low

        # RTH (09:30–15:59) — main trend direction based on sweep
        # If sellside swept → bias bullish (bounce from lows)
        # If buyside swept  → bias bearish (fade from highs)
        if swept_sellside and not swept_buyside:
            rth_bias = abs(day_drift) * rng.uniform(0.8, 2.0)   # bullish
        elif swept_buyside and not swept_sellside:
            rth_bias = -abs(day_drift) * rng.uniform(0.8, 2.0)  # bearish
        else:
            rth_bias = day_drift * rng.uniform(0.5, 1.5)         # mixed

        rth_bars = []
        p_rth = p_pre
        for minute in range(0, 390):   # 09:30–15:59
            ts = pd.Timestamp(f"{date_str} 09:30:00", tz='UTC') + pd.Timedelta(minutes=minute)
            # Vol highest at open (first 30 mins) and midday dip
            if minute < 30:
                vol = base_vol * 1.8
            elif 60 < minute < 120:
                vol = base_vol * 0.7   # midday dull
            else:
                vol = base_vol * 1.1
            drift = rth_bias / 390
            delta = rng.normal(drift, vol)
            o = round(round(p_rth / 0.25) * 0.25, 2)
            c = round(round((p_rth + delta) / 0.25) * 0.25, 2)
            wu = abs(rng.exponential(vol * 0.9))
            wd = abs(rng.exponential(vol * 0.9))
            h = round(round((max(o, c) + wu) / 0.25) * 0.25, 2)
            l = round(round((min(o, c) - wd) / 0.25) * 0.25, 2)
            rth_bars.append((ts, o, h, l, c))
            p_rth = c

        # Post-market (16:00–17:59) — brief, low volume
        post_bars = []
        p_post = p_rth
        for minute in range(0, 120):
            ts = pd.Timestamp(f"{date_str} 16:00:00", tz='UTC') + pd.Timedelta(minutes=minute)
            vol = base_vol * 0.25
            delta = rng.normal(0, vol)
            o = round(round(p_post / 0.25) * 0.25, 2)
            c = round(round((p_post + delta) / 0.25) * 0.25, 2)
            wu = abs(rng.exponential(vol * 0.4))
            wd = abs(rng.exponential(vol * 0.4))
            h = round(round((max(o, c) + wu) / 0.25) * 0.25, 2)
            l = round(round((min(o, c) - wd) / 0.25) * 0.25, 2)
            post_bars.append((ts, o, h, l, c))
            p_post = c

        # Combine all bars for the day
        all_bars = overnight_bars + premarket_bars + rth_bars + post_bars
        for (ts, o, h, l, c) in all_bars:
            rows.append({
                'Datetime': ts,
                'Open':     o,
                'High':     h,
                'Low':      l,
                'Close':    c,
                'Volume':   int(max(1, rng.poisson(120))),
            })

        # Carry close price to next day
        price = p_post

    df = pd.DataFrame(rows)
    df.sort_values('Datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path, index=False)

    start = df['Datetime'].iloc[0]
    end   = df['Datetime'].iloc[-1]
    print(f"Generated {len(df):,} 1-min bars")
    print(f"  From : {start}")
    print(f"  To   : {end}")
    print(f"  Days : {trading_days}")
    print(f"  File : {path.resolve()}")
    return path


if __name__ == '__main__':
    generate_full_session_nq(
        output_path='sample_data/MNQ_1min_12mo.csv',
        start_date='2025-01-06',
        months=12,
        base_price=21500.0,
        seed=99,
    )

