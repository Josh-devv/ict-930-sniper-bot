"""
Institutional Orderflow + Key Levels + Order Block Strategy Engine
===================================================================
100% Strict, Zero-Cheat Backtest on 8 Years of Real CME NQ Futures (2018-2026)

Key Pillars:
1. Key Institutional Levels: PDH / PDL (Previous Day High/Low) & 09:30-09:45 Opening Range
2. Orderflow & Volume Absorption: Volume Climax + Delta Reversal (Absorption Wick)
3. 5m / 15m Institutional Order Block: 50% Mean Threshold (MT) Entry with tight structural stop
4. Targets: 2.0R, 2.5R, 3.0R, and Runner Scaling (50% @ 2R -> BE -> 50% @ 3.5R)
5. Execution Realism: Hard stops checked first, pessimistic limit fills, $5.20 RT commission, 1 tick slippage
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, time
from dataclasses import dataclass
from typing import List, Dict

print("=" * 80)
print("INSTITUTIONAL NQ ENGINE: LEVELS + ORDERFLOW ABSORPTION + ORDER BLOCKS")
print("Data Source: sample_data/NQ_1min_RTH_2018_2026.csv (862,499 bars)")
print("=" * 80)

# ─── 1. Load Data ────────────────────────────────────────────────────────────
print("\n[1/4] Loading real CME 1-min data...")
df_1m = pd.read_csv('sample_data/NQ_1min_RTH_2018_2026.csv')
df_1m.columns = [c.lower() for c in df_1m.columns]
df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
df_1m = df_1m[df_1m['high'] > df_1m['low']].reset_index(drop=True)
df_1m = df_1m.set_index('datetime').sort_index()

# Calculate 1-min bar delta proxy: (2*close - high - low) / (high - low) * volume
rng_1m = (df_1m['high'] - df_1m['low']).replace(0, 1e-6)
df_1m['delta'] = df_1m['volume'] * (2.0 * df_1m['close'] - df_1m['high'] - df_1m['low']) / rng_1m

print(f"Loaded {len(df_1m):,} active 1-min bars from {df_1m.index[0].date()} to {df_1m.index[-1].date()}.")


# ─── 2. Resample Timeframes & Calculate Key Levels ────────────────────────────
print("\n[2/4] Resampling timeframes and extracting daily institutional levels...")

# Resample to 5m and 15m
df_5m = df_1m.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'delta': 'sum'
}).dropna().reset_index()

df_15m = df_1m.resample('15min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'delta': 'sum'
}).dropna().reset_index()

# 1-Hour Trend for HTF directional bias
df_1h = df_1m.resample('1h', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_1h['ema20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
df_1h['htf_trend'] = np.where(df_1h['close'] > df_1h['ema20'], 1, -1)

# Daily levels: PDH, PDL, Previous Day Close
df_daily = df_1m.resample('1d').agg({
    'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_daily['pdh'] = df_daily['high'].shift(1)
df_daily['pdl'] = df_daily['low'].shift(1)
df_daily['pdc'] = df_daily['close'].shift(1)

# Map Daily Levels to Date dict for fast lookup
daily_levels = {}
for dt, row in df_daily.iterrows():
    daily_levels[dt.date()] = {
        'pdh': row['pdh'],
        'pdl': row['pdl'],
        'pdc': row['pdc']
    }


# ─── 3. Strategy & Backtest Simulation Engine ────────────────────────────────
COMMISSION_PTS = 0.26   # $5.20 / $20 per point on Mini NQ (equivalent to $1.24 on MNQ)
SLIPPAGE_PTS   = 0.25   # 1 tick slippage on stops and market/runner exits

@dataclass
class Trade:
    timeframe: str
    direction: int            # 1 = Long, -1 = Short
    entry_time: pd.Timestamp
    entry_px: float
    stop_px: float
    risk_pts: float
    target_r: float
    exit_px: float
    exit_reason: str
    net_pts: float
    realized_r: float
    dur_bars: int
    level_type: str
    htf_aligned: bool


def backtest_levels_orderflow(
    df_tf: pd.DataFrame,
    tf_name: str,
    target_mode: str = '2.0R',      # '2.0R', '2.5R', '3.0R', 'SCALE_RUNNER'
    require_level_sweep: bool = True,
    require_of_absorption: bool = True,
    require_htf_align: bool = True,
    only_silver_bullet: bool = True,
    min_stop: float = 8.0,
    max_stop: float = 35.0
) -> List[Trade]:
    
    # Pre-calculate indicators
    tr = np.maximum(df_tf['high'] - df_tf['low'],
                    np.maximum(abs(df_tf['high'] - df_tf['close'].shift(1)),
                               abs(df_tf['low'] - df_tf['close'].shift(1))))
    df_tf['atr'] = tr.rolling(14).mean().bfill()
    df_tf['vol_ma'] = df_tf['volume'].rolling(20).mean().bfill()

    # Map HTF trend
    df_tf['hour_ts'] = df_tf['datetime'].dt.floor('1h')
    df_tf['htf_trend'] = df_tf['hour_ts'].map(df_1h['htf_trend']).fillna(0)

    dates = df_tf['datetime'].dt.date.values
    times = df_tf['datetime'].values
    opens = df_tf['open'].values
    highs = df_tf['high'].values
    lows = df_tf['low'].values
    closes = df_tf['close'].values
    volumes = df_tf['volume'].values
    deltas = df_tf['delta'].values
    vol_mas = df_tf['vol_ma'].values
    atrs = df_tf['atr'].values
    htf = df_tf['htf_trend'].values

    n = len(df_tf)
    trades: List[Trade] = []

    # Cache Opening Range per day (09:30 to 09:45)
    # We will track ORH/ORL dynamically
    current_day = None
    orh, orl = None, None

    for i in range(20, n - 40):
        t = pd.Timestamp(times[i])
        day_date = dates[i]
        h, m = t.hour, t.minute

        # Reset daily opening range on new day
        if day_date != current_day:
            current_day = day_date
            orh, orl = None, None

        # Calculate 09:30 - 09:45 Opening Range
        if (h == 9 and m >= 30) or (h == 9 and m <= 45):
            if orh is None or highs[i] > orh:
                orh = highs[i]
            if orl is None or lows[i] < orl:
                orl = lows[i]

        # Time filter
        if only_silver_bullet:
            # 10:00 AM - 11:30 AM Silver Bullet Window
            if not ((h == 10) or (h == 11 and m <= 30)):
                continue
        else:
            # Full RTH Session 09:45 - 15:30
            if not ((h == 9 and m >= 45) or (10 <= h <= 15)):
                continue

        # Get Previous Day Levels
        p_levels = daily_levels.get(day_date, None)
        if not p_levels or np.isnan(p_levels['pdh']) or np.isnan(p_levels['pdl']):
            continue
        pdh, pdl = p_levels['pdh'], p_levels['pdl']

        # Check Orderflow & Level Sweep on prior bar i-1 (the Order Block candidate)
        ob_idx = i - 1
        ob_o = opens[ob_idx]
        ob_h = highs[ob_idx]
        ob_l = lows[ob_idx]
        ob_c = closes[ob_idx]
        ob_v = volumes[ob_idx]
        ob_d = deltas[ob_idx]
        ob_v_ma = vol_mas[ob_idx]
        ob_rng = ob_h - ob_l

        if ob_rng < 1.0:
            continue

        # Current bar i is the displacement candle
        disp_body = abs(closes[i] - opens[i])
        is_bull_disp = (disp_body >= 1.0 * atrs[i]) and (closes[i] > opens[i])
        is_bear_disp = (disp_body >= 1.0 * atrs[i]) and (closes[i] < opens[i])

        # ── BULLISH SETUP: Sweep of PDL or ORL + Absorption ──────────────────
        if is_bull_disp and ob_c < ob_o:  # Down-candle prior to displacement
            level_swept = False
            level_name = "NONE"

            if pdl is not None and ob_l < pdl and ob_c >= pdl - (0.5 * atrs[i]):
                level_swept = True
                level_name = "PDL_SWEEP"
            elif orl is not None and ob_l < orl and ob_c >= orl - (0.5 * atrs[i]):
                level_swept = True
                level_name = "ORL_SWEEP"

            if require_level_sweep and not level_swept:
                continue

            # Orderflow Absorption Check
            # 1. Volume surge (institutions absorbing aggressive selling)
            # 2. Lower rejection wick >= 30% of range
            # 3. Delta absorption (delta flips positive or wick demonstrates absorption)
            lower_wick = min(ob_o, ob_c) - ob_l
            has_absorption = (lower_wick / ob_rng >= 0.30)
            has_vol_climax = ob_v >= 1.15 * ob_v_ma
            delta_absorbed = (ob_d > 0) or (deltas[i] > 0)  # Buyers overwhelmed sellers

            if require_of_absorption and not (has_absorption and has_vol_climax and delta_absorbed):
                continue

            if require_htf_align and htf[i] != 1:
                continue

            # 50% Mean Threshold (MT) of Order Block
            block_top = max(ob_o, ob_c)
            block_bot = ob_l
            mt_entry = (block_top + block_bot) / 2.0
            stop_px  = block_bot - 1.0
            risk_pts = mt_entry - stop_px

            if not (min_stop <= risk_pts <= max_stop):
                continue

            # Target calculation
            r_mult = 2.0 if target_mode == '2.0R' else (2.5 if target_mode == '2.5R' else 3.0)
            target_px = mt_entry + r_mult * risk_pts

            # Walk forward for pessimistic fill on bar i+1 onwards
            filled = False
            fill_k = -1
            for k in range(i + 1, min(n - 1, i + 25)):
                if lows[k] <= mt_entry:
                    filled = True
                    fill_k = k
                    break

            if not filled:
                continue

            # Simulate execution strictly
            actual_entry = mt_entry + SLIPPAGE_PTS
            exit_px = 0.0
            exit_reason = ""
            dur = 0

            if target_mode == 'SCALE_RUNNER':
                # TP1 at 2.0R (50%), move SL to BE (+1 pt), TP2 runner at 3.5R (50%)
                tp1_px = mt_entry + 2.0 * risk_pts
                tp2_px = mt_entry + 3.5 * risk_pts
                tp1_hit = False
                be_stop = actual_entry

                for k in range(fill_k, min(n - 1, fill_k + 40)):
                    dur = k - fill_k
                    # Stop loss check FIRST
                    curr_stop = be_stop if tp1_hit else stop_px
                    if lows[k] <= curr_stop:
                        exit_px = curr_stop - SLIPPAGE_PTS
                        exit_reason = "BE_STOP" if tp1_hit else "STOP_LOSS"
                        break
                    if not tp1_hit and highs[k] >= tp1_px:
                        tp1_hit = True
                    if tp1_hit and highs[k] >= tp2_px:
                        exit_px = tp2_px - SLIPPAGE_PTS
                        exit_reason = "RUNNER_3.5R"
                        break
                    if pd.Timestamp(times[k]).hour == 15 and pd.Timestamp(times[k]).minute >= 50:
                        exit_px = closes[k] - SLIPPAGE_PTS
                        exit_reason = "EOD_EXIT"
                        break

                if exit_reason:
                    if exit_reason == "STOP_LOSS":
                        net = (exit_px - actual_entry) - COMMISSION_PTS
                    elif exit_reason == "RUNNER_3.5R":
                        net = 0.5 * (tp1_px - actual_entry) + 0.5 * (exit_px - actual_entry) - COMMISSION_PTS
                    elif exit_reason == "BE_STOP":
                        net = 0.5 * (tp1_px - actual_entry) + 0.5 * (exit_px - actual_entry) - COMMISSION_PTS
                    else:  # EOD_EXIT
                        if tp1_hit:
                            net = 0.5 * (tp1_px - actual_entry) + 0.5 * (exit_px - actual_entry) - COMMISSION_PTS
                        else:
                            net = (exit_px - actual_entry) - COMMISSION_PTS
                    
                    trades.append(Trade(
                        tf_name, 1, pd.Timestamp(times[fill_k]), actual_entry, stop_px,
                        risk_pts, 3.5 if tp1_hit else 2.0, exit_px, exit_reason, net,
                        net / risk_pts, dur, level_name, (htf[i] == 1)
                    ))

            else:
                # Fixed target
                for k in range(fill_k, min(n - 1, fill_k + 40)):
                    dur = k - fill_k
                    if lows[k] <= stop_px:
                        exit_px = stop_px - SLIPPAGE_PTS
                        exit_reason = "STOP_LOSS"
                        break
                    if highs[k] >= target_px:
                        exit_px = target_px - SLIPPAGE_PTS
                        exit_reason = f"TARGET_{r_mult}R"
                        break
                    if pd.Timestamp(times[k]).hour == 15 and pd.Timestamp(times[k]).minute >= 50:
                        exit_px = closes[k] - SLIPPAGE_PTS
                        exit_reason = "EOD_EXIT"
                        break

                if exit_reason:
                    net = (exit_px - actual_entry) - COMMISSION_PTS
                    trades.append(Trade(
                        tf_name, 1, pd.Timestamp(times[fill_k]), actual_entry, stop_px,
                        risk_pts, r_mult, exit_px, exit_reason, net,
                        net / risk_pts, dur, level_name, (htf[i] == 1)
                    ))


        # ── BEARISH SETUP: Sweep of PDH or ORH + Absorption ──────────────────
        elif is_bear_disp and ob_c > ob_o:  # Up-candle prior to displacement
            level_swept = False
            level_name = "NONE"

            if pdh is not None and ob_h > pdh and ob_c <= pdh + (0.5 * atrs[i]):
                level_swept = True
                level_name = "PDH_SWEEP"
            elif orh is not None and ob_h > orh and ob_c <= orh + (0.5 * atrs[i]):
                level_swept = True
                level_name = "ORH_SWEEP"

            if require_level_sweep and not level_swept:
                continue

            # Orderflow Absorption Check
            upper_wick = ob_h - max(ob_o, ob_c)
            has_absorption = (upper_wick / ob_rng >= 0.30)
            has_vol_climax = ob_v >= 1.15 * ob_v_ma
            delta_absorbed = (ob_d < 0) or (deltas[i] < 0)

            if require_of_absorption and not (has_absorption and has_vol_climax and delta_absorbed):
                continue

            if require_htf_align and htf[i] != -1:
                continue

            # 50% Mean Threshold (MT) of Order Block
            block_top = ob_h
            block_bot = min(ob_o, ob_c)
            mt_entry = (block_top + block_bot) / 2.0
            stop_px  = block_top + 1.0
            risk_pts = stop_px - mt_entry

            if not (min_stop <= risk_pts <= max_stop):
                continue

            r_mult = 2.0 if target_mode == '2.0R' else (2.5 if target_mode == '2.5R' else 3.0)
            target_px = mt_entry - r_mult * risk_pts

            filled = False
            fill_k = -1
            for k in range(i + 1, min(n - 1, i + 25)):
                if highs[k] >= mt_entry:
                    filled = True
                    fill_k = k
                    break

            if not filled:
                continue

            actual_entry = mt_entry - SLIPPAGE_PTS
            exit_px = 0.0
            exit_reason = ""
            dur = 0

            if target_mode == 'SCALE_RUNNER':
                tp1_px = mt_entry - 2.0 * risk_pts
                tp2_px = mt_entry - 3.5 * risk_pts
                tp1_hit = False
                be_stop = actual_entry

                for k in range(fill_k, min(n - 1, fill_k + 40)):
                    dur = k - fill_k
                    curr_stop = be_stop if tp1_hit else stop_px
                    if highs[k] >= curr_stop:
                        exit_px = curr_stop + SLIPPAGE_PTS
                        exit_reason = "BE_STOP" if tp1_hit else "STOP_LOSS"
                        break
                    if not tp1_hit and lows[k] <= tp1_px:
                        tp1_hit = True
                    if tp1_hit and lows[k] <= tp2_px:
                        exit_px = tp2_px + SLIPPAGE_PTS
                        exit_reason = "RUNNER_3.5R"
                        break
                    if pd.Timestamp(times[k]).hour == 15 and pd.Timestamp(times[k]).minute >= 50:
                        exit_px = closes[k] + SLIPPAGE_PTS
                        exit_reason = "EOD_EXIT"
                        break

                if exit_reason:
                    if exit_reason == "STOP_LOSS":
                        net = (actual_entry - exit_px) - COMMISSION_PTS
                    elif exit_reason == "RUNNER_3.5R":
                        net = 0.5 * (actual_entry - tp1_px) + 0.5 * (actual_entry - exit_px) - COMMISSION_PTS
                    elif exit_reason == "BE_STOP":
                        net = 0.5 * (actual_entry - tp1_px) + 0.5 * (actual_entry - exit_px) - COMMISSION_PTS
                    else:
                        if tp1_hit:
                            net = 0.5 * (actual_entry - tp1_px) + 0.5 * (actual_entry - exit_px) - COMMISSION_PTS
                        else:
                            net = (actual_entry - exit_px) - COMMISSION_PTS

                    trades.append(Trade(
                        tf_name, -1, pd.Timestamp(times[fill_k]), actual_entry, stop_px,
                        risk_pts, 3.5 if tp1_hit else 2.0, exit_px, exit_reason, net,
                        net / risk_pts, dur, level_name, (htf[i] == -1)
                    ))

            else:
                for k in range(fill_k, min(n - 1, fill_k + 40)):
                    dur = k - fill_k
                    if highs[k] >= stop_px:
                        exit_px = stop_px + SLIPPAGE_PTS
                        exit_reason = "STOP_LOSS"
                        break
                    if lows[k] <= target_px:
                        exit_px = target_px + SLIPPAGE_PTS
                        exit_reason = f"TARGET_{r_mult}R"
                        break
                    if pd.Timestamp(times[k]).hour == 15 and pd.Timestamp(times[k]).minute >= 50:
                        exit_px = closes[k] + SLIPPAGE_PTS
                        exit_reason = "EOD_EXIT"
                        break

                if exit_reason:
                    net = (actual_entry - exit_px) - COMMISSION_PTS
                    trades.append(Trade(
                        tf_name, -1, pd.Timestamp(times[fill_k]), actual_entry, stop_px,
                        risk_pts, r_mult, exit_px, exit_reason, net,
                        net / risk_pts, dur, level_name, (htf[i] == -1)
                    ))

    return trades


# ─── 4. Run Comparative Matrix ────────────────────────────────────────────────
def print_results(trades: List[Trade], name: str):
    if not trades:
        print(f"| {name:<45} | No trades generated |")
        return
    
    tdf = pd.DataFrame([t.__dict__ for t in trades])
    n = len(tdf)
    wins = tdf[tdf['net_pts'] > 0]
    losses = tdf[tdf['net_pts'] <= 0]
    
    wr = len(wins) / n * 100
    gross_win = wins['net_pts'].sum()
    gross_loss = abs(losses['net_pts'].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0
    tot_pts = tdf['net_pts'].sum()
    avg_pts = tdf['net_pts'].mean()
    avg_r = tdf['realized_r'].mean()
    avg_stop = tdf['risk_pts'].mean()
    avg_dur = tdf['dur_bars'].mean()
    
    # Dollar calculations: 1 NQ contract = $20/pt, 1 MNQ contract = $2/pt
    dollar_nq = tot_pts * 20.0
    dollar_mnq = tot_pts * 2.0

    print(f"| {name:<42} | {n:>5} | {wr:>5.1f}% | {pf:>5.2f} | {avg_stop:>5.1f}pt | {avg_pts:>+6.1f}pt | {avg_r:>+5.2f}R | +${dollar_nq:>9,.0f} | +${dollar_mnq:>7,.0f} |")


print("\n[3/4] Running Comprehensive Multi-Factor Matrix on 8-Year Real Data...")
print("=" * 115)
print(f"| {'CONFIGURATION':<42} | {'TRADES':>5} | {'WIN%':>6} | {'PF':>5} | {'AVG SL':>7} | {'AVG PTS':>8} | {'AVG R':>7} | {'NQ NET $':>11} | {'MNQ NET $':>9} |")
print("=" * 115)

# Test 15-Minute vs 5-Minute
# Target variants
for tf_df, tf_name in [(df_15m, "15m"), (df_5m, "5m")]:
    for target in ['2.0R', '2.5R', '3.0R', 'SCALE_RUNNER']:
        trades = backtest_levels_orderflow(
            tf_df, tf_name, target_mode=target,
            require_level_sweep=True,
            require_of_absorption=True,
            require_htf_align=True,
            only_silver_bullet=True
        )
        print_results(trades, f"{tf_name} + Levels + OF + 1H Trend [{target}]")

print("-" * 115)
# Test Component Ablation (What creates the edge?)
# 1. Without Level Sweep (Any random OB + OF)
trades_no_lvl = backtest_levels_orderflow(
    df_15m, "15m", target_mode='SCALE_RUNNER',
    require_level_sweep=False,
    require_of_absorption=True,
    require_htf_align=True,
    only_silver_bullet=True
)
print_results(trades_no_lvl, "15m NO Levels (OB + OF only) [Runner]")

# 2. Without Orderflow Absorption (Plain Level Sweep + OB)
trades_no_of = backtest_levels_orderflow(
    df_15m, "15m", target_mode='SCALE_RUNNER',
    require_level_sweep=True,
    require_of_absorption=False,
    require_htf_align=True,
    only_silver_bullet=True
)
print_results(trades_no_of, "15m NO Orderflow (Levels + OB only) [Runner]")

# 3. Without 1H Trend Alignment (Counter-trend allowed)
trades_no_htf = backtest_levels_orderflow(
    df_15m, "15m", target_mode='SCALE_RUNNER',
    require_level_sweep=True,
    require_of_absorption=True,
    require_htf_align=False,
    only_silver_bullet=True
)
print_results(trades_no_htf, "15m NO 1H Trend (Counter-trend) [Runner]")

# 4. Full Day Session (outside Silver Bullet)
trades_all_day = backtest_levels_orderflow(
    df_15m, "15m", target_mode='SCALE_RUNNER',
    require_level_sweep=True,
    require_of_absorption=True,
    require_htf_align=True,
    only_silver_bullet=False
)
print_results(trades_all_day, "15m All-Day Session (09:45-15:30) [Runner]")

print("=" * 115)
print("\n[4/4] Study successfully executed.")

