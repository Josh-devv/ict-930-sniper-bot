"""
Tony Iyke Market Structure, Liquidity & POI Engine (Refined for CME NQ Futures)
Tested on 8.8 Years of Real CME NQ Futures Data (2018-2026)
==============================================================================

User Question / Goal:
"CAN WE TRY MARKET STRUCTURE WITH INSTEAD... TRADING WITH THE TREND COULD BE 3-5MIN SETUPS...
WITH TONY IYKE STRATEGY... THE STRUCTURE, LIQUIDITY, AND POINT OF INTEREST STRATEGY
FOR SMALLER TIMEFRAME BUT HIGH TIMEFRAME CONTEXT"

Tony Iyke's Exact Institutional SMC Framework:
1. High Timeframe (HTF) Context:
   - 1-Hour & 15-Minute Trend: Bullish Market Structure (Higher Highs / Higher Lows / above 20 EMA).
   - HTF Discount / Premium POI:
     * In an Uptrend: We ONLY look for buys in the DISCOUNT zone (below the 50% equilibrium of the recent 15m dealing range).
     * In a Downtrend: We ONLY look for sells in the PREMIUM zone (above the 50% equilibrium of the recent 15m dealing range).

2. Inducement (IDM) Liquidity:
   - Retail enters prematurely at minor internal pullbacks.
   - Price sweeps this minor internal low (IDM) into the Discount POI.

3. Lower Timeframe (LTF) CHoCH (Change of Character):
   - A 3m or 5m displacement bar closes strongly in the direction of the trend with delta absorption.

4. Entry & Risk:
   - Passive limit at the 50% Mean Threshold (MT) of the Order Block.
   - Structural Stop: 1.0 pt below the swept Inducement low.
   - Take Profit: 1.5R and 2.0R.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List

print("=" * 80)
print("TESTING REFINED TONY IYKE SMC MARKET STRUCTURE (DISCOUNT/PREMIUM + IDM + CHOCH)")
print("=" * 80)

# 1. Load Data
df_1m = pd.read_csv('sample_data/NQ_1min_RTH_2018_2026.csv')
df_1m.columns = [c.lower() for c in df_1m.columns]
df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
df_1m = df_1m[df_1m['high'] > df_1m['low']].reset_index(drop=True)
df_1m = df_1m.set_index('datetime').sort_index()

# 1-min Delta proxy
rng_1m = (df_1m['high'] - df_1m['low']).replace(0, 1e-6)
df_1m['delta'] = df_1m['volume'] * (2.0 * df_1m['close'] - df_1m['high'] - df_1m['low']) / rng_1m

# 1H Trend
df_1h = df_1m.resample('1h', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_1h['ema20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
df_1h['htf_trend'] = np.where(df_1h['close'] > df_1h['ema20'], 1, -1)

COMMISSION_PTS = 0.26
SLIPPAGE_PTS   = 0.25

@dataclass
class Trade:
    year: int
    entry_time: pd.Timestamp
    direction: int
    entry_px: float
    stop_px: float
    risk_pts: float
    exit_px: float
    exit_reason: str
    net_pts: float
    real_r: float
    dur_bars: int
    setup_name: str

def run_tony_iyke_structure_engine(
    tf_str: str = '5min',
    target_r: float = 1.5,
    use_discount_premium: bool = True,
    use_absorption: bool = True,
    swing_idm_len: int = 5
) -> List[Trade]:

    df_tf = df_1m.resample(tf_str, closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'delta': 'sum'
    }).dropna().reset_index()

    tr = np.maximum(df_tf['high'] - df_tf['low'],
                    np.maximum(abs(df_tf['high'] - df_tf['close'].shift(1)),
                               abs(df_tf['low'] - df_tf['close'].shift(1))))
    df_tf['atr'] = tr.rolling(14).mean().bfill()
    df_tf['hour_ts'] = df_tf['datetime'].dt.floor('1h')
    df_tf['htf_trend'] = df_tf['hour_ts'].map(df_1h['htf_trend']).fillna(0)

    # Dealing Range Equilibrium (50% Discount / Premium) over 20 bars
    df_tf['range_high'] = df_tf['high'].shift(1).rolling(20).max()
    df_tf['range_low']  = df_tf['low'].shift(1).rolling(20).min()
    df_tf['equilibrium'] = (df_tf['range_high'] + df_tf['range_low']) / 2.0

    # Inducement Swings (IDM: minor local pullbacks)
    df_tf['idm_low']  = df_tf['low'].shift(1).rolling(swing_idm_len).min()
    df_tf['idm_high'] = df_tf['high'].shift(1).rolling(swing_idm_len).max()

    times = df_tf['datetime'].values
    opens = df_tf['open'].values
    highs = df_tf['high'].values
    lows = df_tf['low'].values
    closes = df_tf['close'].values
    deltas = df_tf['delta'].values
    atrs = df_tf['atr'].values
    htf = df_tf['htf_trend'].values
    eq = df_tf['equilibrium'].values
    idm_lows = df_tf['idm_low'].values
    idm_highs = df_tf['idm_high'].values

    n = len(df_tf)
    trades: List[Trade] = []
    in_trade_until = -1

    for i in range(swing_idm_len + 15, n - 40):
        if i <= in_trade_until:
            continue

        t = pd.Timestamp(times[i])
        h, m = t.hour, t.minute

        # RTH: 09:45 to 15:30
        if not ((h == 9 and m >= 45) or (10 <= h < 15) or (h == 15 and m <= 30)):
            continue

        ob_idx = i - 1
        ob_o, ob_h, ob_l, ob_c = opens[ob_idx], highs[ob_idx], lows[ob_idx], closes[ob_idx]
        ob_rng = ob_h - ob_l

        if ob_rng < 1.0:
            continue

        disp_body = abs(closes[i] - opens[i])
        is_bull_disp = (disp_body >= 0.70 * atrs[i]) and (closes[i] > opens[i])
        is_bear_disp = (disp_body >= 0.70 * atrs[i]) and (closes[i] < opens[i])

        # ── BULLISH SETUP (Trend Uptrend -> Discount POI -> Swept IDM -> CHoCH) ──
        if is_bull_disp and ob_c < ob_o:
            # 1. HTF Trend Alignment
            if htf[i] != 1:
                continue

            # 2. Discount POI Filter (Price must be in Discount zone, below 50% Dealing Range Equilibrium)
            if use_discount_premium and (ob_l > eq[ob_idx]):
                continue  # Reject: buying at Premium is retail behavior

            # 3. Inducement Sweep: Pierced minor internal swing low
            if np.isnan(idm_lows[ob_idx]) or ob_l >= idm_lows[ob_idx]:
                continue
            if ob_c < idm_lows[ob_idx] - 0.75 * atrs[i]:
                continue  # Broke structure down without rejection

            # 4. Absorption / Institutional Sponsorship: Rejection wick or positive delta
            lower_wick = min(ob_o, ob_c) - ob_l
            wick_ratio = lower_wick / ob_rng
            if use_absorption:
                has_absorp = (wick_ratio >= 0.20) or (deltas[ob_idx] > 0) or (deltas[i] > 0)
                if not has_absorp:
                    continue

            # 5. Order Block 50% Mean Threshold (MT) Entry
            block_top = max(ob_o, ob_c)
            block_bot = ob_l
            mt_entry = (block_top + block_bot) / 2.0
            stop_px = block_bot - 1.0
            risk_pts = mt_entry - stop_px

            if not (8.0 <= risk_pts <= 25.0):
                continue

            target_px = mt_entry + (risk_pts * target_r)

            # Limit fill simulation
            filled = False
            fill_k = -1
            for k in range(i + 1, min(n - 1, i + 15)):
                if lows[k] <= mt_entry:
                    filled = True
                    fill_k = k
                    break
                if highs[k] >= target_px:
                    break

            if not filled:
                continue

            exit_px = 0.0
            exit_reason = 'EOD'
            dur = 0
            for k in range(fill_k, min(n - 1, fill_k + 50)):
                dur += 1
                if lows[k] <= stop_px:
                    exit_px = stop_px
                    exit_reason = 'STOP'
                    in_trade_until = k
                    break
                if highs[k] >= target_px:
                    exit_px = target_px
                    exit_reason = 'TARGET'
                    in_trade_until = k
                    break

            if exit_px == 0.0:
                exit_px = closes[min(n - 1, fill_k + 49)]
                in_trade_until = min(n - 1, fill_k + 49)

            net_pts = (exit_px - mt_entry) - COMMISSION_PTS - SLIPPAGE_PTS
            trades.append(Trade(
                year=t.year, entry_time=pd.Timestamp(times[fill_k]), direction=1,
                entry_px=mt_entry, stop_px=stop_px, risk_pts=risk_pts, exit_px=exit_px,
                exit_reason=exit_reason, net_pts=net_pts, real_r=net_pts / risk_pts,
                dur_bars=dur, setup_name=f"TonyIyke_Bull_{tf_str}"
            ))

        # ── BEARISH SETUP (Trend Downtrend -> Premium POI -> Swept IDM -> CHoCH) ──
        elif is_bear_disp and ob_c > ob_o:
            if htf[i] != -1:
                continue

            # Premium POI Filter (Price must be in Premium zone, above 50% Equilibrium)
            if use_discount_premium and (ob_h < eq[ob_idx]):
                continue  # Reject: selling at Discount is retail behavior

            if np.isnan(idm_highs[ob_idx]) or ob_h <= idm_highs[ob_idx]:
                continue
            if ob_c > idm_highs[ob_idx] + 0.75 * atrs[i]:
                continue

            upper_wick = ob_h - max(ob_o, ob_c)
            wick_ratio = upper_wick / ob_rng
            if use_absorption:
                has_absorp = (wick_ratio >= 0.20) or (deltas[ob_idx] < 0) or (deltas[i] < 0)
                if not has_absorp:
                    continue

            block_top = ob_h
            block_bot = min(ob_o, ob_c)
            mt_entry = (block_top + block_bot) / 2.0
            stop_px = block_top + 1.0
            risk_pts = stop_px - mt_entry

            if not (8.0 <= risk_pts <= 25.0):
                continue

            target_px = mt_entry - (risk_pts * target_r)

            filled = False
            fill_k = -1
            for k in range(i + 1, min(n - 1, i + 15)):
                if highs[k] >= mt_entry:
                    filled = True
                    fill_k = k
                    break
                if lows[k] <= target_px:
                    break

            if not filled:
                continue

            exit_px = 0.0
            exit_reason = 'EOD'
            dur = 0
            for k in range(fill_k, min(n - 1, fill_k + 50)):
                dur += 1
                if highs[k] >= stop_px:
                    exit_px = stop_px
                    exit_reason = 'STOP'
                    in_trade_until = k
                    break
                if lows[k] <= target_px:
                    exit_px = target_px
                    exit_reason = 'TARGET'
                    in_trade_until = k
                    break

            if exit_px == 0.0:
                exit_px = closes[min(n - 1, fill_k + 49)]
                in_trade_until = min(n - 1, fill_k + 49)

            net_pts = (mt_entry - exit_px) - COMMISSION_PTS - SLIPPAGE_PTS
            trades.append(Trade(
                year=t.year, entry_time=pd.Timestamp(times[fill_k]), direction=-1,
                entry_px=mt_entry, stop_px=stop_px, risk_pts=risk_pts, exit_px=exit_px,
                exit_reason=exit_reason, net_pts=net_pts, real_r=net_pts / risk_pts,
                dur_bars=dur, setup_name=f"TonyIyke_Bear_{tf_str}"
            ))

    return trades

def print_results(trades: List[Trade], title: str):
    if not trades:
        print(f"\n{title}: NO TRADES GENERATED")
        return
    tdf = pd.DataFrame([t.__dict__ for t in trades])
    n = len(tdf)
    wins = tdf[tdf['net_pts'] > 0]
    losses = tdf[tdf['net_pts'] <= 0]
    wr = len(wins) / n * 100.0
    tot_gain = wins['net_pts'].sum()
    tot_loss = abs(losses['net_pts'].sum())
    pf = tot_gain / tot_loss if tot_loss > 0 else 99.0
    net_pts = tdf['net_pts'].sum()
    net_usd = net_pts * 20.0
    avg_sl = tdf['risk_pts'].mean()
    tpy = n / 8.8

    years = sorted(tdf['year'].unique())
    green_years = sum(tdf[tdf['year'] == y]['net_pts'].sum() > 0 for y in years)

    print(f"\n{'='*75}")
    print(f"CONFIGURATION: {title}")
    print(f"{'='*75}")
    print(f"Total Trades:        {n} ({tpy:.1f} trades/year, ~{tpy/52:.1f} trades/week)")
    print(f"Win Rate:            {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Profit Factor:       {pf:.2f}")
    print(f"Net Points:          +{net_pts:.1f} pts (${net_usd:,.2f} on 1 NQ / ${net_usd/10:,.2f} MNQ)")
    print(f"Avg Stop Loss:       {avg_sl:.1f} pts (${avg_sl*20:.1f} NQ / ${avg_sl*2:.1f} MNQ)")
    print(f"Yearly Consistency:  {green_years}/{len(years)} Green Years")

# RUN COMPREHENSIVE EXPERIMENTS
# Experiment 1: 5-Minute Tony Iyke (Discount/Premium POI + Inducement Sweep + CHoCH + 1.5R)
t1 = run_tony_iyke_structure_engine(tf_str='5min', target_r=1.5, use_discount_premium=True)
print_results(t1, "1. TONY IYKE 5-MIN (Discount/Premium POI + 5m IDM Sweep + CHoCH @ 1.5R)")

# Experiment 2: 5-Minute Tony Iyke @ 2.0R Target
t2 = run_tony_iyke_structure_engine(tf_str='5min', target_r=2.0, use_discount_premium=True)
print_results(t2, "2. TONY IYKE 5-MIN (Discount/Premium POI + 5m IDM Sweep + CHoCH @ 2.0R)")

# Experiment 3: 3-Minute Tony Iyke (Faster execution, more frequent IDM sweeps)
t3 = run_tony_iyke_structure_engine(tf_str='3min', target_r=1.5, use_discount_premium=True)
print_results(t3, "3. TONY IYKE 3-MIN (Discount/Premium POI + 3m IDM Sweep + CHoCH @ 1.5R)")

# Experiment 4: 3-Minute Tony Iyke @ 2.0R Target
t4 = run_tony_iyke_structure_engine(tf_str='3min', target_r=2.0, use_discount_premium=True)
print_results(t4, "4. TONY IYKE 3-MIN (Discount/Premium POI + 3m IDM Sweep + CHoCH @ 2.0R)")

# Experiment 5: 5-Minute Pure Trend & IDM (Active Weekly Setup without dealing range constraint)
t5 = run_tony_iyke_structure_engine(tf_str='5min', target_r=1.5, use_discount_premium=False)
print_results(t5, "5. ACTIVE 5-MIN TREND & IDM SWEEP (All Swings in 1H Trend @ 1.5R)")

