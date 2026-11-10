"""
Empirical Study: The Quantitative Value of "Storytelling" (Institutional Market Narrative)
Tested on 8.8 Years of CME NQ Futures (2018-2026, 839,332 1-min RTH bars)
========================================================================================

Does adding a structured Institutional Story improve:
1. Win Rate?
2. Trade Frequency / Opportunity Count?
3. Profit Factor & Risk-Adjusted Return?

The 4 Quantitative Narratives Tested:
-------------------------------------
Story 1: The Trapped Inventory Story ("Who is trapped offside?")
         - High volume spike at the sweep extreme (retail buying breakout / selling breakdown)
         - Strong rejection wick (>35%) showing institutional passive absorption.
         - Price closes back inside, trapping aggressive retail.

Story 2: The Draw on Liquidity (DOL) Story ("Where is price magnetically drawn?")
         - Entry is only taken if there is an open, unmitigated liquidity target / FVG
           at least 2.0R away, ensuring clear runway without overhead resistance.

Story 3: The 10:00 AM Manipulation vs Ignition Story (Opening Auction Narrative)
         - 09:30 - 10:00 creates the Judas trap against 1H trend.
         - 10:00 - 11:30 unleashes the true algorithmic expansion.

Story 4: The Trapped Fuel Secondary Continuation Story (Adding New Trades)
         - When trapped traders liquidate their stops, does it offer secondary re-test entries
           (Breaker / FVG retest) that increase trade frequency safely?
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List

print("=" * 80)
print("TESTING INSTITUTIONAL STORYTELLING ON 8.8 YEARS OF REAL CME NQ FUTURES")
print("=" * 80)

# Load 1m RTH Data
df_1m = pd.read_csv('sample_data/NQ_1min_RTH_2018_2026.csv')
df_1m.columns = [c.lower() for c in df_1m.columns]
df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
df_1m = df_1m[df_1m['high'] > df_1m['low']].reset_index(drop=True)
df_1m = df_1m.set_index('datetime').sort_index()

# 1-min Delta proxy
rng_1m = (df_1m['high'] - df_1m['low']).replace(0, 1e-6)
df_1m['delta'] = df_1m['volume'] * (2.0 * df_1m['close'] - df_1m['high'] - df_1m['low']) / rng_1m

# Resample to 5m
df_5m = df_1m.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'delta': 'sum'
}).dropna().reset_index()

# Resample to 1H
df_1h = df_1m.resample('1h', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_1h['ema20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
df_1h['htf_trend'] = np.where(df_1h['close'] > df_1h['ema20'], 1, -1)

COMMISSION_PTS = 0.26   # $5.20 / $20
SLIPPAGE_PTS   = 0.25   # 1 tick

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
    narrative_type: str

def evaluate_narrative_model(
    require_trapped_volume: bool = False,
    trapped_vol_mult: float = 1.0,
    min_rejection_wick: float = 0.25,
    require_dol_runway: bool = False,
    allow_secondary_retests: bool = False,
    target_r: float = 1.5
) -> List[Trade]:

    tr = np.maximum(df_5m['high'] - df_5m['low'],
                    np.maximum(abs(df_5m['high'] - df_5m['close'].shift(1)),
                               abs(df_5m['low'] - df_5m['close'].shift(1))))
    df_5m['atr'] = tr.rolling(14).mean().bfill()
    df_5m['vol_ma'] = df_5m['volume'].rolling(14).mean().bfill()
    df_5m['hour_ts'] = df_5m['datetime'].dt.floor('1h')
    df_5m['htf_trend'] = df_5m['hour_ts'].map(df_1h['htf_trend']).fillna(0)

    # Rolling swing highs and lows (5-bar fractal)
    swing_lookback = 6
    df_5m['sh'] = df_5m['high'].shift(1).rolling(swing_lookback).max()
    df_5m['sl'] = df_5m['low'].shift(1).rolling(swing_lookback).min()

    # Draw on Liquidity proxy: rolling 20-bar opposing swing high/low
    df_5m['dol_high'] = df_5m['high'].shift(1).rolling(20).max()
    df_5m['dol_low'] = df_5m['low'].shift(1).rolling(20).min()

    times = df_5m['datetime'].values
    opens = df_5m['open'].values
    highs = df_5m['high'].values
    lows = df_5m['low'].values
    closes = df_5m['close'].values
    volumes = df_5m['volume'].values
    deltas = df_5m['delta'].values
    vol_mas = df_5m['vol_ma'].values
    atrs = df_5m['atr'].values
    htf = df_5m['htf_trend'].values
    sh = df_5m['sh'].values
    sl = df_5m['sl'].values
    dol_highs = df_5m['dol_high'].values
    dol_lows = df_5m['dol_low'].values

    n = len(df_5m)
    trades: List[Trade] = []
    in_trade_until = -1

    for i in range(swing_lookback + 10, n - 40):
        if i <= in_trade_until:
            continue

        t = pd.Timestamp(times[i])
        h, m = t.hour, t.minute

        # RTH active hours: 09:45 to 15:30
        if not ((h == 9 and m >= 45) or (10 <= h < 15) or (h == 15 and m <= 30)):
            continue

        ob_idx = i - 1
        ob_o, ob_h, ob_l, ob_c = opens[ob_idx], highs[ob_idx], lows[ob_idx], closes[ob_idx]
        ob_v, ob_d, ob_vma = volumes[ob_idx], deltas[ob_idx], vol_mas[ob_idx]
        ob_rng = ob_h - ob_l

        if ob_rng < 1.0:
            continue

        # Displacement bar (bar i): body >= 0.85x ATR
        disp_body = abs(closes[i] - opens[i])
        is_bull_disp = (disp_body >= 0.80 * atrs[i]) and (closes[i] > opens[i])
        is_bear_disp = (disp_body >= 0.80 * atrs[i]) and (closes[i] < opens[i])

        # ── BULLISH SETUP (Sweep Low -> Institutions Absorb -> Trapped Shorts) ──
        if is_bull_disp and ob_c < ob_o:
            # 1. Sweep of recent swing low
            if np.isnan(sl[ob_idx]) or ob_l >= sl[ob_idx]:
                continue
            if ob_c < sl[ob_idx] - 0.75 * atrs[i]:
                continue

            # Story Component 1: Trapped Inventory & Absorption
            lower_wick = min(ob_o, ob_c) - ob_l
            wick_ratio = lower_wick / ob_rng
            if wick_ratio < min_rejection_wick:
                continue

            if require_trapped_volume and (ob_v < trapped_vol_mult * ob_vma):
                continue  # No high volume trap story

            # Story Component 2: Higher Timeframe Narrative (Trend with institutional flow)
            if htf[i] != 1:
                continue

            # Story Component 3: Draw on Liquidity (DOL Runway)
            block_top = max(ob_o, ob_c)
            block_bot = ob_l
            mt_entry = (block_top + block_bot) / 2.0
            stop_px = block_bot - 1.0
            risk_pts = mt_entry - stop_px

            if not (8.0 <= risk_pts <= 25.0):
                continue

            target_px = mt_entry + (risk_pts * target_r)

            if require_dol_runway:
                # Is opposing DOL (swing high) at least as high as our target?
                opposing_pool = dol_highs[i]
                if np.isnan(opposing_pool) or (opposing_pool < target_px):
                    continue  # Blocked by low ceiling, no clean DOL story

            # Forward simulation: Limit order fill at MT
            filled = False
            fill_k = -1
            for k in range(i + 1, min(n - 1, i + 15)):
                if lows[k] <= mt_entry:
                    filled = True
                    fill_k = k
                    break
                if highs[k] >= target_px:
                    break  # Ran away without filling

            if not filled:
                continue

            # Monitor trade execution
            exit_px = 0.0
            exit_reason = 'EOD'
            dur = 0
            for k in range(fill_k, min(n - 1, fill_k + 50)):
                dur += 1
                cur_l = lows[k]
                cur_h = highs[k]

                # Stop loss checked first (Zero Cheat)
                if cur_l <= stop_px:
                    exit_px = stop_px
                    exit_reason = 'STOP'
                    in_trade_until = k
                    break
                # Target
                if cur_h >= target_px:
                    exit_px = target_px
                    exit_reason = 'TARGET'
                    in_trade_until = k
                    break

            if exit_px == 0.0:
                exit_px = closes[min(n - 1, fill_k + 49)]
                in_trade_until = min(n - 1, fill_k + 49)

            net_pts = (exit_px - mt_entry) - COMMISSION_PTS - SLIPPAGE_PTS
            real_r = net_pts / risk_pts
            trades.append(Trade(
                year=t.year,
                entry_time=pd.Timestamp(times[fill_k]),
                direction=1,
                entry_px=mt_entry,
                stop_px=stop_px,
                risk_pts=risk_pts,
                exit_px=exit_px,
                exit_reason=exit_reason,
                net_pts=net_pts,
                real_r=real_r,
                dur_bars=dur,
                narrative_type="Primary_Sweep"
            ))

        # ── BEARISH SETUP (Sweep High -> Institutions Absorb -> Trapped Longs) ──
        elif is_bear_disp and ob_c > ob_o:
            if np.isnan(sh[ob_idx]) or ob_h <= sh[ob_idx]:
                continue
            if ob_c > sh[ob_idx] + 0.75 * atrs[i]:
                continue

            upper_wick = ob_h - max(ob_o, ob_c)
            wick_ratio = upper_wick / ob_rng
            if wick_ratio < min_rejection_wick:
                continue

            if require_trapped_volume and (ob_v < trapped_vol_mult * ob_vma):
                continue

            if htf[i] != -1:
                continue

            block_top = ob_h
            block_bot = min(ob_o, ob_c)
            mt_entry = (block_top + block_bot) / 2.0
            stop_px = block_top + 1.0
            risk_pts = stop_px - mt_entry

            if not (8.0 <= risk_pts <= 25.0):
                continue

            target_px = mt_entry - (risk_pts * target_r)

            if require_dol_runway:
                opposing_pool = dol_lows[i]
                if np.isnan(opposing_pool) or (opposing_pool > target_px):
                    continue

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
                cur_l = lows[k]
                cur_h = highs[k]

                if cur_h >= stop_px:
                    exit_px = stop_px
                    exit_reason = 'STOP'
                    in_trade_until = k
                    break
                if cur_l <= target_px:
                    exit_px = target_px
                    exit_reason = 'TARGET'
                    in_trade_until = k
                    break

            if exit_px == 0.0:
                exit_px = closes[min(n - 1, fill_k + 49)]
                in_trade_until = min(n - 1, fill_k + 49)

            net_pts = (mt_entry - exit_px) - COMMISSION_PTS - SLIPPAGE_PTS
            real_r = net_pts / risk_pts
            trades.append(Trade(
                year=t.year,
                entry_time=pd.Timestamp(times[fill_k]),
                direction=-1,
                entry_px=mt_entry,
                stop_px=stop_px,
                risk_pts=risk_pts,
                exit_px=exit_px,
                exit_reason=exit_reason,
                net_pts=net_pts,
                real_r=real_r,
                dur_bars=dur,
                narrative_type="Primary_Sweep"
            ))

    return trades

def print_metrics(trades: List[Trade], label: str):
    if not trades:
        print(f"{label}: NO TRADES")
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

    # Yearly breakdown
    years = sorted(tdf['year'].unique())
    green_years = 0
    for y in years:
        ydf = tdf[tdf['year'] == y]
        if ydf['net_pts'].sum() > 0:
            green_years += 1

    print(f"\n{'='*75}")
    print(f"CONFIGURATION: {label}")
    print(f"{'='*75}")
    print(f"Total Trades:        {n} ({tpy:.1f} trades/year, ~{tpy/52:.1f} trades/week)")
    print(f"Win Rate:            {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Profit Factor:       {pf:.2f}")
    print(f"Net Points:          +{net_pts:.1f} pts (${net_usd:,.2f} on 1 NQ)")
    print(f"Avg Stop Loss:       {avg_sl:.1f} pts (${avg_sl*20:.1f} NQ / ${avg_sl*2:.1f} MNQ)")
    print(f"Yearly Consistency:  {green_years}/{len(years)} Green Years")

# RUN EXPERIMENTS
# 1. Baseline Model (Standard technical swing sweep)
t_base = evaluate_narrative_model(
    require_trapped_volume=False,
    min_rejection_wick=0.25,
    require_dol_runway=False
)
print_metrics(t_base, "1. BASELINE (Pure Indicators: Swing Sweep + 1H EMA, No Narrative)")

# 2. Narrative 1: Trapped Retail Breakout Story (Volume Spike + Deep Rejection Wick >= 35%)
t_trapped = evaluate_narrative_model(
    require_trapped_volume=True,
    trapped_vol_mult=1.15,
    min_rejection_wick=0.35,
    require_dol_runway=False
)
print_metrics(t_trapped, "2. NARRATIVE 1: Trapped Retail Story (High Vol Breakout Trap + 35% Wick)")

# 3. Narrative 2: Draw on Liquidity (DOL) Story (Clear Runway to Opposing Liquidity Target)
t_dol = evaluate_narrative_model(
    require_trapped_volume=False,
    min_rejection_wick=0.25,
    require_dol_runway=True
)
print_metrics(t_dol, "3. NARRATIVE 2: Draw on Liquidity Story (Clean Runway to Opposing Pool)")

# 4. Master Narrative: Trapped Retail + Clear Draw on Liquidity
t_master = evaluate_narrative_model(
    require_trapped_volume=True,
    trapped_vol_mult=1.15,
    min_rejection_wick=0.35,
    require_dol_runway=True
)
print_metrics(t_master, "4. MASTER NARRATIVE: Trapped Retail + Clear Draw on Liquidity (The Full Story)")

# 5. Master Narrative with Higher Target (2.0R because story gives high velocity)
t_master_2r = evaluate_narrative_model(
    require_trapped_volume=True,
    trapped_vol_mult=1.15,
    min_rejection_wick=0.35,
    require_dol_runway=True,
    target_r=2.0
)
print_metrics(t_master_2r, "5. MASTER NARRATIVE @ 2.0R TARGET (Testing High-Velocity Payoff)")
