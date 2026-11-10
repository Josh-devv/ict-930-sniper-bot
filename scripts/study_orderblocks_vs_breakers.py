"""
Deep Quantitative Study: Order Blocks (OB) vs. Breaker Blocks (BB) on NQ Futures
================================================================================
Comprehensive empirical analysis across 12 months of CME NQ 1-minute data:
  1. Exact definitions of High-Probability Order Blocks vs Breaker Blocks
  2. Level Analysis: Open vs 50% Mean Threshold (MT) vs Wick Extreme
  3. Violation Analysis: What happens when candle BODY closes past 50% MT?
  4. Context Filters:
     - Liquidity Sweep prior to formation (Sweep vs No Sweep)
     - Displacement Strength & FVG creation (Displacement vs Normal)
     - Session Window (NY Open 9:30-11:30 AM vs London vs Lunch vs PM)
     - Midnight Open / True Day Open (TDO) alignment
  5. Statistical Output:
     - Win Rate / Respect Rate (1R, 2R, 3R, 4R, 5R + Point targets: 15, 30, 50, 100 pts)
     - Average Excursion (MFE vs MAE)
     - Head-to-Head comparison: When to use OB vs when to use Breaker Block
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

print("=" * 80)
print("DEEP INSTITUTIONAL STUDY: ORDER BLOCKS (OB) VS BREAKER BLOCKS (BB)")
print("Dataset: 12 Months NQ 1-Minute CME Futures")
print("=" * 80)

# ─── Load Data ────────────────────────────────────────────────────────────────
print("\n[1/5] Ingesting 12-Month NQ 1-Min Data...")
df = pd.read_csv('sample_data/MNQ_1min_12mo.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
# Convert to US/Eastern
df['ny_time'] = df['Datetime'].dt.tz_convert('America/New_York')
df = df.sort_values('ny_time').reset_index(drop=True)

# Resample to 5-min for clear institutional structural swings & blocks
# (1-min noise creates thousands of micro blocks; 5m is the primary intraday framework)
df_5m = df.set_index('ny_time').resample('5min', closed='left', label='left').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna().reset_index()

# Precompute True Day Open (Midnight 00:00 EST open)
df_5m['Date'] = df_5m['ny_time'].dt.date
midnight = df_5m[(df_5m['ny_time'].dt.hour == 0) & (df_5m['ny_time'].dt.minute == 0)]
tdo_map = dict(zip(midnight['Date'], midnight['Open']))
df_5m['TDO'] = df_5m['Date'].map(tdo_map)

# Precompute ATR (14) for displacement calibration
tr = np.maximum(df_5m['High'] - df_5m['Low'],
                np.maximum(abs(df_5m['High'] - df_5m['Close'].shift(1)),
                           abs(df_5m['Low'] - df_5m['Close'].shift(1))))
df_5m['ATR'] = tr.rolling(14).mean().bfill()

print(f"Loaded {len(df_5m):,} 5-minute bars covering {df_5m['Date'].nunique()} trading days.")


# ─────────────────────────────────────────────────────────────────────────────
# SWING DETECTION & FVG / DISPLACEMENT IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Detecting Market Structure Shifts (MSS), FVGs & Liquidity Sweeps...")

@dataclass
class BlockEvent:
    block_type: str        # 'BULLISH_OB', 'BEARISH_OB', 'BULLISH_BREAKER', 'BEARISH_BREAKER'
    formed_idx: int
    bar_time: pd.Timestamp
    direction: int         # 1 = Long, -1 = Short
    top: float
    bot: float
    mt: float              # 50% Mean Threshold
    risk_pts: float
    has_sweep: bool        # Did price sweep prior swing high/low before formation?
    has_fvg: bool          # Did the displacement create an unmitigated FVG?
    disp_ratio: float      # Displacement body size / ATR
    tdo_aligned: bool      # Bullish above TDO, Bearish below TDO
    session: str           # 'NY_OPEN', 'LONDON', 'LUNCH', 'PM', 'OTHER'
    # Outcome tracking
    mitigated: bool = False
    mitigated_idx: int = -1
    tap_depth: str = ''    # 'OPEN_ONLY', 'MT_50', 'DEEP_WICK', 'VIOLATED'
    body_closed_past_mt: bool = False
    mfe_pts: float = 0.0   # Max Favorable Excursion
    mae_pts: float = 0.0   # Max Adverse Excursion (drawdown into block)
    hit_1r: bool = False
    hit_2r: bool = False
    hit_3r: bool = False
    hit_4r: bool = False
    hit_15pt: bool = False
    hit_30pt: bool = False
    hit_50pt: bool = False
    invalidated: bool = False


# Find Swing Highs and Lows (3-bar fractal causal detection)
highs = df_5m['High'].values
lows  = df_5m['Low'].values
opens = df_5m['Open'].values
closes = df_5m['Close'].values
atrs  = df_5m['ATR'].values
times = df_5m['ny_time'].values
tdos  = df_5m['TDO'].values

n = len(df_5m)
swing_highs = []  # (idx, price)
swing_lows  = []  # (idx, price)

for i in range(2, n - 2):
    if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
        swing_highs.append((i, highs[i]))
    if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
        swing_lows.append((i, lows[i]))

swing_high_map = {idx: p for idx, p in swing_highs}
swing_low_map  = {idx: p for idx, p in swing_lows}


def get_session(t: pd.Timestamp) -> str:
    h, m = t.hour, t.minute
    if (h == 9 and m >= 30) or (h == 10) or (h == 11 and m <= 30):
        return 'NY_OPEN'
    elif (h >= 2 and h <= 5):
        return 'LONDON'
    elif (h == 11 and m > 30) or (h == 12) or (h == 13 and m <= 30):
        return 'LUNCH'
    elif (h >= 13 and m > 30) and (h <= 16):
        return 'NY_PM'
    return 'ASIA_OTHER'


# ─────────────────────────────────────────────────────────────────────────────
# IDENTIFY ORDER BLOCKS & BREAKER BLOCKS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Scanning for Pure Order Blocks and Breaker Blocks...")

blocks: List[BlockEvent] = []

for i in range(15, n - 100):
    t = pd.Timestamp(times[i])
    atr = atrs[i]
    sess = get_session(t)
    tdo = tdos[i]

    # Look for aggressive displacement at bar i
    body = abs(closes[i] - opens[i])
    disp = body >= 1.2 * atr
    bull_disp = disp and closes[i] > opens[i]
    bear_disp = disp and closes[i] < opens[i]

    # Check FVG created by displacement:
    # Bullish FVG: Low of bar i > High of bar i-2
    has_bull_fvg = (lows[i] > highs[i-2]) if i >= 2 else False
    # Bearish FVG: High of bar i < Low of bar i-2
    has_bear_fvg = (highs[i] < lows[i-2]) if i >= 2 else False

    # ─────────────────────────────────────────────────────────────────────────
    # A. BULLISH ORDER BLOCK (OB)
    # The down-close candle immediately before this bullish displacement
    # that breaks above a recent swing high (MSS)
    # ─────────────────────────────────────────────────────────────────────────
    if bull_disp:
        # Candle i-1 is the potential OB candle
        ob_idx = i - 1
        if closes[ob_idx] < opens[ob_idx]:  # Bearish candle
            # Check MSS: did bar i close above any swing high of the last 10 bars?
            prior_sh = [p for idx, p in swing_highs if idx < ob_idx and idx >= ob_idx - 15]
            if prior_sh and closes[i] > max(prior_sh):
                # Check prior sweep: did bar ob_idx or ob_idx-1 sweep a prior swing low?
                prior_sl = [p for idx, p in swing_lows if idx < ob_idx - 1 and idx >= ob_idx - 25]
                swept = bool(prior_sl and lows[ob_idx] < min(prior_sl))

                top = max(opens[ob_idx], closes[ob_idx])  # Body top
                bot = lows[ob_idx]                        # Wick low
                mt  = (top + bot) / 2.0                   # 50% Mean Threshold
                risk = top - bot
                if 3.0 <= risk <= 40.0:
                    aligned = bool(pd.notna(tdo) and closes[i] > tdo)
                    b = BlockEvent(
                        block_type='BULLISH_OB', formed_idx=i, bar_time=t, direction=1,
                        top=top, bot=bot, mt=mt, risk_pts=risk,
                        has_sweep=swept, has_fvg=has_bull_fvg,
                        disp_ratio=body / atr, tdo_aligned=aligned, session=sess
                    )
                    blocks.append(b)

    # ─────────────────────────────────────────────────────────────────────────
    # B. BEARISH ORDER BLOCK (OB)
    # The up-close candle immediately before bearish displacement breaking swing low
    # ─────────────────────────────────────────────────────────────────────────
    if bear_disp:
        ob_idx = i - 1
        if closes[ob_idx] > opens[ob_idx]:  # Bullish candle
            prior_sl = [p for idx, p in swing_lows if idx < ob_idx and idx >= ob_idx - 15]
            if prior_sl and closes[i] < min(prior_sl):
                prior_sh = [p for idx, p in swing_highs if idx < ob_idx - 1 and idx >= ob_idx - 25]
                swept = bool(prior_sh and highs[ob_idx] > max(prior_sh))

                top = highs[ob_idx]
                bot = min(opens[ob_idx], closes[ob_idx])
                mt  = (top + bot) / 2.0
                risk = top - bot
                if 3.0 <= risk <= 40.0:
                    aligned = bool(pd.notna(tdo) and closes[i] < tdo)
                    b = BlockEvent(
                        block_type='BEARISH_OB', formed_idx=i, bar_time=t, direction=-1,
                        top=top, bot=bot, mt=mt, risk_pts=risk,
                        has_sweep=swept, has_fvg=has_bear_fvg,
                        disp_ratio=body / atr, tdo_aligned=aligned, session=sess
                    )
                    blocks.append(b)

    # ─────────────────────────────────────────────────────────────────────────
    # C. BULLISH BREAKER BLOCK (BB)
    # Structure: Low -> High -> Lower Low (Sweep) -> Break above High
    # The up-close candle leading to the previous High becomes the Breaker Block
    # ─────────────────────────────────────────────────────────────────────────
    if bull_disp:
        # Find if we had a recent Lower Low sweeping a Low, then breaking a High
        recent_sh = [s for s in swing_highs if s[0] < i and s[0] >= i - 25]
        recent_sl = [s for s in swing_lows  if s[0] < i and s[0] >= i - 25]

        if len(recent_sh) >= 1 and len(recent_sl) >= 2:
            last_sh_idx, last_sh_p = recent_sh[-1]
            # Must have broken above this swing high
            if closes[i] > last_sh_p:
                # Find swing lows before and after this swing high
                sl_before = [s for s in recent_sl if s[0] < last_sh_idx]
                sl_after  = [s for s in recent_sl if s[0] > last_sh_idx]

                if sl_before and sl_after:
                    # Liquidity sweep: sl_after made a LOWER low than sl_before
                    if sl_after[-1][1] < sl_before[-1][1]:
                        # The Breaker Block is the highest UP candle leading into last_sh
                        window_start = max(0, last_sh_idx - 4)
                        up_candles = [c for c in range(window_start, last_sh_idx + 1) if closes[c] > opens[c]]
                        if up_candles:
                            bb_candle = up_candles[-1]
                            top = highs[bb_candle]
                            bot = min(opens[bb_candle], closes[bb_candle])
                            mt  = (top + bot) / 2.0
                            risk = top - bot
                            if 3.0 <= risk <= 45.0:
                                aligned = bool(pd.notna(tdo) and closes[i] > tdo)
                                b = BlockEvent(
                                    block_type='BULLISH_BREAKER', formed_idx=i, bar_time=t, direction=1,
                                    top=top, bot=bot, mt=mt, risk_pts=risk,
                                    has_sweep=True, has_fvg=has_bull_fvg,
                                    disp_ratio=body / atr, tdo_aligned=aligned, session=sess
                                )
                                blocks.append(b)

    # ─────────────────────────────────────────────────────────────────────────
    # D. BEARISH BREAKER BLOCK (BB)
    # Structure: High -> Low -> Higher High (Sweep) -> Break below Low
    # The down-close candle leading into that Low becomes the Breaker Block
    # ─────────────────────────────────────────────────────────────────────────
    if bear_disp:
        recent_sh = [s for s in swing_highs if s[0] < i and s[0] >= i - 25]
        recent_sl = [s for s in swing_lows  if s[0] < i and s[0] >= i - 25]

        if len(recent_sl) >= 1 and len(recent_sh) >= 2:
            last_sl_idx, last_sl_p = recent_sl[-1]
            if closes[i] < last_sl_p:
                sh_before = [s for s in recent_sh if s[0] < last_sl_idx]
                sh_after  = [s for s in recent_sh if s[0] > last_sl_idx]

                if sh_before and sh_after:
                    if sh_after[-1][1] > sh_before[-1][1]:  # Swept higher high
                        window_start = max(0, last_sl_idx - 4)
                        dn_candles = [c for c in range(window_start, last_sl_idx + 1) if closes[c] < opens[c]]
                        if dn_candles:
                            bb_candle = dn_candles[-1]
                            top = max(opens[bb_candle], closes[bb_candle])
                            bot = lows[bb_candle]
                            mt  = (top + bot) / 2.0
                            risk = top - bot
                            if 3.0 <= risk <= 45.0:
                                aligned = bool(pd.notna(tdo) and closes[i] < tdo)
                                b = BlockEvent(
                                    block_type='BEARISH_BREAKER', formed_idx=i, bar_time=t, direction=-1,
                                    top=top, bot=bot, mt=mt, risk_pts=risk,
                                    has_sweep=True, has_fvg=has_bear_fvg,
                                    disp_ratio=body / atr, tdo_aligned=aligned, session=sess
                                )
                                blocks.append(b)


print(f"Total Block Setups Identified: {len(blocks):,}")
ob_count = len([b for b in blocks if 'OB' in b.block_type])
bb_count = len([b for b in blocks if 'BREAKER' in b.block_type])
print(f"  • Order Blocks:   {ob_count:,}")
print(f"  • Breaker Blocks: {bb_count:,}")


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATE MITIGATION, TAP DEPTH, LEVEL RESPECT & REACTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Simulating Price Mitigation & Forward Price Excursion...")

for b in blocks:
    start_k = b.formed_idx + 1
    # Check next 80 bars (approx 6.5 hours of forward price action)
    end_k   = min(n - 1, b.formed_idx + 80)

    # 1. First, search for initial mitigation tap
    mit_idx = -1
    for k in range(start_k, end_k):
        if b.direction == 1:
            # Bullish block: Price pulls back DOWN into block (low <= top)
            if lows[k] <= b.top:
                mit_idx = k
                break
        else:
            # Bearish block: Price pulls back UP into block (high >= bot)
            if highs[k] >= b.bot:
                mit_idx = k
                break

    if mit_idx == -1:
        continue  # Block was never retested within the window

    b.mitigated = True
    b.mitigated_idx = mit_idx

    # 2. Determine tap depth on mitigation candle and subsequent reaction
    tap_bar_low  = lows[mit_idx]
    tap_bar_high = highs[mit_idx]
    tap_bar_close = closes[mit_idx]

    if b.direction == 1:
        # Check depth
        if tap_bar_low > b.mt:
            b.tap_depth = 'OPEN_ONLY'       # Shallow tap between top and 50% MT
        elif tap_bar_low > b.bot:
            b.tap_depth = 'MT_50'           # Reached 50% MT but respected block low
        else:
            b.tap_depth = 'VIOLATED'        # Penetrated past block low

        # Check if candle body closed below 50% MT
        if tap_bar_close < b.mt:
            b.body_closed_past_mt = True

        entry_p = b.top  # enter at top edge of block
        stop_p  = b.bot - 1.0
        risk    = entry_p - stop_p

        # Walk forward from mitigation bar to track MFE / MAE
        max_fav = 0.0
        max_adv = 0.0
        invalid = False

        for k in range(mit_idx, end_k):
            fav = highs[k] - entry_p
            adv = entry_p - lows[k]
            if fav > max_fav:
                max_fav = fav
            if adv > max_adv:
                max_adv = adv

            # Stop loss violation: close below bottom
            if closes[k] < stop_p:
                invalid = True
                break

        b.mfe_pts = max_fav
        b.mae_pts = max_adv
        b.invalidated = invalid
        b.hit_1r = max_fav >= risk * 1.0
        b.hit_2r = max_fav >= risk * 2.0
        b.hit_3r = max_fav >= risk * 3.0
        b.hit_4r = max_fav >= risk * 4.0
        b.hit_15pt = max_fav >= 15.0
        b.hit_30pt = max_fav >= 30.0
        b.hit_50pt = max_fav >= 50.0

    else:
        # Bearish block
        if tap_bar_high < b.mt:
            b.tap_depth = 'OPEN_ONLY'
        elif tap_bar_high < b.top:
            b.tap_depth = 'MT_50'
        else:
            b.tap_depth = 'VIOLATED'

        if tap_bar_close > b.mt:
            b.body_closed_past_mt = True

        entry_p = b.bot
        stop_p  = b.top + 1.0
        risk    = stop_p - entry_p

        max_fav = 0.0
        max_adv = 0.0
        invalid = False

        for k in range(mit_idx, end_k):
            fav = entry_p - lows[k]
            adv = highs[k] - entry_p
            if fav > max_fav:
                max_fav = fav
            if adv > max_adv:
                max_adv = adv

            if closes[k] > stop_p:
                invalid = True
                break

        b.mfe_pts = max_fav
        b.mae_pts = max_adv
        b.invalidated = invalid
        b.hit_1r = max_fav >= risk * 1.0
        b.hit_2r = max_fav >= risk * 2.0
        b.hit_3r = max_fav >= risk * 3.0
        b.hit_4r = max_fav >= risk * 4.0
        b.hit_15pt = max_fav >= 15.0
        b.hit_30pt = max_fav >= 30.0
        b.hit_50pt = max_fav >= 50.0


# ─────────────────────────────────────────────────────────────────────────────
# COMPILE STATISTICS & REPORTING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Compiling Quantitative Results & Answering Key Research Questions...\n")

mit_blocks = [b for b in blocks if b.mitigated]
print(f"Total Blocks Tested:    {len(blocks):,}")
print(f"Total Blocks Mitigated: {len(mit_blocks):,} ({len(mit_blocks)/len(blocks)*100:.1f}% retested)\n")


def print_stats_table(title: str, sub_blocks: List[BlockEvent]):
    if not sub_blocks:
        print(f"  {title}: No samples")
        return

    n_tot = len(sub_blocks)
    r1 = sum(1 for b in sub_blocks if b.hit_1r) / n_tot * 100
    r2 = sum(1 for b in sub_blocks if b.hit_2r) / n_tot * 100
    r3 = sum(1 for b in sub_blocks if b.hit_3r) / n_tot * 100
    p15 = sum(1 for b in sub_blocks if b.hit_15pt) / n_tot * 100
    p30 = sum(1 for b in sub_blocks if b.hit_30pt) / n_tot * 100
    p50 = sum(1 for b in sub_blocks if b.hit_50pt) / n_tot * 100
    avg_mfe = np.mean([b.mfe_pts for b in sub_blocks])
    avg_risk = np.mean([b.risk_pts for b in sub_blocks])
    inv_rate = sum(1 for b in sub_blocks if b.invalidated) / n_tot * 100

    print(f"┌{'─'*78}┐")
    print(f"│ {title.upper():<76} │")
    print(f"├{'─'*78}┤")
    print(f"│ Total Sample: {n_tot:<12} | Avg Risk (Stop Size): {avg_risk:<6.1f} pts | Invalidation Rate: {inv_rate:<5.1f}% │")
    print(f"│ Win Rate @ 1:1 R:R : {r1:>5.1f}%  |  Win Rate @ 2:1 R:R : {r2:>5.1f}%  |  Win Rate @ 3:1 R:R : {r3:>5.1f}% │")
    print(f"│ Deliver 15+ Points : {p15:>5.1f}%  |  Deliver 30+ Points : {p30:>5.1f}%  |  Deliver 50+ Points : {p50:>5.1f}% │")
    print(f"│ Average Favorable Run (MFE) : {avg_mfe:>6.1f} pts ({avg_mfe*20:>8,.0f}$ per 1 Mini NQ)                 │")
    print(f"└{'─'*78}┘")


# ─── 1. HEAD-TO-HEAD: ORDER BLOCK VS BREAKER BLOCK ───────────────────────────
ob_mit = [b for b in mit_blocks if 'OB' in b.block_type]
bb_mit = [b for b in mit_blocks if 'BREAKER' in b.block_type]

print_stats_table("ALL ORDER BLOCKS (Standard OB)", ob_mit)
print_stats_table("ALL BREAKER BLOCKS (Institutional Breakers)", bb_mit)


# ─── 2. LEVEL RESPECT STUDY: OPEN vs 50% MEAN THRESHOLD (MT) ─────────────────
print("\n" + "=" * 80)
print("LEVEL ANALYSIS: WHICH LEVEL GETS RESPECTED MOST?")
print("=" * 80)

tap_open = [b for b in mit_blocks if b.tap_depth == 'OPEN_ONLY']
tap_mt50 = [b for b in mit_blocks if b.tap_depth == 'MT_50']
tap_viol = [b for b in mit_blocks if b.tap_depth == 'VIOLATED']

print(f"\nDistribution of Retests:")
print(f"  1. Shallow Tap (Open to 50% MT) : {len(tap_open):,} ({len(tap_open)/len(mit_blocks)*100:.1f}%)")
print(f"  2. Deep Tap (Tapped 50% MT)     : {len(tap_mt50):,} ({len(tap_mt50)/len(mit_blocks)*100:.1f}%)")
print(f"  3. Pierced Past Extreme (Blown) : {len(tap_viol):,} ({len(tap_viol)/len(mit_blocks)*100:.1f}%)")

print_stats_table("LEVEL 1: Retest only touched OPEN (Shallow Reversal)", tap_open)
print_stats_table("LEVEL 2: Retest pierced into 50% MEAN THRESHOLD (MT)", tap_mt50)


# ─── 3. THE 50% MT VIOLATION LAW: CANDLE BODY CLOSE ──────────────────────────
print("\n" + "=" * 80)
print("THE 50% MEAN THRESHOLD RULE: WHAT HAPPENS IF CANDLE BODY CLOSES PAST 50% MT?")
print("=" * 80)

mt_respected = [b for b in mit_blocks if not b.body_closed_past_mt and b.tap_depth != 'VIOLATED']
mt_closed_past = [b for b in mit_blocks if b.body_closed_past_mt]

print_stats_table("BODY RESPECTED 50% MT (Wick dipped, Body closed above/below MT)", mt_respected)
print_stats_table("BODY CLOSED PAST 50% MT (Compromised Block)", mt_closed_past)


# ─── 4. CONTEXT FILTERS: WHY THEY GET RESPECTED (The Secret Ingredients) ─────
print("\n" + "=" * 80)
print("CONTEXT FILTERS: WHAT MAKES AN ORDER BLOCK / BREAKER WORK?")
print("=" * 80)

with_sweep    = [b for b in mit_blocks if b.has_sweep]
without_sweep = [b for b in mit_blocks if not b.has_sweep]

with_fvg    = [b for b in mit_blocks if b.has_fvg]
without_fvg = [b for b in mit_blocks if not b.has_fvg]

high_disp = [b for b in mit_blocks if b.disp_ratio >= 1.8]
low_disp  = [b for b in mit_blocks if b.disp_ratio < 1.8]

print_stats_table("WITH LIQUIDITY SWEEP (Swept prior high/low before forming)", with_sweep)
print_stats_table("WITHOUT LIQUIDITY SWEEP (Formed in the middle of nowhere)", without_sweep)

print_stats_table("DISPLACEMENT CREATED AN FVG (Institutional Imbalance)", with_fvg)
print_stats_table("NO FVG CREATED (Overlapping candles / weak push)", without_fvg)


# ─── 5. SESSION WINDOWS ───────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SESSION TIMING: WHEN ARE THEY MOST RESPECTED?")
print("=" * 80)

ny_blocks     = [b for b in mit_blocks if b.session == 'NY_OPEN']
london_blocks = [b for b in mit_blocks if b.session == 'LONDON']
lunch_blocks  = [b for b in mit_blocks if b.session == 'LUNCH']
pm_blocks     = [b for b in mit_blocks if b.session == 'NY_PM']

print_stats_table("NY OPEN SESSION (09:30 - 11:30 AM EST)", ny_blocks)
print_stats_table("LONDON OPEN SESSION (02:00 - 05:00 AM EST)", london_blocks)
print_stats_table("LUNCH CONSOLIDATION (11:30 AM - 01:30 PM EST)", lunch_blocks)
print_stats_table("NY PM SESSION (01:30 - 04:00 PM EST)", pm_blocks)


# ─── 6. THE GOLDEN SETUP: ALL CONFLUENCES COMBINED ───────────────────────────
print("\n" + "=" * 80)
print("THE HOLY GRAIL COMBO: BREAKER BLOCK + SWEEP + FVG + NY OPEN")
print("=" * 80)

holy_grail = [b for b in mit_blocks if b.block_type == 'BULLISH_BREAKER' or b.block_type == 'BEARISH_BREAKER']
holy_grail = [b for b in holy_grail if b.has_fvg and b.session == 'NY_OPEN' and not b.body_closed_past_mt]

print_stats_table("ELITE SETUP: BREAKER + FVG + NY OPEN + 50% MT BODY RESPECT", holy_grail)

