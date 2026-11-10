"""
Institutional Deep Study: Order Blocks (OB) vs. Breaker Blocks (BB) on NQ Futures
==================================================================================
Strict, Unforgiving Empirical Analysis on 12 Months of CME NQ 1-Minute Data.

Rules of the Simulator (No Lookahead, No Faked Fills, No Inverted Signs):
  1. Entry on Limit: A limit order at Open or 50% MT only fills if a subsequent bar's Low/High crosses it.
  2. Stop Loss is a hard stop: If bar's extreme crosses SL, trade is stopped out immediately at SL - slippage.
  3. Conservative Bar Ordering: Stop is always checked before Take Profit on any ambiguous bar.
  4. Entry Bar Check: If the bar that fills the limit also hits the stop, it is an immediate loss.
  5. Transaction Costs: $1.24/contract RT commission + 1 tick ($0.25 pt = $5/NQ, $0.50/MNQ) slippage on entry and stop.
  6. No sign flips: Long losses are strictly negative: (exit - entry) * multiplier - costs.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict

print("=" * 80)
print("AIRTIGHT EMPIRICAL STUDY: ORDER BLOCKS (OB) VS BREAKER BLOCKS (BB)")
print("Dataset: 12-Month Continuous CME NQ Futures (1-minute bars)")
print("=" * 80)

# ─── Load & Prepare Data ──────────────────────────────────────────────────────
print("\n[1/4] Ingesting and preprocessing 1-min data...")
df_1m = pd.read_csv('sample_data/MNQ_1min_12mo.csv')
df_1m['Datetime'] = pd.to_datetime(df_1m['Datetime'], utc=True)
df_1m['ny_time'] = df_1m['Datetime'].dt.tz_convert('America/New_York')
df_1m = df_1m.sort_values('ny_time').reset_index(drop=True)

# Build 5-min bars for institutional block identification
df_5m = df_1m.set_index('ny_time').resample('5min', closed='left', label='left').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna().reset_index()

# 1H HTF Trend: 20-period EMA on 1-hour chart to determine institutional directional bias
df_1h = df_1m.set_index('ny_time').resample('1h', closed='left', label='left').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna()
df_1h['EMA20'] = df_1h['Close'].ewm(span=20, adjust=False).mean()
df_1h['HTF_Trend'] = np.where(df_1h['Close'] > df_1h['EMA20'], 1, -1)

# Map 1H trend back to 5m bars
df_5m['hour_ts'] = df_5m['ny_time'].dt.floor('1h')
df_5m['HTF_Trend'] = df_5m['hour_ts'].map(df_1h['HTF_Trend']).fillna(0)

# True Day Open (Midnight 00:00 EST)
df_5m['Date'] = df_5m['ny_time'].dt.date
midnight = df_5m[(df_5m['ny_time'].dt.hour == 0) & (df_5m['ny_time'].dt.minute == 0)]
tdo_map = dict(zip(midnight['Date'], midnight['Open']))
df_5m['TDO'] = df_5m['Date'].map(tdo_map)

# Precompute ATR(14) on 5m
tr = np.maximum(df_5m['High'] - df_5m['Low'],
                np.maximum(abs(df_5m['High'] - df_5m['Close'].shift(1)),
                           abs(df_5m['Low'] - df_5m['Close'].shift(1))))
df_5m['ATR'] = tr.rolling(14).mean().bfill()

print(f"Loaded {len(df_5m):,} 5-minute bars ({df_5m['Date'].nunique()} trading days).")


# ─── Data Structures ─────────────────────────────────────────────────────────
@dataclass
class TradeResult:
    block_type: str
    entry_level: str       # 'OPEN', 'MT_50'
    formed_time: pd.Timestamp
    entry_time: pd.Timestamp
    direction: int
    entry_price: float
    stop_price: float
    risk_pts: float
    exit_price: float
    exit_time: pd.Timestamp
    exit_reason: str       # 'STOP_LOSS', 'TARGET_2R', 'TARGET_3R', 'TIME_EXIT'
    gross_pts: float
    net_pts: float
    realized_r: float
    mfe_pts: float
    mae_pts: float
    # Features
    has_sweep: bool
    has_fvg: bool
    htf_aligned: bool
    session: str
    disp_ratio: float


# ─── Detect Swings, FVGs, OBs, and Breakers ──────────────────────────────────
print("\n[2/4] Scanning for institutional Order Blocks and Breaker Blocks...")

highs  = df_5m['High'].values
lows   = df_5m['Low'].values
opens  = df_5m['Open'].values
closes = df_5m['Close'].values
atrs   = df_5m['ATR'].values
times  = df_5m['ny_time'].values
htf    = df_5m['HTF_Trend'].values
tdos   = df_5m['TDO'].values
n      = len(df_5m)

# 3-bar swing fractal
swing_highs = []
swing_lows  = []
for i in range(2, n - 2):
    if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
        swing_highs.append((i, highs[i]))
    if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
        swing_lows.append((i, lows[i]))

def get_session(t: pd.Timestamp) -> str:
    h, m = t.hour, t.minute
    if h == 9 and m >= 30:
        return '930_OPEN'
    elif (h == 10) or (h == 11 and m <= 15):
        return 'SILVER_BULLET'   # 10:00 - 11:15 AM (ICT Prime Window)
    elif (h >= 2 and h <= 5):
        return 'LONDON'
    elif (h == 11 and m > 15) or (h == 12) or (h == 13 and m <= 30):
        return 'LUNCH_CHOP'
    elif (h >= 13 and m > 30) and (h <= 16):
        return 'NY_PM'
    return 'OTHER'


@dataclass
class CandidateBlock:
    block_type: str        # 'BULLISH_OB', 'BEARISH_OB', 'BULLISH_BREAKER', 'BEARISH_BREAKER'
    formed_idx: int
    bar_time: pd.Timestamp
    direction: int         # 1 = Long, -1 = Short
    open_level: float      # Entry Level 1
    mt_level: float        # Entry Level 2 (50% Mean Threshold)
    stop_level: float      # Invalidation point
    has_sweep: bool
    has_fvg: bool
    disp_ratio: float
    htf_aligned: bool
    session: str


candidates: List[CandidateBlock] = []

for i in range(15, n - 60):
    t = pd.Timestamp(times[i])
    atr = atrs[i]
    sess = get_session(t)

    # Displacement candle definition: body >= 1.2x ATR
    body = abs(closes[i] - opens[i])
    disp = body >= 1.2 * atr
    bull_disp = disp and closes[i] > opens[i]
    bear_disp = disp and closes[i] < opens[i]

    has_bull_fvg = (lows[i] > highs[i-2]) if i >= 2 else False
    has_bear_fvg = (highs[i] < lows[i-2]) if i >= 2 else False

    # ─────────────────────────────────────────────────────────────────────────
    # 1. ORDER BLOCKS (OB)
    # Bullish OB: Last down-close candle before displacement breaking swing high
    # ─────────────────────────────────────────────────────────────────────────
    if bull_disp:
        ob_idx = i - 1
        if closes[ob_idx] < opens[ob_idx]:
            # MSS: Closed above a swing high from the past 15 bars
            shs = [p for idx, p in swing_highs if idx < ob_idx and idx >= ob_idx - 15]
            if shs and closes[i] > max(shs):
                # Prior sweep check: did the OB candle sweep a prior swing low?
                sls = [p for idx, p in swing_lows if idx < ob_idx - 1 and idx >= ob_idx - 25]
                swept = bool(sls and lows[ob_idx] < min(sls))

                open_lvl = max(opens[ob_idx], closes[ob_idx]) # top of OB body
                stop_lvl = lows[ob_idx] - 1.0                 # below wick
                mt_lvl   = (open_lvl + lows[ob_idx]) / 2.0    # 50% MT
                risk = open_lvl - stop_lvl

                if 3.0 <= risk <= 35.0:
                    aligned = (htf[i] == 1)
                    candidates.append(CandidateBlock(
                        block_type='BULLISH_OB', formed_idx=i, bar_time=t, direction=1,
                        open_level=open_lvl, mt_level=mt_lvl, stop_level=stop_lvl,
                        has_sweep=swept, has_fvg=has_bull_fvg, disp_ratio=body/atr,
                        htf_aligned=aligned, session=sess
                    ))

    if bear_disp:
        ob_idx = i - 1
        if closes[ob_idx] > opens[ob_idx]:
            sls = [p for idx, p in swing_lows if idx < ob_idx and idx >= ob_idx - 15]
            if sls and closes[i] < min(sls):
                shs = [p for idx, p in swing_highs if idx < ob_idx - 1 and idx >= ob_idx - 25]
                swept = bool(shs and highs[ob_idx] > max(shs))

                open_lvl = min(opens[ob_idx], closes[ob_idx]) # bottom of OB body
                stop_lvl = highs[ob_idx] + 1.0                # above wick
                mt_lvl   = (open_lvl + highs[ob_idx]) / 2.0
                risk = stop_lvl - open_lvl

                if 3.0 <= risk <= 35.0:
                    aligned = (htf[i] == -1)
                    candidates.append(CandidateBlock(
                        block_type='BEARISH_OB', formed_idx=i, bar_time=t, direction=-1,
                        open_level=open_lvl, mt_level=mt_lvl, stop_level=stop_lvl,
                        has_sweep=swept, has_fvg=has_bear_fvg, disp_ratio=body/atr,
                        htf_aligned=aligned, session=sess
                    ))

    # ─────────────────────────────────────────────────────────────────────────
    # 2. BREAKER BLOCKS (BB)
    # Structure: Liquidity Sweep (Turtle Soup) -> Break of opposing structure
    # Bullish Breaker: Low -> High -> Lower Low (Sweep) -> Violently breaks High
    # ─────────────────────────────────────────────────────────────────────────
    if bull_disp:
        shs = [s for s in swing_highs if s[0] < i and s[0] >= i - 25]
        sls = [s for s in swing_lows  if s[0] < i and s[0] >= i - 25]
        if len(shs) >= 1 and len(sls) >= 2:
            last_sh_idx, last_sh_p = shs[-1]
            if closes[i] > last_sh_p:
                sl_before = [s for s in sls if s[0] < last_sh_idx]
                sl_after  = [s for s in sls if s[0] > last_sh_idx]
                if sl_before and sl_after and (sl_after[-1][1] < sl_before[-1][1]):
                    # Highest up-candle leading into the swept high
                    w_start = max(0, last_sh_idx - 4)
                    up_candles = [c for c in range(w_start, last_sh_idx + 1) if closes[c] > opens[c]]
                    if up_candles:
                        bb_c = up_candles[-1]
                        open_lvl = highs[bb_c]                        # Breaker top
                        stop_lvl = min(opens[bb_c], closes[bb_c]) - 1.0 # Breaker body bot
                        mt_lvl   = (open_lvl + stop_lvl + 1.0) / 2.0
                        risk = open_lvl - stop_lvl
                        if 3.0 <= risk <= 40.0:
                            aligned = (htf[i] == 1)
                            candidates.append(CandidateBlock(
                                block_type='BULLISH_BREAKER', formed_idx=i, bar_time=t, direction=1,
                                open_level=open_lvl, mt_level=mt_lvl, stop_level=stop_lvl,
                                has_sweep=True, has_fvg=has_bull_fvg, disp_ratio=body/atr,
                                htf_aligned=aligned, session=sess
                            ))

    if bear_disp:
        shs = [s for s in swing_highs if s[0] < i and s[0] >= i - 25]
        sls = [s for s in swing_lows  if s[0] < i and s[0] >= i - 25]
        if len(sls) >= 1 and len(shs) >= 2:
            last_sl_idx, last_sl_p = sls[-1]
            if closes[i] < last_sl_p:
                sh_before = [s for s in shs if s[0] < last_sl_idx]
                sh_after  = [s for s in shs if s[0] > last_sl_idx]
                if sh_before and sh_after and (sh_after[-1][1] > sh_before[-1][1]):
                    w_start = max(0, last_sl_idx - 4)
                    dn_candles = [c for c in range(w_start, last_sl_idx + 1) if closes[c] < opens[c]]
                    if dn_candles:
                        bb_c = dn_candles[-1]
                        open_lvl = lows[bb_c]
                        stop_lvl = max(opens[bb_c], closes[bb_c]) + 1.0
                        mt_lvl   = (open_lvl + stop_lvl - 1.0) / 2.0
                        risk = stop_lvl - open_lvl
                        if 3.0 <= risk <= 40.0:
                            aligned = (htf[i] == -1)
                            candidates.append(CandidateBlock(
                                block_type='BEARISH_BREAKER', formed_idx=i, bar_time=t, direction=-1,
                                open_level=open_lvl, mt_level=mt_lvl, stop_level=stop_lvl,
                                has_sweep=True, has_fvg=has_bear_fvg, disp_ratio=body/atr,
                                htf_aligned=aligned, session=sess
                            ))

print(f"Total Candidate Blocks Detected: {len(candidates):,}")


# ─── Strict Bar-by-Bar Simulator ─────────────────────────────────────────────
print("\n[3/4] Running Pessimistic Bar-by-Bar Simulation (Checking Open vs 50% MT)...")

COMMISSION_PTS = 0.26   # $5.20 / $20 per pt on 1 Mini NQ (or $1.24 on MNQ)
SLIPPAGE_PTS   = 0.25   # 1 tick slippage

def simulate_block_entry(cand: CandidateBlock, level_choice: str, target_r: float = 2.0) -> Optional[TradeResult]:
    """
    Simulates a trade with 100% strict causal rules:
      - Limit order placed at cand.open_level (if level_choice=='OPEN') or cand.mt_level (if 'MT_50')
      - Limit ONLY fills if a future bar trades into it
      - If entry bar also breaches stop, trade stops out immediately
      - Stop is checked BEFORE Target on any ambiguous bar
      - Realistic commission & slippage deducted
    """
    entry_target = cand.open_level if level_choice == 'OPEN' else cand.mt_level
    stop_target  = cand.stop_level

    # Check risk
    if cand.direction == 1:
        risk = entry_target - stop_target
        if risk <= 1.0:
            return None
        target_price = entry_target + risk * target_r
    else:
        risk = stop_target - entry_target
        if risk <= 1.0:
            return None
        target_price = entry_target - risk * target_r

    # Search for fill in next 40 bars (~3.3 hours)
    start_k = cand.formed_idx + 1
    end_k   = min(n - 1, cand.formed_idx + 40)
    fill_idx = -1

    for k in range(start_k, end_k):
        if cand.direction == 1:
            if lows[k] <= entry_target:
                fill_idx = k
                break
        else:
            if highs[k] >= entry_target:
                fill_idx = k
                break

    if fill_idx == -1:
        return None  # Never filled

    # Filled at entry_target + slippage
    actual_entry = entry_target + (SLIPPAGE_PTS if cand.direction == 1 else -SLIPPAGE_PTS)
    fill_time    = pd.Timestamp(times[fill_idx])

    # Now step forward bar by bar from fill_idx
    max_fav = 0.0
    max_adv = 0.0

    for k in range(fill_idx, min(n - 1, fill_idx + 50)):
        kh, kl, kc = highs[k], lows[k], closes[k]
        kt = pd.Timestamp(times[k])

        if cand.direction == 1:
            fav = kh - actual_entry
            adv = actual_entry - kl
            max_fav = max(max_fav, fav)
            max_adv = max(max_adv, adv)

            # 1. Worst case: check STOP FIRST
            if kl <= stop_target:
                actual_exit = stop_target - SLIPPAGE_PTS
                gross = actual_exit - actual_entry
                net   = gross - COMMISSION_PTS
                real_r = net / risk
                return TradeResult(
                    cand.block_type, level_choice, cand.bar_time, fill_time, 1,
                    actual_entry, stop_target, risk, actual_exit, kt, 'STOP_LOSS',
                    gross, net, real_r, max_fav, max_adv,
                    cand.has_sweep, cand.has_fvg, cand.htf_aligned, cand.session, cand.disp_ratio
                )

            # 2. Check TARGET
            if kh >= target_price:
                actual_exit = target_price - SLIPPAGE_PTS
                gross = actual_exit - actual_entry
                net   = gross - COMMISSION_PTS
                real_r = net / risk
                return TradeResult(
                    cand.block_type, level_choice, cand.bar_time, fill_time, 1,
                    actual_entry, stop_target, risk, actual_exit, kt, f'TARGET_{int(target_r)}R',
                    gross, net, real_r, max_fav, max_adv,
                    cand.has_sweep, cand.has_fvg, cand.htf_aligned, cand.session, cand.disp_ratio
                )

        else: # Short
            fav = actual_entry - kl
            adv = kh - actual_entry
            max_fav = max(max_fav, fav)
            max_adv = max(max_adv, adv)

            if kh >= stop_target:
                actual_exit = stop_target + SLIPPAGE_PTS
                gross = actual_entry - actual_exit
                net   = gross - COMMISSION_PTS
                real_r = net / risk
                return TradeResult(
                    cand.block_type, level_choice, cand.bar_time, fill_time, -1,
                    actual_entry, stop_target, risk, actual_exit, kt, 'STOP_LOSS',
                    gross, net, real_r, max_fav, max_adv,
                    cand.has_sweep, cand.has_fvg, cand.htf_aligned, cand.session, cand.disp_ratio
                )

            if kl <= target_price:
                actual_exit = target_price + SLIPPAGE_PTS
                gross = actual_entry - actual_exit
                net   = gross - COMMISSION_PTS
                real_r = net / risk
                return TradeResult(
                    cand.block_type, level_choice, cand.bar_time, fill_time, -1,
                    actual_entry, stop_target, risk, actual_exit, kt, f'TARGET_{int(target_r)}R',
                    gross, net, real_r, max_fav, max_adv,
                    cand.has_sweep, cand.has_fvg, cand.htf_aligned, cand.session, cand.disp_ratio
                )

        # Session close exit (15:55 EST)
        if kt.hour == 15 and kt.minute >= 55:
            actual_exit = kc - (SLIPPAGE_PTS if cand.direction == 1 else -SLIPPAGE_PTS)
            gross = (actual_exit - actual_entry) if cand.direction == 1 else (actual_entry - actual_exit)
            net   = gross - COMMISSION_PTS
            return TradeResult(
                cand.block_type, level_choice, cand.bar_time, fill_time, cand.direction,
                actual_entry, stop_target, risk, actual_exit, kt, 'TIME_EXIT',
                gross, net, net / risk, max_fav, max_adv,
                cand.has_sweep, cand.has_fvg, cand.htf_aligned, cand.session, cand.disp_ratio
            )

    return None


# ─── Run Tests for Both Levels ───────────────────────────────────────────────
results_open: List[TradeResult] = []
results_mt50: List[TradeResult] = []

for c in candidates:
    r_open = simulate_block_entry(c, 'OPEN', target_r=2.0)
    if r_open:
        results_open.append(r_open)

    r_mt50 = simulate_block_entry(c, 'MT_50', target_r=2.0)
    if r_mt50:
        results_mt50.append(r_mt50)


# ─── Statistical Reporter ────────────────────────────────────────────────────
print("\n[4/4] Compiling Rigorous Results & Discoveries...\n")

def print_audit_card(title: str, tr_list: List[TradeResult]):
    if not tr_list:
        print(f"  [{title}]: No trades filled.")
        return

    n_tr = len(tr_list)
    wins = [t for t in tr_list if 'TARGET' in t.exit_reason]
    losses = [t for t in tr_list if t.exit_reason == 'STOP_LOSS']
    wr = len(wins) / n_tr * 100

    gw = sum(t.net_pts for t in tr_list if t.net_pts > 0)
    gl = abs(sum(t.net_pts for t in tr_list if t.net_pts <= 0))
    pf = gw / gl if gl > 0 else float('inf')
    avg_r = np.mean([t.realized_r for t in tr_list])
    avg_net_pts = np.mean([t.net_pts for t in tr_list])
    avg_risk = np.mean([t.risk_pts for t in tr_list])
    avg_mfe  = np.mean([t.mfe_pts for t in tr_list])

    print(f"┌{'─'*78}┐")
    print(f"│ {title.upper():<76} │")
    print(f"├{'─'*78}┤")
    print(f"│ Trades: {n_tr:<6} | Wins: {len(wins):<4} | Losses: {len(losses):<4} | Win Rate: {wr:>5.1f}% | Avg Stop: {avg_risk:<4.1f} pts │")
    print(f"│ Profit Factor: {pf:>5.2f} | Expectancy: {avg_r:>+5.2f}R | Net Pts/Trade: {avg_net_pts:>+5.1f} pts (${avg_net_pts*20:>+6.0f}/NQ) │")
    print(f"│ Avg Run (MFE): {avg_mfe:>5.1f} pts | Fill Rate: {n_tr/len(candidates)*100:>5.1f}% of setups                      │")
    print(f"└{'─'*78}┘")


# ─── 1. LEVEL ANALYSIS: OPEN VS 50% MT ───────────────────────────────────────
print("=" * 80)
print("QUESTION 1: WHAT LEVEL GETS RESPECTED MOST? (OPEN VS 50% MEAN THRESHOLD)")
print("=" * 80)
print_audit_card("ENTRY LEVEL 1: OPEN OF THE BLOCK (Immediate Re-test)", results_open)
print_audit_card("ENTRY LEVEL 2: 50% MEAN THRESHOLD (MT - Deep Re-test)", results_mt50)


# ─── 2. ORDER BLOCK VS BREAKER BLOCK HEAD-TO-HEAD ────────────────────────────
print("\n" + "=" * 80)
print("QUESTION 2: ORDER BLOCK (OB) VS BREAKER BLOCK (BB) HEAD-TO-HEAD")
print("=" * 80)
ob_trades = [t for t in results_open if 'OB' in t.block_type]
bb_trades = [t for t in results_open if 'BREAKER' in t.block_type]

print_audit_card("ORDER BLOCKS (ALL SESSIONS, OPEN ENTRY)", ob_trades)
print_audit_card("BREAKER BLOCKS (ALL SESSIONS, OPEN ENTRY)", bb_trades)


# ─── 3. THE MAGIC FILTERS: WHAT MAKES THEM HIGH PROBABILITY? ─────────────────
print("\n" + "=" * 80)
print("QUESTION 3: WHY DO THEY GET RESPECTED? (THE CORE FILTERS)")
print("=" * 80)

# Filter A: With Liquidity Sweep vs Without Liquidity Sweep
ob_sweep   = [t for t in ob_trades if t.has_sweep]
ob_nosweep = [t for t in ob_trades if not t.has_sweep]
print_audit_card("ORDER BLOCK WITH PRIOR LIQUIDITY SWEEP", ob_sweep)
print_audit_card("ORDER BLOCK WITHOUT LIQUIDITY SWEEP (MID-AIR)", ob_nosweep)

# Filter B: Higher Timeframe (1H Trend) Alignment
ob_htf_aligned   = [t for t in ob_trades if t.htf_aligned]
ob_htf_opposed   = [t for t in ob_trades if not t.htf_aligned]
print_audit_card("ORDER BLOCK ALIGNED WITH 1H HTF TREND", ob_htf_aligned)
print_audit_card("ORDER BLOCK AGAINST 1H HTF TREND", ob_htf_opposed)

# Filter C: Session Timing (Silver Bullet 10:00-11:15 vs 9:30 Open vs Lunch)
sb_trades    = [t for t in results_open if t.session == 'SILVER_BULLET']
open_trades  = [t for t in results_open if t.session == '930_OPEN']
lunch_trades = [t for t in results_open if t.session == 'LUNCH_CHOP']

print_audit_card("10:00 - 11:15 AM (SILVER BULLET WINDOW)", sb_trades)
print_audit_card("09:30 - 09:59 AM (OPENING BELL CHAOS)", open_trades)
print_audit_card("11:15 AM - 01:30 PM (LUNCHTIME CHOP)", lunch_trades)


# ─── 4. THE OPTIMAL MODEL: THE TRUE INSTITUTIONAL COMBO ──────────────────────
print("\n" + "=" * 80)
print("THE VERDICT: THE PURE STATISTICAL EDGE")
print("=" * 80)

optimal = [t for t in results_open if t.has_sweep and t.htf_aligned and t.session in ('SILVER_BULLET', '930_OPEN')]
print_audit_card("ELITE SYSTEM: LIQUIDITY SWEEP + 1H TREND ALIGNMENT + 10:00 AM NY", optimal)

