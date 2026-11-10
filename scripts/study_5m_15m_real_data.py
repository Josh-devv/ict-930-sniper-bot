"""
Deep Study: 5-Minute vs. 15-Minute Institutional Order Blocks on Real 8-Year CME Data
=====================================================================================
Analyzes 8 Years of Real CME NQ 1-Min Data (2018-2026):
  1. Head-to-Head Comparison: 5-Minute vs 15-Minute Blocks
  2. Stop Loss Dimensions: How tight can stops truly be?
  3. When do they ACTUALLY work with high win rates?
     - 10:00 - 11:30 AM Silver Bullet vs 9:30 Open
     - Prior Liquidity Sweep vs Mid-air formation
     - 1-Hour HTF Trend Alignment vs Counter-trend
  4. Real-world Performance (Hard Stops, Realistic Limit Fills, $5.20 RT Fees + Slippage)
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

print("=" * 80)
print("INSTITUTIONAL STUDY: 5-MIN VS 15-MIN BLOCKS ON REAL 8-YEAR CME NQ DATA")
print("Data Source: sample_data/NQ_1min_RTH_2018_2026.csv (862,499 bars)")
print("=" * 80)

# ─── 1. Load Real Data ────────────────────────────────────────────────────────
print("\n[1/4] Ingesting 8-year real CME dataset...")
df_1m = pd.read_csv('sample_data/NQ_1min_RTH_2018_2026.csv')
df_1m.columns = [c.lower() for c in df_1m.columns]
df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])

# Filter out holiday zero-volume flat bars
df_1m = df_1m[df_1m['high'] > df_1m['low']].reset_index(drop=True)
df_1m = df_1m.set_index('datetime').sort_index()

print(f"Loaded {len(df_1m):,} active 1-min bars from {df_1m.index[0].date()} to {df_1m.index[-1].date()}.")


# ─── 2. Build 5m, 15m, and 1H Bars ───────────────────────────────────────────
print("\n[2/4] Resampling into 5-minute, 15-minute and 1-Hour HTF frames...")

df_5m = df_1m.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

df_15m = df_1m.resample('15min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

# 1-Hour Trend (EMA20) for macro institutional directional filter
df_1h = df_1m.resample('1h', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_1h['ema20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
df_1h['htf_trend'] = np.where(df_1h['close'] > df_1h['ema20'], 1, -1)


# ─── 3. Unified Block Backtest Engine ─────────────────────────────────────────
COMMISSION_PTS = 0.26   # $5.20 / $20 pt on Mini NQ (or $1.24 on MNQ)
SLIPPAGE_PTS   = 0.25   # 1 tick slippage

@dataclass
class TradeSummary:
    timeframe: str
    entry_level: str
    direction: int
    entry_time: pd.Timestamp
    entry_px: float
    stop_px: float
    risk_pts: float
    exit_px: float
    exit_reason: str
    net_pts: float
    realized_r: float
    duration_bars: int
    has_sweep: bool
    htf_aligned: bool
    session: str


def run_timeframe_study(df_tf: pd.DataFrame, tf_name: str, min_risk: float = 3.0, max_risk: float = 40.0) -> List[TradeSummary]:
    print(f"\nAnalyzing {tf_name} Institutional Setups across {len(df_tf):,} bars...")

    # Calculate ATR
    tr = np.maximum(df_tf['high'] - df_tf['low'],
                    np.maximum(abs(df_tf['high'] - df_tf['close'].shift(1)),
                               abs(df_tf['low'] - df_tf['close'].shift(1))))
    df_tf['atr'] = tr.rolling(14).mean().bfill()

    # Map 1H trend
    df_tf['hour_ts'] = df_tf['datetime'].dt.floor('1h')
    df_tf['htf_trend'] = df_tf['hour_ts'].map(df_1h['htf_trend']).fillna(0)

    highs  = df_tf['high'].values
    lows   = df_tf['low'].values
    opens  = df_tf['open'].values
    closes = df_tf['close'].values
    atrs   = df_tf['atr'].values
    times  = df_tf['datetime'].values
    htf    = df_tf['htf_trend'].values
    n      = len(df_tf)

    # 3-bar fractal swings
    swing_highs = []
    swing_lows  = []
    for i in range(2, n - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
            swing_highs.append((i, highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
            swing_lows.append((i, lows[i]))

    trades: List[TradeSummary] = []

    for i in range(15, n - 40):
        t = pd.Timestamp(times[i])
        atr = atrs[i]
        h, m = t.hour, t.minute

        # Classify session
        if (h == 9 and m >= 30) or (h == 10 and m < 0):
            sess = '930_OPEN'
        elif (h == 10) or (h == 11 and m <= 30):
            sess = 'SILVER_BULLET'
        elif (h >= 13 and m >= 30) and (h <= 15 and m <= 45):
            sess = 'NY_PM'
        else:
            sess = 'OTHER'

        # Displacement detection: body >= 1.25x ATR
        body = abs(closes[i] - opens[i])
        disp = body >= 1.25 * atr
        bull_disp = disp and closes[i] > opens[i]
        bear_disp = disp and closes[i] < opens[i]

        # ── BULLISH ORDER BLOCK ──────────────────────────────────────────────
        if bull_disp:
            ob_idx = i - 1
            if closes[ob_idx] < opens[ob_idx]:
                # MSS check
                shs = [p for idx, p in swing_highs if idx < ob_idx and idx >= ob_idx - 15]
                if shs and closes[i] > max(shs):
                    sls = [p for idx, p in swing_lows if idx < ob_idx - 1 and idx >= ob_idx - 20]
                    swept = bool(sls and lows[ob_idx] < min(sls))

                    # 50% Mean Threshold Entry
                    block_top = max(opens[ob_idx], closes[ob_idx])
                    block_bot = lows[ob_idx]
                    mt_entry  = (block_top + block_bot) / 2.0
                    stop_px   = block_bot - 1.0
                    risk      = mt_entry - stop_px

                    if min_risk <= risk <= max_risk:
                        aligned = (htf[i] == 1)
                        target_px = mt_entry + 2.0 * risk

                        # Walk forward to find fill and outcome
                        filled = False
                        fill_k = -1
                        for k in range(i + 1, min(n - 1, i + 30)):
                            if lows[k] <= mt_entry:
                                filled = True
                                fill_k = k
                                break

                        if filled:
                            actual_en = mt_entry + SLIPPAGE_PTS
                            exit_px = 0.0
                            exit_re = ''
                            dur = 0

                            for k in range(fill_k, min(n - 1, fill_k + 40)):
                                dur = k - fill_k
                                # Hard stop checked first
                                if lows[k] <= stop_px:
                                    exit_px = stop_px - SLIPPAGE_PTS
                                    exit_re = 'STOP_LOSS'
                                    break
                                if highs[k] >= target_px:
                                    exit_px = target_px - SLIPPAGE_PTS
                                    exit_re = 'TARGET_2R'
                                    break
                                if pd.Timestamp(times[k]).hour == 15 and pd.Timestamp(times[k]).minute >= 55:
                                    exit_px = closes[k] - SLIPPAGE_PTS
                                    exit_re = 'TIME_EXIT'
                                    break

                            if exit_re:
                                gross = exit_px - actual_en
                                net   = gross - COMMISSION_PTS
                                trades.append(TradeSummary(
                                    tf_name, 'MT_50', 1, pd.Timestamp(times[fill_k]),
                                    actual_en, stop_px, risk, exit_px, exit_re,
                                    net, net / risk, dur, swept, aligned, sess
                                ))

        # ── BEARISH ORDER BLOCK ─────────────────────────────────────────────
        if bear_disp:
            ob_idx = i - 1
            if closes[ob_idx] > opens[ob_idx]:
                sls = [p for idx, p in swing_lows if idx < ob_idx and idx >= ob_idx - 15]
                if sls and closes[i] < min(sls):
                    shs = [p for idx, p in swing_highs if idx < ob_idx - 1 and idx >= ob_idx - 20]
                    swept = bool(shs and highs[ob_idx] > max(shs))

                    block_bot = min(opens[ob_idx], closes[ob_idx])
                    block_top = highs[ob_idx]
                    mt_entry  = (block_top + block_bot) / 2.0
                    stop_px   = block_top + 1.0
                    risk      = stop_px - mt_entry

                    if min_risk <= risk <= max_risk:
                        aligned = (htf[i] == -1)
                        target_px = mt_entry - 2.0 * risk

                        filled = False
                        fill_k = -1
                        for k in range(i + 1, min(n - 1, i + 30)):
                            if highs[k] >= mt_entry:
                                filled = True
                                fill_k = k
                                break

                        if filled:
                            actual_en = mt_entry - SLIPPAGE_PTS
                            exit_px = 0.0
                            exit_re = ''
                            dur = 0

                            for k in range(fill_k, min(n - 1, fill_k + 40)):
                                dur = k - fill_k
                                if highs[k] >= stop_px:
                                    exit_px = stop_px + SLIPPAGE_PTS
                                    exit_re = 'STOP_LOSS'
                                    break
                                if lows[k] <= target_px:
                                    exit_px = target_px + SLIPPAGE_PTS
                                    exit_re = 'TARGET_2R'
                                    break
                                if pd.Timestamp(times[k]).hour == 15 and pd.Timestamp(times[k]).minute >= 55:
                                    exit_px = closes[k] + SLIPPAGE_PTS
                                    exit_re = 'TIME_EXIT'
                                    break

                            if exit_re:
                                gross = actual_en - exit_px
                                net   = gross - COMMISSION_PTS
                                trades.append(TradeSummary(
                                    tf_name, 'MT_50', -1, pd.Timestamp(times[fill_k]),
                                    actual_en, stop_px, risk, exit_px, exit_re,
                                    net, net / risk, dur, swept, aligned, sess
                                ))

    return trades


# ─── Execute 5m and 15m Studies ───────────────────────────────────────────────
t_5m  = run_timeframe_study(df_5m,  "5-Minute Blocks",  min_risk=3.0, max_risk=30.0)
t_15m = run_timeframe_study(df_15m, "15-Minute Blocks", min_risk=5.0, max_risk=50.0)


# ─── Reporting Helper ────────────────────────────────────────────────────────
def print_card(title: str, tr_list: List[TradeSummary], tf_bar_minutes: int):
    if not tr_list:
        print(f"  [{title}]: No trades.")
        return
    n_t = len(tr_list)
    wins = [t for t in tr_list if t.exit_reason == 'TARGET_2R']
    losses = [t for t in tr_list if t.exit_reason == 'STOP_LOSS']
    wr = len(wins) / n_t * 100

    gw = sum(t.net_pts for t in tr_list if t.net_pts > 0)
    gl = abs(sum(t.net_pts for t in tr_list if t.net_pts <= 0))
    pf = gw / gl if gl > 0 else float('inf')
    avg_r = np.mean([t.realized_r for t in tr_list])
    avg_stop = np.mean([t.risk_pts for t in tr_list])
    avg_target = avg_stop * 2.0
    avg_dur_min = np.mean([t.duration_bars for t in tr_list]) * tf_bar_minutes
    net_total_pts = sum(t.net_pts for t in tr_list)

    print(f"┌{'─'*78}┐")
    print(f"│ {title.upper():<76} │")
    print(f"├{'─'*78}┤")
    print(f"│ Total Trades: {n_t:<6} | Wins: {len(wins):<4} | Losses: {len(losses):<4} | Win Rate: {wr:>5.1f}% (at 2:1 R:R)     │")
    print(f"│ Profit Factor: {pf:>5.2f} | Expectancy: {avg_r:>+5.2f}R | Net Pts Total: {net_total_pts:>+8.1f} pts (${net_total_pts*20:>+10,.0f}) │")
    print(f"│ Avg Stop Size: {avg_stop:>4.1f} pts ($ {avg_stop*20:>4.0f}/NQ) | Avg Target: {avg_target:>5.1f} pts ($ {avg_target*20:>5.0f}/NQ)         │")
    print(f"│ Average Trade Duration: {avg_dur_min:>4.0f} minutes ({avg_dur_min/tf_bar_minutes:>3.1f} bars)                              │")
    print(f"└{'─'*78}┘")


print("\n" + "=" * 80)
print("RESULTS: 5-MINUTE VS 15-MINUTE HEAD-TO-HEAD ON REAL CME DATA (2018-2026)")
print("=" * 80)
print_card("ALL 5-MINUTE BLOCKS (50% MT ENTRY)", t_5m, 5)
print_card("ALL 15-MINUTE BLOCKS (50% MT ENTRY)", t_15m, 15)

print("\n" + "=" * 80)
print("HIGH WIN-RATE CONDITIONS: WHEN DO THEY ACTUALLY WORK?")
print("=" * 80)

# Filter 1: 1H Trend Aligned
t5_htf = [t for t in t_5m if t.htf_aligned]
t15_htf = [t for t in t_15m if t.htf_aligned]
print_card("5M BLOCKS ALIGNED WITH 1H HTF TREND", t5_htf, 5)
print_card("15M BLOCKS ALIGNED WITH 1H HTF TREND", t15_htf, 15)

# Filter 2: Silver Bullet Window (10:00 - 11:30 AM NY)
t5_sb = [t for t in t_5m if t.session == 'SILVER_BULLET']
t15_sb = [t for t in t_15m if t.session == 'SILVER_BULLET']
print_card("5M BLOCKS IN 10:00-11:30 AM SILVER BULLET WINDOW", t5_sb, 5)
print_card("15M BLOCKS IN 10:00-11:30 AM SILVER BULLET WINDOW", t15_sb, 15)

# Filter 3: The Golden Setup: 1H Trend Aligned + Silver Bullet Window
t5_elite = [t for t in t_5m if t.htf_aligned and t.session == 'SILVER_BULLET']
t15_elite = [t for t in t_15m if t.htf_aligned and t.session == 'SILVER_BULLET']
print_card("ELITE 5M: 1H TREND ALIGNED + 10:00-11:30 AM WINDOW", t5_elite, 5)
print_card("ELITE 15M: 1H TREND ALIGNED + 10:00-11:30 AM WINDOW", t15_elite, 15)

