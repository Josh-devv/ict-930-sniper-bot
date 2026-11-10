"""
High-Frequency Tony Iyke Market Structure & Liquidity Engine
Tested on 8.8 Years of Real CME NQ Futures (2018-2026, 839,332 1-min RTH bars)
==============================================================================

User Goal:
"THIS WORKS WELL BUT THE TRADE COUNT IS TOO SMALL"
Target: 100 - 250+ trades/year (2 to 5 trades every week, almost daily setups)
while maintaining profitability (PF > 1.50, Win Rate 52-60%, tight stops).

Bottlenecks Diagnosed in Previous Run:
1. Strict 50% MT Limit Fill requirement missed ~40-50% of fast moves that didn't retrace deep.
2. 5-bar lookback on 5m (25 mins) was too slow to catch intra-hour swings.
3. 0.70x ATR displacement requirement filtered out clean micro-CHoCH candles.

Architecture of High-Frequency Tony Iyke:
- Timeframes tested: 2-Minute, 3-Minute, and 5-Minute
- Multiple setups per session (Morning 09:30-11:30, Mid-Day 11:30-13:30, PM 13:30-15:45)
- Inducement: 3-bar and 4-bar rolling fractal swings
- Flexible Entry:
    A) 50% MT Limit Order (Pessimistic Retest)
    B) Direct Market / Next Bar Open Entry on confirmed CHoCH
    C) 25% Retest Entry (Quarter Threshold)
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List

print("=" * 80)
print("HIGH-FREQUENCY TONY IYKE SMC ENGINE (TARGET: 100-250+ TRADES/YEAR)")
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

# 1H HTF Trend
df_1h = df_1m.resample('1h', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_1h['ema20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
df_1h['htf_trend'] = np.where(df_1h['close'] > df_1h['ema20'], 1, -1)

# 15m HTF Trend for faster confirmation
df_15m = df_1m.resample('15min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_15m['ema20'] = df_15m['close'].ewm(span=20, adjust=False).mean()
df_15m['htf_trend'] = np.where(df_15m['close'] > df_15m['ema20'], 1, -1)

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
    entry_type: str

def run_hf_tony_iyke(
    timeframe: str = '3min',
    htf_mode: str = '1h',        # '1h' or '15m'
    idm_lookback: int = 4,       # Faster fractal swing
    entry_mode: str = 'limit_25',# 'limit_50', 'limit_25', or 'market_close'
    min_disp_atr: float = 0.50,  # Lower threshold for micro-displacement
    min_wick_ratio: float = 0.18,# Rejection wick >= 18%
    target_r: float = 1.5,
    min_stop: float = 7.0,
    max_stop: float = 22.0
) -> List[Trade]:

    df_tf = df_1m.resample(timeframe, closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'delta': 'sum'
    }).dropna().reset_index()

    tr = np.maximum(df_tf['high'] - df_tf['low'],
                    np.maximum(abs(df_tf['high'] - df_tf['close'].shift(1)),
                               abs(df_tf['low'] - df_tf['close'].shift(1))))
    df_tf['atr'] = tr.rolling(14).mean().bfill()

    if htf_mode == '1h':
        df_tf['htf_ts'] = df_tf['datetime'].dt.floor('1h')
        df_tf['htf_trend'] = df_tf['htf_ts'].map(df_1h['htf_trend']).fillna(0)
    else:
        df_tf['htf_ts'] = df_tf['datetime'].dt.floor('15min')
        df_tf['htf_trend'] = df_tf['htf_ts'].map(df_15m['htf_trend']).fillna(0)

    # Inducement Swings
    df_tf['idm_low']  = df_tf['low'].shift(1).rolling(idm_lookback).min()
    df_tf['idm_high'] = df_tf['high'].shift(1).rolling(idm_lookback).max()

    times = df_tf['datetime'].values
    opens = df_tf['open'].values
    highs = df_tf['high'].values
    lows = df_tf['low'].values
    closes = df_tf['close'].values
    deltas = df_tf['delta'].values
    atrs = df_tf['atr'].values
    htf = df_tf['htf_trend'].values
    idm_lows = df_tf['idm_low'].values
    idm_highs = df_tf['idm_high'].values

    n = len(df_tf)
    trades: List[Trade] = []
    in_trade_until = -1

    for i in range(idm_lookback + 5, n - 30):
        if i <= in_trade_until:
            continue

        t = pd.Timestamp(times[i])
        h, m = t.hour, t.minute

        # Extended RTH window: 09:35 to 15:45
        if not ((h == 9 and m >= 35) or (10 <= h < 15) or (h == 15 and m <= 45)):
            continue

        ob_idx = i - 1
        ob_o, ob_h, ob_l, ob_c = opens[ob_idx], highs[ob_idx], lows[ob_idx], closes[ob_idx]
        ob_rng = ob_h - ob_l

        if ob_rng < 0.75:
            continue

        disp_body = abs(closes[i] - opens[i])
        is_bull_disp = (disp_body >= min_disp_atr * atrs[i]) and (closes[i] > opens[i])
        is_bear_disp = (disp_body >= min_disp_atr * atrs[i]) and (closes[i] < opens[i])

        # ── BULLISH SETUP ──
        if is_bull_disp and ob_c < ob_o:
            if htf[i] != 1:
                continue

            # Inducement sweep
            if np.isnan(idm_lows[ob_idx]) or ob_l >= idm_lows[ob_idx]:
                continue
            if ob_c < idm_lows[ob_idx] - 0.75 * atrs[i]:
                continue

            # Absorption
            lower_wick = min(ob_o, ob_c) - ob_l
            wick_ratio = lower_wick / ob_rng
            has_absorp = (wick_ratio >= min_wick_ratio) or (deltas[ob_idx] > 0) or (deltas[i] > 0)
            if not has_absorp:
                continue

            # Determine Entry & Stop
            stop_px = ob_l - 1.0
            block_top = max(ob_o, ob_c)
            block_bot = ob_l

            if entry_mode == 'market_close':
                entry_px = closes[i]
                fill_k = i + 1
                filled = True
            elif entry_mode == 'limit_25':
                # 25% shallow pullback from high of block
                entry_px = block_top - 0.25 * (block_top - block_bot)
                filled = False
                fill_k = -1
            else: # limit_50 MT
                entry_px = (block_top + block_bot) / 2.0
                filled = False
                fill_k = -1

            risk_pts = entry_px - stop_px
            if not (min_stop <= risk_pts <= max_stop):
                continue

            target_px = entry_px + (risk_pts * target_r)

            if not filled:
                for k in range(i + 1, min(n - 1, i + 12)):
                    if lows[k] <= entry_px:
                        filled = True
                        fill_k = k
                        break
                    if highs[k] >= target_px:
                        break

            if not filled:
                continue

            # Monitor trade execution
            exit_px = 0.0
            exit_reason = 'EOD'
            dur = 0
            for k in range(fill_k, min(n - 1, fill_k + 40)):
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
                exit_px = closes[min(n - 1, fill_k + 39)]
                in_trade_until = min(n - 1, fill_k + 39)

            net_pts = (exit_px - entry_px) - COMMISSION_PTS - SLIPPAGE_PTS
            trades.append(Trade(
                year=t.year, entry_time=pd.Timestamp(times[fill_k]), direction=1,
                entry_px=entry_px, stop_px=stop_px, risk_pts=risk_pts, exit_px=exit_px,
                exit_reason=exit_reason, net_pts=net_pts, real_r=net_pts / risk_pts,
                dur_bars=dur, entry_type=entry_mode
            ))

        # ── BEARISH SETUP ──
        elif is_bear_disp and ob_c > ob_o:
            if htf[i] != -1:
                continue

            if np.isnan(idm_highs[ob_idx]) or ob_h <= idm_highs[ob_idx]:
                continue
            if ob_c > idm_highs[ob_idx] + 0.75 * atrs[i]:
                continue

            upper_wick = ob_h - max(ob_o, ob_c)
            wick_ratio = upper_wick / ob_rng
            has_absorp = (wick_ratio >= min_wick_ratio) or (deltas[ob_idx] < 0) or (deltas[i] < 0)
            if not has_absorp:
                continue

            stop_px = ob_h + 1.0
            block_top = ob_h
            block_bot = min(ob_o, ob_c)

            if entry_mode == 'market_close':
                entry_px = closes[i]
                fill_k = i + 1
                filled = True
            elif entry_mode == 'limit_25':
                entry_px = block_bot + 0.25 * (block_top - block_bot)
                filled = False
                fill_k = -1
            else: # limit_50 MT
                entry_px = (block_top + block_bot) / 2.0
                filled = False
                fill_k = -1

            risk_pts = stop_px - entry_px
            if not (min_stop <= risk_pts <= max_stop):
                continue

            target_px = entry_px - (risk_pts * target_r)

            if not filled:
                for k in range(i + 1, min(n - 1, i + 12)):
                    if highs[k] >= entry_px:
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
            for k in range(fill_k, min(n - 1, fill_k + 40)):
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
                exit_px = closes[min(n - 1, fill_k + 39)]
                in_trade_until = min(n - 1, fill_k + 39)

            net_pts = (entry_px - exit_px) - COMMISSION_PTS - SLIPPAGE_PTS
            trades.append(Trade(
                year=t.year, entry_time=pd.Timestamp(times[fill_k]), direction=-1,
                entry_px=entry_px, stop_px=stop_px, risk_pts=risk_pts, exit_px=exit_px,
                exit_reason=exit_reason, net_pts=net_pts, real_r=net_pts / risk_pts,
                dur_bars=dur, entry_type=entry_mode
            ))

    return trades

def print_hf_stats(trades: List[Trade], title: str):
    if not trades:
        print(f"\n{title}: NO TRADES")
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
    tpw = tpy / 52.0

    years = sorted(tdf['year'].unique())
    green_years = sum(tdf[tdf['year'] == y]['net_pts'].sum() > 0 for y in years)

    print(f"\n{'='*75}")
    print(f"CONFIGURATION: {title}")
    print(f"{'='*75}")
    print(f"Total Trades (8.8y): {n} ({tpy:.1f} trades/year | ~{tpw:.1f} trades/week | ~{tpy/252:.2f} trades/day)")
    print(f"Win Rate:            {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Profit Factor:       {pf:.2f}")
    print(f"Net Points:          +{net_pts:.1f} pts (${net_usd:,.2f} on 1 NQ / ${net_usd/10:,.2f} MNQ)")
    print(f"Avg Stop Loss:       {avg_sl:.1f} pts (${avg_sl*20:.1f} NQ / ${avg_sl*2:.1f} MNQ)")
    print(f"Yearly Consistency:  {green_years}/{len(years)} Green Years")

# ==============================================================================
# RUN HIGH-FREQUENCY SWEEP MATRIX
# ==============================================================================

# Config A: 3-Minute Limit 25% Retest (Shallow pullback catches more trades)
t_3m_lim25 = run_hf_tony_iyke(timeframe='3min', htf_mode='1h', entry_mode='limit_25', target_r=1.5)
print_hf_stats(t_3m_lim25, "A. 3-MINUTE SMC (Limit 25% Shallow Pullback @ 1.5R)")

# Config B: 3-Minute Limit 50% MT
t_3m_lim50 = run_hf_tony_iyke(timeframe='3min', htf_mode='1h', entry_mode='limit_50', target_r=1.5)
print_hf_stats(t_3m_lim50, "B. 3-MINUTE SMC (Limit 50% MT Retest @ 1.5R)")

# Config C: 2-Minute SMC (Ultra-active day trading: 2m fractal swings)
t_2m_lim50 = run_hf_tony_iyke(timeframe='2min', htf_mode='1h', entry_mode='limit_50', target_r=1.5)
print_hf_stats(t_2m_lim50, "C. 2-MINUTE SMC (Limit 50% MT @ 1.5R)")

# Config D: 2-Minute Limit 25% Retest
t_2m_lim25 = run_hf_tony_iyke(timeframe='2min', htf_mode='1h', entry_mode='limit_25', target_r=1.5)
print_hf_stats(t_2m_lim25, "D. 2-MINUTE SMC (Limit 25% Shallow Pullback @ 1.5R)")

# Config E: 3-Minute SMC with 15-Minute HTF Trend (Dual-MTF SMC for daily setups)
t_3m_htf15m = run_hf_tony_iyke(timeframe='3min', htf_mode='15m', entry_mode='limit_50', target_r=1.5)
print_hf_stats(t_3m_htf15m, "E. 3-MINUTE SMC (15m HTF Trend + 3m IDM Sweep + 50% MT @ 1.5R)")

