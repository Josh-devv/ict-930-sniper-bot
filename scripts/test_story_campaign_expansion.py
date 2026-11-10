"""
Institutional Narrative Campaign: Expanding Trade Count via Story-Driven Structure
Tested on 8.8 Years of CME NQ Futures (2018-2026)
===================================================================================

The Quantitative Question:
"Does storytelling improve the amount of trades we take or win rate?"

Our initial test showed that using storytelling as a STRICT HYPER-FILTER destroyed trade count
(dropping from 65 to 11 trades) and lowered win rate (from 70% to 54%) because retail 'filters'
miss the subtle reality of institutional iceberg execution.

HOWEVER, in real institutional prop desks, the "Story" is used as a REGIME BIAS ENGINE:
Once Chapter 1 (Sweep + Absorption + 1H Alignment) confirms the market story:
- Entry 1: Primary Sweep at 50% MT of the Order Block.
- Entry 2: Secondary Continuation Retest (FVG / Breaker) as trapped liquidity unwinds.

Let's test if this Narrative Campaign:
1. Doubles the trade frequency (from 60-70 to 120-150+ trades/year).
2. Maintains the 58%+ Win Rate and >1.70 Profit Factor.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List

print("=" * 80)
print("TESTING NARRATIVE CAMPAIGN: PRIMARY SWEEP + SECONDARY CONTINUATION RETESTS")
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
    trade_type: str

def run_campaign_simulation(enable_secondary_entries: bool = False, target_r: float = 1.5):
    tr = np.maximum(df_5m['high'] - df_5m['low'],
                    np.maximum(abs(df_5m['high'] - df_5m['close'].shift(1)),
                               abs(df_5m['low'] - df_5m['close'].shift(1))))
    df_5m['atr'] = tr.rolling(14).mean().bfill()
    df_5m['hour_ts'] = df_5m['datetime'].dt.floor('1h')
    df_5m['htf_trend'] = df_5m['hour_ts'].map(df_1h['htf_trend']).fillna(0)

    # 5-bar rolling swing lookback
    swing_lookback = 6
    df_5m['sh'] = df_5m['high'].shift(1).rolling(swing_lookback).max()
    df_5m['sl'] = df_5m['low'].shift(1).rolling(swing_lookback).min()

    times = df_5m['datetime'].values
    opens = df_5m['open'].values
    highs = df_5m['high'].values
    lows = df_5m['low'].values
    closes = df_5m['close'].values
    deltas = df_5m['delta'].values
    atrs = df_5m['atr'].values
    htf = df_5m['htf_trend'].values
    sh = df_5m['sh'].values
    sl = df_5m['sl'].values

    n = len(df_5m)
    trades: List[Trade] = []
    in_trade_until = -1

    for i in range(swing_lookback + 5, n - 40):
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
        is_bull_disp = (disp_body >= 0.85 * atrs[i]) and (closes[i] > opens[i])
        is_bear_disp = (disp_body >= 0.85 * atrs[i]) and (closes[i] < opens[i])

        # ── BULLISH SETUP ──
        if is_bull_disp and ob_c < ob_o:
            if np.isnan(sl[ob_idx]) or ob_l >= sl[ob_idx]:
                continue
            if ob_c < sl[ob_idx] - 0.75 * atrs[i]:
                continue

            lower_wick = min(ob_o, ob_c) - ob_l
            wick_ratio = lower_wick / ob_rng
            if not ((wick_ratio >= 0.25) or (deltas[ob_idx] > 0) or (deltas[i] > 0)):
                continue

            if htf[i] != 1:
                continue

            # Primary Entry: 50% MT
            block_top = max(ob_o, ob_c)
            block_bot = ob_l
            mt_entry = (block_top + block_bot) / 2.0
            stop_px = block_bot - 1.0
            risk_pts = mt_entry - stop_px

            if not (8.0 <= risk_pts <= 25.0):
                continue

            target_px = mt_entry + (risk_pts * target_r)

            # Check fill
            filled = False
            fill_k = -1
            for k in range(i + 1, min(n - 1, i + 15)):
                if lows[k] <= mt_entry:
                    filled = True
                    fill_k = k
                    break
                if highs[k] >= target_px:
                    break

            if filled:
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
                    exit_reason=exit_reason, net_pts=net_pts, real_r=net_pts/risk_pts,
                    dur_bars=dur, trade_type="Primary_Sweep"
                ))

            # ── SECONDARY NARRATIVE ENTRY: Continuation Retest ──
            # If primary was filled and hit target, OR if price broke higher without filling MT (missed entry),
            # the narrative says: "Trapped sellers are liquidating, trend is active."
            # We look for the 1st pullback into the displacement bar's high/open (Breaker/FVG retest)
            if enable_secondary_entries and (not filled or exit_reason == 'TARGET'):
                # Look for a secondary pullback within bars i+2 to i+10
                sec_entry_level = highs[ob_idx] # Breakout retest level
                sec_stop = lows[i] - 1.0 # Protected by displacement candle low
                sec_risk = sec_entry_level - sec_stop
                if 8.0 <= sec_risk <= 22.0:
                    sec_target = sec_entry_level + (sec_risk * target_r)
                    start_search = max(i + 2, in_trade_until + 1)
                    sec_filled = False
                    sec_fill_k = -1
                    for k in range(start_search, min(n - 1, start_search + 10)):
                        if lows[k] <= sec_entry_level and highs[k] > sec_entry_level:
                            sec_filled = True
                            sec_fill_k = k
                            break
                        if highs[k] >= sec_target:
                            break

                    if sec_filled:
                        s_exit_px = 0.0
                        s_reason = 'EOD'
                        s_dur = 0
                        for k in range(sec_fill_k, min(n - 1, sec_fill_k + 40)):
                            s_dur += 1
                            if lows[k] <= sec_stop:
                                s_exit_px = sec_stop
                                s_reason = 'STOP'
                                in_trade_until = k
                                break
                            if highs[k] >= sec_target:
                                s_exit_px = sec_target
                                s_reason = 'TARGET'
                                in_trade_until = k
                                break
                        if s_exit_px == 0.0:
                            s_exit_px = closes[min(n - 1, sec_fill_k + 39)]
                            in_trade_until = min(n - 1, sec_fill_k + 39)

                        s_net = (s_exit_px - sec_entry_level) - COMMISSION_PTS - SLIPPAGE_PTS
                        trades.append(Trade(
                            year=t.year, entry_time=pd.Timestamp(times[sec_fill_k]), direction=1,
                            entry_px=sec_entry_level, stop_px=sec_stop, risk_pts=sec_risk, exit_px=s_exit_px,
                            exit_reason=s_reason, net_pts=s_net, real_r=s_net/sec_risk,
                            dur_bars=s_dur, trade_type="Secondary_Continuation"
                        ))

        # ── BEARISH SETUP ──
        elif is_bear_disp and ob_c > ob_o:
            if np.isnan(sh[ob_idx]) or ob_h <= sh[ob_idx]:
                continue
            if ob_c > sh[ob_idx] + 0.75 * atrs[i]:
                continue

            upper_wick = ob_h - max(ob_o, ob_c)
            wick_ratio = upper_wick / ob_rng
            if not ((wick_ratio >= 0.25) or (deltas[ob_idx] < 0) or (deltas[i] < 0)):
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

            filled = False
            fill_k = -1
            for k in range(i + 1, min(n - 1, i + 15)):
                if highs[k] >= mt_entry:
                    filled = True
                    fill_k = k
                    break
                if lows[k] <= target_px:
                    break

            if filled:
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
                    exit_reason=exit_reason, net_pts=net_pts, real_r=net_pts/risk_pts,
                    dur_bars=dur, trade_type="Primary_Sweep"
                ))

            # ── SECONDARY BEARISH NARRATIVE ENTRY ──
            if enable_secondary_entries and (not filled or exit_reason == 'TARGET'):
                sec_entry_level = lows[ob_idx]
                sec_stop = highs[i] + 1.0
                sec_risk = sec_stop - sec_entry_level
                if 8.0 <= sec_risk <= 22.0:
                    sec_target = sec_entry_level - (sec_risk * target_r)
                    start_search = max(i + 2, in_trade_until + 1)
                    sec_filled = False
                    sec_fill_k = -1
                    for k in range(start_search, min(n - 1, start_search + 10)):
                        if highs[k] >= sec_entry_level and lows[k] < sec_entry_level:
                            sec_filled = True
                            sec_fill_k = k
                            break
                        if lows[k] <= sec_target:
                            break

                    if sec_filled:
                        s_exit_px = 0.0
                        s_reason = 'EOD'
                        s_dur = 0
                        for k in range(sec_fill_k, min(n - 1, sec_fill_k + 40)):
                            s_dur += 1
                            if highs[k] >= sec_stop:
                                s_exit_px = sec_stop
                                s_reason = 'STOP'
                                in_trade_until = k
                                break
                            if lows[k] <= sec_target:
                                s_exit_px = sec_target
                                s_reason = 'TARGET'
                                in_trade_until = k
                                break
                        if s_exit_px == 0.0:
                            s_exit_px = closes[min(n - 1, sec_fill_k + 39)]
                            in_trade_until = min(n - 1, sec_fill_k + 39)

                        s_net = (sec_entry_level - s_exit_px) - COMMISSION_PTS - SLIPPAGE_PTS
                        trades.append(Trade(
                            year=t.year, entry_time=pd.Timestamp(times[sec_fill_k]), direction=-1,
                            entry_px=sec_entry_level, stop_px=sec_stop, risk_pts=sec_risk, exit_px=s_exit_px,
                            exit_reason=s_reason, net_pts=s_net, real_r=s_net/sec_risk,
                            dur_bars=s_dur, trade_type="Secondary_Continuation"
                        ))

    return trades

def print_campaign_results(trades: List[Trade], title: str):
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
    print(f"RESULT: {title}")
    print(f"{'='*75}")
    print(f"Total Trades:        {n} ({tpy:.1f} trades/year, ~{tpy/52:.1f} trades/week)")
    print(f"Win Rate:            {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Profit Factor:       {pf:.2f}")
    print(f"Net Points:          +{net_pts:.1f} pts (${net_usd:,.2f} on 1 NQ)")
    print(f"Avg Stop Loss:       {avg_sl:.1f} pts (${avg_sl*20:.1f} NQ / ${avg_sl*2:.1f} MNQ)")
    print(f"Yearly Consistency:  {green_years}/{len(years)} Green Years")

    if 'trade_type' in tdf.columns:
        print("\nBreakdown by Entry Type:")
        for t_type in tdf['trade_type'].unique():
            sub = tdf[tdf['trade_type'] == t_type]
            sub_w = sub[sub['net_pts'] > 0]
            sub_l = sub[sub['net_pts'] <= 0]
            sub_wr = len(sub_w) / len(sub) * 100.0
            sub_g = sub_w['net_pts'].sum()
            sub_ls = abs(sub_l['net_pts'].sum())
            sub_pf = sub_g / sub_ls if sub_ls > 0 else 99.0
            print(f"  - {t_type:<25}: {len(sub):>3} trades | WR: {sub_wr:.1f}% | PF: {sub_pf:.2f} | Net: +{sub['net_pts'].sum():.1f} pts")

# RUN COMPARISON
t_single = run_campaign_simulation(enable_secondary_entries=False, target_r=1.5)
print_campaign_results(t_single, "1. SINGLE ENTRY (Primary Sweep Only - Baseline)")

t_campaign = run_campaign_simulation(enable_secondary_entries=True, target_r=1.5)
print_campaign_results(t_campaign, "2. NARRATIVE CAMPAIGN (Primary Sweep + Secondary Continuation Retest)")

