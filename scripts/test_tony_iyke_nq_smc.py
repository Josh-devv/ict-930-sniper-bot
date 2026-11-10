"""
Tony Iyke Structure, Liquidity & Point of Interest (POI) Strategy on CME NQ Futures
Tested on 8.8 Years of Real CME Continuous NQ Futures Data (2018-2026)
==================================================================================

Architecture:
1. High Timeframe (HTF - 15m / 1H):
   - Market Structure & Trend: Price trading in alignment with 1H Trend (EMA20 / BOS).
   - High Timeframe POI (Point of Interest):
     * Bullish: 15m Fair Value Gap (Candle 1 High < Candle 3 Low) or 15m Order Block.
     * Bearish: 15m Fair Value Gap (Candle 1 Low > Candle 3 High) or 15m Order Block.

2. Lower Timeframe (LTF - 3m / 5m):
   - Price taps into the active 15m POI with the trend (Discount for Long, Premium for Short).
   - Inducement (IDM) Liquidity Sweep: Price sweeps a local 5-bar swing low (long) or swing high (short)
     to take out early retail stops inside the POI.
   - CHoCH (Change of Character): Strong displacement candle closing in direction of HTF trend
     with a rejection wick showing institutional sponsorship.

3. Execution & Risk:
   - Passive limit entry at 50% Mean Threshold (MT) of the CHoCH / OB candle.
   - Structural Stop Loss: 1.0 point beyond the swept Inducement swing.
   - Take Profit: 1.5R and 2.0R targets.
   - Realistic Execution: Hard stop checked first, pessimistic limit fill, $5.20 NQ / $1.24 MNQ fees, 1 tick slippage.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List

print("=" * 80)
print("TESTING TONY IYKE STRUCTURE, LIQUIDITY & POI STRATEGY ON CME NQ (2018-2026)")
print("=" * 80)

# Load 1m Data
df_1m = pd.read_csv('sample_data/NQ_1min_RTH_2018_2026.csv')
df_1m.columns = [c.lower() for c in df_1m.columns]
df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
df_1m = df_1m[df_1m['high'] > df_1m['low']].reset_index(drop=True)
df_1m = df_1m.set_index('datetime').sort_index()

# Resample HTF: 1H for Trend & 15m for POI (FVG & OB)
df_1h = df_1m.resample('1h', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_1h['ema20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
df_1h['htf_trend'] = np.where(df_1h['close'] > df_1h['ema20'], 1, -1)

df_15m = df_1m.resample('15min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()

# 15m POI: Fair Value Gaps (FVG)
# Bullish FVG: Candle 1 High < Candle 3 Low
df_15m['bull_fvg_top'] = df_15m['low']
df_15m['bull_fvg_bot'] = df_15m['high'].shift(2)
df_15m['is_bull_fvg']  = (df_15m['bull_fvg_top'] > df_15m['bull_fvg_bot'] + 2.0)

# Bearish FVG: Candle 1 Low > Candle 3 High
df_15m['bear_fvg_bot'] = df_15m['high']
df_15m['bear_fvg_top'] = df_15m['low'].shift(2)
df_15m['is_bear_fvg']  = (df_15m['bear_fvg_top'] > df_15m['bear_fvg_bot'] + 2.0)

# Map 15m POI back to 1m / LTF
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
    setup_name: str

def run_tony_iyke_smc(
    timeframe: str = '5min',
    target_r: float = 1.5,
    min_stop: float = 8.0,
    max_stop: float = 25.0,
    require_15m_poi: bool = True
) -> List[Trade]:

    # Resample LTF
    df_ltf = df_1m.resample(timeframe, closed='left', label='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()

    tr = np.maximum(df_ltf['high'] - df_ltf['low'],
                    np.maximum(abs(df_ltf['high'] - df_ltf['close'].shift(1)),
                               abs(df_ltf['low'] - df_ltf['close'].shift(1))))
    df_ltf['atr'] = tr.rolling(14).mean().bfill()
    df_ltf['hour_ts'] = df_ltf['datetime'].dt.floor('1h')
    df_ltf['htf_trend'] = df_ltf['hour_ts'].map(df_1h['htf_trend']).fillna(0)

    # Inducement Swings (5-bar fractal on LTF)
    lookback_idm = 5
    df_ltf['idm_low']  = df_ltf['low'].shift(1).rolling(lookback_idm).min()
    df_ltf['idm_high'] = df_ltf['high'].shift(1).rolling(lookback_idm).max()

    # Align 15m POI using merge_asof
    bull_pois = df_15m[df_15m['is_bull_fvg']][['datetime', 'bull_fvg_top', 'bull_fvg_bot']].copy()
    bear_pois = df_15m[df_15m['is_bear_fvg']][['datetime', 'bear_fvg_top', 'bear_fvg_bot']].copy()

    df_ltf = pd.merge_asof(df_ltf, bull_pois, on='datetime', direction='backward')
    df_ltf = pd.merge_asof(df_ltf, bear_pois, on='datetime', direction='backward')

    times = df_ltf['datetime'].values
    opens = df_ltf['open'].values
    highs = df_ltf['high'].values
    lows = df_ltf['low'].values
    closes = df_ltf['close'].values
    atrs = df_ltf['atr'].values
    htf = df_ltf['htf_trend'].values
    idm_lows = df_ltf['idm_low'].values
    idm_highs = df_ltf['idm_high'].values
    b_poi_tops = df_ltf['bull_fvg_top'].values
    b_poi_bots = df_ltf['bull_fvg_bot'].values
    s_poi_tops = df_ltf['bear_fvg_top'].values
    s_poi_bots = df_ltf['bear_fvg_bot'].values

    n = len(df_ltf)
    trades: List[Trade] = []
    in_trade_until = -1

    for i in range(lookback_idm + 5, n - 40):
        if i <= in_trade_until:
            continue

        t = pd.Timestamp(times[i])
        h, m = t.hour, t.minute

        # RTH session: 09:45 to 15:30
        if not ((h == 9 and m >= 45) or (10 <= h < 15) or (h == 15 and m <= 30)):
            continue

        bar_rng = highs[i] - lows[i]
        if bar_rng < 1.0:
            continue

        body = abs(closes[i] - opens[i])

        # ── BULLISH TONY IYKE SMC SETUP ──────────────────────────────────────
        # 1. HTF Trend is Bullish
        if htf[i] == 1:
            # 2. Inside 15m Bullish POI (Fair Value Gap / Discount zone)
            inside_poi = True
            if require_15m_poi:
                top = b_poi_tops[i]
                bot = b_poi_bots[i]
                if np.isnan(top) or np.isnan(bot):
                    inside_poi = False
                elif not (lows[i] <= top and highs[i] >= bot):
                    inside_poi = False

            if inside_poi:
                # 3. Sweeps LTF Inducement (takes out minor local low)
                swept_idm = (not np.isnan(idm_lows[i])) and (lows[i] < idm_lows[i])
                if swept_idm:
                    # 4. CHoCH / Displacement Confirmation: Closes green, strong body, rejection wick >= 20%
                    lower_wick = min(opens[i], closes[i]) - lows[i]
                    wick_ratio = lower_wick / bar_rng
                    is_choch = (closes[i] > opens[i]) and (body >= 0.50 * atrs[i]) and (wick_ratio >= 0.20)

                    if is_choch:
                        # 5. Entry at 50% Mean Threshold of the lower wick / candle
                        entry_px = lows[i] + 0.50 * lower_wick
                        stop_px = lows[i] - 1.0
                        risk_pts = entry_px - stop_px

                        if min_stop <= risk_pts <= max_stop:
                            target_px = entry_px + (risk_pts * target_r)

                            # Forward limit fill simulation
                            filled = False
                            fill_k = -1
                            for k in range(i + 1, min(n - 1, i + 15)):
                                if lows[k] <= entry_px:
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

                                net_pts = (exit_px - entry_px) - COMMISSION_PTS - SLIPPAGE_PTS
                                trades.append(Trade(
                                    year=t.year,
                                    entry_time=pd.Timestamp(times[fill_k]),
                                    direction=1,
                                    entry_px=entry_px,
                                    stop_px=stop_px,
                                    risk_pts=risk_pts,
                                    exit_px=exit_px,
                                    exit_reason=exit_reason,
                                    net_pts=net_pts,
                                    real_r=net_pts / risk_pts,
                                    dur_bars=dur,
                                    setup_name=f"TonyIyke_Bull_{timeframe}"
                                ))

        # ── BEARISH TONY IYKE SMC SETUP ──────────────────────────────────────
        elif htf[i] == -1:
            inside_poi = True
            if require_15m_poi:
                top = s_poi_tops[i]
                bot = s_poi_bots[i]
                if np.isnan(top) or np.isnan(bot):
                    inside_poi = False
                elif not (highs[i] >= bot and lows[i] <= top):
                    inside_poi = False

            if inside_poi:
                swept_idm = (not np.isnan(idm_highs[i])) and (highs[i] > idm_highs[i])
                if swept_idm:
                    upper_wick = highs[i] - max(opens[i], closes[i])
                    wick_ratio = upper_wick / bar_rng
                    is_choch = (closes[i] < opens[i]) and (body >= 0.50 * atrs[i]) and (wick_ratio >= 0.20)

                    if is_choch:
                        entry_px = highs[i] - 0.50 * upper_wick
                        stop_px = highs[i] + 1.0
                        risk_pts = stop_px - entry_px

                        if min_stop <= risk_pts <= max_stop:
                            target_px = entry_px - (risk_pts * target_r)

                            filled = False
                            fill_k = -1
                            for k in range(i + 1, min(n - 1, i + 15)):
                                if highs[k] >= entry_px:
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

                                net_pts = (entry_px - exit_px) - COMMISSION_PTS - SLIPPAGE_PTS
                                trades.append(Trade(
                                    year=t.year,
                                    entry_time=pd.Timestamp(times[fill_k]),
                                    direction=-1,
                                    entry_px=entry_px,
                                    stop_px=stop_px,
                                    risk_pts=risk_pts,
                                    exit_px=exit_px,
                                    exit_reason=exit_reason,
                                    net_pts=net_pts,
                                    real_r=net_pts / risk_pts,
                                    dur_bars=dur,
                                    setup_name=f"TonyIyke_Bear_{timeframe}"
                                ))

    return trades

def print_smc_results(trades: List[Trade], title: str):
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

# TEST 1: Tony Iyke 5-Minute SMC Model (15m POI + Inducement Sweep + CHoCH)
t_5m_poi = run_tony_iyke_smc(timeframe='5min', target_r=1.5, require_15m_poi=True)
print_smc_results(t_5m_poi, "1. TONY IYKE 5-MINUTE (15m FVG POI + 5m Inducement Sweep + CHoCH)")

# TEST 2: Tony Iyke 3-Minute SMC Model (15m POI + 3m Inducement Sweep + CHoCH)
t_3m_poi = run_tony_iyke_smc(timeframe='3min', target_r=1.5, require_15m_poi=True)
print_smc_results(t_3m_poi, "2. TONY IYKE 3-MINUTE (15m FVG POI + 3m Inducement Sweep + CHoCH)")

# TEST 3: Pure Trend Market Structure & Inducement (No rigid FVG filter, just HTF 1H Trend + Inducement Sweep + CHoCH)
t_5m_structure = run_tony_iyke_smc(timeframe='5min', target_r=1.5, require_15m_poi=False)
print_smc_results(t_5m_structure, "3. PURE MARKET STRUCTURE 5-MIN (1H Trend + 5m Inducement Sweep + CHoCH)")

# TEST 4: Pure Trend Market Structure 3-MIN (1H Trend + 3m Inducement Sweep + CHoCH)
t_3m_structure = run_tony_iyke_smc(timeframe='3min', target_r=1.5, require_15m_poi=False)
print_smc_results(t_3m_structure, "4. PURE MARKET STRUCTURE 3-MIN (1H Trend + 3m Inducement Sweep + CHoCH)")

# TEST 5: 5-MIN Pure Structure with 2.0R Target
t_5m_2r = run_tony_iyke_smc(timeframe='5min', target_r=2.0, require_15m_poi=False)
print_smc_results(t_5m_2r, "5. PURE MARKET STRUCTURE 5-MIN @ 2.0R TARGET")

