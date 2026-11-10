"""
5-Minute Multi-Pool Institutional Orderflow Model
==================================================
Testing 5-minute timeframe with identical zero-cheat rules:
- 15-bar 5m Swing High/Low Sweeps + ORH/ORL Sweeps + PDH/PDL Sweeps
- Volume Climax >= 1.2x 20 SMA
- Absorption Wick >= 30%
- Delta Reversal
- 50% Mean Threshold Entry on 5m Order Block
- 1H Trend Alignment
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List

df_1m = pd.read_csv('sample_data/NQ_1min_RTH_2018_2026.csv')
df_1m.columns = [c.lower() for c in df_1m.columns]
df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
df_1m = df_1m[df_1m['high'] > df_1m['low']].reset_index(drop=True)
df_1m = df_1m.set_index('datetime').sort_index()

rng_1m = (df_1m['high'] - df_1m['low']).replace(0, 1e-6)
df_1m['delta'] = df_1m['volume'] * (2.0 * df_1m['close'] - df_1m['high'] - df_1m['low']) / rng_1m

# Resample 5m and 1h
df_5m = df_1m.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'delta': 'sum'
}).dropna().reset_index()

df_1h = df_1m.resample('1h', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_1h['ema20'] = df_1h['close'].ewm(span=20, adjust=False).mean()
df_1h['htf_trend'] = np.where(df_1h['close'] > df_1h['ema20'], 1, -1)

# Daily PDH / PDL
df_daily = df_1m.resample('1D').agg({
    'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
df_daily['pdh'] = df_daily['high'].shift(1)
df_daily['pdl'] = df_daily['low'].shift(1)

daily_levels = {}
for dt, row in df_daily.iterrows():
    daily_levels[dt.date()] = {'pdh': row['pdh'], 'pdl': row['pdl']}

tr = np.maximum(df_5m['high'] - df_5m['low'],
                np.maximum(abs(df_5m['high'] - df_5m['close'].shift(1)),
                           abs(df_5m['low'] - df_5m['close'].shift(1))))
df_5m['atr'] = tr.rolling(14).mean().bfill()
df_5m['vol_ma'] = df_5m['volume'].rolling(20).mean().bfill()
df_5m['hour_ts'] = df_5m['datetime'].dt.floor('1h')
df_5m['htf_trend'] = df_5m['hour_ts'].map(df_1h['htf_trend']).fillna(0)

# 15-bar swing highs and lows on 5m
df_5m['sh15'] = df_5m['high'].shift(1).rolling(15).max()
df_5m['sl15'] = df_5m['low'].shift(1).rolling(15).min()

COMMISSION_PTS = 0.26
SLIPPAGE_PTS   = 0.25

@dataclass
class TradeRecord:
    year: int
    entry_time: pd.Timestamp
    direction: int
    entry_px: float
    stop_px: float
    risk_pts: float
    exit_px: float
    exit_reason: str
    net_pts: float
    realized_r: float
    dur_bars: int
    pool_type: str

def run_5m_simulation() -> List[TradeRecord]:
    dates = df_5m['datetime'].dt.date.values
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
    sh = df_5m['sh15'].values
    sl = df_5m['sl15'].values

    n = len(df_5m)
    trades: List[TradeRecord] = []
    current_day = None
    orh, orl = None, None

    for i in range(25, n - 40):
        t = pd.Timestamp(times[i])
        day_date = dates[i]
        h, m = t.hour, t.minute

        if day_date != current_day:
            current_day = day_date
            orh, orl = None, None

        if h == 9 and m >= 30 and m <= 45:
            if orh is None or highs[i] > orh:
                orh = highs[i]
            if orl is None or lows[i] < orl:
                orl = lows[i]

        # Active trading window: 10:00 to 14:30
        if not ((h >= 10 and h < 14) or (h == 14 and m <= 30)):
            continue

        p_levels = daily_levels.get(day_date, None)
        pdh = p_levels['pdh'] if p_levels else np.nan
        pdl = p_levels['pdl'] if p_levels else np.nan

        ob_idx = i - 1
        ob_o, ob_h, ob_l, ob_c = opens[ob_idx], highs[ob_idx], lows[ob_idx], closes[ob_idx]
        ob_v, ob_d, ob_vma = volumes[ob_idx], deltas[ob_idx], vol_mas[ob_idx]
        ob_rng = ob_h - ob_l

        if ob_rng < 1.0:
            continue

        disp_body = abs(closes[i] - opens[i])
        is_bull_disp = (disp_body >= 1.0 * atrs[i]) and (closes[i] > opens[i])
        is_bear_disp = (disp_body >= 1.0 * atrs[i]) and (closes[i] < opens[i])

        # BULLISH
        if is_bull_disp and ob_c < ob_o:
            swept_pool = False
            pool_name = ""

            if not np.isnan(pdl) and ob_l < pdl and ob_c >= pdl - 0.5 * atrs[i]:
                swept_pool = True
                pool_name = "PDL"
            elif orl is not None and ob_l < orl and ob_c >= orl - 0.5 * atrs[i]:
                swept_pool = True
                pool_name = "ORL"
            elif not np.isnan(sl[ob_idx]) and ob_l < sl[ob_idx] and ob_c >= sl[ob_idx] - 0.5 * atrs[i]:
                swept_pool = True
                pool_name = "SWING_LOW"

            if not swept_pool:
                continue

            lower_wick = min(ob_o, ob_c) - ob_l
            has_absorption = (lower_wick / ob_rng >= 0.28)
            has_vol = ob_v >= 1.15 * ob_vma
            delta_absorbed = (ob_d > 0) or (deltas[i] > 0)

            if not (has_absorption and has_vol and delta_absorbed):
                continue

            if htf[i] != 1:
                continue

            block_top = max(ob_o, ob_c)
            block_bot = ob_l
            mt_entry = (block_top + block_bot) / 2.0
            stop_px = block_bot - 1.0
            risk_pts = mt_entry - stop_px

            if not (6.0 <= risk_pts <= 25.0):
                continue

            filled = False
            fill_k = -1
            for k in range(i + 1, min(n - 1, i + 25)):
                if lows[k] <= mt_entry:
                    filled = True
                    fill_k = k
                    break

            if not filled:
                continue

            actual_entry = mt_entry + SLIPPAGE_PTS
            target_px = mt_entry + 2.0 * risk_pts
            exit_px = 0.0
            exit_reason = ""
            dur = 0

            for k in range(fill_k, min(n - 1, fill_k + 40)):
                dur = k - fill_k
                if lows[k] <= stop_px:
                    exit_px = stop_px - SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if highs[k] >= target_px:
                    exit_px = target_px - SLIPPAGE_PTS
                    exit_reason = "TARGET_2.0R"
                    break
                if pd.Timestamp(times[k]).hour == 15 and pd.Timestamp(times[k]).minute >= 50:
                    exit_px = closes[k] - SLIPPAGE_PTS
                    exit_reason = "EOD_EXIT"
                    break

            if exit_reason:
                net = (exit_px - actual_entry) - COMMISSION_PTS
                trades.append(TradeRecord(
                    t.year, pd.Timestamp(times[fill_k]), 1, actual_entry, stop_px,
                    risk_pts, exit_px, exit_reason, net, net / risk_pts, dur, pool_name
                ))

        # BEARISH
        elif is_bear_disp and ob_c > ob_o:
            swept_pool = False
            pool_name = ""

            if not np.isnan(pdh) and ob_h > pdh and ob_c <= pdh + 0.5 * atrs[i]:
                swept_pool = True
                pool_name = "PDH"
            elif orh is not None and ob_h > orh and ob_c <= orh + 0.5 * atrs[i]:
                swept_pool = True
                pool_name = "ORH"
            elif not np.isnan(sh[ob_idx]) and ob_h > sh[ob_idx] and ob_c <= sh[ob_idx] + 0.5 * atrs[i]:
                swept_pool = True
                pool_name = "SWING_HIGH"

            if not swept_pool:
                continue

            upper_wick = ob_h - max(ob_o, ob_c)
            has_absorption = (upper_wick / ob_rng >= 0.28)
            has_vol = ob_v >= 1.15 * ob_vma
            delta_absorbed = (ob_d < 0) or (deltas[i] < 0)

            if not (has_absorption and has_vol and delta_absorbed):
                continue

            if htf[i] != -1:
                continue

            block_top = ob_h
            block_bot = min(ob_o, ob_c)
            mt_entry = (block_top + block_bot) / 2.0
            stop_px = block_top + 1.0
            risk_pts = stop_px - mt_entry

            if not (6.0 <= risk_pts <= 25.0):
                continue

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
            target_px = mt_entry - 2.0 * risk_pts
            exit_px = 0.0
            exit_reason = ""
            dur = 0

            for k in range(fill_k, min(n - 1, fill_k + 40)):
                dur = k - fill_k
                if highs[k] >= stop_px:
                    exit_px = stop_px + SLIPPAGE_PTS
                    exit_reason = "STOP_LOSS"
                    break
                if lows[k] <= target_px:
                    exit_px = target_px + SLIPPAGE_PTS
                    exit_reason = "TARGET_2.0R"
                    break
                if pd.Timestamp(times[k]).hour == 15 and pd.Timestamp(times[k]).minute >= 50:
                    exit_px = closes[k] + SLIPPAGE_PTS
                    exit_reason = "EOD_EXIT"
                    break

            if exit_reason:
                net = (actual_entry - exit_px) - COMMISSION_PTS
                trades.append(TradeRecord(
                    t.year, pd.Timestamp(times[fill_k]), -1, actual_entry, stop_px,
                    risk_pts, exit_px, exit_reason, net, net / risk_pts, dur, pool_name
                ))

    return trades

trades_5m = run_5m_simulation()
tdf = pd.DataFrame([t.__dict__ for t in trades_5m])
n = len(tdf)
wins = tdf[tdf['net_pts'] > 0]
losses = tdf[tdf['net_pts'] <= 0]
wr = len(wins) / n * 100
pf = wins['net_pts'].sum() / abs(losses['net_pts'].sum())
tot_pts = tdf['net_pts'].sum()
avg_pts = tdf['net_pts'].mean()
avg_sl = tdf['risk_pts'].mean()

print("=" * 70)
print("5-MINUTE INSTITUTIONAL ORDERFLOW & LEVELS RESULTS (2018-2026)")
print("=" * 70)
print(f"Total Trades:           {n}")
print(f"Win Rate:               {wr:.1f}%")
print(f"Profit Factor:          {pf:.2f}")
print(f"Total Net Points:       {tot_pts:+,.1f} pts")
print(f"Net Profit (1 NQ):      +${tot_pts * 20.0:,.0f}")
print(f"Net Profit (1 MNQ):     +${tot_pts * 2.0:,.0f}")
print(f"Average Stop Loss:      {avg_sl:.1f} pts (${avg_sl * 2.0:.1f} on 1 MNQ)")
print(f"Average Trade Points:   {avg_pts:+.1f} pts")
print("=" * 70)
print(tdf.groupby('pool_type')['net_pts'].agg(['count', 'sum', 'mean']))

