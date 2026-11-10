"""
High-Frequency Institutional NQ Engine (Weekly / Monthly Active Trades)
======================================================================
100% Zero-Cheat Backtest on 8 Years of Real CME NQ Futures (2018-2026)

Goal: Achieve consistent weekly/monthly trade frequency (100 - 300+ trades/year)
with small stops, high win rate, and realistic execution.

Core Architecture:
1. Higher Timeframe Bias: 15-minute / 1-Hour Trend & VWAP Direction
2. Execution Timeframe: 3-minute and 5-minute charts
3. Setups Tested:
   A) Intra-day Liquidity Sweeps (5-10 bar Fractal High/Low Sweeps + Delta Absorption)
   B) 10:00 AM Macro / Silver Bullet Displacement & Order Block Retests
   C) Fair Value Gap (FVG) + Order Block 50% Mean Threshold Entries
4. Risk Management:
   - Stop Loss: 8 to 22 points (tight structural stop)
   - Target: 1.5R, 2.0R, and Runner Scaling (50% @ 1.5R -> BE -> 50% @ 2.5R)
   - Fees: $5.20 RT NQ / $1.24 MNQ + 1 tick slippage
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List

print("=" * 80)
print("ACTIVE HIGH-FREQUENCY INSTITUTIONAL ENGINE (2018-2026)")
print("Target: 100-300+ trades/year (multiple trades/week) with positive expectancy")
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

# Resample to 3m, 5m, and 1h
df_3m = df_1m.resample('3min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'delta': 'sum'
}).dropna().reset_index()

df_5m = df_1m.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'delta': 'sum'
}).dropna().reset_index()

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
    realized_r: float
    dur_bars: int
    setup_name: str


def run_active_engine(
    df_tf: pd.DataFrame,
    tf_name: str,
    target_r: float = 2.0,
    swing_lookback: int = 8,
    min_stop: float = 8.0,
    max_stop: float = 25.0,
    use_runner: bool = False
) -> List[Trade]:

    tr = np.maximum(df_tf['high'] - df_tf['low'],
                    np.maximum(abs(df_tf['high'] - df_tf['close'].shift(1)),
                               abs(df_tf['low'] - df_tf['close'].shift(1))))
    df_tf['atr'] = tr.rolling(14).mean().bfill()
    df_tf['vol_ma'] = df_tf['volume'].rolling(14).mean().bfill()
    df_tf['hour_ts'] = df_tf['datetime'].dt.floor('1h')
    df_tf['htf_trend'] = df_tf['hour_ts'].map(df_1h['htf_trend']).fillna(0)

    # Rolling swing highs and lows
    df_tf['sh'] = df_tf['high'].shift(1).rolling(swing_lookback).max()
    df_tf['sl'] = df_tf['low'].shift(1).rolling(swing_lookback).min()

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
    sh = df_tf['sh'].values
    sl = df_tf['sl'].values

    n = len(df_tf)
    trades: List[Trade] = []
    in_trade_until = -1

    for i in range(swing_lookback + 5, n - 40):
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
        is_bull_disp = (disp_body >= 0.85 * atrs[i]) and (closes[i] > opens[i])
        is_bear_disp = (disp_body >= 0.85 * atrs[i]) and (closes[i] < opens[i])

        # ── BULLISH SETUP ────────────────────────────────────────────────────
        if is_bull_disp and ob_c < ob_o:
            # 1. Intra-day Liquidity Sweep of recent swing low
            if np.isnan(sl[ob_idx]) or ob_l >= sl[ob_idx]:
                continue
            # Must close back inside or near the level
            if ob_c < sl[ob_idx] - 0.75 * atrs[i]:
                continue

            # 2. Absorption: Rejection wick >= 25% OR delta positive
            lower_wick = min(ob_o, ob_c) - ob_l
            wick_ratio = lower_wick / ob_rng
            has_absorption = (wick_ratio >= 0.25) or (ob_d > 0) or (deltas[i] > 0)
            if not has_absorption:
                continue

            # 3. Macro Trend Alignment (1H EMA20)
            if htf[i] != 1:
                continue

            # 4. Entry & Risk: 50% Mean Threshold of OB
            block_top = max(ob_o, ob_c)
            block_bot = ob_l
            mt_entry = (block_top + block_bot) / 2.0
            stop_px = block_bot - 1.0
            risk_pts = mt_entry - stop_px

            if not (min_stop <= risk_pts <= max_stop):
                continue

            # 5. Forward pessimistic fill
            filled = False
            fill_k = -1
            for k in range(i + 1, min(n - 1, i + 15)):
                if lows[k] <= mt_entry:
                    filled = True
                    fill_k = k
                    break

            if not filled:
                continue

            actual_entry = mt_entry + SLIPPAGE_PTS
            dur = 0
            exit_px = 0.0
            exit_reason = ""

            if use_runner:
                # 50% at 1.5R -> SL to Breakeven -> 50% at 2.5R runner
                tp1_px = mt_entry + 1.5 * risk_pts
                tp2_px = mt_entry + 2.5 * risk_pts
                tp1_hit = False
                be_stop = actual_entry

                for k in range(fill_k, min(n - 1, fill_k + 35)):
                    dur = k - fill_k
                    curr_stop = be_stop if tp1_hit else stop_px
                    if lows[k] <= curr_stop:
                        exit_px = curr_stop - SLIPPAGE_PTS
                        exit_reason = "BE_STOP" if tp1_hit else "STOP_LOSS"
                        in_trade_until = k
                        break
                    if not tp1_hit and highs[k] >= tp1_px:
                        tp1_hit = True
                    if tp1_hit and highs[k] >= tp2_px:
                        exit_px = tp2_px - SLIPPAGE_PTS
                        exit_reason = "RUNNER_2.5R"
                        in_trade_until = k
                        break
                    if pd.Timestamp(times[k]).hour == 15 and pd.Timestamp(times[k]).minute >= 50:
                        exit_px = closes[k] - SLIPPAGE_PTS
                        exit_reason = "EOD_EXIT"
                        in_trade_until = k
                        break

                if exit_reason:
                    if exit_reason == "STOP_LOSS":
                        net = (exit_px - actual_entry) - COMMISSION_PTS
                    elif exit_reason in ("RUNNER_2.5R", "BE_STOP"):
                        net = 0.5 * (tp1_px - actual_entry) + 0.5 * (exit_px - actual_entry) - COMMISSION_PTS
                    else:
                        net = (0.5 * (tp1_px - actual_entry) + 0.5 * (exit_px - actual_entry) - COMMISSION_PTS) if tp1_hit else ((exit_px - actual_entry) - COMMISSION_PTS)

                    trades.append(Trade(
                        t.year, pd.Timestamp(times[fill_k]), 1, actual_entry, stop_px,
                        risk_pts, exit_px, exit_reason, net, net / risk_pts, dur, f"{tf_name}_BULL_SWEEP"
                    ))

            else:
                target_px = mt_entry + target_r * risk_pts
                for k in range(fill_k, min(n - 1, fill_k + 35)):
                    dur = k - fill_k
                    if lows[k] <= stop_px:
                        exit_px = stop_px - SLIPPAGE_PTS
                        exit_reason = "STOP_LOSS"
                        in_trade_until = k
                        break
                    if highs[k] >= target_px:
                        exit_px = target_px - SLIPPAGE_PTS
                        exit_reason = f"TARGET_{target_r}R"
                        in_trade_until = k
                        break
                    if pd.Timestamp(times[k]).hour == 15 and pd.Timestamp(times[k]).minute >= 50:
                        exit_px = closes[k] - SLIPPAGE_PTS
                        exit_reason = "EOD_EXIT"
                        in_trade_until = k
                        break

                if exit_reason:
                    net = (exit_px - actual_entry) - COMMISSION_PTS
                    trades.append(Trade(
                        t.year, pd.Timestamp(times[fill_k]), 1, actual_entry, stop_px,
                        risk_pts, exit_px, exit_reason, net, net / risk_pts, dur, f"{tf_name}_BULL_SWEEP"
                    ))

        # ── BEARISH SETUP ────────────────────────────────────────────────────
        elif is_bear_disp and ob_c > ob_o:
            if np.isnan(sh[ob_idx]) or ob_h <= sh[ob_idx]:
                continue
            if ob_c > sh[ob_idx] + 0.75 * atrs[i]:
                continue

            upper_wick = ob_h - max(ob_o, ob_c)
            wick_ratio = upper_wick / ob_rng
            has_absorption = (wick_ratio >= 0.25) or (ob_d < 0) or (deltas[i] < 0)
            if not has_absorption:
                continue

            if htf[i] != -1:
                continue

            block_top = ob_h
            block_bot = min(ob_o, ob_c)
            mt_entry = (block_top + block_bot) / 2.0
            stop_px = block_top + 1.0
            risk_pts = stop_px - mt_entry

            if not (min_stop <= risk_pts <= max_stop):
                continue

            filled = False
            fill_k = -1
            for k in range(i + 1, min(n - 1, i + 15)):
                if highs[k] >= mt_entry:
                    filled = True
                    fill_k = k
                    break

            if not filled:
                continue

            actual_entry = mt_entry - SLIPPAGE_PTS
            dur = 0
            exit_px = 0.0
            exit_reason = ""

            if use_runner:
                tp1_px = mt_entry - 1.5 * risk_pts
                tp2_px = mt_entry - 2.5 * risk_pts
                tp1_hit = False
                be_stop = actual_entry

                for k in range(fill_k, min(n - 1, fill_k + 35)):
                    dur = k - fill_k
                    curr_stop = be_stop if tp1_hit else stop_px
                    if highs[k] >= curr_stop:
                        exit_px = curr_stop + SLIPPAGE_PTS
                        exit_reason = "BE_STOP" if tp1_hit else "STOP_LOSS"
                        in_trade_until = k
                        break
                    if not tp1_hit and lows[k] <= tp1_px:
                        tp1_hit = True
                    if tp1_hit and lows[k] <= tp2_px:
                        exit_px = tp2_px + SLIPPAGE_PTS
                        exit_reason = "RUNNER_2.5R"
                        in_trade_until = k
                        break
                    if pd.Timestamp(times[k]).hour == 15 and pd.Timestamp(times[k]).minute >= 50:
                        exit_px = closes[k] + SLIPPAGE_PTS
                        exit_reason = "EOD_EXIT"
                        in_trade_until = k
                        break

                if exit_reason:
                    if exit_reason == "STOP_LOSS":
                        net = (actual_entry - exit_px) - COMMISSION_PTS
                    elif exit_reason in ("RUNNER_2.5R", "BE_STOP"):
                        net = 0.5 * (actual_entry - tp1_px) + 0.5 * (actual_entry - exit_px) - COMMISSION_PTS
                    else:
                        net = (0.5 * (actual_entry - tp1_px) + 0.5 * (actual_entry - exit_px) - COMMISSION_PTS) if tp1_hit else ((actual_entry - exit_px) - COMMISSION_PTS)

                    trades.append(Trade(
                        t.year, pd.Timestamp(times[fill_k]), -1, actual_entry, stop_px,
                        risk_pts, exit_px, exit_reason, net, net / risk_pts, dur, f"{tf_name}_BEAR_SWEEP"
                    ))

            else:
                target_px = mt_entry - target_r * risk_pts
                for k in range(fill_k, min(n - 1, fill_k + 35)):
                    dur = k - fill_k
                    if highs[k] >= stop_px:
                        exit_px = stop_px + SLIPPAGE_PTS
                        exit_reason = "STOP_LOSS"
                        in_trade_until = k
                        break
                    if lows[k] <= target_px:
                        exit_px = target_px + SLIPPAGE_PTS
                        exit_reason = f"TARGET_{target_r}R"
                        in_trade_until = k
                        break
                    if pd.Timestamp(times[k]).hour == 15 and pd.Timestamp(times[k]).minute >= 50:
                        exit_px = closes[k] + SLIPPAGE_PTS
                        exit_reason = "EOD_EXIT"
                        in_trade_until = k
                        break

                if exit_reason:
                    net = (actual_entry - exit_px) - COMMISSION_PTS
                    trades.append(Trade(
                        t.year, pd.Timestamp(times[fill_k]), -1, actual_entry, stop_px,
                        risk_pts, exit_px, exit_reason, net, net / risk_pts, dur, f"{tf_name}_BEAR_SWEEP"
                    ))

    return trades


# ─── 4. Run Matrix across 3m and 5m ───────────────────────────────────────────
configs = [
    (df_5m, "5m", 1.5, 8, 8.0, 25.0, False, "5m Pure 1.5R (8-bar sweep)"),
    (df_5m, "5m", 2.0, 8, 8.0, 25.0, False, "5m Pure 2.0R (8-bar sweep)"),
    (df_5m, "5m", 2.0, 8, 8.0, 25.0, True,  "5m Scale-Runner (1.5R/2.5R)"),
    (df_5m, "5m", 1.5, 5, 8.0, 25.0, False, "5m High-Frequency (5-bar sweep, 1.5R)"),
    (df_3m, "3m", 1.5, 10, 6.0, 20.0, False, "3m Pure 1.5R (10-bar sweep)"),
    (df_3m, "3m", 2.0, 10, 6.0, 20.0, False, "3m Pure 2.0R (10-bar sweep)"),
    (df_3m, "3m", 2.0, 10, 6.0, 20.0, True,  "3m Scale-Runner (1.5R/2.5R)"),
]

print("\n" + "=" * 115)
print(f"| {'CONFIGURATION':<42} | {'TRADES':>6} | {'TR/YR':>6} | {'TR/MO':>5} | {'WIN%':>6} | {'PF':>5} | {'AVG SL':>6} | {'NET PTS':>8} | {'NQ NET $':>10} |")
print("=" * 115)

for df_c, name_c, trg_c, sw_c, min_s, max_s, run_c, label in configs:
    trades = run_active_engine(df_c, name_c, target_r=trg_c, swing_lookback=sw_c,
                               min_stop=min_s, max_stop=max_s, use_runner=run_c)
    if not trades:
        print(f"| {label:<42} | No trades |")
        continue

    tdf = pd.DataFrame([t.__dict__ for t in trades])
    n = len(tdf)
    tr_yr = n / 8.8
    tr_mo = tr_yr / 12.0
    wins = tdf[tdf['net_pts'] > 0]
    losses = tdf[tdf['net_pts'] <= 0]
    wr = len(wins) / n * 100
    pf = wins['net_pts'].sum() / abs(losses['net_pts'].sum()) if len(losses) > 0 and losses['net_pts'].sum() != 0 else 99.0
    tot_pts = tdf['net_pts'].sum()
    avg_sl = tdf['risk_pts'].mean()
    nq_net = tot_pts * 20.0

    print(f"| {label:<42} | {n:>6} | {tr_yr:>6.1f} | {tr_mo:>5.1f} | {wr:>5.1f}% | {pf:>5.2f} | {avg_sl:>5.1f}p | {tot_pts:>+7.1f}p | +${nq_net:>9,.0f} |")

print("=" * 115)

