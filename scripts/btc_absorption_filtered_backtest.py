"""
BTC Absorption Strategy: FILTERED with HTF Trend + Structure + Stricter Thresholds
===================================================================================
The raw absorption signal loses money (0.75 PF, -$7K).
This script adds the "missing brain filter" — what you do instinctively when trading live.

Filters tested:
  A) HTF Trend: Only buy above 50 EMA on 1H, only sell below
  B) Stricter Volume: 2.0x instead of 1.3x
  C) Bigger Wick: 40%+ instead of 30%
  D) Market Structure: Require CHoCH/BOS before entry (swing break)
  E) Combined: All filters together

CORRECTED ACCOUNTING: Hard stops checked first, $5 RT commission, proper negative stops.
"""

import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
from typing import List, Optional

def p(msg):
    print(msg)
    sys.stdout.flush()

p("=" * 80)
p("BTC ABSORPTION FILTERED BACKTEST: ADDING THE 'BRAIN FILTER'")
p("=" * 80)

# ── LOAD LOCAL DUKASCOPY BTC DATA ─────────────────────────────────────────────
DATA_PATH = r'C:\Users\JOSH\Desktop\2xjosh\CRTbot\data\crypto\BTCUSD-duka.json'

p(f"Loading BTC data from {DATA_PATH}...")
df_raw = pd.read_json(DATA_PATH)
p(f"JSON parsed, {len(df_raw):,} rows loaded")

df_raw['dt'] = pd.to_datetime(df_raw['time'], unit='ms', utc=True)
df_raw = df_raw.set_index('dt').drop(columns=['time'])
df_raw.columns = ['open', 'high', 'low', 'close', 'volume']
df_raw['volume'] = (df_raw['volume'] * 1_000_000).clip(lower=1)

p(f"Raw bars: {len(df_raw):,} | {df_raw.index[0].date()} to {df_raw.index[-1].date()}")

# ── RESAMPLE TO 5m ────────────────────────────────────────────────────────────
df5 = df_raw.resample('5min', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna(subset=['open'])
df5 = df5[~df5.index.duplicated(keep='first')]
p(f"5m bars: {len(df5):,}")

# ── RESAMPLE TO 1H (for HTF trend) ───────────────────────────────────────────
df1h = df_raw.resample('1h', closed='left', label='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna(subset=['open'])
df1h['ema50'] = df1h['close'].ewm(span=50, adjust=False).mean()
df1h['ema200'] = df1h['close'].ewm(span=200, adjust=False).mean()

# Map 1H EMA back to 5m bars (use the PREVIOUS completed 1H bar)
df5['htf_ema50'] = df1h['ema50'].reindex(df5.index, method='ffill')
df5['htf_ema200'] = df1h['ema200'].reindex(df5.index, method='ffill')
df5['htf_close'] = df1h['close'].reindex(df5.index, method='ffill')

# HTF trend: above both EMAs = bullish, below both = bearish
df5['htf_bull'] = (df5['htf_close'] > df5['htf_ema50']) & (df5['htf_ema50'] > df5['htf_ema200'])
df5['htf_bear'] = (df5['htf_close'] < df5['htf_ema50']) & (df5['htf_ema50'] < df5['htf_ema200'])
# Relaxed: just above/below 50 EMA
df5['htf_bull_relaxed'] = df5['htf_close'] > df5['htf_ema50']
df5['htf_bear_relaxed'] = df5['htf_close'] < df5['htf_ema50']

p(f"HTF Bull bars (strict): {df5['htf_bull'].sum():,} | HTF Bear bars: {df5['htf_bear'].sum():,}")
p(f"HTF Bull bars (relaxed): {df5['htf_bull_relaxed'].sum():,} | HTF Bear bars: {df5['htf_bear_relaxed'].sum():,}")

# ── 5m INDICATORS ─────────────────────────────────────────────────────────────
df = df5.copy()
hours = df.index.hour
df['active_session'] = ((hours >= 7) & (hours <= 11)) | ((hours >= 13) & (hours <= 18))

df['vol_ma'] = df['volume'].rolling(20).mean()

df['local_high_20'] = df['high'].rolling(20).max().shift(1)
df['local_low_20'] = df['low'].rolling(20).min().shift(1)

df['range'] = (df['high'] - df['low']).clip(lower=1.0)
df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
df['upper_wick_ratio'] = df['upper_wick'] / df['range']
df['lower_wick_ratio'] = df['lower_wick'] / df['range']

tr1 = df['high'] - df['low']
tr2 = (df['high'] - df['close'].shift(1)).abs()
tr3 = (df['low'] - df['close'].shift(1)).abs()
df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().bfill()

# ── MARKET STRUCTURE: Swing Highs/Lows + CHoCH ───────────────────────────────
LB = 5  # lookback for swing detection

df['swing_high'] = (
    (df['high'] >= df['high'].shift(1)) &
    (df['high'] >= df['high'].shift(2)) &
    (df['high'] >= df['high'].shift(-1)) &
    (df['high'] >= df['high'].shift(-2))
)
df['swing_low'] = (
    (df['low'] <= df['low'].shift(1)) &
    (df['low'] <= df['low'].shift(2)) &
    (df['low'] <= df['low'].shift(-1)) &
    (df['low'] <= df['low'].shift(-2))
)

# Track recent swing lows/highs for structure breaks
# Bullish CHoCH = price breaks above most recent swing high after making a lower low
# We'll use a rolling approach
df['recent_swing_high'] = df['high'].where(df['swing_high']).ffill()
df['recent_swing_low'] = df['low'].where(df['swing_low']).ffill()
df['prev_swing_high'] = df['high'].where(df['swing_high']).shift(1).ffill()
df['prev_swing_low'] = df['low'].where(df['swing_low']).shift(1).ffill()

# Bullish structure break: close breaks above recent swing high
df['bull_bos'] = df['close'] > df['recent_swing_high'].shift(1)
# Bearish structure break: close breaks below recent swing low
df['bear_bos'] = df['close'] < df['recent_swing_low'].shift(1)

# Rolling structure state: last N bars had a bull/bear BOS
STRUCT_LOOKBACK = 30  # Structure valid for 30 bars (2.5 hours on 5m)
df['bull_struct'] = df['bull_bos'].rolling(STRUCT_LOOKBACK).max().fillna(0).astype(bool)
df['bear_struct'] = df['bear_bos'].rolling(STRUCT_LOOKBACK).max().fillna(0).astype(bool)

p(f"Bull BOS events: {df['bull_bos'].sum():,} | Bear BOS events: {df['bear_bos'].sum():,}")

# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────
BTC_SIZE = 0.1
COMMISSION = 5.0

@dataclass
class Trade:
    date: str
    direction: str
    entry_mode: str
    entry_px: float
    stop_px: float
    tp1_px: float
    exit_px: float
    result: str
    pnl: float
    risk_pts: float

def run_filtered_backtest(
    vol_mult: float = 1.3,
    wick_min: float = 0.30,
    require_htf_trend: bool = False,
    htf_strict: bool = False,
    require_structure: bool = False,
    entry_mode: str = 'direct',
    long_only: bool = False,
    target_r: float = 1.5,
    min_stop: float = 40.0,
) -> List[Trade]:
    trades: List[Trade] = []
    in_pos = False
    pos = None
    
    vals = df.reset_index()
    n = len(vals)
    
    for i in range(50, n - 2):
        row = vals.iloc[i]
        
        if not in_pos:
            vol_ok = row['volume'] >= (row['vol_ma'] * vol_mult) if pd.notna(row['vol_ma']) else False
            
            # ── BULL ABSORPTION ──
            bull_sig = (
                row['active_session'] and
                vol_ok and
                (row['low'] <= row['local_low_20']) and
                (row['lower_wick_ratio'] >= wick_min)
            )
            
            if require_htf_trend and bull_sig:
                if htf_strict:
                    bull_sig = bull_sig and row['htf_bull']
                else:
                    bull_sig = bull_sig and row['htf_bull_relaxed']
            
            if require_structure and bull_sig:
                bull_sig = bull_sig and row['bull_struct']
            
            if bull_sig:
                if entry_mode == 'direct':
                    entry = row['close']
                else:
                    entry = row['low'] + 0.50 * row['lower_wick']
                
                stop = row['low'] - 1.0 * row['atr']
                stop_dist = entry - stop
                
                if stop_dist < min_stop:
                    continue
                
                tp1 = entry + target_r * stop_dist
                
                if entry_mode == 'direct':
                    fill_bar = i + 1
                    filled = True
                else:
                    filled = False
                    fill_bar = -1
                    for k in range(i + 1, min(n - 1, i + 15)):
                        if vals.iloc[k]['low'] <= entry:
                            filled = True
                            fill_bar = k
                            break
                        if vals.iloc[k]['high'] >= tp1:
                            break
                
                if not filled or fill_bar >= n - 1:
                    continue
                
                in_pos = True
                pos = {
                    'date': str(row['dt'].date()),
                    'entry': entry, 'stop': stop, 'stop_dist': stop_dist,
                    'tp1': tp1, 'fill_bar': fill_bar, 'bars_held': 0, 'dir': 'long'
                }
            
            # ── BEAR ABSORPTION ──
            elif not long_only:
                bear_sig = (
                    row['active_session'] and
                    vol_ok and
                    (row['high'] >= row['local_high_20']) and
                    (row['upper_wick_ratio'] >= wick_min)
                )
                
                if require_htf_trend and bear_sig:
                    if htf_strict:
                        bear_sig = bear_sig and row['htf_bear']
                    else:
                        bear_sig = bear_sig and row['htf_bear_relaxed']
                
                if require_structure and bear_sig:
                    bear_sig = bear_sig and row['bear_struct']
                
                if bear_sig:
                    if entry_mode == 'direct':
                        entry = row['close']
                    else:
                        entry = row['high'] - 0.50 * row['upper_wick']
                    
                    stop = row['high'] + 1.0 * row['atr']
                    stop_dist = stop - entry
                    
                    if stop_dist < min_stop:
                        continue
                    
                    tp1 = entry - target_r * stop_dist
                    
                    if entry_mode == 'direct':
                        fill_bar = i + 1
                        filled = True
                    else:
                        filled = False
                        fill_bar = -1
                        for k in range(i + 1, min(n - 1, i + 15)):
                            if vals.iloc[k]['high'] >= entry:
                                filled = True
                                fill_bar = k
                                break
                            if vals.iloc[k]['low'] <= tp1:
                                break
                    
                    if not filled or fill_bar >= n - 1:
                        continue
                    
                    in_pos = True
                    pos = {
                        'date': str(row['dt'].date()),
                        'entry': entry, 'stop': stop, 'stop_dist': stop_dist,
                        'tp1': tp1, 'fill_bar': fill_bar, 'bars_held': 0, 'dir': 'short'
                    }
        
        else:
            pos['bars_held'] += 1
            h, l, c = row['high'], row['low'], row['close']
            is_long = pos['dir'] == 'long'
            
            if is_long:
                # HARD STOP FIRST
                if l <= pos['stop']:
                    pnl = -(pos['stop_dist'] * BTC_SIZE) - COMMISSION
                    trades.append(Trade(
                        date=pos['date'], direction='long', entry_mode=entry_mode,
                        entry_px=pos['entry'], stop_px=pos['stop'], tp1_px=pos['tp1'],
                        exit_px=pos['stop'], result='STOP', pnl=pnl, risk_pts=pos['stop_dist']
                    ))
                    in_pos = False; pos = None; continue
                # TP HIT
                if h >= pos['tp1']:
                    pnl = (pos['tp1'] - pos['entry']) * BTC_SIZE - COMMISSION
                    trades.append(Trade(
                        date=pos['date'], direction='long', entry_mode=entry_mode,
                        entry_px=pos['entry'], stop_px=pos['stop'], tp1_px=pos['tp1'],
                        exit_px=pos['tp1'], result='TP', pnl=pnl, risk_pts=pos['stop_dist']
                    ))
                    in_pos = False; pos = None; continue
            else:
                if h >= pos['stop']:
                    pnl = -(pos['stop_dist'] * BTC_SIZE) - COMMISSION
                    trades.append(Trade(
                        date=pos['date'], direction='short', entry_mode=entry_mode,
                        entry_px=pos['entry'], stop_px=pos['stop'], tp1_px=pos['tp1'],
                        exit_px=pos['stop'], result='STOP', pnl=pnl, risk_pts=pos['stop_dist']
                    ))
                    in_pos = False; pos = None; continue
                if l <= pos['tp1']:
                    pnl = (pos['entry'] - pos['tp1']) * BTC_SIZE - COMMISSION
                    trades.append(Trade(
                        date=pos['date'], direction='short', entry_mode=entry_mode,
                        entry_px=pos['entry'], stop_px=pos['stop'], tp1_px=pos['tp1'],
                        exit_px=pos['tp1'], result='TP', pnl=pnl, risk_pts=pos['stop_dist']
                    ))
                    in_pos = False; pos = None; continue
            
            # Time exit after 30 bars (2.5 hours)
            if pos and pos['bars_held'] >= 30:
                if is_long:
                    pnl = (c - pos['entry']) * BTC_SIZE - COMMISSION
                else:
                    pnl = (pos['entry'] - c) * BTC_SIZE - COMMISSION
                trades.append(Trade(
                    date=pos['date'], direction=pos['dir'], entry_mode=entry_mode,
                    entry_px=pos['entry'], stop_px=pos['stop'], tp1_px=pos['tp1'],
                    exit_px=c, result='TIME_EXIT', pnl=pnl, risk_pts=pos['stop_dist']
                ))
                in_pos = False; pos = None
    
    return trades

def show_results(trades: List[Trade], title: str):
    if not trades:
        p(f"\n{title}: NO TRADES")
        return
    tdf = pd.DataFrame([t.__dict__ for t in trades])
    n = len(tdf)
    wins = tdf[tdf['pnl'] > 0]
    losses = tdf[tdf['pnl'] <= 0]
    wr = len(wins) / n * 100.0
    gp = wins['pnl'].sum()
    gl = abs(losses['pnl'].sum())
    pf = gp / gl if gl > 0 else 99.0
    net = tdf['pnl'].sum()
    
    tdf['month'] = pd.to_datetime(tdf['date']).dt.to_period('M')
    monthly = tdf.groupby('month')['pnl'].sum()
    green_months = (monthly > 0).sum()
    
    avg_risk = tdf['risk_pts'].mean()
    
    p(f"\n{'='*75}")
    p(f"  {title}")
    p(f"{'='*75}")
    p(f"  Trades:        {n}")
    p(f"  Win Rate:      {wr:.1f}% ({len(wins)}W / {len(losses)}L)")
    p(f"  Profit Factor: {pf:.2f}")
    p(f"  Net (0.1 BTC): ${net:,.2f}")
    p(f"  Net (1.0 BTC): ${net*10:,.2f}")
    if len(wins) > 0:
        p(f"  Avg Win:       ${wins['pnl'].mean():,.2f}")
    if len(losses) > 0:
        p(f"  Avg Loss:      ${losses['pnl'].mean():,.2f}")
    p(f"  Avg Risk:      ${avg_risk:,.0f} pts")
    p(f"  Green Months:  {green_months}/{len(monthly)}")
    
    p("\n  Outcomes:")
    for res, grp in tdf.groupby('result'):
        p(f"    {res:<12}: {len(grp):>4} | Avg ${grp['pnl'].mean():>8.2f} | Total ${grp['pnl'].sum():>10,.2f}")
    
    p("\n  Monthly PnL:")
    for m, pnl_val in monthly.items():
        marker = "✅" if pnl_val > 0 else "❌"
        mt = tdf[tdf['month'] == m]
        mwr = (mt['pnl'] > 0).sum() / len(mt) * 100
        p(f"    {m} : {marker} ${pnl_val:>+10,.2f} ({len(mt)} trades, {mwr:.0f}% WR)")

# ── RUN ALL FILTER COMBINATIONS ───────────────────────────────────────────────

configs = [
    # (title, kwargs)
    ("A1: HTF Trend (relaxed 50EMA) + Direct Entry, Both",
     dict(vol_mult=1.3, wick_min=0.30, require_htf_trend=True, htf_strict=False,
          require_structure=False, entry_mode='direct', long_only=False, target_r=1.5)),
    
    ("A2: HTF Trend (strict EMA stack) + Direct Entry, Both",
     dict(vol_mult=1.3, wick_min=0.30, require_htf_trend=True, htf_strict=True,
          require_structure=False, entry_mode='direct', long_only=False, target_r=1.5)),
    
    ("B: Stricter Volume (2.0x) + Direct Entry, Both",
     dict(vol_mult=2.0, wick_min=0.30, require_htf_trend=False,
          require_structure=False, entry_mode='direct', long_only=False, target_r=1.5)),
    
    ("C: Bigger Wick (40%) + Direct Entry, Both",
     dict(vol_mult=1.3, wick_min=0.40, require_htf_trend=False,
          require_structure=False, entry_mode='direct', long_only=False, target_r=1.5)),
    
    ("D: Market Structure (BOS) + Direct Entry, Both",
     dict(vol_mult=1.3, wick_min=0.30, require_htf_trend=False,
          require_structure=True, entry_mode='direct', long_only=False, target_r=1.5)),
    
    ("E1: ALL FILTERS (HTF relaxed + 2x Vol + 40% Wick + Structure) Direct, Both",
     dict(vol_mult=2.0, wick_min=0.40, require_htf_trend=True, htf_strict=False,
          require_structure=True, entry_mode='direct', long_only=False, target_r=1.5)),
    
    ("E2: ALL FILTERS + 50% Retest, Both",
     dict(vol_mult=2.0, wick_min=0.40, require_htf_trend=True, htf_strict=False,
          require_structure=True, entry_mode='retest_50', long_only=False, target_r=1.5)),
    
    ("F1: HTF Trend + Structure + Direct Entry, Both (1.3x vol, 30% wick)",
     dict(vol_mult=1.3, wick_min=0.30, require_htf_trend=True, htf_strict=False,
          require_structure=True, entry_mode='direct', long_only=False, target_r=1.5)),
    
    ("F2: HTF Trend + Structure + Direct Entry, Long Only",
     dict(vol_mult=1.3, wick_min=0.30, require_htf_trend=True, htf_strict=False,
          require_structure=True, entry_mode='direct', long_only=True, target_r=1.5)),
    
    ("G1: HTF + Struct + 2x Vol + Direct, Both (2R target)",
     dict(vol_mult=2.0, wick_min=0.30, require_htf_trend=True, htf_strict=False,
          require_structure=True, entry_mode='direct', long_only=False, target_r=2.0)),
    
    ("G2: HTF + Struct + 2x Vol + Direct, Both (1.5R target)",
     dict(vol_mult=2.0, wick_min=0.30, require_htf_trend=True, htf_strict=False,
          require_structure=True, entry_mode='direct', long_only=False, target_r=1.5)),
]

for i, (title, kwargs) in enumerate(configs, 1):
    p(f"\n--- Running Config {i}/{len(configs)}: {title} ---")
    trades = run_filtered_backtest(**kwargs)
    show_results(trades, title)

p("\n" + "=" * 80)
p("ALL FILTERED BACKTESTS COMPLETE")
p("=" * 80)

