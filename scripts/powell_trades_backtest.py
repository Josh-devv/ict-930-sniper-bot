"""
Powell Trades — Complete Multi-Model Backtest Engine
=====================================================
Implements ALL 3 of Powell's core strategies on NQ 1-min data:

  MODEL 1 — Rejection Block / Wick Theory (Top & Bottom Reversal)
  ---------------------------------------------------------------
  - Wait for price to sweep a key level (Asia hi/lo, PDH/PDL, equal H/L)
  - Identify the Rejection Block (the sweeping wick candle that CLOSES back)
  - Enter at the CE (50% midpoint of the RB wick)
  - Stop: just beyond the wick extreme (very tight: 5–15 pts)
  - Target: opposing session liquidity (3R–7R typical)
  - Window: 9:30–11:00 AM EST (Silver Bullet alignment)

  MODEL 2 — True Day Open (TDO) + CISD + Displacement
  -----------------------------------------------------
  - True Day Open = midnight (00:00 EST) candle open
  - Bias: price above TDO = bullish, below = bearish
  - Wait for AMD manipulation leg (fake move away from TDO)
  - Confirm CISD: sharp candle closes back across the TDO or structure break
  - Enter on retest of FVG/OB left behind by displacement
  - Stop: behind manipulation swing (tight)
  - Target: external liquidity (PDH, PDL, or HTF swing)

  MODEL 3 — AMD Silver Bullet (10:00–11:00 AM NY)
  -------------------------------------------------
  - Accumulation: Asia + London builds range
  - Manipulation: 9:30–10:00 AM fakes one direction (sweeps high/low)
  - Distribution: 10:00–11:00 AM Silver Bullet window is true direction
  - Entry: FVG/OB formed during displacement on 1-min chart
  - Stop: behind Manipulation swing
  - Target: DOL (opposing liquidity from the range)

All 3 models are tested independently and compared.
Results include: Win Rate, PF, Avg R:R, Max DD, FundedNext sim.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from data.loader import NQDataLoader
from config import StrategyConfig, MNQContractSpec, NQContractSpec
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PowellTrade:
    model:         str
    entry_time:    pd.Timestamp
    direction:     int          # 1 = Long, -1 = Short
    entry_price:   float
    stop_loss:     float
    take_profit:   float
    risk_pts:      float
    exit_price:    float = 0.0
    exit_reason:   str   = ''
    gross_pts:     float = 0.0
    realized_r:    float = 0.0
    net_pnl:       float = 0.0  # 1 Mini NQ at $20/pt − $10.20 commission+slippage


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading NQ 1-min data …")
loader = NQDataLoader(target_timezone='America/New_York')
df = loader.load_csv('sample_data/MNQ_1min_20260120_20260415.csv',
                     source_timezone='UTC')

# Pre-calculate Prior Day High/Low
daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
df['PDH'] = daily['High'].shift(1).reindex(df.index, method='ffill')
df['PDL'] = daily['Low'].shift(1).reindex(df.index, method='ffill')

# True Day Open = midnight (00:00) candle open each trading day
midnight_mask = (df.index.hour == 0) & (df.index.minute == 0)
midnight_opens = df.loc[midnight_mask, 'Open'].rename('TDO')
midnight_opens.index = midnight_opens.index.normalize()       # date only
df['TDO'] = df.index.normalize().map(midnight_opens)          # broadcast to all rows

dates     = pd.Series(df.index.date, index=df.index)
unique_days = dates.unique()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: simulate intrabar trade exit
# ─────────────────────────────────────────────────────────────────────────────
POINT_VALUE  = 20.0    # 1 Mini NQ
COMMISSION   = 10.20   # RT commission + slippage

def _simulate_exit(df: pd.DataFrame, entry_idx: int, direction: int,
                   entry_price: float, stop_loss: float,
                   take_profit: float, risk_pts: float,
                   be_trigger_pct: float = 0.50) -> PowellTrade:
    """Walk forward bar-by-bar from entry, applying BE and exit logic."""
    be_price    = entry_price  # start at entry
    be_triggered = False
    be_level    = entry_price + direction * risk_pts * be_trigger_pct * abs(take_profit - entry_price) / abs(take_profit - entry_price)
    # simpler: BE triggered when price moves 50% of the way from entry to TP
    tp_dist = abs(take_profit - entry_price)
    be_trigger_price = entry_price + direction * tp_dist * be_trigger_pct

    for i in range(entry_idx + 1, min(entry_idx + 400, len(df))):
        bar = df.iloc[i]
        bar_time = df.index[i]

        # Time exit at 15:45
        if bar_time.hour > 15 or (bar_time.hour == 15 and bar_time.minute >= 45):
            exit_p  = bar['Open']
            gpts    = (exit_p - entry_price) * direction
            net     = gpts * POINT_VALUE - COMMISSION
            return PowellTrade('', bar_time, direction, entry_price, stop_loss,
                               take_profit, risk_pts, exit_p, 'TIME_EXIT',
                               gpts, gpts / risk_pts if risk_pts else 0, net)

        # Check BE trigger
        if direction == 1 and not be_triggered and bar['High'] >= be_trigger_price:
            be_triggered = True
            stop_loss = entry_price + 0.25  # 1 tick above entry
        if direction == -1 and not be_triggered and bar['Low'] <= be_trigger_price:
            be_triggered = True
            stop_loss = entry_price - 0.25

        # Worst-case: check SL first
        if direction == 1:
            if bar['Low'] <= stop_loss:
                exit_p  = stop_loss
                gpts    = (exit_p - entry_price)
                reason  = 'BREAKEVEN' if be_triggered else 'STOP_LOSS'
                net     = gpts * POINT_VALUE - COMMISSION
                return PowellTrade('', bar_time, direction, entry_price, stop_loss,
                                   take_profit, risk_pts, exit_p, reason,
                                   gpts, gpts / risk_pts if risk_pts else 0, net)
            if bar['High'] >= take_profit:
                exit_p  = take_profit
                gpts    = exit_p - entry_price
                net     = gpts * POINT_VALUE - COMMISSION
                return PowellTrade('', bar_time, direction, entry_price, stop_loss,
                                   take_profit, risk_pts, exit_p, 'TARGET',
                                   gpts, gpts / risk_pts if risk_pts else 0, net)
        else:
            if bar['High'] >= stop_loss:
                exit_p  = stop_loss
                gpts    = (entry_price - exit_p)
                reason  = 'BREAKEVEN' if be_triggered else 'STOP_LOSS'
                net     = gpts * POINT_VALUE - COMMISSION
                return PowellTrade('', bar_time, direction, entry_price, stop_loss,
                                   take_profit, risk_pts, exit_p, reason,
                                   gpts, gpts / risk_pts if risk_pts else 0, net)
            if bar['Low'] <= take_profit:
                exit_p  = take_profit
                gpts    = entry_price - exit_p
                net     = gpts * POINT_VALUE - COMMISSION
                return PowellTrade('', bar_time, direction, entry_price, stop_loss,
                                   take_profit, risk_pts, exit_p, 'TARGET',
                                   gpts, gpts / risk_pts if risk_pts else 0, net)

    # No exit found — close at last bar
    exit_p  = df['Close'].iloc[-1]
    gpts    = (exit_p - entry_price) * direction
    net     = gpts * POINT_VALUE - COMMISSION
    return PowellTrade('', df.index[-1], direction, entry_price, stop_loss,
                       take_profit, risk_pts, exit_p, 'EOD',
                       gpts, gpts / risk_pts if risk_pts else 0, net)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Rejection Block / Wick Theory
# ─────────────────────────────────────────────────────────────────────────────
def run_model1_rejection_block(df: pd.DataFrame, dates, unique_days) -> List[PowellTrade]:
    """
    Catches session tops/bottoms.
    1. Mark Asia session (00:00–08:30) high/low + PDH/PDL as key liquidity
    2. In 9:30–11:00 AM window, look for a candle that:
       - Sweeps above Asia high OR PDH (buyside sweep)  → potential short
       - Sweeps below Asia low  OR PDL (sellside sweep) → potential long
       - Candle CLOSES back inside (away from extreme)  → confirmed RB
    3. Entry at CE (50% of the wick body)
    4. Stop: 2 ticks beyond the wick extreme
    5. Target: 4.0R (or opposing session level)
    """
    trades = []
    MIN_WICK_PTS  = 3.0   # wick must be at least 3 NQ pts to matter
    R_MULTIPLE    = 4.0   # target 4R (tight stop → big reward)
    BE_PCT        = 0.40  # move to BE at 40% of TP distance

    for day in unique_days:
        day_idx = np.where(dates == day)[0]
        if len(day_idx) < 30:
            continue
        day_ts  = df.index[day_idx]

        # Asia range (00:00–08:30)
        asia_mask = (day_ts.time >= pd.to_datetime('00:00').time()) & \
                    (day_ts.time <  pd.to_datetime('08:30').time())
        asia_idx  = [day_idx[j] for j, m in enumerate(asia_mask) if m]
        if len(asia_idx) < 10:
            continue
        asia_high = df['High'].iloc[asia_idx].max()
        asia_low  = df['Low'].iloc[asia_idx].min()
        pdh       = df['PDH'].iloc[day_idx[0]]
        pdl       = df['PDL'].iloc[day_idx[0]]
        key_high  = max(asia_high, pdh) if pd.notna(pdh) else asia_high
        key_low   = min(asia_low,  pdl) if pd.notna(pdl) else asia_low

        # Silver Bullet window 9:30–11:00
        sb_mask  = (day_ts.time >= pd.to_datetime('09:30').time()) & \
                   (day_ts.time <= pd.to_datetime('11:00').time())
        sb_idx   = [day_idx[j] for j, m in enumerate(sb_mask) if m]
        if len(sb_idx) < 5:
            continue

        traded_today = False
        for pos, g in enumerate(sb_idx):
            if traded_today:
                break
            bar     = df.iloc[g]
            b_high  = bar['High']
            b_low   = bar['Low']
            b_open  = bar['Open']
            b_close = bar['Close']
            b_time  = df.index[g]

            # ── SHORT SETUP: wick sweeps above key high, closes back below ──
            # Wick above key_high exists?
            wick_up   = b_high - max(b_open, b_close)   # upper wick size
            body_size = abs(b_close - b_open)
            if (b_high > key_high and                   # swept key high
                wick_up >= MIN_WICK_PTS and             # meaningful wick
                b_close < b_high - MIN_WICK_PTS * 0.5 and  # closed away from extreme
                b_close < b_open):                      # bearish close (rejection)

                # Rejection Block: body of candle (open to close)
                rb_high = max(b_open, b_close)
                rb_low  = min(b_open, b_close)
                ce      = (rb_high + rb_low) / 2        # consequential encroachment
                sl      = b_high + 0.50                 # stop beyond wick tip + buffer
                risk    = sl - ce
                if risk < 2.0 or risk > 25.0:           # guard against bad setups
                    continue
                tp      = ce - risk * R_MULTIPLE

                # Look for retracement to CE in next 15 bars
                for future in sb_idx[pos+1 : pos+16]:
                    fb = df.iloc[future]
                    if fb['High'] >= ce and fb['Open'] <= ce:
                        t = _simulate_exit(df, future, -1, ce, sl, tp, risk, BE_PCT)
                        t.model      = 'M1_RB_Short'
                        t.entry_time = df.index[future]
                        trades.append(t)
                        traded_today = True
                        break
                continue   # skip long check on same candle

            # ── LONG SETUP: wick sweeps below key low, closes back above ──
            wick_dn   = min(b_open, b_close) - b_low    # lower wick size
            if (b_low < key_low and                     # swept key low
                wick_dn >= MIN_WICK_PTS and             # meaningful wick
                b_close > b_low + MIN_WICK_PTS * 0.5 and   # closed away from extreme
                b_close > b_open):                      # bullish close (rejection)

                rb_high = max(b_open, b_close)
                rb_low  = min(b_open, b_close)
                ce      = (rb_high + rb_low) / 2
                sl      = b_low - 0.50
                risk    = ce - sl
                if risk < 2.0 or risk > 25.0:
                    continue
                tp      = ce + risk * R_MULTIPLE

                for future in sb_idx[pos+1 : pos+16]:
                    fb = df.iloc[future]
                    if fb['Low'] <= ce and fb['Open'] >= ce:
                        t = _simulate_exit(df, future, 1, ce, sl, tp, risk, BE_PCT)
                        t.model      = 'M1_RB_Long'
                        t.entry_time = df.index[future]
                        trades.append(t)
                        traded_today = True
                        break

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — True Day Open + CISD + Displacement
# ─────────────────────────────────────────────────────────────────────────────
def run_model2_tdo_cisd(df: pd.DataFrame, dates, unique_days) -> List[PowellTrade]:
    """
    1. True Day Open = midnight (00:00) open → primary bias anchor
    2. Bias: price above TDO → bullish, below → bearish
    3. 9:30–10:30 AM: watch for manipulation leg OPPOSITE to bias
       (fake move below TDO in bull bias → sells stops → CISD → long)
    4. CISD confirmation: a 1-min candle BODY closes back across TDO
    5. Entry: retest of FVG or OB left behind by the displacement candle
    6. Stop: behind manipulation swing extreme
    7. Target: 3.0R to external liquidity
    """
    trades = []
    R_MULTIPLE = 3.0
    BE_PCT     = 0.50

    for day in unique_days:
        day_idx = np.where(dates == day)[0]
        if len(day_idx) < 30:
            continue
        day_ts = df.index[day_idx]

        tdo = df['TDO'].iloc[day_idx[0]]
        if pd.isna(tdo):
            continue

        pdh = df['PDH'].iloc[day_idx[0]]
        pdl = df['PDL'].iloc[day_idx[0]]

        # Morning manipulation window 9:30 to 10:30
        manip_mask = (day_ts.time >= pd.to_datetime('09:30').time()) & \
                     (day_ts.time <= pd.to_datetime('10:30').time())
        manip_idx  = [day_idx[j] for j, m in enumerate(manip_mask) if m]
        if len(manip_idx) < 10:
            continue

        # Entry window: 10:00–11:00
        entry_mask = (day_ts.time >= pd.to_datetime('10:00').time()) & \
                     (day_ts.time <= pd.to_datetime('11:00').time())
        entry_idx  = [day_idx[j] for j, m in enumerate(entry_mask) if m]
        if not entry_idx:
            continue

        # Determine bias via TDO
        open_9_30  = df['Open'].iloc[manip_idx[0]]
        bull_bias  = open_9_30 > tdo   # price opened above TDO → bullish bias

        traded_today = False
        manip_low  = float('inf')
        manip_high = float('-inf')
        cisd_confirmed = False
        cisd_bar_idx   = -1
        fvg_low = fvg_high = 0.0

        for pos, g in enumerate(manip_idx):
            if traded_today:
                break
            bar    = df.iloc[g]
            b_time = df.index[g]

            # Track manipulation range
            manip_low  = min(manip_low,  bar['Low'])
            manip_high = max(manip_high, bar['High'])

            if not cisd_confirmed:
                # CISD: For bull bias — look for a candle that dipped below TDO
                # and then CLOSES back above TDO (body closed above = CISD bullish)
                if bull_bias:
                    if bar['Low'] < tdo and bar['Close'] > tdo:
                        # Check for FVG: gap between previous bar high and current bar low
                        if pos >= 2:
                            prev_bar  = df.iloc[manip_idx[pos - 1]]
                            prev2_bar = df.iloc[manip_idx[pos - 2]]
                            # FVG: prev2 high < current low (gap = imbalance)
                            if bar['Low'] > prev2_bar['High']:
                                fvg_low  = prev2_bar['High']
                                fvg_high = bar['Low']
                            else:
                                # Use order block (last bearish candle before this move)
                                fvg_low  = min(prev_bar['Open'], prev_bar['Close'])
                                fvg_high = max(prev_bar['Open'], prev_bar['Close'])
                        cisd_confirmed = True
                        cisd_bar_idx   = g
                        manip_swing_low = manip_low  # manipulation swept this low
                else:  # bear bias
                    if bar['High'] > tdo and bar['Close'] < tdo:
                        if pos >= 2:
                            prev_bar  = df.iloc[manip_idx[pos - 1]]
                            prev2_bar = df.iloc[manip_idx[pos - 2]]
                            if bar['High'] < prev2_bar['Low']:
                                fvg_low  = bar['High']
                                fvg_high = prev2_bar['Low']
                            else:
                                fvg_low  = min(prev_bar['Open'], prev_bar['Close'])
                                fvg_high = max(prev_bar['Open'], prev_bar['Close'])
                        cisd_confirmed  = True
                        cisd_bar_idx    = g
                        manip_swing_high = manip_high
            else:
                # After CISD — look for retest of FVG/OB
                if b_time.hour < 10:
                    continue   # only enter at 10:00+ to avoid noise
                ce = (fvg_low + fvg_high) / 2

                if bull_bias:
                    sl   = manip_low - 0.50
                    risk = ce - sl
                    if risk < 3.0 or risk > 35.0:
                        continue
                    tp = ce + risk * R_MULTIPLE
                    # Price retraces into FVG zone from above
                    if bar['Low'] <= fvg_high and bar['Open'] >= fvg_low:
                        entry_p = fvg_high  # enter at top of FVG (limit)
                        sl2     = manip_low - 0.50
                        risk2   = entry_p - sl2
                        if risk2 < 3.0 or risk2 > 35.0:
                            continue
                        tp2 = entry_p + risk2 * R_MULTIPLE
                        t = _simulate_exit(df, g, 1, entry_p, sl2, tp2, risk2, BE_PCT)
                        t.model      = 'M2_TDO_Long'
                        t.entry_time = b_time
                        trades.append(t)
                        traded_today = True
                else:
                    ce   = (fvg_low + fvg_high) / 2
                    sl   = manip_high + 0.50
                    risk = sl - ce
                    if risk < 3.0 or risk > 35.0:
                        continue
                    tp = ce - risk * R_MULTIPLE
                    # Price retraces into FVG from below
                    if bar['High'] >= fvg_low and bar['Open'] <= fvg_high:
                        entry_p = fvg_low
                        sl2     = manip_high + 0.50
                        risk2   = sl2 - entry_p
                        if risk2 < 3.0 or risk2 > 35.0:
                            continue
                        tp2 = entry_p - risk2 * R_MULTIPLE
                        t = _simulate_exit(df, g, -1, entry_p, sl2, tp2, risk2, BE_PCT)
                        t.model      = 'M2_TDO_Short'
                        t.entry_time = b_time
                        trades.append(t)
                        traded_today = True

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — AMD Silver Bullet (10:00–11:00 AM Distribution)
# ─────────────────────────────────────────────────────────────────────────────
def run_model3_amd_silver_bullet(df: pd.DataFrame, dates, unique_days) -> List[PowellTrade]:
    """
    A — Accumulation: Asia/London builds the range (00:00–09:29)
    M — Manipulation: 9:30–10:00 fakes one direction (Judas swing)
    D — Distribution: 10:00–11:00 Silver Bullet true direction + entry

    Entry trigger: first 1-min FVG in the distribution direction
    after price breaks the manipulation high/low with displacement.
    Stop: behind Manipulation swing
    Target: 3.5R to opposing liquidity
    """
    trades = []
    R_MULTIPLE = 3.5
    BE_PCT     = 0.45

    for day in unique_days:
        day_idx = np.where(dates == day)[0]
        if len(day_idx) < 30:
            continue
        day_ts = df.index[day_idx]

        # Accumulation range (00:00–09:29)
        acc_mask = (day_ts.time >= pd.to_datetime('00:00').time()) & \
                   (day_ts.time <  pd.to_datetime('09:30').time())
        acc_idx  = [day_idx[j] for j, m in enumerate(acc_mask) if m]
        if len(acc_idx) < 20:
            continue
        acc_high = df['High'].iloc[acc_idx].max()
        acc_low  = df['Low'].iloc[acc_idx].min()
        acc_mid  = (acc_high + acc_low) / 2

        # Manipulation phase (9:30–9:59)
        manip_mask = (day_ts.time >= pd.to_datetime('09:30').time()) & \
                     (day_ts.time <  pd.to_datetime('10:00').time())
        manip_idx  = [day_idx[j] for j, m in enumerate(manip_mask) if m]
        if len(manip_idx) < 5:
            continue
        manip_high = df['High'].iloc[manip_idx].max()
        manip_low  = df['Low'].iloc[manip_idx].min()

        # Determine Judas direction: whichever leg is bigger (the fake)
        up_sweep   = manip_high - acc_high   # how far above accumulation
        down_sweep = acc_low - manip_low     # how far below accumulation
        if up_sweep <= 1.0 and down_sweep <= 1.0:
            continue   # no meaningful sweep happened

        # If up_sweep > down_sweep: buyside was swept → true direction is SHORT
        # If down_sweep >= up_sweep: sellside was swept → true direction is LONG
        if up_sweep > down_sweep:
            true_direction  = -1  # short
            manip_swing     = manip_high  # SL behind this
            dol_target_area = acc_low     # target sell-side liquidity
        else:
            true_direction  = 1   # long
            manip_swing     = manip_low
            dol_target_area = acc_high    # target buy-side liquidity

        # Distribution / Silver Bullet window 10:00–11:00
        dist_mask = (day_ts.time >= pd.to_datetime('10:00').time()) & \
                    (day_ts.time <= pd.to_datetime('11:00').time())
        dist_idx  = [day_idx[j] for j, m in enumerate(dist_mask) if m]
        if len(dist_idx) < 5:
            continue

        traded_today  = False
        displacement_seen = False

        for pos, g in enumerate(dist_idx[:-1]):
            if traded_today:
                break
            if pos < 1:
                continue

            bar       = df.iloc[g]
            prev_bar  = df.iloc[dist_idx[pos - 1]]
            b_time    = df.index[g]

            # Detect displacement candle in true direction
            body = abs(bar['Close'] - bar['Open'])
            if true_direction == 1:
                is_disp = bar['Close'] > bar['Open'] and body >= 5.0 and \
                          bar['Close'] > prev_bar['High']
            else:
                is_disp = bar['Close'] < bar['Open'] and body >= 5.0 and \
                          bar['Close'] < prev_bar['Low']

            if is_disp and not displacement_seen:
                displacement_seen = True
                continue  # next bar — look for FVG entry

            if displacement_seen:
                # FVG: check if bar before prev and current form an imbalance
                if pos >= 2:
                    prev2_bar = df.iloc[dist_idx[pos - 2]]
                    if true_direction == 1:
                        # Bullish FVG: prev2 high < current low
                        if bar['Low'] > prev2_bar['High']:
                            fvg_low  = prev2_bar['High']
                            fvg_high = bar['Low']
                            # wait for price to pull back into fvg
                            for future in dist_idx[pos+1 : pos+10]:
                                fb = df.iloc[future]
                                if fb['Low'] <= fvg_high and fb['Open'] >= fvg_low:
                                    entry_p = (fvg_low + fvg_high) / 2
                                    sl      = manip_swing - 1.0
                                    risk    = entry_p - sl
                                    if risk < 3.0 or risk > 40.0:
                                        break
                                    tp = entry_p + risk * R_MULTIPLE
                                    t  = _simulate_exit(df, future, 1, entry_p, sl, tp, risk, BE_PCT)
                                    t.model      = 'M3_SB_Long'
                                    t.entry_time = df.index[future]
                                    trades.append(t)
                                    traded_today = True
                                    break
                            if traded_today:
                                break
                    else:
                        # Bearish FVG: prev2 low > current high
                        if bar['High'] < prev2_bar['Low']:
                            fvg_low  = bar['High']
                            fvg_high = prev2_bar['Low']
                            for future in dist_idx[pos+1 : pos+10]:
                                fb = df.iloc[future]
                                if fb['High'] >= fvg_low and fb['Open'] <= fvg_high:
                                    entry_p = (fvg_low + fvg_high) / 2
                                    sl      = manip_swing + 1.0
                                    risk    = sl - entry_p
                                    if risk < 3.0 or risk > 40.0:
                                        break
                                    tp = entry_p - risk * R_MULTIPLE
                                    t  = _simulate_exit(df, future, -1, entry_p, sl, tp, risk, BE_PCT)
                                    t.model      = 'M3_SB_Short'
                                    t.entry_time = df.index[future]
                                    trades.append(t)
                                    traded_today = True
                                    break
                            if traded_today:
                                break
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Run all 3 models
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning Model 1 — Rejection Block / Wick Theory …")
m1_trades = run_model1_rejection_block(df, dates, unique_days)

print("Running Model 2 — True Day Open + CISD + Displacement …")
m2_trades = run_model2_tdo_cisd(df, dates, unique_days)

print("Running Model 3 — AMD Silver Bullet (10:00–11:00 AM) …")
m3_trades = run_model3_amd_silver_bullet(df, dates, unique_days)


# ─────────────────────────────────────────────────────────────────────────────
# Stats helper
# ─────────────────────────────────────────────────────────────────────────────
def print_stats(name: str, trades: List[PowellTrade], point_val=20.0):
    if not trades:
        print(f"\n{'='*60}")
        print(f"  {name}: No trades generated")
        return

    wins   = [t for t in trades if t.exit_reason == 'TARGET']
    be     = [t for t in trades if t.exit_reason == 'BREAKEVEN']
    losses = [t for t in trades if t.exit_reason == 'STOP_LOSS']
    eod    = [t for t in trades if t.exit_reason in ('TIME_EXIT','EOD')]

    wr     = len(wins) / len(trades) * 100
    net    = sum(t.net_pnl for t in trades)
    gw     = sum(t.gross_pts * point_val for t in trades if t.gross_pts > 0)
    gl     = abs(sum(t.gross_pts * point_val for t in trades if t.gross_pts <= 0))
    pf     = gw / gl if gl > 0 else float('inf')
    avg_r  = np.mean([t.realized_r for t in trades])
    avg_rr_w = np.mean([t.realized_r for t in wins]) if wins else 0

    cum    = np.cumsum([t.net_pnl for t in trades])
    running_max = np.maximum.accumulate(cum)
    dd_series   = running_max - cum
    max_dd = dd_series.max() if len(dd_series) else 0

    avg_risk_pts  = np.mean([t.risk_pts for t in trades])
    avg_risk_usd  = avg_risk_pts * point_val
    avg_win_usd   = np.mean([t.gross_pts * point_val for t in wins]) if wins else 0
    avg_loss_usd  = np.mean([abs(t.gross_pts) * point_val for t in losses]) if losses else 0

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Total Trades     : {len(trades)}")
    print(f"  Wins (Full TP)   : {len(wins)} ({wr:.1f}%)")
    print(f"  Breakeven        : {len(be)}  ({len(be)/len(trades)*100:.1f}%)")
    print(f"  Losses (SL)      : {len(losses)}  ({len(losses)/len(trades)*100:.1f}%)")
    print(f"  Time/EOD Exits   : {len(eod)}")
    print(f"  ─────────────────────────────")
    print(f"  Profit Factor    : {pf:.2f}")
    print(f"  Avg R/trade      : {avg_r:+.2f}R")
    print(f"  Avg R (wins only): {avg_rr_w:+.2f}R")
    print(f"  Avg Risk (pts)   : {avg_risk_pts:.1f} pts")
    print(f"  Avg Risk ($)     : ${avg_risk_usd:.0f}")
    print(f"  Avg Win ($)      : +${avg_win_usd:.0f}")
    print(f"  Avg Loss ($)     : -${avg_loss_usd:.0f}")
    print(f"  Net P&L (1 NQ)   : ${net:+.2f}")
    print(f"  Max Drawdown     : ${max_dd:.2f}")
    print(f"  ─────────────────────────────")

    # FundedNext $50K simulation — 1 Mini NQ
    bal  = 50000.0
    peak = 50000.0
    mdd  = 0.0
    passed = False
    pass_t = None
    pass_date = None
    for i, t in enumerate(trades):
        bal += t.net_pnl
        if bal > peak:
            peak = bal
        dd = peak - bal
        if dd > mdd:
            mdd = dd
        if bal >= 52500.0 and not passed:
            passed   = True
            pass_t   = i + 1
            pass_date = t.entry_time.date() if hasattr(t.entry_time, 'date') else t.entry_time
        if bal <= 48500.0:  # breached DD limit
            print(f"  ⛔ FAILED: Breached $1,500 DD limit at trade #{i+1}")
            break

    print(f"  FundedNext $50K  : {'✅ PASSED' if passed else '❌ Not passed'}")
    if passed:
        print(f"  Pass trade       : #{pass_t} on {pass_date}")
        print(f"  DD at pass       : ${mdd:.2f} (limit $1,500)")
    print(f"  Final Balance    : ${bal:,.2f}")

    # Trade detail (first 15 trades)
    print(f"\n  {'#':>3}  {'Date':<22}  {'Model':<14}  {'Dir':>5}  {'Risk':>5}  {'R':>6}  {'Exit':<12}  {'PnL':>8}")
    print(f"  {'─'*85}")
    for i, t in enumerate(trades[:20]):
        d  = t.entry_time.strftime('%b %d %H:%M') if hasattr(t.entry_time,'strftime') else str(t.entry_time)
        di = '↑ LONG' if t.direction == 1 else '↓ SHORT'
        print(f"  {i+1:>3}  {d:<22}  {t.model:<14}  {di:>7}  {t.risk_pts:>4.1f}  {t.realized_r:>+5.2f}R  {t.exit_reason:<12}  {t.net_pnl:>+8.2f}")
    if len(trades) > 20:
        print(f"  ... (+{len(trades)-20} more trades)")


print_stats("MODEL 1 — Rejection Block / Wick Theory", m1_trades)
print_stats("MODEL 2 — True Day Open + CISD + Displacement", m2_trades)
print_stats("MODEL 3 — AMD Silver Bullet (10:00–11:00 AM)", m3_trades)

# ─────────────────────────────────────────────────────────────────────────────
# COMBINED MODEL — best signals from all 3
# ─────────────────────────────────────────────────────────────────────────────
# Combine only 1 trade per day (pick the one with best R:R setup)
# Group by date, keep only 1 per day with highest realized_r potential
from collections import defaultdict

all_trades_by_day = defaultdict(list)
for t in m1_trades + m2_trades + m3_trades:
    key = t.entry_time.date() if hasattr(t.entry_time, 'date') else str(t.entry_time)[:10]
    all_trades_by_day[key].append(t)

combined = []
for day_key in sorted(all_trades_by_day.keys()):
    day_trades = all_trades_by_day[day_key]
    # Prefer: M1 (tightest stop, highest R) → M3 → M2
    order = ['M1_RB_Short','M1_RB_Long','M3_SB_Long','M3_SB_Short','M2_TDO_Long','M2_TDO_Short']
    picked = None
    for preferred in order:
        for t in day_trades:
            if t.model == preferred:
                picked = t
                break
        if picked:
            break
    if not picked:
        picked = day_trades[0]
    combined.append(picked)

print_stats("COMBINED — Best Signal Per Day (All 3 Models)", combined)
