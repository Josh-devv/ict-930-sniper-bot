"""
NQ 9:30 VWAP Sniper — MTF Telegram Alert Bot v3.0
===================================================
5-Minute VWAP Bands (the ZONE) + 3-Minute Candle (the TRIGGER)

Alert Stages:
  Stage 1 — 09:25-09:29 NY: Heads-up with 5m VWAP band levels
  Stage 2 — 09:30-09:32 NY: LIVE TOUCH — 3m candle just swept the 5m band
  Stage 3 — 09:33-09:36 NY: CONFIRMED — 3m candle closed with wick rejection
                             (Entry, Stop, TP1=VWAP, TP2=Opposite Band)
  Stage 4 — 09:39, 09:42 NY: Catch follow-on 3m signals inside the window

WHY 3m + 5m:
  The 5m VWAP bands define the institutional zone.
  A 3m candle confirms the rejection 2 minutes sooner — tighter stops, earlier entry.

Data Source: Yahoo Finance NQ=F (1m + 3m feeds, free, no API key needed)
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ─── Credentials ──────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8290114978:AAETL_UeHqbu8r4n2jHvDXxW-VKuEnvyp0c")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "5642644592")

# ─── Timezone ─────────────────────────────────────────────────────────────────
NY_TZ    = ZoneInfo("America/New_York")
LAGOS_TZ = ZoneInfo("Africa/Lagos")

# ─── Strategy Parameters (Sealed — do not change) ─────────────────────────────
SIGMA       = 2.0    # VWAP band deviation multiplier
MIN_BAND_PT = 15.0   # Minimum VWAP-to-band distance in points (use 12.0 for USTEC CFD)
WICK_MIN    = 0.35   # Minimum rejection wick ratio (35% of candle range)
STOP_BUFFER = 1.5    # Points beyond the wick extreme for stop loss
MIN_STOP    = 4.0    # Minimum stop distance in points
MAX_STOP    = 60.0   # Maximum stop distance in points
MIN_RR      = 1.5    # Minimum room-to-run to VWAP (R:R filter)
ADAPT_THRESH = 20.0  # 20-pt rule: candle range <= 20 → market; > 20 → limit
MIN_CRANGE  = 5.0    # Minimum 3m candle range to qualify

# ─── State (prevents duplicate alerts per session) ────────────────────────────
_stage2_sent = False   # live touch warning
_stage3_sent = False   # confirmed signal


def send_telegram(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        ok = r.status_code == 200
        if not ok:
            print(f"Telegram error {r.status_code}: {r.text[:200]}")
        return ok
    except Exception as e:
        print(f"Telegram send failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_yahoo(interval: str, range_str: str) -> pd.DataFrame | None:
    """Fetch NQ futures OHLCV from Yahoo Finance."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/NQ=F?interval={interval}&range={range_str}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NQBot/3.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            print(f"Yahoo Finance returned {r.status_code}")
            return None
        data = r.json()
        chart  = data["chart"]["result"][0]
        ohlcv  = chart["indicators"]["quote"][0]
        df = pd.DataFrame({
            "ts":     pd.to_datetime(chart["timestamp"], unit="s", utc=True),
            "open":   ohlcv["open"],
            "high":   ohlcv["high"],
            "low":    ohlcv["low"],
            "close":  ohlcv["close"],
            "volume": ohlcv["volume"],
        }).dropna()
        df["ny_time"] = df["ts"].dt.tz_convert(NY_TZ)
        return df
    except Exception as e:
        print(f"Error fetching {interval} data: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5-MINUTE SESSION VWAP (Anchored at 9:30 AM NY)
# This defines the ZONE. Bands are computed on 5m bars only.
# ─────────────────────────────────────────────────────────────────────────────
def compute_5m_vwap(df_1m: pd.DataFrame):
    """
    Resample 1m data to 5m, then compute cumulative session VWAP anchored at 9:30 AM.
    Returns the 5m DataFrame with vwap, upper_b, lower_b columns.
    """
    today = datetime.now(NY_TZ).date()
    df = df_1m[df_1m["ny_time"].dt.date == today].copy()
    if df.empty:
        df = df_1m.copy()

    # Resample 1m OHLCV → 5m bars
    df = df.set_index("ny_time")
    df_5m = df.resample("5min", closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    df_5m = df_5m.reset_index()
    df_5m = df_5m.rename(columns={"ny_time": "bar_time"})

    # Anchor at 9:30 AM — only compute VWAP from that point forward
    anchor = df_5m[
        (df_5m["bar_time"].dt.hour == 9) &
        (df_5m["bar_time"].dt.minute == 30)
    ].index
    if len(anchor) == 0:
        # Try any bar >= 9:30 today
        anchor = df_5m[df_5m["bar_time"].dt.hour >= 9].index
        if len(anchor) == 0:
            return None
    start_idx = anchor[0]
    df_5m = df_5m.loc[start_idx:].copy()

    tp  = (df_5m["high"] + df_5m["low"] + df_5m["close"]) / 3.0
    vol = df_5m["volume"].replace(0, 1.0)
    cum_vol  = vol.cumsum()
    cum_pv   = (tp * vol).cumsum()
    cum_pv2  = (tp * tp * vol).cumsum()

    vwap     = cum_pv / cum_vol
    variance = np.maximum(0.0, cum_pv2 / cum_vol - vwap ** 2)
    stdev    = np.sqrt(variance)

    df_5m["vwap"]    = vwap.values
    df_5m["upper_b"] = (vwap + SIGMA * stdev).values
    df_5m["lower_b"] = (vwap - SIGMA * stdev).values
    return df_5m


# ─────────────────────────────────────────────────────────────────────────────
# 3-MINUTE CANDLE TRIGGER CHECK
# Checks if any recent 3m candle swept a 5m band and closed with rejection.
# ─────────────────────────────────────────────────────────────────────────────
def check_3m_signal(df_1m: pd.DataFrame, vwap_val: float, lower_b: float, upper_b: float):
    """
    Resample 1m → 3m, find the most recent completed 3m candle that swept
    either the lower or upper 5m band and closed with a wick rejection.

    Returns a dict with signal details or None if no signal.
    """
    today = datetime.now(NY_TZ).date()
    df = df_1m[df_1m["ny_time"].dt.date == today].copy()
    if df.empty:
        return None

    df = df.set_index("ny_time")
    df_3m = df.resample("3min", closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna().reset_index().rename(columns={"ny_time": "bar_time"})

    # Only check bars from 9:30 onwards
    df_3m = df_3m[df_3m["bar_time"].dt.hour >= 9].copy()
    df_3m = df_3m[
        (df_3m["bar_time"].dt.hour > 9) |
        (df_3m["bar_time"].dt.minute >= 30)
    ]

    # Look at the last few completed 3m bars (not the current live bar)
    if len(df_3m) < 2:
        return None

    # Check the most recent completed bar (index -2 is last confirmed closed bar)
    # Also check -3 to catch signals we might have missed in the last run
    for i in [-2, -3]:
        if abs(i) > len(df_3m):
            continue
        c = df_3m.iloc[i]

        crange    = c["high"] - c["low"]
        down_wick = min(c["open"], c["close"]) - c["low"]   # tail below body
        up_wick   = c["high"] - max(c["open"], c["close"])  # tail above body

        if crange < MIN_CRANGE:
            continue

        band_width_lower = vwap_val - lower_b
        band_width_upper = upper_b  - vwap_val

        # ── LONG SETUP ───────────────────────────────────────────────────────
        if (band_width_lower >= MIN_BAND_PT and
                c["low"]   <= lower_b and          # swept below the band
                c["close"] >  lower_b and          # closed back above band
                c["close"] >  c["open"] and        # bullish candle
                down_wick  >= crange * WICK_MIN):  # strong rejection tail

            # 20-Point Adaptive Entry Rule
            use_market = crange <= ADAPT_THRESH
            entry = c["close"] if use_market else (c["low"] + 0.50 * down_wick)
            stop  = c["low"] - STOP_BUFFER
            risk  = entry - stop

            if MIN_STOP <= risk <= MAX_STOP and (vwap_val - entry) >= risk * MIN_RR:
                return {
                    "direction":   "LONG",
                    "bar_time":    c["bar_time"],
                    "entry":       entry,
                    "stop":        stop,
                    "tp1":         vwap_val,
                    "tp2":         upper_b,
                    "risk_pts":    risk,
                    "wick_ratio":  down_wick / crange,
                    "crange":      crange,
                    "entry_type":  "BUY MARKET" if use_market else "BUY LIMIT (50% Wick)",
                    "tp1_pts":     vwap_val - entry,
                    "tp2_pts":     upper_b  - entry,
                }

        # ── SHORT SETUP ──────────────────────────────────────────────────────
        if (band_width_upper >= MIN_BAND_PT and
                c["high"]  >= upper_b and          # swept above the band
                c["close"] <  upper_b and          # closed back below band
                c["close"] <  c["open"] and        # bearish candle
                up_wick    >= crange * WICK_MIN):  # strong rejection tail

            use_market = crange <= ADAPT_THRESH
            entry = c["close"] if use_market else (c["high"] - 0.50 * up_wick)
            stop  = c["high"] + STOP_BUFFER
            risk  = stop - entry

            if MIN_STOP <= risk <= MAX_STOP and (entry - vwap_val) >= risk * MIN_RR:
                return {
                    "direction":   "SHORT",
                    "bar_time":    c["bar_time"],
                    "entry":       entry,
                    "stop":        stop,
                    "tp1":         vwap_val,
                    "tp2":         lower_b,
                    "risk_pts":    risk,
                    "wick_ratio":  up_wick / crange,
                    "crange":      crange,
                    "entry_type":  "SELL MARKET" if use_market else "SELL LIMIT (50% Wick)",
                    "tp1_pts":     entry - vwap_val,
                    "tp2_pts":     entry - lower_b,
                }

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def check_nq_now():
    global _stage2_sent, _stage3_sent

    now_ny  = datetime.now(NY_TZ)
    now_wat = datetime.now(LAGOS_TZ)
    hour, minute = now_ny.hour, now_ny.minute

    # Only run on weekdays
    if now_ny.weekday() >= 5:
        print("Weekend — markets closed.")
        return

    time_str = now_ny.strftime("%H:%M NY") + " / " + now_wat.strftime("%H:%M WAT")

    # ── Fetch data ────────────────────────────────────────────────────────────
    df_1m = _fetch_yahoo("1m", "1d")
    if df_1m is None or len(df_1m) < 5:
        print("Could not fetch NQ 1m data.")
        return

    # Reset state flags at start of each new day
    today = now_ny.date()
    if not hasattr(check_nq_now, "_last_date") or check_nq_now._last_date != today:
        check_nq_now._last_date = today
        _stage2_sent = False
        _stage3_sent = False

    # ── Compute 5m VWAP zone ─────────────────────────────────────────────────
    df_5m = compute_5m_vwap(df_1m)
    if df_5m is None or len(df_5m) == 0:
        print("Could not compute 5m VWAP.")
        return

    latest  = df_5m.iloc[-1]
    vwap_v  = float(latest["vwap"])
    upper_v = float(latest["upper_b"])
    lower_v = float(latest["lower_b"])
    band_w  = vwap_v - lower_v

    cur_price = float(df_1m.iloc[-1]["close"])
    print(f"[{time_str}] NQ: {cur_price:,.2f} | VWAP: {vwap_v:,.2f} | "
          f"Lower: {lower_v:,.2f} | Upper: {upper_v:,.2f} | Band: {band_w:.1f} pts")

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 1 — 09:25-09:29 NY: Pre-session Heads-Up
    # ═════════════════════════════════════════════════════════════════════════
    if hour == 9 and 25 <= minute <= 29:
        opens_in = 30 - minute
        msg = (
            f"⏰ <b>NQ 9:30 OPEN IN {opens_in} MINUTE{'S' if opens_in != 1 else ''}!</b>\n\n"
            f"<b>Time:</b> {time_str}\n"
            f"{'─' * 38}\n"
            f"📍 <b>Current NQ:</b> <code>{cur_price:,.2f}</code>\n"
            f"🎯 <b>5m Session VWAP:</b> <code>{vwap_v:,.2f}</code>\n"
            f"🟢 <b>Lower Band (–2.0σ):</b> <code>{lower_v:,.2f}</code>  ← BUY ZONE\n"
            f"🔴 <b>Upper Band (+2.0σ):</b> <code>{upper_v:,.2f}</code>  ← SELL ZONE\n"
            f"📏 <b>Band Width:</b> <code>{band_w:.1f} pts</code>\n"
            f"{'─' * 38}\n"
            f"👀 <i>Wait for price to touch either band at the bell.\n"
            f"Entry confirmed on the <b>3-minute candle close</b> with wick rejection.</i>"
        )
        print("→ Sending Stage 1 Heads-Up")
        send_telegram(msg)
        return

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 2 — 09:30-09:35 NY: LIVE TOUCH WARNING (3m candle sweeping the band)
    # ═════════════════════════════════════════════════════════════════════════
    if hour == 9 and 30 <= minute <= 35 and not _stage2_sent:
        recent_lows  = df_1m["low"].iloc[-6:].min()   # last ~5-6 1m bars
        recent_highs = df_1m["high"].iloc[-6:].max()

        touched_lower = recent_lows  <= lower_v and band_w >= MIN_BAND_PT
        touched_upper = recent_highs >= upper_v and (upper_v - vwap_v) >= MIN_BAND_PT

        if touched_lower or touched_upper:
            side = "LOWER (BUY)" if touched_lower else "UPPER (SELL)"
            extreme = f"{recent_lows:,.2f}" if touched_lower else f"{recent_highs:,.2f}"
            band_hit = f"{lower_v:,.2f}" if touched_lower else f"{upper_v:,.2f}"
            action = "BUY" if touched_lower else "SELL"
            emoji = "🟢" if touched_lower else "🔴"

            msg = (
                f"⚡ <b>NQ LIVE TOUCH — {side} BAND SWEPT!</b>\n\n"
                f"<b>Time:</b> {time_str}\n"
                f"{'─' * 38}\n"
                f"{emoji} <b>Band Hit:</b> <code>{band_hit}</code>\n"
                f"📍 <b>Price Extreme:</b> <code>{extreme}</code>\n"
                f"🎯 <b>VWAP Target:</b> <code>{vwap_v:,.2f}</code>\n"
                f"{'─' * 38}\n"
                f"⚠️ <b>Absorption forming RIGHT NOW!</b>\n"
                f"Watch the <b>3m candle</b> that is currently building.\n\n"
                f"<b>When it closes:</b>\n"
                f"  • Bullish wick ≥ 35% → <b>{action} SIGNAL INCOMING</b>\n"
                f"  • 3m range ≤ 20 pts → <b>MARKET at close</b>\n"
                f"  • 3m range > 20 pts → <b>LIMIT at 50% wick</b>\n\n"
                f"<i>Confirmed alert fires on 3m candle close.</i>"
            )
            print("→ Sending Stage 2 Live Touch Warning")
            send_telegram(msg)
            _stage2_sent = True
        return

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 3 — 09:33-09:45 NY: CONFIRMED 3m SIGNAL
    # Checks every 3 minutes (the bot runs every 5m via cron, so we catch it
    # on the first run after the 3m candle closes at :33, :36, :39, :42, :45)
    # ═════════════════════════════════════════════════════════════════════════
    if hour == 9 and 33 <= minute <= 59 and not _stage3_sent:
        sig = check_3m_signal(df_1m, vwap_v, lower_v, upper_v)
        if sig:
            # Dollar sizing: 5 MNQ, $2/pt per contract
            # TP1: scale out 3 MNQ; TP2: runner with 2 MNQ
            risk_usd = sig["risk_pts"] * 5 * 2         # max risk (all 5)
            tp1_usd  = sig["tp1_pts"]  * 3 * 2         # 3 MNQ hit TP1
            tp2_usd  = sig["tp2_pts"]  * 2 * 2         # 2 MNQ runner

            d       = sig["direction"]
            arrow   = "🟢 LONG" if d == "LONG" else "🔴 SHORT"
            emoji2  = "🚀" if d == "LONG" else "🎯"
            sig_time = sig["bar_time"].strftime("%H:%M") + " NY"

            msg = (
                f"{emoji2} <b>NQ 3m SNIPER — {arrow} CONFIRMED!</b>\n\n"
                f"<b>Time:</b> {time_str}\n"
                f"<b>Signal Bar:</b> {sig_time} 3m candle\n"
                f"{'─' * 38}\n"
                f"🔹 <b>{sig['entry_type']}:</b> <code>{sig['entry']:,.2f}</code>\n"
                f"🔻 <b>STOP LOSS:</b> <code>{sig['stop']:,.2f}</code>  "
                f"(-{sig['risk_pts']:.1f} pts / -<b>${risk_usd:.0f}</b>)\n"
                f"{'─' * 38}\n"
                f"🏆 <b>TP1 → VWAP (scale 3 MNQ, then BE+1):</b>\n"
                f"   <code>{sig['tp1']:,.2f}</code>  "
                f"(+{sig['tp1_pts']:.1f} pts / +<b>${tp1_usd:.0f}</b>)\n\n"
                f"🎯 <b>TP2 → Opposite Band (runner 2 MNQ):</b>\n"
                f"   <code>{sig['tp2']:,.2f}</code>  "
                f"(+{sig['tp2_pts']:.1f} pts / +<b>${tp2_usd:.0f}</b>)\n"
                f"{'─' * 38}\n"
                f"📊 <b>Rejection Wick:</b> {sig['wick_ratio']*100:.0f}%  |  "
                f"<b>Range:</b> {sig['crange']:.1f} pts\n"
                f"🎯 <b>VWAP:</b> <code>{vwap_v:,.2f}</code>\n"
                f"{'─' * 38}\n"
                f"<i>5 MNQ = $10/pt total. Scale 60% at TP1, let 40% run to TP2.</i>"
            )
            print(f"→ Sending Stage 3 Confirmed {d} Signal")
            send_telegram(msg)
            _stage3_sent = True
        else:
            print(f"[{time_str}] No qualified 3m signal yet.")
        return

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 4 — 10:00-11:30 NY: Extended window follow-on signals
    # Same 3m trigger logic — catches re-tests and second entries in the window
    # ═════════════════════════════════════════════════════════════════════════
    if (hour == 10 or (hour == 11 and minute <= 30)) and not _stage3_sent:
        sig = check_3m_signal(df_1m, vwap_v, lower_v, upper_v)
        if sig:
            risk_usd = sig["risk_pts"] * 5 * 2
            tp1_usd  = sig["tp1_pts"]  * 3 * 2
            tp2_usd  = sig["tp2_pts"]  * 2 * 2

            d       = sig["direction"]
            arrow   = "🟢 LONG" if d == "LONG" else "🔴 SHORT"
            emoji2  = "🚀" if d == "LONG" else "🎯"
            sig_time = sig["bar_time"].strftime("%H:%M") + " NY"

            msg = (
                f"{emoji2} <b>NQ 3m FOLLOW-ON — {arrow} SIGNAL!</b>\n\n"
                f"<b>Time:</b> {time_str}\n"
                f"<b>Signal Bar:</b> {sig_time} 3m candle\n"
                f"{'─' * 38}\n"
                f"🔹 <b>{sig['entry_type']}:</b> <code>{sig['entry']:,.2f}</code>\n"
                f"🔻 <b>STOP LOSS:</b> <code>{sig['stop']:,.2f}</code>  "
                f"(-{sig['risk_pts']:.1f} pts / -<b>${risk_usd:.0f}</b>)\n"
                f"{'─' * 38}\n"
                f"🏆 <b>TP1 → VWAP:</b> <code>{sig['tp1']:,.2f}</code>  "
                f"(+{sig['tp1_pts']:.1f} pts / +<b>${tp1_usd:.0f}</b>)\n"
                f"🎯 <b>TP2 → Opp. Band:</b> <code>{sig['tp2']:,.2f}</code>  "
                f"(+{sig['tp2_pts']:.1f} pts / +<b>${tp2_usd:.0f}</b>)\n"
                f"{'─' * 38}\n"
                f"📊 Wick: {sig['wick_ratio']*100:.0f}%  |  Range: {sig['crange']:.1f} pts\n"
                f"<i>Follow-on 3m entry — 1 trade/day rule still applies.</i>"
            )
            print(f"→ Sending Stage 4 Follow-On {d} Signal")
            send_telegram(msg)
            _stage3_sent = True

    print(f"[{time_str}] Outside active alert window or already signalled.")


if __name__ == "__main__":
    check_nq_now()
