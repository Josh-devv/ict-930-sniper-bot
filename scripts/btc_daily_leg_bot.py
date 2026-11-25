"""
BTC 15m Daily-Leg Precision — Telegram Alert Bot
=================================================
VP6 + SMC/Pullback + Delta + Daily Open — FINAL-F

This is a SEPARATE bot from the existing BTC Orderflow Absorption bot.
Tag: [BTC 15m DAILY LEG]
=================================================
"""

import os
import json
import time
import requests
import logging
from datetime import datetime, timezone, timedelta

# ========== CONFIG ==========
TELEGRAM_BOT_TOKEN = "8290114978:AAETL_UeHqbu8r4n2jHvDXxW-VKuEnvyp0c"
TELEGRAM_CHAT_ID   = "5642644592"

SYMBOL   = "BTCUSDT"
INTERVAL = "15m"
TAG      = "[BTC 15m DAILY LEG]"

# Session: London open to NY close (07-18 UTC)
SESSION_START_UTC = 7
SESSION_END_UTC   = 18
WEEKDAYS_ONLY     = True
MAX_TRADES_DAY    = 2

# EMAs
EMA20_LEN  = 20
EMA50_LEN  = 50

# Volume Profile
VP_LOOKBACK = 96   # 24 hours of 15m bars
VP_BINS     = 50
VP_PROX_ATR = 0.8  # proximity in ATR multiples

# Orderflow
DELTA_THRESHOLD = 0.1

# Risk/Exit
RR1          = 1.5
RR2          = 4.0
ATR_SL_PAD   = 0.4
MIN_RISK_PTS = 100.0

# SMC
SMC_LOOKBACK   = 18
MIN_BODY_RATIO = 0.45
DIST_CAP       = 8.0

# Polling
POLL_INTERVAL = 60  # seconds

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("BTC_DailyLeg")


# ========== BINANCE KLINES ==========
def fetch_klines(symbol, interval, limit=200):
    """Fetch klines from Binance public API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        klines = []
        for k in data:
            klines.append({
                "time":   int(k[0]),
                "open":   float(k[1]),
                "high":   float(k[2]),
                "low":    float(k[3]),
                "close":  float(k[4]),
                "volume": float(k[5]),
            })
        return klines
    except Exception as e:
        log.error(f"Kline fetch error: {e}")
        return []


def fetch_1h_klines(symbol, limit=200):
    """Fetch 1H klines for HTF EMA."""
    return fetch_klines(symbol, "1h", limit)


# ========== INDICATORS ==========
def ema(values, length):
    """Compute EMA."""
    out = [values[0]]
    k = 2.0 / (length + 1)
    for i in range(1, len(values)):
        out.append(values[i] * k + out[-1] * (1 - k))
    return out


def atr(highs, lows, closes, length=14):
    """Compute ATR."""
    trs = [highs[0] - lows[0]]
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    # SMA of TR
    out = []
    for i in range(len(trs)):
        window = trs[max(0, i - length + 1):i + 1]
        out.append(sum(window) / len(window))
    return out


def compute_vwap_session(klines):
    """Compute session VWAP (resets daily)."""
    vwap = []
    cum_pv, cum_vol = 0.0, 0.0
    prev_day = None
    for k in klines:
        dt = datetime.fromtimestamp(k["time"] / 1000, tz=timezone.utc)
        day = dt.date()
        if day != prev_day:
            cum_pv, cum_vol = 0.0, 0.0
            prev_day = day
        tp = (k["high"] + k["low"] + k["close"]) / 3.0
        cum_pv += tp * k["volume"]
        cum_vol += k["volume"]
        vwap.append(cum_pv / cum_vol if cum_vol > 0 else k["close"])
    return vwap


def compute_poc(klines, lookback=96, num_bins=50):
    """Compute rolling POC."""
    poc = [None] * len(klines)
    for i in range(lookback, len(klines)):
        window = klines[i - lookback:i]
        p_min = min(k["low"] for k in window)
        p_max = max(k["high"] for k in window)
        p_range = p_max - p_min
        if p_range < 50:
            continue
        bin_size = p_range / num_bins
        bin_vols = [0.0] * num_bins
        for k in window:
            tp = (k["high"] + k["low"] + k["close"]) / 3.0
            bi = max(0, min(num_bins - 1, int((tp - p_min) / bin_size)))
            bin_vols[bi] += k["volume"]
        max_bin = max(range(num_bins), key=lambda x: bin_vols[x])
        poc[i] = p_min + (max_bin + 0.5) * bin_size
    return poc


def compute_daily_open(klines):
    """Compute daily open price."""
    opens = [None] * len(klines)
    prev_day = None
    cur_open = None
    for i, k in enumerate(klines):
        dt = datetime.fromtimestamp(k["time"] / 1000, tz=timezone.utc)
        day = dt.date()
        if day != prev_day:
            cur_open = k["open"]
            prev_day = day
        opens[i] = cur_open
    return opens


# ========== SMC SIGNAL ==========
def check_signal(idx, klines, ema20_arr, ema50_arr, htf_ema, atr_arr):
    """Check for BUY/SELL signal at index idx."""
    if idx < SMC_LOOKBACK + 3:
        return None

    c_val = klines[idx]["close"]
    o_val = klines[idx]["open"]
    h_val = klines[idx]["high"]
    l_val = klines[idx]["low"]
    cur_atr = atr_arr[idx]

    if htf_ema is None or cur_atr == 0:
        return None

    # Distance check
    if abs(c_val - htf_ema) / cur_atr > DIST_CAP:
        return None

    body = abs(c_val - o_val)
    rng = max(1.0, h_val - l_val)

    # Sweep detection (last 4 bars vs prior 15)
    recent_lows  = [klines[j]["low"]  for j in range(idx - 3, idx + 1)]
    recent_highs = [klines[j]["high"] for j in range(idx - 3, idx + 1)]
    prior_lows   = [klines[j]["low"]  for j in range(idx - SMC_LOOKBACK, idx - 3)]
    prior_highs  = [klines[j]["high"] for j in range(idx - SMC_LOOKBACK, idx - 3)]

    swept_low  = min(recent_lows) < min(prior_lows)
    swept_high = max(recent_highs) > max(prior_highs)

    # MSS/FVG proxies
    prev_highs_3 = [klines[j]["high"] for j in range(idx - 3, idx)]
    prev_lows_3  = [klines[j]["low"]  for j in range(idx - 3, idx)]

    bull_smc = (swept_low and
                (l_val > klines[idx - 2]["high"] or c_val > max(prev_highs_3)) and
                c_val > o_val and c_val > htf_ema)
    bear_smc = (swept_high and
                (h_val < klines[idx - 2]["low"] or c_val < min(prev_lows_3)) and
                c_val < o_val and c_val < htf_ema)

    # Pullback
    bull_pb = (c_val > htf_ema and ema20_arr[idx] > ema50_arr[idx] and
               l_val <= ema20_arr[idx] and c_val > ema20_arr[idx] and
               c_val > o_val and body / rng >= MIN_BODY_RATIO and
               c_val > klines[idx - 1]["high"])
    bear_pb = (c_val < htf_ema and ema20_arr[idx] < ema50_arr[idx] and
               h_val >= ema20_arr[idx] and c_val < ema20_arr[idx] and
               c_val < o_val and body / rng >= MIN_BODY_RATIO and
               c_val < klines[idx - 1]["low"])

    if bull_smc or bull_pb:
        stop = min(recent_lows) - ATR_SL_PAD * cur_atr
        return ("BUY", c_val, stop)
    elif bear_smc or bear_pb:
        stop = max(recent_highs) + ATR_SL_PAD * cur_atr
        return ("SELL", c_val, stop)

    return None


# ========== TELEGRAM ==========
def send_telegram(msg):
    """Send message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            log.info("Telegram sent OK")
        else:
            log.error(f"Telegram error {r.status_code}: {r.text}")
    except Exception as e:
        log.error(f"Telegram send failed: {e}")


def format_alert(signal, entry, stop, tp1, tp2, risk):
    """Format alert message."""
    direction_emoji = "🟢" if signal == "BUY" else "🔴"
    rr_tp1 = RR1
    rr_tp2 = RR2

    msg = (
        f"{TAG}\n"
        f"{direction_emoji} <b>{signal} BTCUSDT</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📍 Entry: <code>{entry:,.1f}</code>\n"
        f"🛑 Stop: <code>{stop:,.1f}</code>\n"
        f"🎯 TP1 ({rr_tp1}R): <code>{tp1:,.1f}</code>\n"
        f"🏆 TP2 ({rr_tp2}R): <code>{tp2:,.1f}</code>\n"
        f"📏 Risk: <code>{risk:,.1f}</code> pts\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"⏱ TF: 15m | VP6+Delta+DO\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )
    return msg


# ========== 1H HTF EMA ==========
def get_htf_ema(klines_1h, target_time_ms):
    """Get the latest closed 1H EMA50 value before target_time_ms."""
    closes = [k["close"] for k in klines_1h]
    if len(closes) < 50:
        return None
    ema_vals = ema(closes, 50)
    # Find the last 1H bar that closed before this 15m bar
    for i in range(len(klines_1h) - 1, -1, -1):
        # 1H bar close time is open_time + 3600000
        bar_close_ms = klines_1h[i]["time"] + 3_600_000
        if bar_close_ms <= target_time_ms:
            return ema_vals[i]
    return ema_vals[-2] if len(ema_vals) >= 2 else None


# ========== MAIN LOOP ==========
def is_active_session():
    """Check if we're in an active trading session."""
    now = datetime.now(timezone.utc)
    if WEEKDAYS_ONLY and now.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    if not (SESSION_START_UTC <= now.hour <= SESSION_END_UTC):
        return False
    return True


def main():
    log.info(f"Starting {TAG} Bot")
    log.info(f"Session: {SESSION_START_UTC}-{SESSION_END_UTC} UTC, Weekdays: {WEEKDAYS_ONLY}")
    log.info(f"Max trades/day: {MAX_TRADES_DAY}")
    log.info(f"Filters: VP6 (POC/VWAP {VP_PROX_ATR}xATR) + Delta ({DELTA_THRESHOLD}) + Daily Open")
    log.info(f"Exit: {RR1}R TP1 → BE → {RR2}R TP2")

    send_telegram(
        f"{TAG}\n"
        f"🤖 <b>Bot Started</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Session: {SESSION_START_UTC}:00-{SESSION_END_UTC}:00 UTC\n"
        f"Filters: VP6 + Delta + Daily Open\n"
        f"Exit: {RR1}R/{RR2}R | Max {MAX_TRADES_DAY}/day\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Status: ✅ Running"
    )

    last_signal_bar = 0
    trades_today = 0
    last_date = None

    while True:
        try:
            if not is_active_session():
                log.debug("Outside session, sleeping...")
                time.sleep(POLL_INTERVAL)
                continue

            # Reset daily trade counter
            today = datetime.now(timezone.utc).date()
            if today != last_date:
                trades_today = 0
                last_date = today
                log.info(f"New trading day: {today}")

            if trades_today >= MAX_TRADES_DAY:
                log.debug(f"Max trades ({MAX_TRADES_DAY}) reached for today")
                time.sleep(POLL_INTERVAL)
                continue

            # Fetch data
            klines = fetch_klines(SYMBOL, INTERVAL, 200)
            klines_1h = fetch_1h_klines(SYMBOL, 200)

            if len(klines) < VP_LOOKBACK + 20 or len(klines_1h) < 60:
                log.warning("Not enough kline data")
                time.sleep(POLL_INTERVAL)
                continue

            # Compute indicators
            closes = [k["close"] for k in klines]
            highs  = [k["high"]  for k in klines]
            lows   = [k["low"]   for k in klines]

            ema20_arr = ema(closes, EMA20_LEN)
            ema50_arr = ema(closes, EMA50_LEN)
            atr_arr   = atr(highs, lows, closes, 14)
            vwap_arr  = compute_vwap_session(klines)
            poc_arr   = compute_poc(klines, VP_LOOKBACK, VP_BINS)
            do_arr    = compute_daily_open(klines)

            # Check the LAST CLOSED bar (index -2, since -1 is still forming)
            idx = len(klines) - 2
            bar_time_ms = klines[idx]["time"]

            # Avoid duplicate signals
            if bar_time_ms <= last_signal_bar:
                time.sleep(POLL_INTERVAL)
                continue

            # HTF EMA
            htf_ema_val = get_htf_ema(klines_1h, bar_time_ms)

            # Check SMC signal
            sig_result = check_signal(idx, klines, ema20_arr, ema50_arr, htf_ema_val, atr_arr)
            if sig_result is None:
                time.sleep(POLL_INTERVAL)
                continue

            sig, entry, stop = sig_result
            risk = abs(entry - stop)

            if risk < MIN_RISK_PTS:
                time.sleep(POLL_INTERVAL)
                continue

            # VP Filter: near POC or VWAP
            cur_atr = atr_arr[idx]
            poc_val = poc_arr[idx]
            vwap_val = vwap_arr[idx]
            near_poc  = poc_val is not None and abs(entry - poc_val) <= VP_PROX_ATR * cur_atr
            near_vwap = abs(entry - vwap_val) <= VP_PROX_ATR * cur_atr
            if not (near_poc or near_vwap):
                log.debug(f"VP filter rejected: POC={poc_val}, VWAP={vwap_val:.1f}, entry={entry:.1f}")
                time.sleep(POLL_INTERVAL)
                continue

            # Delta filter
            bar_range = klines[idx]["high"] - klines[idx]["low"]
            if bar_range > 0:
                delta_val = 2.0 * (klines[idx]["close"] - klines[idx]["low"]) / bar_range - 1.0
            else:
                delta_val = 0.0

            if sig == "BUY" and delta_val < DELTA_THRESHOLD:
                log.debug(f"Delta filter rejected BUY: delta={delta_val:.3f}")
                time.sleep(POLL_INTERVAL)
                continue
            if sig == "SELL" and delta_val > -DELTA_THRESHOLD:
                log.debug(f"Delta filter rejected SELL: delta={delta_val:.3f}")
                time.sleep(POLL_INTERVAL)
                continue

            # Daily Open filter
            day_open = do_arr[idx]
            if day_open is not None:
                if sig == "BUY" and entry < day_open:
                    log.debug(f"Daily Open filter rejected BUY: entry={entry:.1f} < DO={day_open:.1f}")
                    time.sleep(POLL_INTERVAL)
                    continue
                if sig == "SELL" and entry > day_open:
                    log.debug(f"Daily Open filter rejected SELL: entry={entry:.1f} > DO={day_open:.1f}")
                    time.sleep(POLL_INTERVAL)
                    continue

            # ===== SIGNAL PASSED ALL FILTERS =====
            if sig == "BUY":
                tp1 = entry + RR1 * risk
                tp2 = entry + RR2 * risk
            else:
                tp1 = entry - RR1 * risk
                tp2 = entry - RR2 * risk

            last_signal_bar = bar_time_ms
            trades_today += 1

            bar_dt = datetime.fromtimestamp(bar_time_ms / 1000, tz=timezone.utc)
            log.info(f"SIGNAL: {sig} @ {entry:.1f} | SL {stop:.1f} | TP1 {tp1:.1f} | TP2 {tp2:.1f} | Risk {risk:.1f} | Trade #{trades_today}")

            msg = format_alert(sig, entry, stop, tp1, tp2, risk)
            send_telegram(msg)

        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
