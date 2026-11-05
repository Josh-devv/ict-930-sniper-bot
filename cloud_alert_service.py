"""
24/7 Cloud Trading Alert Service & Webhook Bridge
=================================================
Runs continuously on Render, Railway, or VPS.
Monitors:
  1. NQ 9:30 MTF Sniper (5m VWAP Zone + 3m Trigger)
  2. BTC 5m Orderflow Absorption (OKX with Yahoo fallback)
  3. TradingView Webhook Bridge (/webhook endpoint)

Free 24/7 Cloud Architecture:
  - Binds to $PORT for cloud healthchecks
  - Responds to GET / with live status (keeps Render awake with UptimeRobot)
  - Accepts POST /webhook from TradingView to relay instant alerts
"""

import os
import json
import time
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from http.server import HTTPServer, BaseHTTPRequestHandler

# ─── Configuration ────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8290114978:AAETL_UeHqbu8r4n2jHvDXxW-VKuEnvyp0c")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "5642644592")
PORT               = int(os.getenv("PORT", "10000"))

NY_TZ    = ZoneInfo("America/New_York")
LAGOS_TZ = ZoneInfo("Africa/Lagos")

# Strategy Constants
NQ_SIGMA        = 2.0
NQ_MIN_BAND_PT  = 15.0
NQ_WICK_MIN     = 0.35
NQ_STOP_BUFFER  = 1.5
NQ_MIN_STOP     = 4.0
NQ_MAX_STOP     = 60.0
NQ_MIN_RR       = 1.5
NQ_ADAPT_THRESH = 20.0
NQ_MIN_CRANGE   = 5.0

BTC_VOL_MULT    = 1.3
BTC_WICK_MIN    = 0.30
BTC_ATR_MULT    = 1.0

# State Tracking
state = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "last_nq_check": None,
    "last_btc_check": None,
    "last_nq_price": None,
    "last_btc_price": None,
    "stage1_sent_date": None,
    "stage2_sent_date": None,
    "stage3_sent_date": None,
    "last_btc_alert_time": None,
    "webhooks_received": 0
}


def send_telegram(text: str) -> bool:
    """Send HTML message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=12)
        if r.status_code != 200:
            print(f"[Telegram Error {r.status_code}]: {r.text[:200]}")
        return r.status_code == 200
    except Exception as e:
        print(f"[Telegram Exception]: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# NQ 9:30 MTF SNIPER ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_nq_1m() -> pd.DataFrame | None:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/NQ=F?interval=1m&range=1d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NQBot/3.0"}
    try:
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
        chart = data["chart"]["result"][0]
        ohlcv = chart["indicators"]["quote"][0]
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
        print(f"[NQ Data Error]: {e}")
        return None


def compute_5m_vwap(df_1m: pd.DataFrame):
    today = datetime.now(NY_TZ).date()
    df = df_1m[df_1m["ny_time"].dt.date == today].copy()
    if df.empty:
        df = df_1m.copy()

    df = df.set_index("ny_time")
    df_5m = df.resample("5min", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index().rename(columns={"ny_time": "bar_time"})

    anchor = df_5m[(df_5m["bar_time"].dt.hour == 9) & (df_5m["bar_time"].dt.minute == 30)].index
    if len(anchor) == 0:
        anchor = df_5m[df_5m["bar_time"].dt.hour >= 9].index
        if len(anchor) == 0:
            return None
    start_idx = anchor[0]
    df_5m = df_5m.loc[start_idx:].copy()

    tp  = (df_5m["high"] + df_5m["low"] + df_5m["close"]) / 3.0
    vol = df_5m["volume"].replace(0, 1.0)
    cum_vol = vol.cumsum()
    cum_pv  = (tp * vol).cumsum()
    cum_pv2 = (tp * tp * vol).cumsum()

    vwap     = cum_pv / cum_vol
    variance = np.maximum(0.0, cum_pv2 / cum_vol - vwap ** 2)
    stdev    = np.sqrt(variance)

    df_5m["vwap"]    = vwap.values
    df_5m["upper_b"] = (vwap + NQ_SIGMA * stdev).values
    df_5m["lower_b"] = (vwap - NQ_SIGMA * stdev).values
    return df_5m


def check_3m_signal(df_1m: pd.DataFrame, vwap_val: float, lower_b: float, upper_b: float):
    today = datetime.now(NY_TZ).date()
    df = df_1m[df_1m["ny_time"].dt.date == today].copy()
    if df.empty:
        return None

    df = df.set_index("ny_time")
    df_3m = df.resample("3min", closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index().rename(columns={"ny_time": "bar_time"})

    df_3m = df_3m[df_3m["bar_time"].dt.hour >= 9].copy()
    df_3m = df_3m[(df_3m["bar_time"].dt.hour > 9) | (df_3m["bar_time"].dt.minute >= 30)]

    if len(df_3m) < 2:
        return None

    for i in [-2, -3]:
        if abs(i) > len(df_3m):
            continue
        c = df_3m.iloc[i]
        crange    = c["high"] - c["low"]
        down_wick = min(c["open"], c["close"]) - c["low"]
        up_wick   = c["high"] - max(c["open"], c["close"])

        if crange < NQ_MIN_CRANGE:
            continue

        band_w_lower = vwap_val - lower_b
        band_w_upper = upper_b - vwap_val

        # Long Setup
        if (band_w_lower >= NQ_MIN_BAND_PT and
                c["low"] <= lower_b and c["close"] > lower_b and
                c["close"] > c["open"] and down_wick >= crange * NQ_WICK_MIN):
            use_market = crange <= NQ_ADAPT_THRESH
            entry = c["close"] if use_market else (c["low"] + 0.50 * down_wick)
            stop  = c["low"] - NQ_STOP_BUFFER
            risk  = entry - stop
            if NQ_MIN_STOP <= risk <= NQ_MAX_STOP and (vwap_val - entry) >= risk * NQ_MIN_RR:
                return {
                    "direction": "LONG", "bar_time": c["bar_time"],
                    "entry": entry, "stop": stop, "tp1": vwap_val, "tp2": upper_b,
                    "risk_pts": risk, "wick_ratio": down_wick / crange, "crange": crange,
                    "entry_type": "BUY MARKET" if use_market else "BUY LIMIT (50% Wick)",
                    "tp1_pts": vwap_val - entry, "tp2_pts": upper_b - entry
                }

        # Short Setup
        if (band_w_upper >= NQ_MIN_BAND_PT and
                c["high"] >= upper_b and c["close"] < upper_b and
                c["close"] < c["open"] and up_wick >= crange * NQ_WICK_MIN):
            use_market = crange <= NQ_ADAPT_THRESH
            entry = c["close"] if use_market else (c["high"] - 0.50 * up_wick)
            stop  = c["high"] + NQ_STOP_BUFFER
            risk  = stop - entry
            if NQ_MIN_STOP <= risk <= NQ_MAX_STOP and (entry - vwap_val) >= risk * NQ_MIN_RR:
                return {
                    "direction": "SHORT", "bar_time": c["bar_time"],
                    "entry": entry, "stop": stop, "tp1": vwap_val, "tp2": lower_b,
                    "risk_pts": risk, "wick_ratio": up_wick / crange, "crange": crange,
                    "entry_type": "SELL MARKET" if use_market else "SELL LIMIT (50% Wick)",
                    "tp1_pts": entry - vwap_val, "tp2_pts": entry - lower_b
                }

    return None


def run_nq_check():
    now_ny  = datetime.now(NY_TZ)
    now_wat = datetime.now(LAGOS_TZ)
    hour, minute = now_ny.hour, now_ny.minute
    today = now_ny.date()

    if now_ny.weekday() >= 5:
        return

    # Active window: 09:20 - 11:35 NY
    if hour < 9 or (hour == 9 and minute < 20) or hour > 11 or (hour == 11 and minute > 35):
        return

    time_str = now_ny.strftime("%H:%M NY") + " / " + now_wat.strftime("%H:%M WAT")

    df_1m = fetch_nq_1m()
    if df_1m is None or len(df_1m) < 5:
        return

    cur_price = float(df_1m.iloc[-1]["close"])
    state["last_nq_price"] = cur_price
    state["last_nq_check"] = time_str

    df_5m = compute_5m_vwap(df_1m)
    if df_5m is None or len(df_5m) == 0:
        return

    latest  = df_5m.iloc[-1]
    vwap_v  = float(latest["vwap"])
    upper_v = float(latest["upper_b"])
    lower_v = float(latest["lower_b"])
    band_w  = vwap_v - lower_v

    # STAGE 1: 09:25 - 09:29 Heads-Up
    if hour == 9 and 25 <= minute <= 29:
        if state["stage1_sent_date"] != today:
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
                f"👀 <i>Wait for price to sweep either band at the bell.\n"
                f"Entry confirmed on <b>3m candle close</b> with wick rejection.</i>"
            )
            if send_telegram(msg):
                state["stage1_sent_date"] = today
                print(f"[NQ Stage 1 Sent] {time_str}")

    # STAGE 2: 09:30 - 09:35 Live Touch
    if hour == 9 and 30 <= minute <= 35:
        if state["stage2_sent_date"] != today:
            recent_lows  = df_1m["low"].iloc[-6:].min()
            recent_highs = df_1m["high"].iloc[-6:].max()
            touched_lower = recent_lows <= lower_v and band_w >= NQ_MIN_BAND_PT
            touched_upper = recent_highs >= upper_v and (upper_v - vwap_v) >= NQ_MIN_BAND_PT

            if touched_lower or touched_upper:
                side   = "LOWER (BUY)" if touched_lower else "UPPER (SELL)"
                extreme = f"{recent_lows:,.2f}" if touched_lower else f"{recent_highs:,.2f}"
                band_hit = f"{lower_v:,.2f}" if touched_lower else f"{upper_v:,.2f}"
                action = "BUY" if touched_lower else "SELL"
                emoji  = "🟢" if touched_lower else "🔴"

                msg = (
                    f"⚡ <b>NQ LIVE TOUCH — {side} BAND SWEPT!</b>\n\n"
                    f"<b>Time:</b> {time_str}\n"
                    f"{'─' * 38}\n"
                    f"{emoji} <b>Band Hit:</b> <code>{band_hit}</code>\n"
                    f"📍 <b>Price Extreme:</b> <code>{extreme}</code>\n"
                    f"🎯 <b>VWAP Target:</b> <code>{vwap_v:,.2f}</code>\n"
                    f"{'─' * 38}\n"
                    f"⚠️ <b>Absorption forming RIGHT NOW!</b>\n"
                    f"Watch the 3m candle currently building.\n"
                    f"• Wick ≥ 35% → {action} incoming\n"
                    f"• Range ≤ 20 pts → Market at close\n"
                    f"• Range > 20 pts → 50% Wick Limit"
                )
                if send_telegram(msg):
                    state["stage2_sent_date"] = today
                    print(f"[NQ Stage 2 Sent] {time_str}")

    # STAGE 3 & 4: Confirmed Signal
    if (hour == 9 and minute >= 33) or hour == 10 or (hour == 11 and minute <= 30):
        if state["stage3_sent_date"] != today:
            sig = check_3m_signal(df_1m, vwap_v, lower_v, upper_v)
            if sig:
                risk_usd = sig["risk_pts"] * 5 * 2
                tp1_usd  = sig["tp1_pts"] * 3 * 2
                tp2_usd  = sig["tp2_pts"] * 2 * 2
                d        = sig["direction"]
                arrow    = "🟢 LONG" if d == "LONG" else "🔴 SHORT"
                emoji2   = "🚀" if d == "LONG" else "🎯"
                sig_time = sig["bar_time"].strftime("%H:%M") + " NY"

                msg = (
                    f"{emoji2} <b>NQ 3m SNIPER — {arrow} CONFIRMED!</b>\n\n"
                    f"<b>Time:</b> {time_str}\n"
                    f"<b>Signal Bar:</b> {sig_time} 3m candle\n"
                    f"{'─' * 38}\n"
                    f"🔹 <b>{sig['entry_type']}:</b> <code>{sig['entry']:,.2f}</code>\n"
                    f"🔻 <b>STOP LOSS:</b> <code>{sig['stop']:,.2f}</code> "
                    f"(-{sig['risk_pts']:.1f} pts / -<b>${risk_usd:.0f}</b>)\n"
                    f"{'─' * 38}\n"
                    f"🏆 <b>TP1 → VWAP (scale 3 MNQ, then BE+1):</b>\n"
                    f"   <code>{sig['tp1']:,.2f}</code> (+{sig['tp1_pts']:.1f} pts / +<b>${tp1_usd:.0f}</b>)\n\n"
                    f"🎯 <b>TP2 → Opposite Band (runner 2 MNQ):</b>\n"
                    f"   <code>{sig['tp2']:,.2f}</code> (+{sig['tp2_pts']:.1f} pts / +<b>${tp2_usd:.0f}</b>)\n"
                    f"{'─' * 38}\n"
                    f"📊 <b>Wick:</b> {sig['wick_ratio']*100:.0f}% | <b>Range:</b> {sig['crange']:.1f} pts\n"
                    f"🎯 <b>VWAP:</b> <code>{vwap_v:,.2f}</code>\n"
                    f"{'─' * 38}\n"
                    f"<i>5 MNQ = $10/pt. Move SL to BE+1 immediately after TP1.</i>"
                )
                if send_telegram(msg):
                    state["stage3_sent_date"] = today
                    print(f"[NQ Stage 3 Sent] {time_str}")


# ─────────────────────────────────────────────────────────────────────────────
# BTC 5m ORDERFLOW ABSORPTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def is_btc_active_session():
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    in_london = (7 <= hour <= 11)
    in_ny     = (13 <= hour <= 18)
    return in_london or in_ny


def fetch_btc_5m() -> pd.DataFrame | None:
    # 1. Try OKX
    try:
        url = "https://www.okx.com/api/v5/market/candles?instId=BTC-USDT&bar=5m&limit=40"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            raw = list(reversed(r.json().get("data", [])))
            df = pd.DataFrame(raw, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'quote_volume', 'taker_volume', 'confirm'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = df[col].astype(float)
            return df
    except Exception:
        pass

    # 2. Seamless Fallback: Yahoo Finance BTC-USD
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD?interval=5m&range=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            chart = r.json()["chart"]["result"][0]
            ohlcv = chart["indicators"]["quote"][0]
            df = pd.DataFrame({
                "open_time": chart["timestamp"],
                "open": ohlcv["open"],
                "high": ohlcv["high"],
                "low": ohlcv["low"],
                "close": ohlcv["close"],
                "volume": ohlcv["volume"],
                "quote_volume": [o * v for o, v in zip(ohlcv["open"], ohlcv["volume"])]
            }).dropna()
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = df[col].astype(float)
            return df
    except Exception as e:
        print(f"[BTC Fallback Error]: {e}")

    return None


def run_btc_check():
    if not is_btc_active_session():
        return

    now_utc = datetime.now(timezone.utc)
    # Check near bar close (minute % 5 == 0 at 5-45s)
    if now_utc.minute % 5 != 0 and now_utc.minute % 5 != 4:
        return

    df = fetch_btc_5m()
    if df is None or len(df) < 25:
        return

    closed_candle = df.iloc[-2]
    candle_time = int(closed_candle['open_time'])

    if state["last_btc_alert_time"] == candle_time:
        return

    state["last_btc_price"] = closed_candle['close']
    state["last_btc_check"] = now_utc.strftime("%H:%M UTC")

    vol_series = df['quote_volume'].iloc[:-1]
    vol_ma = vol_series.rolling(20).mean().iloc[-1]
    vol_climax = (closed_candle['quote_volume'] >= vol_ma * BTC_VOL_MULT)

    local_low = df['low'].iloc[-22:-2].min()
    low_swept = (closed_candle['low'] <= local_low)

    c_range = max(1.0, closed_candle['high'] - closed_candle['low'])
    body_bot = min(closed_candle['open'], closed_candle['close'])
    lower_wick = body_bot - closed_candle['low']
    wick_ratio = lower_wick / c_range

    tr = (df['high'] - df['low']).iloc[:-1]
    atr = tr.rolling(14).mean().iloc[-1]

    if vol_climax and low_swept and (wick_ratio >= BTC_WICK_MIN):
        entry_price = closed_candle['low'] + 0.50 * lower_wick
        stop_price  = closed_candle['low'] - BTC_ATR_MULT * atr
        stop_dist   = entry_price - stop_price

        if stop_dist >= 40.0:
            tp1_price = entry_price + 1.5 * stop_dist
            tp2_price = entry_price + 3.0 * stop_dist

            now_ny = datetime.now(timezone.utc).astimezone(NY_TZ)
            now_str = now_ny.strftime("%Y-%m-%d %H:%M NY")

            alert_text = (
                f"🚀 <b>BTC ORDERFLOW ABSORPTION BUY!</b>\n\n"
                f"<b>Time (NY):</b> {now_str}\n"
                f"<b>Chart:</b> BTC-USD · 5m\n"
                f"{'─' * 38}\n"
                f"🔹 <b>BUY LIMIT (50% Wick):</b> <code>${entry_price:,.2f}</code>\n"
                f"🔻 <b>STOP LOSS:</b> <code>${stop_price:,.2f}</code> (Risk: ${stop_dist:,.2f})\n"
                f"🟢 <b>TP1 (60% Close + BE):</b> <code>${tp1_price:,.2f}</code> (+1.5R)\n"
                f"🎯 <b>TP2 (40% Runner):</b> <code>${tp2_price:,.2f}</code> (+3.0R)\n"
                f"{'─' * 38}\n"
                f"💡 <i>Place Buy Limit order on Exness MT5 now.</i>"
            )
            if send_telegram(alert_text):
                state["last_btc_alert_time"] = candle_time
                print(f"[BTC Absorption Alert Sent] {now_str}")


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND MONITOR LOOP
# ─────────────────────────────────────────────────────────────────────────────
def background_loop():
    print(">> Background Alert Monitor Loop Started.")
    # Send startup confirmation to Telegram
    startup_msg = (
        f"🟢 <b>Trading Alerts Cloud Service Online!</b>\n\n"
        f"✅ <b>NQ 9:30 MTF Sniper:</b> ACTIVE\n"
        f"✅ <b>BTC 5m Absorption:</b> ACTIVE\n"
        f"✅ <b>Webhook Bridge:</b> ACTIVE on port {PORT}\n\n"
        f"<i>Monitoring institutional absorption 24/7 with zero lag.</i>"
    )
    send_telegram(startup_msg)

    while True:
        try:
            run_nq_check()
        except Exception as e:
            print(f"[NQ Check Error]: {e}")

        try:
            run_btc_check()
        except Exception as e:
            print(f"[BTC Check Error]: {e}")

        time.sleep(15)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP WEB SERVER & TRADINGVIEW WEBHOOK BRIDGE
# ─────────────────────────────────────────────────────────────────────────────
class WebhookHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Healthcheck endpoint for UptimeRobot / Render."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        res = {
            "status": "healthy",
            "service": "Trading Alerts Cloud Worker",
            "state": state
        }
        self.wfile.write(json.dumps(res, indent=2).encode("utf-8"))

    def do_POST(self):
        """Direct Webhook endpoint for TradingView or brokers."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""

        state["webhooks_received"] += 1
        print(f"[Webhook Received]: {body[:200]}")

        # Send alert payload directly to Telegram
        if body:
            # Format cleanly if it's plain text or JSON
            alert_msg = f"🔔 <b>TRADINGVIEW WEBHOOK ALERT:</b>\n\n{body}"
            send_telegram(alert_msg)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "delivered"}')

    def log_message(self, format, *args):
        # Silence normal HTTP polling log noise
        return


def run_server():
    server = HTTPServer(("0.0.0.0", PORT), WebhookHandler)
    print(f">> Webhook & Healthcheck Server listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    t = threading.Thread(target=background_loop, daemon=True)
    t.start()
    run_server()
