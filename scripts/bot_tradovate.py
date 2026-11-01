"""
ICT 9:30 VWAP Sniper Bot — Live Tradovate Connection
Runs on any VPS (Linux/Windows) or your local PC
Connects directly to Tradovate API — no TradersPost needed

Requirements:
  pip install aiohttp websockets pandas numpy python-dotenv

Setup:
  1. Create a .env file with your Tradovate credentials (see below)
  2. Run: python bot_tradovate.py
  3. Bot runs every trading day 09:25 AM – 04:00 PM ET automatically
"""

import asyncio
import aiohttp
import json
import os
import math
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("bot_log.txt"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("ICT_BOT")

# ── Tradovate Credentials (set in .env file) ──────────────────────────────────
# Create a file called .env in the same folder with these lines:
#   TRADOVATE_USER=your_email@example.com
#   TRADOVATE_PASS=your_password
#   TRADOVATE_APP_ID=your_app_id
#   TRADOVATE_APP_VERSION=1.0
#   TRADOVATE_DEVICE_ID=your_device_id
#   TRADOVATE_CID=your_cid
#   TRADOVATE_SECRET=your_secret
#   LIVE_MODE=false        ← set to true only when ready to go live

LIVE_MODE    = os.getenv("LIVE_MODE", "false").lower() == "true"
BASE_URL     = "https://live.tradovate.com/v1" if LIVE_MODE else "https://demo.tradovateapi.com/v1"
WS_URL       = "wss://live.tradovate.com/v1/websocket" if LIVE_MODE else "wss://demo.tradovateapi.com/v1/websocket"

ET = ZoneInfo("America/New_York")

# ── Strategy Parameters ────────────────────────────────────────────────────────
SYMBOL       = "NQZ4"       # Update to current front-month NQ contract
VWAP_MULT    = 2.0
MIN_BAND_PT  = 15.0
WICK_PCT     = 0.35
MIN_STOP     = 5.0
MAX_STOP     = 70.0
MIN_RR       = 1.5
STOP_BUF     = 2.0
CONTRACTS    = 1
POINT_VALUE  = 20.0

ENTRY_START  = dtime(9, 30)
ENTRY_END    = dtime(11, 30)
EOD_TIME     = dtime(15, 45)

# ── VWAP State ────────────────────────────────────────────────────────────────
class VWAPState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cum_vol  = 0.0
        self.cum_pv   = 0.0
        self.cum_pv2  = 0.0
        self.vwap     = None
        self.upper    = None
        self.lower    = None
        self.stdev    = 0.0
        log.info("VWAP state reset for new session")

    def update(self, o, h, l, c, v):
        tp = (h + l + c) / 3.0
        v  = max(v, 1.0)
        self.cum_vol += v
        self.cum_pv  += tp * v
        self.cum_pv2 += tp * tp * v
        self.vwap    = self.cum_pv / self.cum_vol
        var_         = max(0.0, self.cum_pv2 / self.cum_vol - self.vwap**2)
        self.stdev   = math.sqrt(var_)
        self.upper   = self.vwap + VWAP_MULT * self.stdev
        self.lower   = self.vwap - VWAP_MULT * self.stdev

# ── Tradovate API Client ───────────────────────────────────────────────────────
class TradovateBot:
    def __init__(self):
        self.session      = None
        self.access_token = None
        self.account_id   = None
        self.vwap         = VWAPState()
        self.traded_today = False
        self.in_trade     = False
        self.last_date    = None
        self.candle_buf   = {}   # partial 3-min candle accumulator

    # ── Authentication ────────────────────────────────────────────────────────
    async def authenticate(self):
        payload = {
            "name":       os.getenv("TRADOVATE_USER"),
            "password":   os.getenv("TRADOVATE_PASS"),
            "appId":      os.getenv("TRADOVATE_APP_ID"),
            "appVersion": os.getenv("TRADOVATE_APP_VERSION", "1.0"),
            "deviceId":   os.getenv("TRADOVATE_DEVICE_ID"),
            "cid":        int(os.getenv("TRADOVATE_CID", "0")),
            "sec":        os.getenv("TRADOVATE_SECRET"),
        }
        async with self.session.post(f"{BASE_URL}/auth/accesstokenrequest", json=payload) as r:
            data = await r.json()
            if "accessToken" not in data:
                raise Exception(f"Auth failed: {data}")
            self.access_token = data["accessToken"]
            log.info(f"Authenticated OK — {'LIVE' if LIVE_MODE else 'DEMO'} mode")

        # Get account ID
        headers = {"Authorization": f"Bearer {self.access_token}"}
        async with self.session.get(f"{BASE_URL}/account/list", headers=headers) as r:
            accounts = await r.json()
            self.account_id = accounts[0]["id"]
            log.info(f"Account ID: {self.account_id}  Name: {accounts[0]['name']}")

    # ── Place Bracket Order ───────────────────────────────────────────────────
    async def place_bracket(self, side: str, entry: float, stop: float, target: float):
        action = "Buy" if side == "long" else "Sell"
        opp    = "Sell" if side == "long" else "Buy"
        headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}

        order = {
            "accountId": self.account_id,
            "action":    action,
            "symbol":    SYMBOL,
            "orderQty":  CONTRACTS,
            "orderType": "Limit",
            "price":     round(entry, 2),
            "isAutomated": True,
            "bracket1": {
                "action":    opp,
                "orderType": "Stop",
                "stopPrice": round(stop, 2),
                "orderQty":  CONTRACTS
            },
            "bracket2": {
                "action":    opp,
                "orderType": "Limit",
                "price":     round(target, 2),
                "orderQty":  CONTRACTS
            }
        }

        async with self.session.post(f"{BASE_URL}/order/placeorder", json=order, headers=headers) as r:
            result = await r.json()
            log.info(f"ORDER PLACED: {side.upper()} | Entry={entry:.2f} | SL={stop:.2f} | TP={target:.2f}")
            log.info(f"Order response: {result}")
            return result

    # ── Close All Positions (EOD) ─────────────────────────────────────────────
    async def close_all(self):
        headers = {"Authorization": f"Bearer {self.access_token}"}
        async with self.session.post(f"{BASE_URL}/order/liquidateposition",
                                     json={"accountId": self.account_id, "isAutomated": True},
                                     headers=headers) as r:
            result = await r.json()
            log.info(f"EOD FLAT: {result}")

    # ── Process a completed 3-min candle ─────────────────────────────────────
    async def on_candle(self, o, h, l, c, v, ts: datetime):
        now_et = ts.astimezone(ET)
        now_t  = now_et.time()
        today  = now_et.date()

        # Reset daily state
        if today != self.last_date:
            self.last_date    = today
            self.traded_today = False
            self.in_trade     = False
            self.vwap.reset()

        # Update VWAP from 09:30 onwards
        if now_t >= ENTRY_START:
            self.vwap.update(o, h, l, c, v)

        # EOD flat
        if now_t >= EOD_TIME and self.in_trade:
            log.info("EOD — closing position")
            await self.close_all()
            self.in_trade = False
            return

        # Skip if already traded or VWAP not warmed up
        if self.traded_today or self.in_trade or self.vwap.vwap is None or self.vwap.stdev < 0.5:
            return
        if not (ENTRY_START <= now_t <= ENTRY_END):
            return

        vwap   = self.vwap.vwap
        upper  = self.vwap.upper
        lower  = self.vwap.lower
        crange = h - l
        up_w   = h - max(o, c)
        dn_w   = min(o, c) - l

        # ── SHORT SIGNAL ──────────────────────────────────────────────────────
        if ((upper - vwap) >= MIN_BAND_PT and h >= upper and crange >= 6.0
                and up_w >= crange * WICK_PCT and c < o):

            entry_p  = (h + max(o, c)) / 2.0
            stop_p   = h + STOP_BUF
            risk_p   = min(max(stop_p - entry_p, 1.0), MAX_STOP)
            target_p = vwap
            reward_p = entry_p - target_p
            rr       = reward_p / risk_p if risk_p > 0 else 0

            if risk_p >= MIN_STOP and reward_p >= risk_p * MIN_RR:
                log.info(f"SHORT SIGNAL | Entry={entry_p:.2f} SL={stop_p:.2f} TP={target_p:.2f} RR={rr:.1f}R")
                await self.place_bracket("short", entry_p, stop_p, target_p)
                self.traded_today = True
                self.in_trade     = True

        # ── LONG SIGNAL ───────────────────────────────────────────────────────
        elif ((vwap - lower) >= MIN_BAND_PT and l <= lower and crange >= 6.0
                and dn_w >= crange * WICK_PCT and c > o):

            entry_p  = (l + min(o, c)) / 2.0
            stop_p   = l - STOP_BUF
            risk_p   = min(max(entry_p - stop_p, 1.0), MAX_STOP)
            target_p = vwap
            reward_p = target_p - entry_p
            rr       = reward_p / risk_p if risk_p > 0 else 0

            if risk_p >= MIN_STOP and reward_p >= risk_p * MIN_RR:
                log.info(f"LONG SIGNAL  | Entry={entry_p:.2f} SL={stop_p:.2f} TP={target_p:.2f} RR={rr:.1f}R")
                await self.place_bracket("long", entry_p, stop_p, target_p)
                self.traded_today = True
                self.in_trade     = True

    # ── WebSocket Market Data Feed ────────────────────────────────────────────
    async def run_market_feed(self):
        import websockets
        log.info(f"Connecting to Tradovate WebSocket: {WS_URL}")

        tick_buf = []   # accumulate ticks into 3-min candles
        candle_open_min = None

        async with websockets.connect(WS_URL) as ws:
            # Auth handshake
            await ws.send(json.dumps({
                "op": "authorize",
                "token": self.access_token
            }))
            await ws.recv()

            # Subscribe to quote feed for NQ
            await ws.send(json.dumps({
                "op": "subscribe",
                "args": [f"md/subscribequote/{SYMBOL}"]
            }))
            log.info(f"Subscribed to {SYMBOL} quote feed")

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    if msg.get("e") != "quote":
                        continue

                    price = msg["d"].get("tradePrice") or msg["d"].get("bidPrice")
                    vol   = msg["d"].get("tradeSize", 1)
                    ts    = datetime.fromisoformat(msg["d"]["timestamp"].replace("Z", "+00:00"))

                    tick_buf.append({"price": price, "vol": vol, "ts": ts})

                    # Build 3-min candles from ticks
                    minute_slot = (ts.minute // 3) * 3
                    if candle_open_min is None:
                        candle_open_min = minute_slot

                    if minute_slot != candle_open_min and len(tick_buf) > 1:
                        # Close the previous candle
                        prices = [t["price"] for t in tick_buf[:-1]]
                        vols   = [t["vol"]   for t in tick_buf[:-1]]
                        o = prices[0]; h = max(prices); l = min(prices); c = prices[-1]
                        v = sum(vols)
                        candle_ts = tick_buf[0]["ts"]
                        await self.on_candle(o, h, l, c, v, candle_ts)
                        tick_buf = [tick_buf[-1]]
                        candle_open_min = minute_slot

                except Exception as e:
                    log.error(f"Feed error: {e}")

    # ── Main Run Loop ──────────────────────────────────────────────────────────
    async def run(self):
        async with aiohttp.ClientSession() as sess:
            self.session = sess
            await self.authenticate()

            while True:
                now_et = datetime.now(ET)
                now_t  = now_et.time()

                # Only run during market hours (09:25 – 16:05 ET)
                if dtime(9, 25) <= now_t <= dtime(16, 5):
                    log.info("Market hours — starting feed...")
                    try:
                        await self.run_market_feed()
                    except Exception as e:
                        log.error(f"Feed disconnected: {e} — reconnecting in 10s")
                        await asyncio.sleep(10)
                else:
                    # Sleep until next 09:25 ET
                    log.info(f"Outside market hours ({now_t}) — sleeping 5 min")
                    await asyncio.sleep(300)


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("  ICT 9:30 VWAP Sniper Bot — Starting")
    log.info(f"  Mode: {'*** LIVE TRADING ***' if LIVE_MODE else 'DEMO / PAPER'}")
    log.info(f"  Symbol: {SYMBOL}  |  Contracts: {CONTRACTS}")
    log.info("=" * 55)
    asyncio.run(TradovateBot().run())

