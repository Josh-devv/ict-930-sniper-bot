"""
ICT 9:30 VWAP Sniper Bot — Professional Prop Firm Edition
Built for Tradovate (FundedNext / TopStep / MyFundedFutures compatible)

Features:
  - 09:30 AM Cash Open Anchored VWAP ± 2.0σ Bands
  - 50% Wick Consequential Encroachment (CE) Limit Entry
  - Sniper Stop Loss + Mean-Reversion Target to VWAP
  - Built-in Prop Firm Risk Guardian (Daily loss & Max Drawdown killswitch)
  - 100% Automated Bracket Orders (Entry, SL, TP placed simultaneously)
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

# ── Directory & Logging Setup ────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/bot_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("ICT_BOT")

# ── Configuration ────────────────────────────────────────────────────────────
LIVE_MODE   = os.getenv("LIVE_MODE", "false").lower() == "true"
SYMBOL      = os.getenv("NQ_SYMBOL", "NQZ4")
CONTRACTS   = int(os.getenv("CONTRACTS", "1"))
BASE_URL    = "https://live.tradovate.com/v1" if LIVE_MODE else "https://demo.tradovateapi.com/v1"
WS_URL      = "wss://live.tradovate.com/v1/websocket" if LIVE_MODE else "wss://demo.tradovateapi.com/v1/websocket"
ET          = ZoneInfo("America/New_York")

# ── Strategy Risk Settings ───────────────────────────────────────────────────
VWAP_MULT   = 2.0
MIN_BAND_PT = 15.0
WICK_PCT    = 0.35
MIN_STOP    = 5.0
MAX_STOP    = 70.0
MIN_RR      = 1.5
STOP_BUF    = 2.0
CANDLE_MINS = 3       # 3-minute execution
ENTRY_START = dtime(9, 30)
ENTRY_END   = dtime(11, 30)
EOD_TIME    = dtime(15, 45)

# ── Prop Firm Safety Guardian ($50k Challenge Parameters) ────────────────────
MAX_DAILY_LOSS_USD   = float(os.getenv("MAX_DAILY_LOSS_USD", "800.0"))    # Hard kill below $1,000 daily limit
MAX_TOTAL_DRAWDOWN   = float(os.getenv("MAX_TOTAL_DRAWDOWN", "1600.0"))   # Hard kill below $2,000 trailing limit

# ── VWAP Calculator ──────────────────────────────────────────────────────────
class VWAP:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cum_vol = 0.0
        self.cum_pv  = 0.0
        self.cum_pv2 = 0.0
        self.value   = None
        self.upper   = None
        self.lower   = None
        self.stdev   = 0.0

    def update(self, h, l, c, v):
        tp = (h + l + c) / 3.0
        v  = max(v, 1.0)
        self.cum_vol += v
        self.cum_pv  += tp * v
        self.cum_pv2 += tp * tp * v
        self.value    = self.cum_pv / self.cum_vol
        var_          = max(0.0, self.cum_pv2 / self.cum_vol - self.value**2)
        self.stdev    = math.sqrt(var_)
        self.upper    = self.value + VWAP_MULT * self.stdev
        self.lower    = self.value - VWAP_MULT * self.stdev

    @property
    def ready(self):
        return self.value is not None and self.stdev >= 0.5

# ── Sniper Bot Engine ────────────────────────────────────────────────────────
class SniperBot:
    def __init__(self):
        self.session       = None
        self.token         = None
        self.account_id    = None
        self.account_name  = None
        self.starting_bal  = None
        self.peak_bal      = None
        self.daily_pnl     = 0.0
        
        self.vwap          = VWAP()
        self.traded_today  = False
        self.in_trade      = False
        self.last_date     = None
        self.tick_buf      = []
        self.candle_slot   = None
        self.kill_switch   = False

    async def auth(self):
        payload = {
            "name":        os.getenv("TRADOVATE_USER"),
            "password":    os.getenv("TRADOVATE_PASS"),
            "appId":       os.getenv("TRADOVATE_APP_ID", "ICT_Sniper_Bot"),
            "appVersion":  os.getenv("TRADOVATE_APP_VERSION", "1.0"),
            "deviceId":    os.getenv("TRADOVATE_DEVICE_ID"),
            "cid":         int(os.getenv("TRADOVATE_CID", "0")),
            "sec":         os.getenv("TRADOVATE_SECRET"),
        }
        async with self.session.post(f"{BASE_URL}/auth/accesstokenrequest", json=payload) as r:
            data = await r.json()
            if "accessToken" not in data:
                raise Exception(f"Auth failed: {data}")
            self.token = data["accessToken"]

        headers = {"Authorization": f"Bearer {self.token}"}
        async with self.session.get(f"{BASE_URL}/account/list", headers=headers) as r:
            accs = await r.json()
            self.account_id   = accs[0]["id"]
            self.account_name = accs[0]["name"]
            
        # Get Balance
        async with self.session.get(f"{BASE_URL}/cashBalance/getCashBalanceSnapshot?accountId={self.account_id}", headers=headers) as r:
            bdata = await r.json()
            bal = bdata.get("netLiquidatingValue") or bdata.get("cashBalance") or 50000.0
            self.starting_bal = float(bal)
            self.peak_bal     = self.starting_bal

        log.info("=" * 60)
        log.info(f"  AUTHENTICATED: Account {self.account_name} (ID: {self.account_id})")
        log.info(f"  Starting Balance   : ${self.starting_bal:,.2f}")
        log.info(f"  Mode               : {'*** LIVE TRADING ***' if LIVE_MODE else 'DEMO / PAPER'}")
        log.info(f"  Daily Loss Guard   : -${MAX_DAILY_LOSS_USD:,.2f}")
        log.info(f"  Total DD Guard     : -${MAX_TOTAL_DRAWDOWN:,.2f}")
        log.info("=" * 60)

    async def check_risk_guard(self):
        """Prop Firm Risk Guardian: locks trading if drawdown thresholds are approached"""
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            async with self.session.get(f"{BASE_URL}/cashBalance/getCashBalanceSnapshot?accountId={self.account_id}", headers=headers) as r:
                bdata = await r.json()
                cur_bal = float(bdata.get("netLiquidatingValue") or bdata.get("cashBalance") or self.starting_bal)
                
                if cur_bal > self.peak_bal:
                    self.peak_bal = cur_bal

                # Daily PnL
                self.daily_pnl = cur_bal - self.starting_bal
                total_dd = self.peak_bal - cur_bal

                # Check Daily Loss
                if self.daily_pnl <= -MAX_DAILY_LOSS_USD:
                    log.warning(f"🚨 RISK GUARD: Daily loss (-${abs(self.daily_pnl):,.2f}) hit limit (-${MAX_DAILY_LOSS_USD:,.2f})!")
                    self.kill_switch = True
                    await self.flat()
                    return False

                # Check Max Drawdown
                if total_dd >= MAX_TOTAL_DRAWDOWN:
                    log.warning(f"🚨 RISK GUARD: Trailing drawdown (-${total_dd:,.2f}) hit limit (-${MAX_TOTAL_DRAWDOWN:,.2f})!")
                    self.kill_switch = True
                    await self.flat()
                    return False

                return True
        except Exception as e:
            log.error(f"Error checking balance: {e}")
            return True

    async def place_bracket(self, side, entry, stop, target):
        if not await self.check_risk_guard():
            log.warning("Risk limit breached. Order blocked by Prop Firm Guardian.")
            return

        action = "Buy" if side == "long" else "Sell"
        opp    = "Sell" if side == "long" else "Buy"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        
        order = {
            "accountId":   self.account_id,
            "action":      action,
            "symbol":      SYMBOL,
            "orderQty":    CONTRACTS,
            "orderType":   "Limit",
            "price":       round(entry, 2),
            "isAutomated": True,
            "bracket1": {"action": opp, "orderType": "Stop",  "stopPrice": round(stop,   2), "orderQty": CONTRACTS},
            "bracket2": {"action": opp, "orderType": "Limit", "price":     round(target, 2), "orderQty": CONTRACTS},
        }
        
        async with self.session.post(f"{BASE_URL}/order/placeorder", json=order, headers=headers) as r:
            result = await r.json()
            rr = abs(target - entry) / abs(stop - entry)
            log.info("=" * 60)
            log.info(f"  ⚡ {action.upper()} BRACKET ORDER PLACED ({SYMBOL})")
            log.info(f"  Entry Price : {entry:.2f}")
            log.info(f"  Stop Loss   : {stop:.2f}  (-{abs(entry-stop):.1f} pts)")
            log.info(f"  Take Profit : {target:.2f}  (+{abs(target-entry):.1f} pts)")
            log.info(f"  R:R Ratio   : {rr:.2f}R")
            log.info("=" * 60)
            return result

    async def flat(self):
        headers = {"Authorization": f"Bearer {self.token}"}
        async with self.session.post(f"{BASE_URL}/order/liquidateposition",
             json={"accountId": self.account_id, "isAutomated": True}, headers=headers) as r:
            log.info("Position liquidated / flattened.")

    async def process_candle(self, o, h, l, c, v, ts):
        now_et = ts.astimezone(ET)
        t      = now_et.time()
        today  = now_et.date()

        # Daily reset at midnight or new day
        if today != self.last_date:
            self.last_date    = today
            self.traded_today = False
            self.in_trade     = False
            self.kill_switch  = False
            self.vwap.reset()
            log.info(f"--- New Trading Session: {today} ---")

        # Accumulate VWAP from 09:30 AM
        if t >= ENTRY_START:
            self.vwap.update(h, l, c, v)

        # EOD flat at 15:45
        if t >= EOD_TIME and self.in_trade:
            log.info("15:45 EST reached — flattening open position for EOD.")
            await self.flat()
            self.in_trade = False
            return

        if self.kill_switch or self.traded_today or self.in_trade or not self.vwap.ready:
            return
        if not (ENTRY_START <= t <= ENTRY_END):
            return

        vw  = self.vwap.value
        up  = self.vwap.upper
        dn  = self.vwap.lower
        cr  = h - l
        uw  = h - max(o, c)
        dw  = min(o, c) - l

        # SHORT SETUP
        if (up - vw) >= MIN_BAND_PT and h >= up and cr >= 6.0 and uw >= cr * WICK_PCT and c < o:
            ep = (h + max(o, c)) / 2.0
            sl = h + STOP_BUF
            rk = min(max(sl - ep, 1.0), MAX_STOP)
            tp = vw
            rw = ep - tp
            if rk >= MIN_STOP and rw >= rk * MIN_RR:
                log.info(f"🎯 SHORT SIGNAL triggered at {t.strftime('%H:%M:%S')} EST!")
                await self.place_bracket("short", ep, sl, tp)
                self.traded_today = self.in_trade = True

        # LONG SETUP
        elif (vw - dn) >= MIN_BAND_PT and l <= dn and cr >= 6.0 and dw >= cr * WICK_PCT and c > o:
            ep = (l + min(o, c)) / 2.0
            sl = l - STOP_BUF
            rk = min(max(ep - sl, 1.0), MAX_STOP)
            tp = vw
            rw = tp - ep
            if rk >= MIN_STOP and rw >= rk * MIN_RR:
                log.info(f"🎯 LONG SIGNAL triggered at {t.strftime('%H:%M:%S')} EST!")
                await self.place_bracket("long", ep, sl, tp)
                self.traded_today = self.in_trade = True

    async def feed(self):
        import websockets
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"op": "authorize", "token": self.token}))
            await ws.recv()
            await ws.send(json.dumps({"op": "subscribe", "args": [f"md/subscribequote/{SYMBOL}"]}))
            log.info(f"Live market data feed connected for {SYMBOL}...")

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    if msg.get("e") != "quote":
                        continue
                    px  = msg["d"].get("tradePrice") or msg["d"].get("bidPrice")
                    vol = msg["d"].get("tradeSize", 1)
                    ts  = datetime.fromisoformat(msg["d"]["timestamp"].replace("Z", "+00:00"))
                    if not px:
                        continue

                    self.tick_buf.append({"px": px, "vol": vol, "ts": ts})
                    slot = (ts.minute // CANDLE_MINS) * CANDLE_MINS

                    if self.candle_slot is None:
                        self.candle_slot = slot

                    if slot != self.candle_slot and len(self.tick_buf) > 1:
                        prev = self.tick_buf[:-1]
                        prices = [t["px"] for t in prev]
                        o = prices[0]; h = max(prices); l = min(prices); c = prices[-1]
                        v = sum(t["vol"] for t in prev)
                        await self.process_candle(o, h, l, c, v, prev[0]["ts"])
                        self.tick_buf    = [self.tick_buf[-1]]
                        self.candle_slot = slot
                except Exception as e:
                    log.error(f"Feed error: {e}")

    async def run(self):
        async with aiohttp.ClientSession() as sess:
            self.session = sess
            await self.auth()
            while True:
                now_t = datetime.now(ET).time()
                # Run between 09:25 AM and 16:05 PM EST
                if dtime(9, 25) <= now_t <= dtime(16, 5):
                    try:
                        await self.feed()
                    except Exception as e:
                        log.error(f"Feed disconnected: {e} — reconnecting in 10s")
                        await asyncio.sleep(10)
                else:
                    await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(SniperBot().run())
