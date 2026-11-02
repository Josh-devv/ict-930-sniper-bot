"""
ICT 9:30 VWAP Sniper — Exness MetaTrader 5 (MT5) Automated Bot
Trades micro-lots (0.01 lot) on USTEC / NAS100 for small accounts ($20 - $100+)
Direct connection to Exness MT5 terminal — No monthly middleman fees.

Requirements:
  pip install MetaTrader5 pandas numpy python-dotenv
"""

import time
import math
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import os

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("logs/exness_bot_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("EXNESS_SNIPER")

# ── Settings ─────────────────────────────────────────────────────────────────
EXNESS_ACCOUNT  = int(os.getenv("EXNESS_ACCOUNT", "0"))
EXNESS_PASSWORD = os.getenv("EXNESS_PASSWORD", "")
EXNESS_SERVER   = os.getenv("EXNESS_SERVER", "Exness-MT5Real")
SYMBOL          = os.getenv("EXNESS_SYMBOL", "USTEC")   # Or USTECm on standard accounts
LOT_SIZE        = float(os.getenv("EXNESS_LOT_SIZE", "0.01"))

VWAP_MULT       = 2.0
MIN_BAND_PT     = 15.0
WICK_PCT        = 0.35
STOP_BUF        = 1.5
MIN_RR          = 1.5

ET = ZoneInfo("America/New_York")

def connect_mt5():
    if mt5 is None:
        log.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
        return False
    if not mt5.initialize():
        log.error(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    if EXNESS_ACCOUNT != 0:
        authorized = mt5.login(EXNESS_ACCOUNT, password=EXNESS_PASSWORD, server=EXNESS_SERVER)
        if not authorized:
            log.error(f"Failed to connect to Exness MT5 account #{EXNESS_ACCOUNT}: {mt5.last_error()}")
            return False
    acc_info = mt5.account_info()
    log.info(f"Connected to Exness MT5! Account: {acc_info.login} | Balance: ${acc_info.balance:.2f} | Leverage: 1:{acc_info.leverage}")
    return True

def calculate_vwap_bands(rates):
    # Rates is a numpy structured array from MT5: time, open, high, low, close, tick_volume
    cum_vol = 0.0
    cum_pv = 0.0
    cum_pv2 = 0.0
    vwap = None
    stdev = 0.0

    for r in rates:
        tp = (r['high'] + r['low'] + r['close']) / 3.0
        v  = max(float(r['tick_volume']), 1.0)
        cum_vol += v
        cum_pv  += tp * v
        cum_pv2 += tp * tp * v

    if cum_vol > 0:
        vwap = cum_pv / cum_vol
        var_ = max(0.0, cum_pv2 / cum_vol - vwap**2)
        stdev = math.sqrt(var_)

    upper = vwap + VWAP_MULT * stdev if vwap else None
    lower = vwap - VWAP_MULT * stdev if vwap else None
    return vwap, upper, lower, stdev

def place_exness_order(action_type, entry_p, sl_p, tp_p):
    price = mt5.symbol_info_tick(SYMBOL).ask if action_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).bid
    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       SYMBOL,
        "volume":       LOT_SIZE,
        "type":         action_type,
        "price":        price,
        "sl":           round(sl_p, 2),
        "tp":           round(tp_p, 2),
        "deviation":    20,
        "magic":        93050,
        "comment":      "ICT 9:30 Sniper",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Order failed: {res.comment} (code {res.retcode})")
        return False
    log.info(f"⚡ ORDER EXECUTED ON EXNESS: {'BUY' if action_type==mt5.ORDER_TYPE_BUY else 'SELL'} {LOT_SIZE} lots of {SYMBOL} at {price} | SL: {sl_p:.2f} | TP: {tp_p:.2f}")
    return True

def run():
    if not connect_mt5():
        return

    log.info(f"Monitoring {SYMBOL} for 9:30 AM EST Cash Open...")
    traded_today = False
    last_day = None

    while True:
        now_et = datetime.now(ET)
        t = now_et.time()
        today = now_et.date()

        if today != last_day:
            last_day = today
            traded_today = False
            log.info(f"--- New Trading Day: {today} ---")

        # Trading window: 09:30 AM to 11:30 AM EST
        if dtime(9, 30) <= t <= dtime(11, 30) and not traded_today:
            # Fetch 3-min rates from MT5 (e.g. past 40 bars)
            rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M3, 1, 40)
            if rates is not None and len(rates) > 10:
                last_bar = rates[-1]
                o, h, l, c = last_bar['open'], last_bar['high'], last_bar['low'], last_bar['close']
                vwap, upper, lower, stdev = calculate_vwap_bands(rates)

                crange = h - l
                up_w   = h - max(o, c)
                dn_w   = min(o, c) - l

                # Short Setup
                if (upper - vwap) >= MIN_BAND_PT and h >= upper and crange >= 6.0 and up_w >= crange * WICK_PCT and c < o:
                    ep = (h + max(o, c)) / 2.0
                    sl = h + STOP_BUF
                    tp = vwap
                    if (sl - ep) >= 5.0 and (ep - tp) >= (sl - ep) * MIN_RR:
                        log.info(f"🎯 9:30 Short Signal on Exness at {t}!")
                        if place_exness_order(mt5.ORDER_TYPE_SELL, ep, sl, tp):
                            traded_today = True

                # Long Setup
                elif (vwap - lower) >= MIN_BAND_PT and l <= lower and crange >= 6.0 and dn_w >= crange * WICK_PCT and c > o:
                    ep = (l + min(o, c)) / 2.0
                    sl = l - STOP_BUF
                    tp = vwap
                    if (ep - sl) >= 5.0 and (tp - ep) >= (ep - sl) * MIN_RR:
                        log.info(f"🎯 9:30 Long Signal on Exness at {t}!")
                        if place_exness_order(mt5.ORDER_TYPE_BUY, ep, sl, tp):
                            traded_today = True

        time.sleep(5)

if __name__ == "__main__":
    run()

