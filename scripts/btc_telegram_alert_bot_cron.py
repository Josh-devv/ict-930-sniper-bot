"""
BTC Orderflow CVD Absorption Single-Check Cron Script
=====================================================
Designed for 100% Free 24/7 Cloud Execution (GitHub Actions Cron / Render).
Runs once every 5 minutes, checks the newly closed 5m bar.
If an institutional absorption setup just closed -> Sends instant Telegram alert!
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8290114978:AAETL_UeHqbu8r4n2jHvDXxW-VKuEnvyp0c")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "5642644592")

SYMBOL   = "BTCUSDT"
INTERVAL = "5m"
VOL_MULT = 1.3
WICK_MIN = 0.30
ATR_MULT = 1.0

def is_active_session():
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    in_london = (hour >= 7 and hour <= 11)
    in_ny     = (hour >= 13 and hour <= 18)
    return in_london or in_ny

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False

def check_latest_candle():
    if not is_active_session():
        print("Outside active trading sessions (London/NY). No check needed.")
        return

    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={INTERVAL}&limit=40"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        print("Failed to fetch klines from Binance")
        return

    data = r.json()
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'tb_base', 'tb_quote', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Closed candle is second to last
    closed_candle = df.iloc[-2]
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    vol_series = df['volume'].iloc[:-1]
    vol_ma = vol_series.rolling(20).mean().iloc[-1]
    vol_climax = (closed_candle['volume'] >= vol_ma * VOL_MULT)

    local_low = df['low'].iloc[-22:-2].min()
    low_swept = (closed_candle['low'] <= local_low)

    c_range = max(1.0, closed_candle['high'] - closed_candle['low'])
    body_bot = min(closed_candle['open'], closed_candle['close'])
    lower_wick = body_bot - closed_candle['low']
    wick_ratio = lower_wick / c_range

    tr = (df['high'] - df['low']).iloc[:-1]
    atr = tr.rolling(14).mean().iloc[-1]

    if vol_climax and low_swept and (wick_ratio >= WICK_MIN):
        entry_price = closed_candle['low'] + 0.50 * lower_wick
        stop_price  = closed_candle['low'] - ATR_MULT * atr
        stop_dist   = entry_price - stop_price

        if stop_dist >= 40.0:
            tp1_price = entry_price + 1.5 * stop_dist
            tp2_price = entry_price + 3.0 * stop_dist

            alert_text = (
                f"🚀 <b>BTC ORDERFLOW ABSORPTION BUY!</b>\n\n"
                f"<b>Time:</b> {now_str}\n"
                f"<b>Timeframe:</b> 5-Minute\n"
                f"----------------------------------------\n"
                f"🔹 <b>BUY LIMIT (50% Wick):</b> <code>${entry_price:,.2f}</code>\n"
                f"🔻 <b>STOP LOSS:</b> <code>${stop_price:,.2f}</code> (Risk: ${stop_dist:,.2f})\n"
                f"🟢 <b>TP1 (60% Close + BE):</b> <code>${tp1_price:,.2f}</code> (+1.5R)\n"
                f"🎯 <b>TP2 (40% Runner):</b> <code>${tp2_price:,.2f}</code> (+3.0R)\n"
                f"----------------------------------------\n"
                f"💡 <i>Place Buy Limit order on Exness MT5 now.</i>"
            )
            print("SIGNAL FIRED!")
            send_telegram_alert(alert_text)
        else:
            print("Setup skipped: stop distance too tight")
    else:
        print(f"Bar closed at {now_str}: No setup.")

if __name__ == "__main__":
    check_latest_candle()
