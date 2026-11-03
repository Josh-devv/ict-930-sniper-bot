"""
BTC Orderflow CVD Absorption Telegram Alert Bot
================================================
Runs 24/7 (Local or Free Cloud Worker)
Fetches live BTC 5-minute candles directly from Binance/Coinbase public REST APIs (zero auth needed).
Monitors London (07:00-11:00 UTC) and NY (13:00-18:00 UTC) sessions.
When Volume Climax + 30% Lower Wick Absorption triggers:
  -> Dispatches instant alert to your Telegram Chat with Entry, Stop, and TPs!
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ==========================================
# CONFIGURATION
# ==========================================
# You can set these via environment variables or paste your tokens here:
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8290114978:AAETL_UeHqbu8r4n2jHvDXxW-VKuEnvyp0c")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "5642644592")

SYMBOL             = "BTCUSDT"
INTERVAL           = "5m"
POLL_SECONDS       = 15      # Check every 15s around bar close
VOL_MULT           = 1.3     # Volume Climax threshold
WICK_MIN           = 0.30    # Min 30% lower wick
LOOKBACK           = 20      # Local sweep lookback
ATR_MULT           = 1.0     # Stop buffer

# Session check (UTC)
def is_active_session():
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    in_london = (hour >= 7 and hour <= 11)
    in_ny     = (hour >= 13 and hour <= 18)
    return in_london or in_ny

def send_telegram_alert(message):
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("[WARNING] Telegram credentials not set. Message that would be sent:\n", message)
        return False
    
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

def fetch_live_klines():
    """Fetch recent 5m candles from Binance public API (no API key needed)"""
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={INTERVAL}&limit=50"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'tb_base', 'tb_quote', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df['dt'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            return df
    except Exception as e:
        print(f"Error fetching klines: {e}")
    return None

def main():
    print(f"==================================================")
    print(f"  BTC Orderflow Absorption Telegram Bot Started")
    print(f"  Symbol: {SYMBOL} | Timeframe: {INTERVAL}")
    print(f"==================================================")

    last_processed_candle_time = None

    # Test Telegram connection at startup
    welcome_msg = (
        "🤖 <b>BTC Orderflow Sniper Bot is ONLINE!</b>\n\n"
        "Monitoring 5-minute candles for institutional absorption setups during London & NY sessions.\n"
        "You will receive instant alerts right here whenever a high-probability trade triggers."
    )
    if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE":
        send_telegram_alert(welcome_msg)
        print("Startup ping sent to Telegram!")

    while True:
        try:
            df = fetch_live_klines()
            if df is not None and len(df) >= 30:
                # The last row df.iloc[-1] is the CURRENT forming candle
                # The row before it df.iloc[-2] is the newly CLOSED candle
                closed_candle = df.iloc[-2]
                candle_time = closed_candle['open_time']

                if candle_time != last_processed_candle_time:
                    last_processed_candle_time = candle_time
                    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

                    # Check session filter
                    if is_active_session():
                        # Calculate Orderflow metrics on closed candle
                        vol_series = df['volume'].iloc[:-1] # all closed bars
                        vol_ma = vol_series.rolling(20).mean().iloc[-1]
                        vol_climax = (closed_candle['volume'] >= vol_ma * VOL_MULT)

                        # Local 20-period low sweep
                        local_low = df['low'].iloc[-22:-2].min()
                        low_swept = (closed_candle['low'] <= local_low)

                        # Range and lower wick
                        c_range = max(1.0, closed_candle['high'] - closed_candle['low'])
                        body_bot = min(closed_candle['open'], closed_candle['close'])
                        lower_wick = body_bot - closed_candle['low']
                        wick_ratio = lower_wick / c_range

                        # ATR
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
                                print("\n" + "="*50)
                                print(f"SIGNAL FIRED AT {now_str}")
                                print(alert_text)
                                print("="*50 + "\n")
                                send_telegram_alert(alert_text)
                            else:
                                print(f"[{now_str}] Setup skipped: stop distance too tight (${stop_dist:.2f})")
                        else:
                            print(f"[{now_str}] Bar closed: No signal (Vol: {closed_candle['volume']:.1f}/{vol_ma*VOL_MULT:.1f}, Wick: {wick_ratio*100:.1f}%)")
                    else:
                        print(f"[{now_str}] Bar closed: Outside active trading session (London/NY). Sleeping...")

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print(f"Loop error: {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()

