"""
NQ 9:30 VWAP Sniper Telegram Alert Bot
======================================
Designed for 100% Free Cloud Execution (GitHub Actions / Local).
Monitors NQ Futures / CFD during the 09:25 - 10:30 AM NY Window.

Alert Stages:
1. 09:25 - 09:29 AM NY: Pre-market Heads-Up (VWAP & Lower Band level announced)
2. 09:30 - 09:34 AM NY: LIVE TOUCH WARNING (Alerts immediately when lower band is pierced)
3. 09:35 AM NY: CONFIRMED SNIPER SIGNAL (Buy Limit Entry, Stop, TP1, TP2)
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8290114978:AAETL_UeHqbu8r4n2jHvDXxW-VKuEnvyp0c")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "5642644592")

NY_TZ = ZoneInfo("America/New_York")
LAGOS_TZ = ZoneInfo("Africa/Lagos")

SIGMA = 2.0
MIN_BAND_PT = 15.0
WICK_MIN = 0.35
STOP_BUFFER = 1.5

def send_telegram(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False

def fetch_nq_data(interval="1m", range_str="1d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/NQ=F?interval={interval}&range={range_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    try:
        chart = data["chart"]["result"][0]
        timestamps = chart["timestamp"]
        ohlcv = chart["indicators"]["quote"][0]
        df = pd.DataFrame({
            "ts": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": ohlcv["open"],
            "high": ohlcv["high"],
            "low": ohlcv["low"],
            "close": ohlcv["close"],
            "volume": ohlcv["volume"]
        }).dropna()
        df["ny_time"] = df["ts"].dt.tz_convert(NY_TZ)
        return df
    except Exception as e:
        print(f"Error parsing NQ data: {e}")
        return None

def compute_daily_vwap(df):
    if df is None or len(df) == 0:
        return None
    
    # Filter bars from today's regular / pre-market session
    today_ny = datetime.now(NY_TZ).date()
    df_today = df[df["ny_time"].dt.date == today_ny].copy()
    if len(df_today) == 0:
        df_today = df.copy() # fallback

    tp = (df_today["high"] + df_today["low"] + df_today["close"]) / 3.0
    vol = df_today["volume"].replace(0, 1.0)
    cum_vol = vol.cumsum()
    cum_pv = (tp * vol).cumsum()
    cum_pv2 = (tp * tp * vol).cumsum()
    
    vwap = cum_pv / cum_vol
    variance = np.maximum(0.0, (cum_pv2 / cum_vol) - (vwap ** 2))
    stdev = np.sqrt(variance)
    
    df_today["vwap"] = vwap
    df_today["upper_b"] = vwap + SIGMA * stdev
    df_today["lower_b"] = vwap - SIGMA * stdev
    return df_today

def check_nq_now():
    now_ny = datetime.now(NY_TZ)
    now_wat = datetime.now(LAGOS_TZ)
    hour, minute = now_ny.hour, now_ny.minute
    
    # We only trade Monday - Friday
    if now_ny.weekday() >= 5:
        print("Weekend - markets closed.")
        return

    # Check 1m data
    df_1m = fetch_nq_data(interval="1m", range_str="1d")
    if df_1m is None or len(df_1m) == 0:
        print("Could not fetch NQ 1m data.")
        return
        
    df_vwap = compute_daily_vwap(df_1m)
    latest = df_vwap.iloc[-1]
    
    cur_price = latest["close"]
    cur_vwap  = latest["vwap"]
    cur_lower = latest["lower_b"]
    
    time_str = now_ny.strftime("%H:%M NY") + " / " + now_wat.strftime("%H:%M WAT")
    
    print(f"[{time_str}] NQ: {cur_price:.2f} | VWAP: {cur_vwap:.2f} | Lower Band: {cur_lower:.2f}")

    # STAGE 1: 09:25 - 09:29 AM Heads Up
    if hour == 9 and 25 <= minute <= 29:
        msg = (
            f"⏰ <b>NQ 9:30 OPEN IN {30 - minute} MINUTES!</b>\n\n"
            f"<b>Time:</b> {time_str}\n"
            f"----------------------------------------\n"
            f"📍 <b>Current NQ:</b> <code>{cur_price:,.2f}</code>\n"
            f"🎯 <b>Daily VWAP:</b> <code>{cur_vwap:,.2f}</code>\n"
            f"🟢 <b>Lower Band (-2.0σ):</b> <code>{cur_lower:,.2f}</code>\n"
            f"----------------------------------------\n"
            f"👀 <i>Watch for a dip below <code>{cur_lower:,.2f}</code> right at the 9:30 AM open bell.</i>"
        )
        print("Sending Stage 1 Heads-Up Alert")
        send_telegram(msg)
        return

    # STAGE 2: 09:30 - 09:34 AM LIVE TOUCH WARNING
    if hour == 9 and 30 <= minute <= 34:
        recent_low = df_vwap["low"].iloc[-5:].min()
        if recent_low <= cur_lower and (cur_vwap - cur_lower) >= MIN_BAND_PT:
            msg = (
                f"⚡ <b>NQ EARLY WARNING: LOWER BAND PIERCED!</b>\n\n"
                f"<b>Time:</b> {time_str}\n"
                f"----------------------------------------\n"
                f"🔴 <b>Price Pierced Band:</b> Low reached <code>{recent_low:,.2f}</code>\n"
                f"🛡️ <b>Lower Band (-2.0σ):</b> <code>{cur_lower:,.2f}</code>\n"
                f"----------------------------------------\n"
                f"⚠️ <b>ACTION:</b> Absorption is forming right now!\n"
                f"Watch the 5m candle closing at 09:35 NY.\n"
                f"Prepare to place your <b>Buy Limit</b> inside the lower wick."
            )
            print("Sending Stage 2 Live Touch Warning")
            send_telegram(msg)
            return

    # STAGE 3: 09:35 AM CONFIRMED 5m BAR CLOSE
    if hour == 9 and minute in [35, 36]:
        df_5m = fetch_nq_data(interval="5m", range_str="1d")
        if df_5m is not None and len(df_5m) >= 2:
            # candle that opened at 9:30 and closed at 9:35
            bar_930 = df_5m[df_5m["ny_time"].dt.hour == 9]
            bar_930 = bar_930[bar_930["ny_time"].dt.minute == 30]
            
            if len(bar_930) > 0:
                c = bar_930.iloc[-1]
                crange = c["high"] - c["low"]
                down_wick = min(c["open"], c["close"]) - c["low"]
                wick_ratio = down_wick / crange if crange > 0 else 0
                swept = c["low"] <= cur_lower
                bullish = c["close"] > c["open"]
                
                if swept and bullish and wick_ratio >= WICK_MIN and crange >= 6.0:
                    entry = c["low"] + 0.50 * down_wick
                    stop  = c["low"] - STOP_BUFFER
                    risk  = entry - stop
                    tp1   = entry + 1.5 * risk
                    tp2   = entry + 3.0 * risk
                    
                    msg = (
                        f"🚀 <b>NQ 9:30 SNIPER CONFIRMED BUY!</b>\n\n"
                        f"<b>Time:</b> {time_str}\n"
                        f"----------------------------------------\n"
                        f"🔹 <b>BUY LIMIT (50% Wick):</b> <code>{entry:,.2f}</code>\n"
                        f"🔻 <b>STOP LOSS:</b> <code>{stop:,.2f}</code> (Risk: {risk:.2f} pts)\n"
                        f"🟢 <b>TP1 (Scale 60% + BE):</b> <code>{tp1:,.2f}</code> (+1.5R)\n"
                        f"🎯 <b>TP2 (Runner 40%):</b> <code>{tp2:,.2f}</code> (+3.0R)\n"
                        f"📊 <b>Rejection Wick:</b> {wick_ratio*100:.0f}%\n"
                        f"----------------------------------------\n"
                        f"💼 <i>Sized for 5 MNQ ($10/pt) or Prop Firm Challenge</i>"
                    )
                    print("Sending Stage 3 Confirmed Setup Alert")
                    send_telegram(msg)
                    return
                else:
                    print(f"9:30 bar did not qualify (Swept={swept}, Bullish={bullish}, Wick={wick_ratio:.2f})")

if __name__ == "__main__":
    check_nq_now()
