"""
Fetch NQ futures data — 5-min bars (60 days available via yfinance)
Then run the full ICT VWAP band reversal backtest on it
"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "yfinance", "-q"], capture_output=True)

import yfinance as yf
import pandas as pd

print("Fetching NQ=F 5-min data (60 days)...")
ticker = yf.Ticker("NQ=F")

# 5-min goes back ~60 days — gives us ~40 trading days
df = ticker.history(period="60d", interval="5m", prepost=False)
df = df.reset_index()
df.columns = [c.lower() for c in df.columns]

# Rename datetime column
dt_col = [c for c in df.columns if "datetime" in c or c == "index" or "timestamp" in c][0]
df = df.rename(columns={dt_col: "dt"})
df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert("America/New_York")
df = df.sort_values("dt").reset_index(drop=True)

# Keep only regular hours 09:30 – 16:00 ET
df["hour"]   = df["dt"].dt.hour
df["minute"] = df["dt"].dt.minute
df["date"]   = df["dt"].dt.date
df = df[(df["hour"] >= 9) & ~((df["hour"]==9) & (df["minute"]<30))].copy()
df = df[df["hour"] < 16].copy()
df = df.reset_index(drop=True)

df.to_csv("sample_data/NQ_5min_60days.csv", index=False)
print(f"Saved {len(df):,} bars across {df['date'].nunique()} trading days")
print(f"Range: {df['dt'].iloc[0]}  ->  {df['dt'].iloc[-1]}")
print(f"Price: {df['low'].min():.0f} – {df['high'].max():.0f}")

