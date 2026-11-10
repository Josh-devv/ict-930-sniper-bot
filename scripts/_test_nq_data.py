import requests, pandas as pd
from zoneinfo import ZoneInfo
NY = ZoneInfo("America/New_York")
url = "https://query1.finance.yahoo.com/v8/finance/chart/NQ=F?interval=1m&range=1d"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
data = r.json()
chart = data["chart"]["result"][0]
ohlcv = chart["indicators"]["quote"][0]
df = pd.DataFrame({
    "ts": pd.to_datetime(chart["timestamp"], unit="s", utc=True),
    "close": ohlcv["close"],
    "volume": ohlcv["volume"]
}).dropna()
df["ny"] = df["ts"].dt.tz_convert(NY)
print("First bar NY:", df["ny"].iloc[0])
print("Last  bar NY:", df["ny"].iloc[-1])
print("Total bars  :", len(df))
print(df.tail(5).to_string())

