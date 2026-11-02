"""
Diagnostic: How often does price sweep Asia High/Low or Prior Day High/Low in London and New York?
And what is the size of the move after the sweep?
"""
import pandas as pd
import numpy as np
from datetime import timedelta, time as dtime

raw = pd.read_csv("sample_data/MNQ_1min_20260120_20260415.csv")
raw["dt"] = pd.to_datetime(raw["datetime"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
raw = raw.sort_values("dt").set_index("dt")

df_5m = raw.resample("5min", closed="left", label="left").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
}).dropna().reset_index()

df_5m["session_date"] = df_5m["dt"].apply(lambda x: (x + timedelta(hours=6)).date() if x.hour >= 18 else x.date())
df_5m["hour"] = df_5m["dt"].dt.hour
df_5m["time"] = df_5m["dt"].dt.time

print(f"Total Globex sessions: {df_5m['session_date'].nunique()}")

stats = []

for sess_date, s_df in df_5m.groupby("session_date"):
    if len(s_df) < 50: continue
    
    # Asian Range (18:00 to 02:00 EST)
    asia = s_df[(s_df["hour"] >= 18) | (s_df["hour"] < 2)]
    if asia.empty: continue
    asia_h = asia["high"].max()
    asia_l = asia["low"].min()
    asia_range = asia_h - asia_l
    
    # London Session (02:00 to 05:00 EST)
    london = s_df[(s_df["time"] >= dtime(2, 0)) & (s_df["time"] <= dtime(5, 0))]
    if london.empty: continue
    london_h = london["high"].max()
    london_l = london["low"].min()
    
    # New York Session (09:30 to 16:00 EST)
    ny = s_df[(s_df["time"] >= dtime(9, 30)) & (s_df["time"] <= dtime(16, 0))]
    if ny.empty: continue
    ny_h = ny["high"].max()
    ny_l = ny["low"].min()
    
    # Sweeps
    london_swept_asia_h = london_h > asia_h
    london_swept_asia_l = london_l < asia_l
    
    ny_swept_asia_h = ny_h > asia_h
    ny_swept_asia_l = ny_l < asia_l
    ny_swept_london_h = ny_h > london_h
    ny_swept_london_l = ny_l < london_l
    
    # Daily High/Low
    full_day_h = s_df["high"].max()
    full_day_l = s_df["low"].min()
    
    hod_in_london = abs(london_h - full_day_h) < 1.0
    lod_in_london = abs(london_l - full_day_l) < 1.0
    hod_in_ny     = abs(ny_h - full_day_h) < 1.0
    lod_in_ny     = abs(ny_l - full_day_l) < 1.0

    stats.append({
        "date": sess_date,
        "asia_range": asia_range,
        "london_swept_asia_h": london_swept_asia_h,
        "london_swept_asia_l": london_swept_asia_l,
        "ny_swept_london_h": ny_swept_london_h,
        "ny_swept_london_l": ny_swept_london_l,
        "hod_in_london": hod_in_london,
        "lod_in_london": lod_in_london,
        "hod_in_ny": hod_in_ny,
        "lod_in_ny": lod_in_ny,
        "day_range": full_day_h - full_day_l
    })

res = pd.DataFrame(stats)
print(f"Total valid days: {len(res)}")
print(f"Average Asia Range: {res['asia_range'].mean():.1f} pts")
print(f"Average Full Day Range: {res['day_range'].mean():.1f} pts")
print("-" * 55)
print(f"London sweeps Asia High: {res['london_swept_asia_h'].mean()*100:.1f}%")
print(f"London sweeps Asia Low:  {res['london_swept_asia_l'].mean()*100:.1f}%")
print(f"NY sweeps London High:   {res['ny_swept_london_h'].mean()*100:.1f}%")
print(f"NY sweeps London Low:    {res['ny_swept_london_l'].mean()*100:.1f}%")
print("-" * 55)
print(f"HOD formed in London:    {res['hod_in_london'].mean()*100:.1f}%")
print(f"LOD formed in London:    {res['lod_in_london'].mean()*100:.1f}%")
print(f"HOD formed in New York:  {res['hod_in_ny'].mean()*100:.1f}%")
print(f"LOD formed in New York:  {res['lod_in_ny'].mean()*100:.1f}%")

