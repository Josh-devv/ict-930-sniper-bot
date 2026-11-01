"""
Train Machine Learning / Rule-Based Model to Predict 10:00 AM Expansion
========================================================================
We evaluate which patterns have the highest predictive probability (Win Rate):
1. Pattern A: Sellside Sweep + Bullish London Trend
2. Pattern B: Buyside Sweep + Bearish London Trend
3. Pattern C: True Day Open (TDO) Confluence (Above TDO + Bullish 9:30 close)
4. Decision Tree / Random Forest / Logistic Regression classifier
"""

import sys
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from scripts.build_predictive_features import build_predictive_dataset

# Build dataset
df = build_predictive_dataset('sample_data/MNQ_1min_20260120_20260415.csv')

print("\n" + "="*70)
print("  PATTERN DISCOVERY & WIN-RATE CORRELATION STUDY")
print("="*70)

# Filter out chop days (target != 0)
df_trend = df[df['outcome'] != 0].copy()

# 1. Study Sellside Sweeps (Swept Low == 1)
sellside_swept = df_trend[df_trend['swept_low'] == 1]
long_wins = len(sellside_swept[sellside_swept['outcome'] == 1])
total_s_swept = len(sellside_swept)
print(f"\n1. When 09:30 AM Swept Sellside (London Low / PDL):")
print(f"   Total Instances : {total_s_swept}")
print(f"   10AM Bull Run   : {long_wins} ({long_wins/total_s_swept*100:.1f}% Win Rate for Longs)")

# 2. Study Buyside Sweeps (Swept High == 1)
buyside_swept = df_trend[df_trend['swept_high'] == 1]
short_wins = len(buyside_swept[buyside_swept['outcome'] == -1])
total_b_swept = len(buyside_swept)
print(f"\n2. When 09:30 AM Swept Buyside (London High / PDH):")
print(f"   Total Instances : {total_b_swept}")
print(f"   10AM Bear Run   : {short_wins} ({short_wins/total_b_swept*100:.1f}% Win Rate for Shorts)")

# 3. CONFLUENCE: Sweep + Rejection Wick (The "Wick Clue")
# If swept sellside AND candle body closed back up (lower wick > 40% of range)
wick_longs = df_trend[(df_trend['swept_low'] == 1) & (df_trend['lower_wick'] > df_trend['range_930'] * 0.35)]
wl_wins = len(wick_longs[wick_longs['outcome'] == 1])
print(f"\n3. CONFLUENCE: Sellside Sweep + Lower Rejection Wick (>35% range):")
print(f"   Total Instances : {len(wick_longs)}")
print(f"   10AM Bull Run   : {wl_wins} ({wl_wins/len(wick_longs)*100:.1f}% Win Rate)")

wick_shorts = df_trend[(df_trend['swept_high'] == 1) & (df_trend['upper_wick'] > df_trend['range_930'] * 0.35)]
ws_wins = len(wick_shorts[wick_shorts['outcome'] == -1])
print(f"\n4. CONFLUENCE: Buyside Sweep + Upper Rejection Wick (>35% range):")
print(f"   Total Instances : {len(wick_shorts)}")
print(f"   10AM Bear Run   : {ws_wins} ({ws_wins/len(wick_shorts)*100:.1f}% Win Rate)")

# 5. CONFLUENCE: True Day Open (TDO)
# Longs only when above TDO, Shorts only when below TDO
tdo_longs = wick_longs[wick_longs['above_tdo'] == 1]
tl_wins = len(tdo_longs[tdo_longs['outcome'] == 1])
if len(tdo_longs) > 0:
    print(f"\n5. TRIPLE CONFLUENCE (Sweep + Wick + Above TDO):")
    print(f"   Total Instances : {len(tdo_longs)}")
    print(f"   10AM Bull Run   : {tl_wins} ({tl_wins/len(tdo_longs)*100:.1f}% Win Rate)")
