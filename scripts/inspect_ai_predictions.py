import sys
sys.path.insert(0, '.')
import pandas as pd
from scripts.build_predictive_features import build_predictive_dataset

df = build_predictive_dataset('sample_data/MNQ_1min_20260120_20260415.csv')
count = 0
correct = 0

for i, r in df.iterrows():
    body = r['body_930']
    lw = r['lower_wick']
    uw = r['upper_wick']
    
    pred = 0
    rule = ""
    if lw >= 35.0 and body >= 10.0:
        pred = 1
        rule = "Lower Wick Absorption"
    elif uw >= 45.0 and body <= -10.0:
        pred = -1
        rule = "Upper Wick Distribution"
    elif lw <= 3.0 and body >= 25.0:
        pred = 1
        rule = "Shaved Bull Trend"
    elif uw <= 3.0 and body <= -25.0:
        pred = -1
        rule = "Shaved Bear Trend"
        
    if pred != 0:
        count += 1
        actual = r['outcome']
        is_win = (pred == actual)
        if is_win:
            correct += 1
        print(f"Date: {r['date']} | Pred: {pred:>2} | Actual: {actual:>2} | Win: {is_win} | Rule: {rule}")

print("-" * 60)
print(f"Total AI Predicted Trades: {count}")
print(f"Accurate Predictions     : {correct} ({correct/count*100:.1f}%)")
