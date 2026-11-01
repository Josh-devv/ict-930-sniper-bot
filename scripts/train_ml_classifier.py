"""
Train Decision Tree & Feature Importance Classifier for 10:00 AM Direction
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from scripts.build_predictive_features import build_predictive_dataset
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

df = build_predictive_dataset('sample_data/MNQ_1min_20260120_20260415.csv')

features = ['l_trend', 'above_tdo', 'swept_high', 'swept_low', 'body_930', 'upper_wick', 'lower_wick', 'range_930']
X = df[features]
y = df['outcome']

# Fit Decision Tree (max depth 3 to avoid overfitting and find core rules)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

print("\n" + "="*70)
print("  DECISION TREE RULES DISCOVERED BY AI")
print("="*70)
tree_rules = export_text(dt, feature_names=features)
print(tree_rules)

# Feature Importance
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X, y)

print("\nTop Predictive Features Ranking:")
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
for f, imp in importances.items():
    print(f"  {f:<15} : {imp*100:.1f}%")

