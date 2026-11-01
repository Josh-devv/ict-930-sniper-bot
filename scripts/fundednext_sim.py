import pandas as pd
import numpy as np

df_trades = pd.read_csv('trade_log.csv')
print(f"Total trades in log: {len(df_trades)}")

# 1 MNQ contract fixed
cum_1mnq = df_trades['net_pnl'].cumsum()
print(f"1 MNQ: Peak = ${cum_1mnq.max():.2f}, Lowest = ${cum_1mnq.min():.2f}, Final = ${cum_1mnq.iloc[-1]:.2f}")

# Test dynamic risk sizing
results = []
for risk_dollar in [50, 75, 100, 125, 150, 200, 250]:
    balance = 50000.0
    peak = 50000.0
    passed = False
    failed = False
    pass_trade = None
    fail_trade = None
    pass_date = None
    fail_date = None
    max_dd = 0.0

    for i, row in df_trades.iterrows():
        pts = row['risk_points']
        # contracts sized so stop loss = risk_dollar
        contracts = max(1, int(round(risk_dollar / (pts * 2.0))))
        net = contracts * row['gross_points'] * 2.0 - contracts * 6.20
        balance += net
        if balance > peak:
            peak = balance
        dd = peak - balance
        if dd > max_dd:
            max_dd = dd

        if balance >= 52500.0 and not passed:
            passed = True
            pass_trade = i + 1
            pass_date = row['exit_time']
        if dd >= 1500.0 and not failed:
            failed = True
            fail_trade = i + 1
            fail_date = row['exit_time']

    results.append({
        'risk_dollar': risk_dollar,
        'passed': passed,
        'pass_trade': pass_trade,
        'pass_date': pass_date,
        'failed': failed,
        'fail_trade': fail_trade,
        'fail_date': fail_date,
        'final_bal': balance,
        'max_dd': max_dd
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
