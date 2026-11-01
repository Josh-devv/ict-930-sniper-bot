import pandas as pd, numpy as np

df = pd.read_csv('trade_log.csv')
print('=== TRADE LOG STOP LOSS ANALYSIS ===')
print(f'Total trades: {len(df)}')
print()
print(f'Avg stop (risk_points)  : {df.risk_points.mean():.1f} pts')
print(f'Min stop                : {df.risk_points.min():.1f} pts')
print(f'Max stop                : {df.risk_points.max():.1f} pts')
print(f'Median stop             : {df.risk_points.median():.1f} pts')
print()
wins   = df[df.exit_reason == 'TARGET_2R']
losses = df[df.exit_reason == 'STOP_LOSS']
bes    = df[df.exit_reason == 'BREAKEVEN']
print(f'Avg gross pts (winners) : +{wins.gross_points.mean():.1f} pts')
print(f'Avg gross pts (losses)  : {losses.gross_points.mean():.1f} pts')
print(f'Avg gross pts (all)     : {df.gross_points.mean():+.1f} pts')
print()
print('--- Dollar P&L breakdown ---')
for label, mult, comm in [('1 MNQ ($2/pt)', 2.0, 6.20), ('1 Mini NQ ($20/pt)', 20.0, 10.20)]:
    nets  = df.gross_points * mult - comm
    wnet  = wins.gross_points * mult - comm
    lnet  = losses.gross_points * mult - comm
    print(f'{label}:')
    print(f'  Avg risk per trade    : ${(df.risk_points * mult).mean():.0f}')
    print(f'  Avg win (net)         : ${wnet.mean():+.2f}')
    print(f'  Avg loss (net)        : ${lnet.mean():+.2f}')
    print(f'  Avg net/trade (all)   : ${nets.mean():+.2f}')
    print(f'  Total net P&L         : ${nets.sum():+.2f}')
    print()

