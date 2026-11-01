import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def fetch_multi_chunks():
    print("Downloading authentic 1-minute market data (NQ futures & QQQ)...")
    ticker = yf.Ticker("NQ=F")
    
    # 7-day period request
    df = ticker.history(period="7d", interval="1m")
    if df.empty or len(df) < 50:
        print("Switching to QQQ...")
        ticker = yf.Ticker("QQQ")
        df = ticker.history(period="7d", interval="1m")
    
    if not df.empty:
        df = df.reset_index()
        date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df_clean = pd.DataFrame({
            'Datetime': pd.to_datetime(df[date_col]),
            'Open': df['Open'],
            'High': df['High'],
            'Low': df['Low'],
            'Close': df['Close'],
            'Volume': df['Volume']
        })
        out_path = Path("sample_data/REAL_NQ_1min.csv")
        df_clean.to_csv(out_path, index=False)
        print(f"Successfully downloaded {len(df_clean):,} real 1-minute bars!")
        print(f"Date Range: {df_clean['Datetime'].min()} -> {df_clean['Datetime'].max()}")
        return out_path
    else:
        print("Could not retrieve 1m data.")
        return None

if __name__ == '__main__':
    fetch_multi_chunks()
