import duckdb

conn = duckdb.connect()
conn.execute("INSTALL httpfs; LOAD httpfs;")

url = 'https://huggingface.co/datasets/mito0o852/OHLCV-1m/resolve/main/data/ohlcv_2024-10.parquet'
res = conn.execute(f"SELECT COUNT(open) as count, MIN(timestamp) as min_t, MAX(timestamp) as max_t FROM '{url}' WHERE ticker = 'QQQ'").df()
print("QQQ in Oct 2024:")
print(res)

# Export QQQ to CSV
print("Exporting QQQ 1-minute data for October 2024...")
conn.execute(f"COPY (SELECT timestamp, open, high, low, close, volume FROM '{url}' WHERE ticker = 'QQQ' ORDER BY timestamp) TO 'sample_data/QQQ_1min_202410.csv' (HEADER, DELIMITER ',')")
print("Saved to sample_data/QQQ_1min_202410.csv!")
