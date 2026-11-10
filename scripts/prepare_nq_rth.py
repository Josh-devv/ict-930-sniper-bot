"""
Stream a large Dukascopy-style JSON file of 1-minute bars and extract the
09:30-15:44 New York session to a compact CSV.

The 2018-2026 NQ file is ~450 MB of JSON on a single line. Loading it whole
needs several GB of RAM, so this reads it in chunks and keeps only complete
records, writing matched RTH bars as it goes.

Usage:
    uv run --with pandas --with numpy python scripts/prepare_nq_rth.py \
        --src C:\\Users\\JOSH\\Desktop\\2xjosh\\CRTbot\\data\\indices\\NQ-duka-2018-2026.json \
        --dst sample_data/NQ_1min_RTH_2018_2026.csv
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


RECORD = re.compile(
    rb'\{"time":(\d+),"open":([-\d.]+),"high":([-\d.]+),"low":([-\d.]+),'
    rb'"close":([-\d.]+),"volume":([-\d.]+)\}'
)


def convert(src: Path, dst: Path, chunk_mb: int = 8) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    buf = b""
    scanned = 0
    kept = 0
    zero_volume = 0
    first_ts = None
    last_ts = None
    t0 = time.time()

    with open(src, "rb") as f, open(dst, "w", newline="", encoding="utf-8") as out:
        out.write("datetime,open,high,low,close,volume\n")

        while True:
            chunk = f.read(chunk_mb * 1024 * 1024)
            if not chunk:
                break
            buf += chunk

            matches = list(RECORD.finditer(buf))
            if not matches:
                continue

            # Keep any trailing partial record for the next read.
            buf = buf[matches[-1].end():]

            raw = np.array(
                [[m.group(i) for i in range(1, 7)] for m in matches],
                dtype=object,
            )
            scanned += len(raw)

            times = raw[:, 0].astype(np.int64)
            dt = pd.to_datetime(times, unit="ms", utc=True).tz_convert("America/New_York")

            minutes = dt.hour * 60 + dt.minute
            mask = (
                (dt.day_of_week < 5)
                & (minutes >= 9 * 60 + 30)
                & (minutes < 15 * 60 + 45)
            )
            if not mask.any():
                continue

            frame = pd.DataFrame(
                {
                    "datetime": dt[mask].tz_localize(None),
                    "open": raw[mask, 1].astype(float),
                    "high": raw[mask, 2].astype(float),
                    "low": raw[mask, 3].astype(float),
                    "close": raw[mask, 4].astype(float),
                    "volume": raw[mask, 5].astype(float),
                }
            )
            zero_volume += int((frame["volume"] <= 0).sum())
            kept += len(frame)
            first_ts = frame["datetime"].iloc[0] if first_ts is None else first_ts
            last_ts = frame["datetime"].iloc[-1]
            frame.to_csv(out, header=False, index=False)

            elapsed = time.time() - t0
            print(
                f"  scanned {scanned:,} | kept {kept:,} | "
                f"{scanned / max(elapsed, 1e-9):,.0f} rec/s",
                end="\r",
            )

    print()
    print(f"Source        : {src}")
    print(f"Output        : {dst}")
    print(f"Records read  : {scanned:,}")
    print(f"RTH bars kept : {kept:,}  ({zero_volume:,} with zero volume)")
    print(f"Range         : {first_ts} -> {last_ts}")
    print(f"Elapsed       : {time.time() - t0:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(description="Extract NY RTH bars from a large JSON file")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--chunk-mb", type=int, default=8)
    args = p.parse_args()
    convert(Path(args.src), Path(args.dst), args.chunk_mb)


if __name__ == "__main__":
    main()
