"""
Robust Historical 1-Minute OHLCV Data Ingestion & Validation for NQ Futures.
Supports arbitrary CSV schemas, timezone conversions to US/Eastern,
integrity checking, and rollover/gap identification.
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple
import pandas as pd
import numpy as np


class NQDataLoader:
    """
    Loads, cleans, validates, and prepares historical 1-minute futures data.
    Ensures format conforms to both backtesting.py requirements (Open, High, Low, Close, Volume)
    and custom institutional backtesting standards.
    """

    def __init__(self, target_timezone: str = "America/New_York"):
        self.target_timezone = target_timezone

    def load_csv(
        self,
        file_path: Union[str, Path],
        datetime_col: Optional[str] = None,
        date_col: Optional[str] = None,
        time_col: Optional[str] = None,
        source_timezone: Optional[str] = None,  # None means assume UTC or target_timezone
        delimiter: str = ",",
    ) -> pd.DataFrame:
        """
        Loads CSV file and normalizes columns into canonical format.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found at: {path.resolve()}")

        df = pd.read_csv(path, sep=delimiter)
        if df.empty:
            raise ValueError(f"Supplied data file is empty: {path.name}")

        df = self._standardize_columns(df, datetime_col, date_col, time_col)
        df = self._parse_and_set_index(df, source_timezone)
        self.validate(df)
        return df

    def _standardize_columns(
        self,
        df: pd.DataFrame,
        datetime_col: Optional[str],
        date_col: Optional[str],
        time_col: Optional[str],
    ) -> pd.DataFrame:
        """
        Maps source columns to canonical Open, High, Low, Close, Volume.
        """
        # Case-insensitive mapping lookup
        col_lower_map = {str(col).lower().strip(): col for col in df.columns}

        # Handle Date & Time splitting if separate
        if date_col and time_col:
            df["Datetime"] = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
            df = df.drop(columns=[date_col, time_col])
        elif datetime_col:
            df = df.rename(columns={datetime_col: "Datetime"})
        else:
            # Auto-detect timestamp column
            candidates = ["datetime", "date_time", "time", "timestamp", "date"]
            found = None
            for c in candidates:
                if c in col_lower_map:
                    found = col_lower_map[c]
                    break
            if found:
                df = df.rename(columns={found: "Datetime"})
            elif "Datetime" not in df.columns:
                # If first column looks like a date/time string
                first_col = df.columns[0]
                df = df.rename(columns={first_col: "Datetime"})

        # Map OHLCV
        mapping = {}
        for target in ["Open", "High", "Low", "Close", "Volume"]:
            tgt_lower = target.lower()
            if tgt_lower in col_lower_map:
                mapping[col_lower_map[tgt_lower]] = target
            elif tgt_lower == "volume" and "vol" in col_lower_map:
                mapping[col_lower_map["vol"]] = "Volume"

        df = df.rename(columns=mapping)

        # Ensure all 4 OHLC columns exist
        missing = [col for col in ["Open", "High", "Low", "Close"] if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required price columns: {missing}. Available: {list(df.columns)}")

        # If Volume is missing, default to 0
        if "Volume" not in df.columns:
            df["Volume"] = 1.0

        return df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

    def _parse_and_set_index(
        self, df: pd.DataFrame, source_timezone: Optional[str]
    ) -> pd.DataFrame:
        """
        Parses timestamps, sorts chronologically, deduplicates, and converts timezone.
        """
        # Convert Datetime to pd.Timestamp
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=(source_timezone == "UTC"))

        # Timezone conversion
        if df["Datetime"].dt.tz is None:
            if source_timezone:
                df["Datetime"] = df["Datetime"].dt.tz_localize(source_timezone)
            else:
                # Default assume target timezone or UTC
                df["Datetime"] = df["Datetime"].dt.tz_localize(self.target_timezone)

        # Convert to target timezone (e.g. America/New_York)
        df["Datetime"] = df["Datetime"].dt.tz_convert(self.target_timezone)

        df = df.set_index("Datetime").sort_index()

        # Deduplicate timestamps keeping the last recorded tick/bar
        if df.index.has_duplicates:
            dup_count = df.index.duplicated().sum()
            df = df[~df.index.duplicated(keep="last")]
            print(f"[NQDataLoader] Info: Removed {dup_count} duplicate timestamp bars.")

        # Cast OHLCV to float
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validates OHLC integrity and invariants. Raises ValueError if data is corrupted.
        """
        if df.empty:
            raise ValueError("Dataframe is empty after parsing.")

        # Check for NaNs
        nan_counts = df[["Open", "High", "Low", "Close"]].isna().sum()
        if nan_counts.any():
            raise ValueError(f"Found NaN values in price data:\n{nan_counts}")

        # Invariant checks:
        # 1. High >= Low
        invalid_hl = df[df["High"] < df["Low"]]
        if not invalid_hl.empty:
            raise ValueError(f"Found {len(invalid_hl)} bars where High < Low.")

        # 2. High >= Open and High >= Close
        invalid_ho = df[df["High"] < df["Open"]]
        invalid_hc = df[df["High"] < df["Close"]]
        if not invalid_ho.empty or not invalid_hc.empty:
            raise ValueError(f"Found bars where High is lower than Open or Close.")

        # 3. Low <= Open and Low <= Close
        invalid_lo = df[df["Low"] > df["Open"]]
        invalid_lc = df[df["Low"] > df["Close"]]
        if not invalid_lo.empty or not invalid_lc.empty:
            raise ValueError(f"Found bars where Low is higher than Open or Close.")

        # 4. Prices must be strictly positive
        if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
            raise ValueError("Found non-positive price values in dataset.")

    def detect_session_gaps_and_rollovers(
        self, df: pd.DataFrame, gap_threshold_points: float = 30.0
    ) -> List[dict]:
        """
        Detects significant price jumps between consecutive bars or sessions.
        Helps flag contract expiration rollover gaps or discontinuous feeds.
        """
        time_diff = df.index.to_series().diff()
        price_diff = (df["Open"] - df["Close"].shift(1)).abs()

        # Flag gaps where time jumped > 1 hour and price jumped significantly
        gap_mask = (time_diff > pd.Timedelta(hours=1)) & (price_diff >= gap_threshold_points)
        gaps = df[gap_mask]

        flagged = []
        for dt, row in gaps.iterrows():
            prev_idx = df.index.get_loc(dt) - 1
            prev_row = df.iloc[prev_idx]
            jump = row["Open"] - prev_row["Close"]
            flagged.append({
                "timestamp": dt,
                "prev_timestamp": df.index[prev_idx],
                "time_gap": time_diff.loc[dt],
                "price_jump": round(jump, 2),
                "is_potential_rollover": abs(jump) >= gap_threshold_points,
            })
        return flagged
