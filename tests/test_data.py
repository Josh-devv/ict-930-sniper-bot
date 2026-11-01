"""
Unit tests for data loader, validation, and integrity invariants.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from data.loader import NQDataLoader


@pytest.fixture
def sample_csv(tmp_path):
    csv_file = tmp_path / "test_nq.csv"
    data = {
        "timestamp": [
            "2024-01-08 09:30:00",
            "2024-01-08 09:31:00",
            "2024-01-08 09:32:00",
        ],
        "open": [16800.0, 16802.0, 16805.0],
        "high": [16805.0, 16806.0, 16808.0],
        "low": [16799.0, 16801.0, 16804.0],
        "close": [16802.0, 16805.0, 16807.0],
        "vol": [100, 150, 120],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file


def test_load_and_standardize_csv(sample_csv):
    loader = NQDataLoader(target_timezone="America/New_York")
    df = loader.load_csv(sample_csv)
    assert len(df) == 3
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert str(df.index.tz) == "America/New_York"
    assert (df["High"] >= df["Low"]).all()


def test_validation_fails_on_corrupt_hl(tmp_path):
    csv_file = tmp_path / "corrupt_hl.csv"
    data = {
        "Datetime": ["2024-01-08 09:30:00"],
        "Open": [16800.0],
        "High": [16790.0],  # High < Low (Corrupted!)
        "Low": [16800.0],
        "Close": [16795.0],
        "Volume": [100],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)
    loader = NQDataLoader()
    with pytest.raises(ValueError, match="High < Low"):
        loader.load_csv(csv_file)
