"""
Generates high-fidelity synthetic 1-minute NQ futures data for system verification.
Creates realistic market sessions (RTH 09:30 - 16:00 NY time), swing points,
and Fair Value Gaps across multiple trading days.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_nq_data(
    output_path: str = "sample_data/nq_1min_sample.csv",
    start_date: str = "2024-01-08",
    days: int = 5,
    base_price: float = 16800.0,
    seed: int = 42,
) -> Path:
    """
    Generates realistic 1-minute OHLCV data.
    """
    np.random.seed(seed)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = []
    opens, highs, lows, closes, volumes = [], [], [], [], []

    current_price = base_price
    date_range = pd.date_range(start=start_date, periods=days, freq="B")  # Business days

    for day in date_range:
        # Trading day: 09:30 to 16:00 (391 1-minute bars)
        day_times = pd.date_range(
            start=f"{day.strftime('%Y-%m-%d')} 09:30:00",
            end=f"{day.strftime('%Y-%m-%d')} 16:00:00",
            freq="1min",
            tz="America/New_York",
        )

        for i, dt in enumerate(day_times):
            # Session open volatility bump
            vol_factor = 2.0 if i < 30 else 1.0

            # Intrabar random walk
            drift = np.sin(i / 25.0) * 0.5
            noise = np.random.normal(0, 1.2 * vol_factor)
            delta = drift + noise

            o = current_price
            c = round(round((o + delta) / 0.25) * 0.25, 2)
            # Create high and low with realistic wick
            wick_up = abs(np.random.exponential(scale=1.0 * vol_factor))
            wick_down = abs(np.random.exponential(scale=1.0 * vol_factor))

            h = round(round((max(o, c) + wick_up) / 0.25) * 0.25, 2)
            l = round(round((min(o, c) - wick_down) / 0.25) * 0.25, 2)
            v = int(max(10, np.random.poisson(lam=150 * vol_factor)))

            timestamps.append(dt)
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(c)
            volumes.append(v)

            current_price = c

    df = pd.DataFrame({
        "Datetime": timestamps,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    })

    df.to_csv(path, index=False)
    print(f"[SampleDataGenerator] Generated {len(df)} 1-minute bars -> {path.resolve()}")
    return path


if __name__ == "__main__":
    generate_synthetic_nq_data()
