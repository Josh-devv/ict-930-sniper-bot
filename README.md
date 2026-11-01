# NQ (E-mini Nasdaq-100) 1-Minute Historical Backtesting System

A reliable, institutional-grade Python backtesting system specifically engineered for **NQ futures 1-minute historical data**.

It provides dual execution engines:
1. **High-Precision Intrabar Simulator (`NQIntrabarSimulator`)**: Purpose-built to faithfully model CME futures contract mechanics ($20/point multiplier, 0.25 tick size), round-turn commissions ($4.50), tick slippage, **causal 1-minute intrabar OHLC price paths**, **3R target**, and **dynamic 25% breakeven stop-loss progression**.
2. **`backtesting.py` Adapter (`BacktestingPyNQStrategy`)**: Subclasses `backtesting.Strategy` for fast execution, native parameter optimization, and interactive Bokeh HTML visual plotting.

---

## Workspace & Environment Overview

- **Python Environment**: CPython 3.11.18 isolated virtual environment located in `.venv/`.
- **Installed Dependencies**: `backtesting` (v0.6.6), `pandas` (v3.1.0), `numpy`, `matplotlib`, `pytest`, `scipy`, `bokeh`.
- **Contract Specs Built-In**:
  - **NQ (E-mini)**: 0.25 tick ($5.00), $20.00/point.
  - **MNQ (Micro)**: 0.25 tick ($0.50), $2.00/point.

---

## Project Structure

```
10am/
├── config.py                     # Centralized contract specs, execution friction, and strategy settings
├── data/
│   ├── __init__.py
│   ├── contract.py               # Futures tick rounding, multiplier, and P&L math
│   └── loader.py                 # 1-minute CSV ingestion, tz conversion (US/Eastern), invariant validation
├── strategy/
│   ├── __init__.py
│   ├── indicators.py             # Causal Swing detection (zero lookahead), FVGs, MSS, Expansion candles
│   ├── rules.py                  # Setup logic, 3R target levels, 25% BE trigger calculation
│   └── backtesting_py_adapter.py # Adapter bridging strategy logic to backtesting.py
├── execution/
│   ├── __init__.py
│   └── simulator.py              # Bar-by-bar futures engine with intrabar path modeling & BE progression
├── analytics/
│   ├── __init__.py
│   ├── metrics.py                # Expectancy (R), Avg R, Win Rate, Profit Factor, Drawdown, Long vs Short
│   └── reporting.py              # Terminal dashboard, trade log CSV export, equity curve chart
├── scripts/
│   ├── generate_sample_data.py   # Realistic synthetic 1-minute NQ sample generator
│   └── run_backtest.py           # CLI backtest runner for intrabar and backtesting.py modes
├── tests/
│   ├── test_contract.py          # Contract tick & P&L unit tests
│   ├── test_data.py              # Ingestion & integrity unit tests
│   ├── test_indicators.py        # Causal pivot & FVG unit tests
│   ├── test_simulator.py         # 3R target & 25% BE progression tests
│   └── test_metrics.py           # Performance analytics unit tests
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Core dependencies
└── README.md                     # Documentation
```

---

## Intrabar Execution: Why backtesting.py vs Custom Simulator

In `backtesting.py`, orders execute candle-by-candle with fixed SL/TP levels set at order inception. In fast-moving 1-minute futures trading:
- **Dynamic 25% Breakeven Progression**: When price moves 25% of the distance toward the 3R target, the stop loss must dynamically adjust to Entry + 1 tick buffer. `backtesting.py` cannot adjust stops mid-candle if the trigger and subsequent price action happen within the same 1-minute bar.
- **Intrabar Path Modeling**: On a green candle (`Close >= Open`), the typical intrabar path is `Open -> Low -> High -> Close`. The custom simulator evaluates whether the initial stop was touched *before* the high was formed, preventing premature stop-outs once BE is triggered.
- **Futures Point Multipliers**: CME NQ contracts earn/lose $20 per full index point (not equity percentage). The simulator handles exact futures point math, exchange fees, and tick slippage.

---

## Running the System

### 1. Run Unit Tests (12/12 Passing)
```powershell
.\.venv\Scripts\pytest.exe -v
```

### 2. Run High-Precision Intrabar Simulation
```powershell
.\.venv\Scripts\python.exe scripts/run_backtest.py --mode intrabar
```
This prints the full terminal dashboard, exports `trade_log.csv`, and saves `equity_curve.png`.

### 3. Run Native backtesting.py with Bokeh Visualization
```powershell
.\.venv\Scripts\python.exe scripts/run_backtest.py --mode backtesting_py
```
This runs `backtesting.py` and saves the interactive HTML chart to `backtesting_py_result.html`.

### 4. Backtest Your Custom CSV Data
Place your NQ 1-minute CSV file in `sample_data/` or any path, and run:
```powershell
.\.venv\Scripts\python.exe scripts/run_backtest.py --data "path/to/your_nq_1min.csv" --mode intrabar
```

---

## Ready for Your Data & Strategy Rules

The foundation is fully operational, verified, and passing all unit and integration tests. When you provide your NQ data and strategy specifications:
1. Data files will be imported via [data/loader.py](file:///c:/Users/JOSHUA/Desktop/10am/data/loader.py).
2. Any specific entry/exit, session macro, SMT, or key-level rules will be mapped directly into [strategy/rules.py](file:///c:/Users/JOSHUA/Desktop/10am/strategy/rules.py).
