"""
Clean 9:30 VWAP Sniper backtest - direct market entry vs 50% wick limit.

This is intentionally separate from the older ad-hoc scripts. It fixes the
known execution issues:

  * long stop-outs are booked as losses, not gains
  * stops are checked before targets on every bar
  * a limit fill bar can only stop out, never hit a target
  * direct entries pay one tick of adverse slippage
  * stops and EOD market exits pay one tick of adverse slippage
  * commissions are charged per contract, round turn

Strategy rules (same family as the 9:30 VWAP sniper):
  * session VWAP anchored at 09:30 ET, 2 sigma bands
  * fade a band sweep with a 35% rejection wick
  * signal window 09:30-11:30 ET, one signal per day
  * stop 1.5 pts beyond the signal candle extreme
  * TP1 = VWAP, TP2 = opposite band
  * 60% off at TP1, 40% runner, runner stop to BE+1 after TP1

Examples:
  python scripts/backtest_930_direct_entry.py --data sample_data/MNQ_1min_20260120_20260415.csv --tf 1 3 5 --entry direct
  python scripts/backtest_930_direct_entry.py --data sample_data/MNQ_1min_20260120_20260415.csv --tf 5 --entry both
  python scripts/backtest_930_direct_entry.py --data sample_data/NQ_5min_60days.csv --tf 5 --entry direct
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


NY_TZ = "America/New_York"


@dataclass(frozen=True)
class Config:
    sigma: float = 2.0
    min_band_pt: float = 15.0
    wick_pct: float = 0.35
    min_crange: float = 6.0
    stop_buffer: float = 1.5
    min_stop: float = 5.0
    max_stop: float = 60.0
    min_rr: float = 1.5
    contracts: int = 5
    pt_value: float = 2.0
    commission_per_contract: float = 1.24
    tick_size: float = 0.25
    slippage_ticks: int = 1
    tp1_pct: float = 0.60
    be_offset: float = 1.0

    @property
    def slip(self) -> float:
        return self.tick_size * self.slippage_ticks

    @property
    def commission(self) -> float:
        return self.commission_per_contract * self.contracts


def load_data(path: str | Path, source_tz: str = "UTC") -> pd.DataFrame:
    df = pd.read_csv(path)
    dt_candidates = [c for c in df.columns if c.lower() in {"datetime", "dt", "date_time", "timestamp"}]
    if not dt_candidates:
        raise ValueError(f"No datetime column found in {path}. Columns: {list(df.columns)}")

    df["dt"] = pd.to_datetime(df[dt_candidates[0]])
    if df["dt"].dt.tz is None:
        df["dt"] = df["dt"].dt.tz_localize(source_tz)
    df["dt"] = df["dt"].dt.tz_convert(NY_TZ).dt.tz_localize(None)
    df = df.sort_values("dt").set_index("dt")

    for col in ("open", "high", "low", "close", "volume"):
        found = [c for c in df.columns if c.lower() == col]
        if not found:
            raise ValueError(f"Missing {col} column in {path}")
        df[col] = pd.to_numeric(df[found[0]], errors="coerce")

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    # Keep volume as-is: zero-volume bars should carry zero weight in the VWAP.
    df = df[~df.index.duplicated(keep="last")]
    df = df[df.index.day_of_week < 5]
    return df


def resample_ohlcv(df: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if tf_min == 1:
        return df.copy()
    return (
        df.resample(f"{tf_min}min", closed="left", label="left")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open"])
    )


def session_vwap(day: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.Series, pd.Series]:
    typical = (day["high"] + day["low"] + day["close"]) / 3.0
    vol = day["volume"]
    cum_vol = vol.cumsum()
    vwap = (typical * vol).cumsum() / cum_vol
    variance = ((typical**2 * vol).cumsum() / cum_vol - vwap**2).clip(lower=0.0)
    std = np.sqrt(variance)
    return vwap, vwap + cfg.sigma * std, vwap - cfg.sigma * std


def find_signal(
    day: pd.DataFrame,
    idx: int,
    vwap: pd.Series,
    upper: pd.Series,
    lower: pd.Series,
    cfg: Config,
    entry_mode: str,
) -> dict | None:
    o, h, l, c = (float(day[col].iloc[idx]) for col in ("open", "high", "low", "close"))
    rng = h - l
    if rng < cfg.min_crange:
        return None

    vw = float(vwap.iloc[idx])
    ub = float(upper.iloc[idx])
    lb = float(lower.iloc[idx])

    # SHORT: sweep upper band, close back below it, bearish candle, upper wick rejection.
    if h >= ub and c < ub and c < o and (ub - vw) >= cfg.min_band_pt:
        wick = h - c
        if wick / rng >= cfg.wick_pct:
            raw_entry = c if entry_mode == "direct" else h - 0.50 * wick
            entry = raw_entry - cfg.slip if entry_mode == "direct" else raw_entry
            stop = h + cfg.stop_buffer
            risk = stop - entry
            tp1, tp2 = vw, lb
            if cfg.min_stop <= risk <= cfg.max_stop and (entry - tp1) >= risk * cfg.min_rr:
                return {
                    "direction": "short",
                    "entry": entry,
                    "stop": stop,
                    "tp1": tp1,
                    "tp2": tp2,
                    "risk": risk,
                }

    # LONG: sweep lower band, close back above it, bullish candle, lower wick rejection.
    elif l <= lb and c > lb and c > o and (vw - lb) >= cfg.min_band_pt:
        wick = c - l
        if wick / rng >= cfg.wick_pct:
            raw_entry = c if entry_mode == "direct" else l + 0.50 * wick
            entry = raw_entry + cfg.slip if entry_mode == "direct" else raw_entry
            stop = l - cfg.stop_buffer
            risk = entry - stop
            tp1, tp2 = vw, ub
            if cfg.min_stop <= risk <= cfg.max_stop and (tp1 - entry) >= risk * cfg.min_rr:
                return {
                    "direction": "long",
                    "entry": entry,
                    "stop": stop,
                    "tp1": tp1,
                    "tp2": tp2,
                    "risk": risk,
                }

    return None


def find_limit_fill(future: pd.DataFrame, direction: str, entry: float) -> int | None:
    for j, (_, bar) in enumerate(future.iterrows()):
        if direction == "short" and float(bar["high"]) >= entry:
            return j
        if direction == "long" and float(bar["low"]) <= entry:
            return j
    return None


def _result(
    date: str,
    direction: str,
    entry: float,
    stop: float,
    tp1: float,
    tp2: float,
    risk: float,
    exit_reason: str,
    pnl: float,
    cfg: Config,
) -> dict:
    risk_dollars = risk * cfg.contracts * cfg.pt_value
    return {
        "date": date,
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "risk_pts": risk,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "r": pnl / risk_dollars if risk_dollars else 0.0,
    }


def simulate_trade(
    day: pd.DataFrame,
    signal_idx: int,
    sig: dict,
    cfg: Config,
    entry_mode: str,
    date: str,
) -> dict | None:
    future = day.iloc[signal_idx + 1 :]
    if len(future) == 0:
        return None

    if entry_mode == "limit":
        fill_idx = find_limit_fill(future, sig["direction"], sig["entry"])
        if fill_idx is None:
            return None
        post = future.iloc[fill_idx:]
    else:
        post = future

    direction = sig["direction"]
    entry = float(sig["entry"])
    stop = float(sig["stop"])
    tp1 = float(sig["tp1"])
    tp2 = float(sig["tp2"])
    risk = float(sig["risk"])

    tp1_contracts = int(round(cfg.contracts * cfg.tp1_pct))
    runner_contracts = cfg.contracts - tp1_contracts
    tp1_pnl = 0.0
    tp1_hit = False
    runner_stop: float | None = None

    for k, (_, bar) in enumerate(post.iterrows()):
        high = float(bar["high"])
        low = float(bar["low"])

        if not tp1_hit:
            # Initial stop is always checked first.
            if direction == "long" and low <= stop:
                exit_px = stop - cfg.slip
                pnl = (exit_px - entry) * cfg.contracts * cfg.pt_value - cfg.commission
                return _result(date, direction, entry, stop, tp1, tp2, risk, "stop", pnl, cfg)
            if direction == "short" and high >= stop:
                exit_px = stop + cfg.slip
                pnl = (entry - exit_px) * cfg.contracts * cfg.pt_value - cfg.commission
                return _result(date, direction, entry, stop, tp1, tp2, risk, "stop", pnl, cfg)

            # A limit fill bar may only stop out. It cannot also hit a target.
            if entry_mode == "limit" and k == 0:
                continue

            if direction == "long" and high >= tp1:
                tp1_hit = True
                tp1_pnl = (tp1 - entry) * tp1_contracts * cfg.pt_value
                runner_stop = entry + cfg.be_offset
            elif direction == "short" and low <= tp1:
                tp1_hit = True
                tp1_pnl = (entry - tp1) * tp1_contracts * cfg.pt_value
                runner_stop = entry - cfg.be_offset
            else:
                continue

            # After TP1, check the new runner stop before TP2 on the same bar.
            if direction == "long" and low <= runner_stop:
                exit_px = runner_stop - cfg.slip
                runner_pnl = (exit_px - entry) * runner_contracts * cfg.pt_value
                pnl = tp1_pnl + runner_pnl - cfg.commission
                return _result(date, direction, entry, stop, tp1, tp2, risk, "tp1+be", pnl, cfg)
            if direction == "short" and high >= runner_stop:
                exit_px = runner_stop + cfg.slip
                runner_pnl = (entry - exit_px) * runner_contracts * cfg.pt_value
                pnl = tp1_pnl + runner_pnl - cfg.commission
                return _result(date, direction, entry, stop, tp1, tp2, risk, "tp1+be", pnl, cfg)

            if direction == "long" and high >= tp2:
                runner_pnl = (tp2 - entry) * runner_contracts * cfg.pt_value
                pnl = tp1_pnl + runner_pnl - cfg.commission
                return _result(date, direction, entry, stop, tp1, tp2, risk, "tp2", pnl, cfg)
            if direction == "short" and low <= tp2:
                runner_pnl = (entry - tp2) * runner_contracts * cfg.pt_value
                pnl = tp1_pnl + runner_pnl - cfg.commission
                return _result(date, direction, entry, stop, tp1, tp2, risk, "tp2", pnl, cfg)

        else:
            # Runner is live. Stop first, then TP2.
            if direction == "long" and low <= runner_stop:
                exit_px = runner_stop - cfg.slip
                runner_pnl = (exit_px - entry) * runner_contracts * cfg.pt_value
                pnl = tp1_pnl + runner_pnl - cfg.commission
                return _result(date, direction, entry, stop, tp1, tp2, risk, "tp1+be", pnl, cfg)
            if direction == "short" and high >= runner_stop:
                exit_px = runner_stop + cfg.slip
                runner_pnl = (entry - exit_px) * runner_contracts * cfg.pt_value
                pnl = tp1_pnl + runner_pnl - cfg.commission
                return _result(date, direction, entry, stop, tp1, tp2, risk, "tp1+be", pnl, cfg)

            if direction == "long" and high >= tp2:
                runner_pnl = (tp2 - entry) * runner_contracts * cfg.pt_value
                pnl = tp1_pnl + runner_pnl - cfg.commission
                return _result(date, direction, entry, stop, tp1, tp2, risk, "tp2", pnl, cfg)
            if direction == "short" and low <= tp2:
                runner_pnl = (entry - tp2) * runner_contracts * cfg.pt_value
                pnl = tp1_pnl + runner_pnl - cfg.commission
                return _result(date, direction, entry, stop, tp1, tp2, risk, "tp2", pnl, cfg)

    # EOD market exit with one tick of adverse slippage.
    last_close = float(post["close"].iloc[-1])
    exit_px = last_close - cfg.slip if direction == "long" else last_close + cfg.slip
    if tp1_hit:
        runner_pnl = (
            (exit_px - entry) * runner_contracts * cfg.pt_value
            if direction == "long"
            else (entry - exit_px) * runner_contracts * cfg.pt_value
        )
        pnl = tp1_pnl + runner_pnl - cfg.commission
        return _result(date, direction, entry, stop, tp1, tp2, risk, "tp1+eod", pnl, cfg)

    pnl = (
        (exit_px - entry) * cfg.contracts * cfg.pt_value
        if direction == "long"
        else (entry - exit_px) * cfg.contracts * cfg.pt_value
    ) - cfg.commission
    return _result(date, direction, entry, stop, tp1, tp2, risk, "eod", pnl, cfg)


def run_backtest(df: pd.DataFrame, tf_min: int, entry_mode: str, cfg: Config) -> pd.DataFrame:
    bars = resample_ohlcv(df, tf_min)
    trades: list[dict] = []

    for date, day in bars.groupby(bars.index.date):
        day = day.between_time("09:30", "15:44")
        if len(day) < 3:
            continue

        vwap, upper, lower = session_vwap(day, cfg)
        signal_bars = day.between_time("09:30", "11:29")

        for ts in signal_bars.index:
            idx = day.index.get_loc(ts)
            sig = find_signal(day, idx, vwap, upper, lower, cfg, entry_mode)
            if sig is None:
                continue

            trade = simulate_trade(day, idx, sig, cfg, entry_mode, str(date))
            if trade is not None:
                trades.append(trade)
            # One signal per day, whether the limit filled or not.
            break

    return pd.DataFrame(trades)


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> tuple[float, float]:
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)])
    return tuple(np.percentile(means, [2.5, 97.5]))


def summarize(trades: pd.DataFrame, label: str, cfg: Config) -> dict:
    if trades.empty:
        print(f"\n{label}: no trades")
        return {}

    pnl = trades["pnl"].to_numpy()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    pf = gross_profit / gross_loss if gross_loss else float("inf")
    equity = np.cumsum(pnl)
    max_dd = float((equity - np.maximum.accumulate(equity)).min())
    ci_low, ci_high = bootstrap_ci(pnl)

    print(f"\n{label}")
    print("-" * len(label))
    print(f"  Trades        : {len(trades):,}")
    print(f"  Win rate      : {(pnl > 0).mean() * 100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Net P&L       : ${pnl.sum():,.2f}")
    print(f"  Avg / trade   : ${pnl.mean():,.2f}  ({trades['r'].mean():+.2f}R)")
    print(f"  95% CI avg    : [${ci_low:,.2f}, ${ci_high:,.2f}]")
    print(f"  Max drawdown  : ${max_dd:,.2f}")
    print(f"  Avg win/loss  : ${wins.mean() if len(wins) else 0:,.2f} / ${losses.mean() if len(losses) else 0:,.2f}")
    print(f"  Long / short  : {int((trades['direction'] == 'long').sum())} / {int((trades['direction'] == 'short').sum())}")
    print("  Exits         : " + ", ".join(f"{k}={v}" for k, v in trades["exit_reason"].value_counts().items()))

    # Year-by-year consistency.
    yearly = trades.copy()
    yearly["year"] = yearly["date"].str.slice(0, 4)
    print("  Yearly        :")
    positive_years = 0
    for year, group in yearly.groupby("year"):
        y_pnl = group["pnl"].to_numpy()
        y_wins = y_pnl[y_pnl > 0].sum()
        y_loss = abs(y_pnl[y_pnl <= 0].sum())
        y_pf = y_wins / y_loss if y_loss else float("inf")
        positive_years += int(y_pnl.sum() > 0)
        print(
            f"    {year}: {len(group):>3} trades  "
            f"WR {(y_pnl > 0).mean() * 100:>5.1f}%  "
            f"PF {y_pf:>5.2f}  "
            f"Net ${y_pnl.sum():>12,.2f}"
        )
    print(f"  Positive years: {positive_years}/{yearly['year'].nunique()}")

    return {
        "trades": len(trades),
        "win_rate": (pnl > 0).mean() * 100,
        "pf": pf,
        "net": pnl.sum(),
        "avg": pnl.mean(),
        "avg_r": trades["r"].mean(),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "max_dd": max_dd,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean 9:30 VWAP Sniper direct-entry backtest")
    p.add_argument("--data", default="sample_data/MNQ_1min_20260120_20260415.csv")
    p.add_argument("--tf", type=int, nargs="+", default=[5], help="Timeframe(s) in minutes, e.g. 1 3 5")
    p.add_argument("--entry", choices=["direct", "limit", "both"], default="direct")
    p.add_argument("--source-tz", default="UTC", help="Timezone to assume when the CSV has no timezone")
    p.add_argument("--start", default=None, help="Optional first date, e.g. 2024-01-01")
    p.add_argument("--end", default=None, help="Optional last date, e.g. 2024-12-31")
    p.add_argument("--save-trades", default=None, help="Optional CSV path for trade-by-trade output")
    p.add_argument("--contracts", type=int, default=5)
    p.add_argument("--point-value", type=float, default=2.0, help="$ per point per contract (MNQ=2, NQ=20)")
    p.add_argument("--commission", type=float, default=1.24, help="Round-turn commission per contract")
    p.add_argument("--slippage-ticks", type=int, default=1)
    p.add_argument("--min-crange", type=float, default=6.0)
    p.add_argument("--min-stop", type=float, default=5.0)
    p.add_argument("--max-stop", type=float, default=60.0)
    p.add_argument("--min-rr", type=float, default=1.5)
    p.add_argument("--band", type=float, default=15.0)
    p.add_argument("--wick", type=float, default=0.35)
    p.add_argument("--stop-buffer", type=float, default=1.5)
    p.add_argument("--tp1-pct", type=float, default=0.60)
    p.add_argument("--be-offset", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        min_crange=args.min_crange,
        min_stop=args.min_stop,
        max_stop=args.max_stop,
        min_rr=args.min_rr,
        min_band_pt=args.band,
        wick_pct=args.wick,
        stop_buffer=args.stop_buffer,
        contracts=args.contracts,
        pt_value=args.point_value,
        commission_per_contract=args.commission,
        slippage_ticks=args.slippage_ticks,
        tp1_pct=args.tp1_pct,
        be_offset=args.be_offset,
    )

    df = load_data(args.data, source_tz=args.source_tz)
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df.index < pd.Timestamp(args.end) + pd.Timedelta(days=1)]
    sessions = df.index.normalize().nunique()
    print("=" * 78)
    print("  9:30 VWAP SNIPER - CLEAN BACKTEST")
    print("=" * 78)
    print(f"  Data          : {args.data}")
    print(f"  Range         : {df.index.min()} -> {df.index.max()}")
    print(f"  Bars / sessions: {len(df):,} / {sessions}")
    print(f"  Costs         : ${cfg.commission_per_contract:.2f}/contract RT, {cfg.slippage_ticks} tick slippage")
    print(f"  Size          : {cfg.contracts} contracts @ ${cfg.pt_value:.2f}/pt")
    print(f"  Entry mode    : {args.entry}")
    print("=" * 78)

    modes = ["direct", "limit"] if args.entry == "both" else [args.entry]
    for tf_min in args.tf:
        for mode in modes:
            trades = run_backtest(df, tf_min, mode, cfg)
            summarize(trades, f"{mode.upper()} ENTRY | {tf_min}m", cfg)
            if args.save_trades and not trades.empty:
                out = Path(args.save_trades)
                tagged = out.with_name(f"{out.stem}_{tf_min}m_{mode}{out.suffix or '.csv'}")
                trades.to_csv(tagged, index=False)
                print(f"  Saved trades  : {tagged}")


if __name__ == "__main__":
    main()
