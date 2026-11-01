"""
Comprehensive Quantitative Metrics for NQ Futures Trading Strategies.
Calculates R-multiples, expectancy, max drawdown, Sharpe/Sortino,
Long vs Short breakdowns, and Time-of-day / Monthly / Yearly distributions.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from execution.simulator import SimulatedTrade


class PerformanceAnalytics:
    """
    Computes institutional performance metrics from a list of completed trades.
    """

    def __init__(self, trades: List[SimulatedTrade], initial_capital: float = 100_000.0):
        self.trades = trades
        self.initial_capital = initial_capital
        self.trades_df = self._to_dataframe(trades)

    def _to_dataframe(self, trades: List[SimulatedTrade]) -> pd.DataFrame:
        """
        Converts simulated trade objects to a pandas DataFrame.
        """
        if not trades:
            return pd.DataFrame()

        records = []
        for t in trades:
            records.append({
                "trade_id": t.trade_id,
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "risk_points": t.risk_points,
                "gross_points": t.gross_points,
                "gross_pnl": t.gross_pnl,
                "fees": t.fees,
                "slippage": t.slippage,
                "net_pnl": t.net_pnl,
                "realized_r": t.realized_r,
                "be_activated": t.be_activated,
                "exit_reason": t.exit_reason,
            })
        df = pd.DataFrame(records)
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        return df

    def compute_summary_statistics(self) -> Dict[str, Any]:
        """
        Returns full quantitative summary of strategy performance.
        """
        df = self.trades_df
        if df.empty:
            return {
                "total_trades": 0,
                "win_count": 0,
                "loss_count": 0,
                "scratch_count": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "net_pnl": 0.0,
                "total_fees": 0.0,
                "total_slippage": 0.0,
                "average_r": 0.0,
                "avg_win_r": 0.0,
                "avg_loss_r": 0.0,
                "expectancy_r": 0.0,
                "max_drawdown_dollars": 0.0,
                "max_drawdown_pct": 0.0,
                "recovery_factor": 0.0,
            }

        total_trades = len(df)
        wins = df[df["net_pnl"] > 0]
        losses = df[df["net_pnl"] < 0]
        scratches = df[df["net_pnl"] == 0]

        win_count = len(wins)
        loss_count = len(losses)
        scratch_count = len(scratches)
        win_rate = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0

        gross_profit = wins["net_pnl"].sum()
        gross_loss = abs(losses["net_pnl"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        total_net_pnl = df["net_pnl"].sum()
        total_fees = df["fees"].sum()
        total_slippage = df["slippage"].sum()

        # R-Multiple Expectancy
        # Expectancy in R = (Win Rate * Avg Win R) - (Loss Rate * Avg Loss R)
        avg_r = df["realized_r"].mean()
        avg_win_r = wins["realized_r"].mean() if not wins.empty else 0.0
        avg_loss_r = abs(losses["realized_r"].mean()) if not losses.empty else 0.0

        p_win = win_count / total_trades
        p_loss = loss_count / total_trades
        expectancy_r = (p_win * avg_win_r) - (p_loss * avg_loss_r)

        # Equity Curve and Drawdown
        df_sorted = df.sort_values("exit_time").copy()
        df_sorted["cumulative_pnl"] = df_sorted["net_pnl"].cumsum()
        df_sorted["equity"] = self.initial_capital + df_sorted["cumulative_pnl"]
        peak = df_sorted["equity"].cummax()
        drawdown_dollars = peak - df_sorted["equity"]
        max_dd_dollars = drawdown_dollars.max()
        drawdown_pct = (drawdown_dollars / peak) * 100.0
        max_dd_pct = drawdown_pct.max()

        # Return on Max Drawdown (Calmar proxy)
        recovery_factor = (total_net_pnl / max_dd_dollars) if max_dd_dollars > 0 else 999.0

        return {
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "scratch_count": scratch_count,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_pnl": round(total_net_pnl, 2),
            "total_fees": round(total_fees, 2),
            "total_slippage": round(total_slippage, 2),
            "average_r": round(avg_r, 2),
            "avg_win_r": round(avg_win_r, 2),
            "avg_loss_r": round(avg_loss_r, 2),
            "expectancy_r": round(expectancy_r, 2),
            "max_drawdown_dollars": round(max_dd_dollars, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "recovery_factor": round(recovery_factor, 2),
        }

    def compute_long_vs_short_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculates separate metrics for Long and Short trades.
        """
        df = self.trades_df
        results = {}
        if df.empty:
            for side in ["LONG", "SHORT"]:
                results[side] = {"count": 0, "win_rate_pct": 0.0, "net_pnl": 0.0, "profit_factor": 0.0, "avg_r": 0.0}
            return results

        for side in ["LONG", "SHORT"]:
            sub = df[df["direction"] == side]
            if sub.empty:
                results[side] = {
                    "count": 0, "win_rate_pct": 0.0, "net_pnl": 0.0, "profit_factor": 0.0, "avg_r": 0.0
                }
            else:
                wins = sub[sub["net_pnl"] > 0]
                losses = sub[sub["net_pnl"] < 0]
                gp = wins["net_pnl"].sum()
                gl = abs(losses["net_pnl"].sum())
                pf = (gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0)
                results[side] = {
                    "count": len(sub),
                    "win_rate_pct": round((len(wins) / len(sub)) * 100.0, 2),
                    "net_pnl": round(sub["net_pnl"].sum(), 2),
                    "profit_factor": round(pf, 2),
                    "avg_r": round(sub["realized_r"].mean(), 2),
                }
        return results

    def compute_hourly_breakdown(self) -> pd.DataFrame:
        """
        Computes performance grouped by entry hour (e.g. 9:00, 10:00, 11:00).
        """
        df = self.trades_df
        if df.empty:
            return pd.DataFrame()
        df_h = df.copy()
        df_h["hour"] = df_h["entry_time"].dt.hour
        grouped = df_h.groupby("hour").agg(
            trades=("trade_id", "count"),
            net_pnl=("net_pnl", "sum"),
            win_rate=("net_pnl", lambda x: round((x > 0).mean() * 100.0, 1)),
            avg_r=("realized_r", "mean"),
        ).reset_index()
        return grouped

    def compute_periodic_breakdown(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (monthly_performance_df, yearly_performance_df).
        """
        df = self.trades_df
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        df_p = df.copy()
        df_p["year"] = df_p["entry_time"].dt.year
        df_p["year_month"] = df_p["entry_time"].dt.to_period("M").astype(str)

        monthly = df_p.groupby("year_month").agg(
            trades=("trade_id", "count"),
            net_pnl=("net_pnl", "sum"),
            win_rate=("net_pnl", lambda x: round((x > 0).mean() * 100.0, 1)),
            avg_r=("realized_r", "mean"),
        ).reset_index()

        yearly = df_p.groupby("year").agg(
            trades=("trade_id", "count"),
            net_pnl=("net_pnl", "sum"),
            win_rate=("net_pnl", lambda x: round((x > 0).mean() * 100.0, 1)),
            avg_r=("realized_r", "mean"),
        ).reset_index()

        return monthly, yearly
