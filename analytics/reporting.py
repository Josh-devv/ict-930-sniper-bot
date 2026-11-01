"""
Performance Reporting & Equity Curve Visualization for NQ Backtest.
Formats terminal statistics tables, exports trade logs, and renders equity curves.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
import matplotlib.pyplot as plt
from execution.simulator import SimulatedTrade
from analytics.metrics import PerformanceAnalytics


class PerformanceReporter:
    """
    Renders institutional performance reports and equity charts.
    """

    def __init__(self, analytics: PerformanceAnalytics):
        self.analytics = analytics

    def print_terminal_summary(self) -> None:
        """
        Outputs clean, institutional-grade terminal performance summary.
        """
        stats = self.analytics.compute_summary_statistics()
        ls_breakdown = self.analytics.compute_long_vs_short_breakdown()

        print("\n" + "=" * 68)
        print("           NQ FUTURES 1-MINUTE BACKTEST PERFORMANCE REPORT")
        print("=" * 68)
        print(f" Total Trades           : {stats['total_trades']:<8} Win Rate            : {stats['win_rate_pct']:.2f}%")
        print(f" Winning Trades         : {stats['win_count']:<8} Losing Trades       : {stats['loss_count']}")
        print(f" Scratch/BE Trades      : {stats['scratch_count']:<8} Profit Factor       : {stats['profit_factor']:.2f}")
        print("-" * 68)
        print(f" Net Realized P&L ($)   : ${stats['net_pnl']:<11,.2f} Gross Profit        : ${stats['gross_profit']:,.2f}")
        print(f" Total Commissions/Fees : ${stats['total_fees']:<11,.2f} Total Slippage ($)  : ${stats['total_slippage']:,.2f}")
        print(f" Max Drawdown ($)       : ${stats['max_drawdown_dollars']:<11,.2f} Max Drawdown (%)    : {stats['max_drawdown_pct']:.2f}%")
        print("-" * 68)
        print(f" Expectancy (in R)      : {stats['expectancy_r']:<8.2f} R  Average R Realized  : {stats['average_r']:.2f} R")
        print(f" Average Win R          : {stats['avg_win_r']:<8.2f} R  Average Loss R      : {stats['avg_loss_r']:.2f} R")
        print("=" * 68)
        print("                      LONG VS SHORT PERFORMANCE")
        print("-" * 68)
        for side in ["LONG", "SHORT"]:
            s = ls_breakdown.get(side, {})
            print(f" {side:<6} | Trades: {s.get('count', 0):<4} | Win Rate: {s.get('win_rate_pct', 0.0):>5.1f}% | Net P&L: ${s.get('net_pnl', 0.0):>9.2f} | PF: {s.get('profit_factor', 0.0):>4.2f} | Avg R: {s.get('avg_r', 0.0):>4.2f}")
        print("=" * 68 + "\n")

    def export_trade_log(self, output_path: str = "trade_log.csv") -> Path:
        """
        Saves individual trade log to CSV.
        """
        df = self.analytics.trades_df
        path = Path(output_path)
        df.to_csv(path, index=False)
        print(f"[PerformanceReporter] Trade log successfully exported to: {path.resolve()}")
        return path

    def save_equity_curve_plot(self, output_path: str = "equity_curve.png") -> Optional[Path]:
        """
        Generates and saves the strategy equity curve and drawdown chart.
        """
        df = self.analytics.trades_df
        if df.empty:
            return None

        df_sorted = df.sort_values("exit_time").copy()
        df_sorted["cumulative_pnl"] = df_sorted["net_pnl"].cumsum()
        equity = self.analytics.initial_capital + df_sorted["cumulative_pnl"]
        peak = equity.cummax()
        drawdown = peak - equity

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

        ax1.plot(df_sorted["exit_time"], equity, label="Strategy Equity ($)", color="#00bcd4", linewidth=1.8)
        ax1.plot(df_sorted["exit_time"], peak, label="High Water Mark", color="#78909c", linestyle="--", alpha=0.7)
        ax1.set_title("NQ 1-Minute Strategy - Equity Curve & Drawdown", fontsize=14, fontweight="bold", pad=12)
        ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper left")

        ax2.fill_between(df_sorted["exit_time"], -drawdown, 0, color="#ef5350", alpha=0.4, label="Drawdown ($)")
        ax2.set_ylabel("Drawdown ($)", fontsize=11)
        ax2.set_xlabel("Date / Time", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="lower left")

        plt.tight_layout()
        path = Path(output_path)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[PerformanceReporter] Equity curve chart saved to: {path.resolve()}")
        return path
