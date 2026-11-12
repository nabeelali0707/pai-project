"""ModelEvaluator: regression metrics and all project visualisations.

Owner: Nabeel Ali

Metrics are implemented directly with NumPy (vectorised) so the maths is
visible; ``tests/test_evaluator.py`` checks them against scikit-learn.
"""

from __future__ import annotations

import os

import matplotlib

if not os.environ.get("DISPLAY") and os.name != "nt" and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")  # headless Linux/CI: render to files only

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

sns.set_theme(style="whitegrid", context="notebook")


class ModelEvaluator:
    """Compare actual and predicted values.

    Attributes:
        actual: True next-day closing prices.
        predicted: Model predictions for the same days.
        label: Name shown in titles (e.g. "Ridge | Enhanced").
    """

    def __init__(self, actual, predicted, label: str = "model"):
        self.actual = np.asarray(actual, dtype=float)
        self.predicted = np.asarray(predicted, dtype=float)
        if self.actual.shape != self.predicted.shape:
            raise ValueError(f"Shape mismatch: actual {self.actual.shape} vs predicted {self.predicted.shape}")
        if self.actual.size == 0:
            raise ValueError("Cannot evaluate empty arrays.")
        self.label = label

    # ---------------------------------------------------------------- metrics
    @property
    def errors(self) -> np.ndarray:
        return self.actual - self.predicted

    def mae(self) -> float:
        """Mean Absolute Error: average size of the miss, in dollars."""
        return float(np.mean(np.abs(self.errors)))

    def rmse(self) -> float:
        """Root Mean Squared Error: like MAE but punishes large misses more."""
        return float(np.sqrt(np.mean(self.errors ** 2)))

    def r2(self) -> float:
        """Coefficient of determination: 1 - SS_res / SS_tot."""
        ss_res = np.sum(self.errors ** 2)
        ss_tot = np.sum((self.actual - self.actual.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def mape(self) -> float:
        """Mean Absolute Percentage Error (%)."""
        return float(np.mean(np.abs(self.errors / self.actual)) * 100)

    def direction_accuracy(self, previous) -> float:
        """Share of days where the predicted up/down move matches the real one.

        Args:
            previous: The close on the day each prediction was made.
        """
        previous = np.asarray(previous, dtype=float)
        real_move = np.sign(self.actual - previous)
        pred_move = np.sign(self.predicted - previous)
        mask = real_move != 0  # ignore days the price did not change at all
        if not mask.any() or not pred_move.any():  # a model that never predicts a move has no direction

            return float("nan")
        return float(np.mean(real_move[mask] == pred_move[mask]))

    def summary(self, previous=None) -> dict:
        """All metrics in one dict (direction accuracy needs ``previous``)."""
        out = {"MAE": self.mae(), "RMSE": self.rmse(), "R2": self.r2(), "MAPE_%": self.mape()}
        if previous is not None:
            out["Direction_Acc"] = self.direction_accuracy(previous)
        return out

    # ------------------------------------------------------------------ plots
    @staticmethod
    def _finish(fig, filename: str | None, show: bool) -> str | None:
        path = None
        if filename:
            os.makedirs(FIGURE_DIR, exist_ok=True)
            path = os.path.join(FIGURE_DIR, filename)
            fig.savefig(path, dpi=130, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_results(self, dates=None, filename: str | None = "actual_vs_predicted.png",
                     show: bool = False, baseline=None, baseline_label: str = "Baseline") -> str | None:
        """Line chart of actual vs predicted prices, plus a residual panel."""
        x = pd.to_datetime(dates) if dates is not None else np.arange(self.actual.size)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(x, self.actual, label="Actual close", color="#1f2937", linewidth=1.6)
        ax1.plot(x, self.predicted, label=f"Predicted ({self.label})", color="#2563eb",
                 linewidth=1.1, alpha=0.9)
        if baseline is not None:
            ax1.plot(x, np.asarray(baseline, dtype=float), label=baseline_label,
                     color="#f59e0b", linewidth=1.0, alpha=0.8, linestyle="--")
        ax1.set_ylabel("Price (USD)")
        # "\$" stops matplotlib from treating the dollar signs as maths markup
        ax1.set_title(f"Next-day close: actual vs predicted (MAE \\${self.mae():.2f}, "
                      f"RMSE \\${self.rmse():.2f}, R² {self.r2():.3f})")
        ax1.legend(loc="upper left")
        ax2.vlines(x, 0, self.errors, colors=np.where(self.errors >= 0, "#16a34a", "#dc2626"),
                   linewidth=1.4 if self.actual.size <= 120 else 0.8)
        ax2.axhline(0, color="#111827", linewidth=0.8)
        ax2.set_ylabel("Error (USD)")
        ax2.set_xlabel("Date")
        fig.tight_layout()
        return self._finish(fig, filename, show)

    @staticmethod
    def plot_comparison(results: pd.DataFrame, filename: str | None = "model_comparison.png",
                        show: bool = False) -> str | None:
        """Grouped bar charts of RMSE and direction accuracy per model and feature set."""
        df = results[results["feature_set"] != "-"].copy()
        naive = results[results["feature_set"] == "-"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(data=df, x="model", y="RMSE", hue="feature_set", ax=axes[0],
                    palette=["#94a3b8", "#2563eb"])
        if not naive.empty:
            axes[0].axhline(naive["RMSE"].iloc[0], color="#dc2626", linestyle="--",
                            label="Naive (no change)")
        lo = min(df["RMSE"].min(), naive["RMSE"].min() if not naive.empty else np.inf)
        hi = max(df["RMSE"].max(), naive["RMSE"].max() if not naive.empty else -np.inf)
        axes[0].set_ylim(lo * 0.97, hi * 1.02)  # zoom in: differences are small
        axes[0].set_title("RMSE on test period (lower is better)")
        axes[0].legend(fontsize=8)
        sns.barplot(data=df, x="model", y="Direction_Acc", hue="feature_set", ax=axes[1],
                    palette=["#94a3b8", "#2563eb"])
        axes[1].axhline(0.5, color="#dc2626", linestyle="--", label="Coin flip (50%)")
        axes[1].set_ylim(0.4, max(0.6, df["Direction_Acc"].max() + 0.03))
        axes[1].set_title("Up/down direction accuracy (higher is better)")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        return ModelEvaluator._finish(fig, filename, show)

    # --------------------------------------------------------- EDA plots
    @staticmethod
    def plot_price_history(df: pd.DataFrame, symbol: str = "", filename: str | None = "price_history.png",
                           show: bool = False) -> str | None:
        """Close price with moving averages and daily volume."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(df.index, df["Close"], label="Close", color="#1f2937", linewidth=1.2)
        for col, color in (("ma_20", "#2563eb"), ("ma_10", "#16a34a"), ("ma_5", "#f59e0b")):
            if col in df:
                ax1.plot(df.index, df[col], label=col.upper().replace("_", " "), color=color,
                         linewidth=0.9, alpha=0.85)
        ax1.set_title(f"{symbol} closing price and moving averages")
        ax1.set_ylabel("Price (USD)")
        ax1.legend(loc="upper left")
        ax2.bar(df.index, df["Volume"] / 1e6, color="#64748b", width=1.0)
        ax2.set_ylabel("Volume (M)")
        fig.tight_layout()
        return ModelEvaluator._finish(fig, filename, show)

    @staticmethod
    def plot_returns_distribution(df: pd.DataFrame, filename: str | None = "returns_distribution.png",
                                  show: bool = False) -> str | None:
        """Histogram of daily returns: they are centred near 0 with fat tails."""
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(df["return_1d"].dropna() * 100, bins=80, kde=True, color="#2563eb", ax=ax)
        ax.set_xlabel("Daily return (%)")
        ax.set_title("Distribution of daily returns")
        fig.tight_layout()
        return ModelEvaluator._finish(fig, filename, show)

    @staticmethod
    def plot_sentiment_distribution(scores: pd.DataFrame, daily: pd.DataFrame | None = None,
                                    filename: str | None = "sentiment_distribution.png",
                                    show: bool = False) -> str | None:
        """Headline score histogram, label counts and the daily sentiment over time."""
        n = 3 if daily is not None else 2
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.8))
        sns.histplot(scores["compound"], bins=50, color="#7c3aed", ax=axes[0])
        axes[0].set_title("Headline compound scores")
        order = ["negative", "neutral", "positive"]
        counts = scores["label"].value_counts().reindex(order, fill_value=0)
        axes[1].bar(order, counts.values, color=["#dc2626", "#94a3b8", "#16a34a"])
        axes[1].set_title("Headlines per sentiment label")
        if daily is not None:
            s = daily["sentiment_mean"]
            axes[2].plot(s.index, s.rolling(30, min_periods=5).mean(), color="#7c3aed")
            axes[2].axhline(0, color="#111827", linewidth=0.8)
            axes[2].set_title("Daily sentiment (30-day rolling mean)")
            axes[2].tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return ModelEvaluator._finish(fig, filename, show)

    @staticmethod
    def plot_correlation(df: pd.DataFrame, columns: list[str], filename: str | None = "feature_correlation.png",
                         show: bool = False) -> str | None:
        """Heatmap of feature correlations (including the target)."""
        corr = df[columns].corr()
        fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(columns), 0.8 + 0.5 * len(columns)))
        sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=len(columns) <= 18,
                    fmt=".2f", annot_kws={"size": 7}, ax=ax, square=True, cbar_kws={"shrink": 0.7})
        ax.set_title("Feature correlation matrix")
        fig.tight_layout()
        return ModelEvaluator._finish(fig, filename, show)
