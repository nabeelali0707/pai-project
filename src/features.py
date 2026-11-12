"""FeatureBuilder: technical indicators + sentiment features + prediction target.

Owners: Zain ul Abideen (price features), Siraj Udin (sentiment merge)

Leakage rules followed here:
    * Every feature on row ``t`` uses only information available at the close
      of trading day ``t`` (rolling windows look backwards only).
    * Headlines dated on a weekend/holiday are assigned to the NEXT trading day,
      never to an earlier one.
    * The target on row ``t`` is the closing price of day ``t + 1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MA_WINDOWS = (5, 10, 20)

PRICE_FEATURES = [
    "return_1d", "return_lag1", "return_lag2", "return_lag3",
    "ma_ratio_5", "ma_ratio_10", "ma_ratio_20",
    "volatility_10", "volume_change", "hl_range",
]
SENTIMENT_FEATURES = [
    "sentiment_mean", "sentiment_ma3", "sentiment_change",
    "news_count_log", "positive_ratio", "negative_ratio",
]
TARGET_RETURN = "target_return"
TARGET_PRICE = "target_close"


class FeatureBuilder:
    """Build the model-ready feature table from prices and daily sentiment.

    Attributes:
        price_data: Cleaned OHLCV DataFrame indexed by trading date.
        sentiment_data: Daily sentiment from ``SentimentAnalyzer.aggregate_by_date``
            (indexed by calendar date), or ``None`` for a price-only table.
        features: The resulting table after :meth:`build`.
    """

    def __init__(self, price_data: pd.DataFrame, sentiment_data: pd.DataFrame | None = None):
        self.price_data = price_data.sort_index().copy()
        self.sentiment_data = sentiment_data
        self.features: pd.DataFrame | None = None

    # -------------------------------------------------------------- price
    def add_returns(self, lags: int = 3) -> pd.DataFrame:
        """Add daily simple return (vectorised NumPy), log return and lagged returns."""
        df = self.price_data
        close = df["Close"].to_numpy(dtype=float)
        ret = np.empty_like(close)
        ret[0] = np.nan
        ret[1:] = close[1:] / close[:-1] - 1.0          # broadcasting, no Python loop
        df["return_1d"] = ret
        df["log_return"] = np.log1p(ret)
        for k in range(1, lags + 1):
            df[f"return_lag{k}"] = df["return_1d"].shift(k)
        return df

    def add_moving_averages(self, windows=MA_WINDOWS) -> pd.DataFrame:
        """Add simple moving averages and the price's distance from each one.

        ``ma_ratio_w = Close / MA_w - 1`` is scale-free, so it means the same
        thing in 2015 (AAPL ~ $27) and 2023 (AAPL ~ $190).
        """
        df = self.price_data
        for w in windows:
            df[f"ma_{w}"] = df["Close"].rolling(w).mean()
            df[f"ma_ratio_{w}"] = df["Close"] / df[f"ma_{w}"] - 1.0
        if "return_1d" not in df:
            self.add_returns()
        df["volatility_10"] = df["return_1d"].rolling(10).std()
        df["volume_change"] = np.log(df["Volume"].replace(0, np.nan)).diff()
        df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
        return df

    # ---------------------------------------------------------- sentiment
    @staticmethod
    def align_to_trading_days(daily: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
        """Move each calendar-date sentiment row onto the next trading day (same day if open).

        Weekend news therefore counts toward Monday's features. Dates after the
        last trading day are dropped. Means are re-weighted by headline count.
        """
        daily = daily.copy()
        daily.index = pd.to_datetime(daily.index).normalize()
        trading = pd.DatetimeIndex(trading_days).normalize()
        pos = np.searchsorted(trading.values, daily.index.values, side="left")
        valid = pos < len(trading)
        daily = daily[valid]
        daily["trade_date"] = trading.values[pos[valid]]

        weight = daily["news_count"]
        weighted_cols = ["sentiment_mean", "positive_ratio", "negative_ratio"]
        for col in weighted_cols:
            daily[col] = daily[col] * weight
        grouped = daily.groupby("trade_date")[weighted_cols + ["news_count"]].sum()
        for col in weighted_cols:
            grouped[col] = grouped[col] / grouped["news_count"]
        grouped.index.name = "Date"
        return grouped

    def merge_features(self) -> pd.DataFrame:
        """Join sentiment onto price rows and derive smoothed sentiment features.

        Trading days without any headline get neutral sentiment (0) and a
        news count of 0.
        """
        df = self.price_data
        if self.sentiment_data is None or self.sentiment_data.empty:
            return df
        aligned = self.align_to_trading_days(self.sentiment_data, df.index)
        df = df.drop(columns=[c for c in aligned.columns if c in df.columns])
        df = df.join(aligned, how="left")
        df[["sentiment_mean", "positive_ratio", "negative_ratio"]] = (
            df[["sentiment_mean", "positive_ratio", "negative_ratio"]].fillna(0.0))
        df["news_count"] = df["news_count"].fillna(0).astype(int)
        df["news_count_log"] = np.log1p(df["news_count"])
        df["sentiment_ma3"] = df["sentiment_mean"].rolling(3, min_periods=1).mean()
        df["sentiment_change"] = df["sentiment_mean"].diff().fillna(0.0)
        self.price_data = df
        return df

    # ------------------------------------------------------------- target
    def add_target(self) -> pd.DataFrame:
        """Add next-day close (``target_close``) and next-day return (``target_return``)."""
        df = self.price_data
        df[TARGET_PRICE] = df["Close"].shift(-1)
        df[TARGET_RETURN] = df[TARGET_PRICE] / df["Close"] - 1.0
        return df

    def build(self) -> pd.DataFrame:
        """Run every step and drop warm-up rows that lack rolling-window history.

        The final row (most recent day) is kept even though its target is
        unknown; it is the input for the next-day prediction. Use
        :meth:`training_rows` to get only rows with a known target.
        """
        self.add_returns()
        self.add_moving_averages()
        self.merge_features()
        self.add_target()
        df = self.price_data
        df = df.dropna(subset=PRICE_FEATURES)
        self.features = df
        return df

    @property
    def sentiment_available(self) -> bool:
        return self.features is not None and all(c in self.features for c in SENTIMENT_FEATURES)

    def feature_columns(self, with_sentiment: bool) -> list[str]:
        """Column names for the baseline (price-only) or enhanced feature set."""
        if with_sentiment and not self.sentiment_available:
            raise ValueError("Sentiment features were not built; run sentiment analysis first.")
        return PRICE_FEATURES + (SENTIMENT_FEATURES if with_sentiment else [])

    def training_rows(self) -> pd.DataFrame:
        """Rows whose next-day target is known (all but the latest day)."""
        if self.features is None:
            self.build()
        return self.features.dropna(subset=[TARGET_RETURN])

    def latest_row(self) -> pd.DataFrame:
        """The most recent trading day, used to forecast the following day."""
        if self.features is None:
            self.build()
        return self.features.iloc[[-1]]
