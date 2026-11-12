"""Preprocessor: cleans price data, handles missing values, splits chronologically.

Owner: Zain ul Abideen
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


class Preprocessor:
    """Clean a daily OHLCV DataFrame and split it without look-ahead.

    Attributes:
        dataframe: The working DataFrame; each step updates it in place of the
            previous version and also returns it, so calls can be chained.
        report: Counts of what each cleaning step changed (for the menu/README).
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()
        self.report: dict[str, int] = {}

    def clean(self) -> pd.DataFrame:
        """Sort by date, drop duplicate dates, coerce numbers, drop invalid prices."""
        df = self.dataframe
        before = len(df)

        df = df[~df.index.duplicated(keep="last")].sort_index()
        self.report["duplicate_dates_removed"] = before - len(df)

        present = [c for c in NUMERIC_COLUMNS if c in df.columns]
        df[present] = df[present].apply(lambda col: pd.to_numeric(col, errors="coerce"))

        # Prices must be strictly positive; zero/negative values are data errors.
        price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]
        invalid = (df[price_cols] <= 0).any(axis=1)
        df.loc[invalid, price_cols] = np.nan
        self.report["invalid_price_rows"] = int(invalid.sum())

        self.dataframe = df
        return df

    def handle_missing(self, max_gap: int = 3) -> pd.DataFrame:
        """Forward-fill short gaps (at most ``max_gap`` days), then drop what remains.

        Forward-filling only uses past values, so it does not leak future prices.
        """
        df = self.dataframe
        self.report["missing_values_before"] = int(df.isna().sum().sum())
        df = df.ffill(limit=max_gap)
        before = len(df)
        df = df.dropna(subset=[c for c in ["Close"] if c in df.columns])
        df = df.dropna()
        self.report["rows_dropped_missing"] = before - len(df)
        self.dataframe = df
        return df

    def run(self) -> pd.DataFrame:
        """Convenience: ``clean()`` then ``handle_missing()``."""
        self.clean()
        return self.handle_missing()

    @staticmethod
    def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
        """Split rows by time: the first ``1 - test_size`` share trains, the rest tests.

        Unlike a random split, every test date is strictly after every training
        date, so the model never sees the future during training.
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        df = df.sort_index()
        cut = int(round(len(df) * (1 - test_size)))
        if cut == 0 or cut == len(df):
            raise ValueError("Not enough rows for the requested split.")
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
