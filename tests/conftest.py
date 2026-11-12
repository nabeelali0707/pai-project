import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def prices():
    """Synthetic 300-day random-walk OHLCV data on business days."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300, name="Date")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.002, len(idx))),
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Adj Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, len(idx)).astype(float),
    }, index=idx)


@pytest.fixture
def headlines():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-04", "2020-01-04", "2020-01-06", "2020-01-07"]),
        "headline": [
            "Apple (NASDAQ: AAPL) shares surge after record iPhone sales",
            "Analysts upgrade Apple, bullish on services growth",
            "Apple downgraded as iPhone demand slump worries investors",
            "Apple to hold annual shareholder meeting in March",
        ],
    })
