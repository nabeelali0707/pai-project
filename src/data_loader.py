"""DataLoader: loads stock prices and financial headlines, with an offline cache.

Owner: Zain ul Abideen

Loading strategy (offline-first fallback):
    1. Try the Yahoo Finance API (yfinance) unless ``offline=True``.
    2. On success, save the data to ``data/cache/<SYMBOL>_prices.csv`` and a
       JSON metadata file describing where/when it came from.
    3. If the network is unavailable (or yfinance fails / returns nothing),
       automatically fall back to the cached CSV.

Headlines are never fetched at runtime; they come from a dataset downloaded in
advance (see ``scripts/fetch_headlines.py``) and stored in the cache folder.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import pandas as pd

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
HEADLINE_COLUMNS = ["date", "headline"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")


class DataValidationError(ValueError):
    """Raised when a loaded dataset is missing required columns or rows."""


class DataLoader:
    """Load, validate and cache market data and headlines for one stock symbol.

    Attributes:
        stock_symbol: Ticker symbol, e.g. ``"AAPL"``.
        file_path: Path of the price CSV used by :meth:`load_csv` (defaults to
            the cache file for ``stock_symbol``).
        data: The most recently loaded price DataFrame (``None`` until loaded).
        headlines: The most recently loaded headline DataFrame.
        source: ``"api"``, ``"cache"`` or ``"csv"``, describing where ``data`` came from.
    """

    def __init__(self, stock_symbol: str = "AAPL", file_path: str | None = None,
                 cache_dir: str = DEFAULT_CACHE_DIR,
                 start: str = "2015-01-01", end: str = "2024-01-31"):
        self.stock_symbol = stock_symbol.upper().strip()
        self.cache_dir = cache_dir
        self.start = start
        self.end = end
        self.file_path = file_path or self.price_cache_path
        self.data: pd.DataFrame | None = None
        self.headlines: pd.DataFrame | None = None
        self.source: str | None = None

    # ------------------------------------------------------------------ paths
    @property
    def price_cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"{self.stock_symbol}_prices.csv")

    @property
    def meta_cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"{self.stock_symbol}_prices_meta.json")

    @property
    def headline_cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"{self.stock_symbol}_headlines.csv.gz")

    # ------------------------------------------------------------- validation
    @staticmethod
    def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
        """Check that ``df`` has a DatetimeIndex and the OHLCV columns."""
        if df is None or df.empty:
            raise DataValidationError("Price data is empty.")
        missing = [c for c in PRICE_COLUMNS if c not in df.columns]
        if missing:
            raise DataValidationError(f"Price data is missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError("Price data must be indexed by date.")
        return df

    @staticmethod
    def validate_headlines(df: pd.DataFrame) -> pd.DataFrame:
        """Check that ``df`` has ``date`` and ``headline`` columns and some rows."""
        if df is None or df.empty:
            raise DataValidationError("Headline data is empty.")
        missing = [c for c in HEADLINE_COLUMNS if c not in df.columns]
        if missing:
            raise DataValidationError(f"Headline data is missing columns: {missing}")
        return df

    # ---------------------------------------------------------------- loading
    def load_api(self, retries: int = 3) -> pd.DataFrame:
        """Download daily prices from Yahoo Finance and refresh the cache.

        Raises:
            ConnectionError: if yfinance is missing, the network fails, or no
                rows are returned.
        """
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ConnectionError("yfinance is not installed") from exc
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # we report failures ourselves

        raw, last_error = None, "no rows returned"
        for attempt in range(1, retries + 1):
            try:
                raw = yf.download(self.stock_symbol, start=self.start, end=self.end,
                                  progress=False, auto_adjust=False, threads=False)
            except Exception as exc:  # yfinance raises many different error types
                raw, last_error = None, str(exc)
            if raw is not None and not raw.empty:
                break
            if attempt < retries:
                time.sleep(2 * attempt)  # Yahoo intermittently rate-limits; back off

        if raw is None or raw.empty:
            raise ConnectionError(
                f"Yahoo Finance gave no data for {self.stock_symbol} after {retries} tries ({last_error})")

        df = self._normalise_api_frame(raw)
        self.validate_prices(df)
        self.data, self.source = df, "api"
        self.save_cache()
        return df

    @staticmethod
    def _normalise_api_frame(raw: pd.DataFrame) -> pd.DataFrame:
        """Flatten yfinance's (Price, Ticker) MultiIndex columns and tidy the index."""
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c) for c in df.columns]
        keep = [c for c in PRICE_COLUMNS + ["Adj Close"] if c in df.columns]
        df = df[keep].round(6)  # same precision as the cache, so online == offline results
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "Date"
        return df

    def load_csv(self, path: str | None = None) -> pd.DataFrame:
        """Load prices from a CSV file (the cache by default) and validate them."""
        path = path or self.file_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"No price file at {path}")
        df = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
        self.validate_prices(df)
        self.data = df
        self.source = "cache" if os.path.abspath(path) == os.path.abspath(self.price_cache_path) else "csv"
        return df

    def save_cache(self) -> str:
        """Write the current price data to CSV plus a JSON metadata file."""
        if self.data is None:
            raise ValueError("Nothing to cache: load data first.")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.data.to_csv(self.price_cache_path, float_format="%.6f")
        meta = {
            "symbol": self.stock_symbol,
            "source": "Yahoo Finance via yfinance",
            "start": self.start,
            "end": self.end,
            "rows": int(len(self.data)),
            "first_date": str(self.data.index.min().date()),
            "last_date": str(self.data.index.max().date()),
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with open(self.meta_cache_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return self.price_cache_path

    def load_prices(self, offline: bool = False, verbose: bool = True) -> pd.DataFrame:
        """Load prices from the API, falling back to the cache when offline.

        Args:
            offline: Skip the network entirely and read the cache.
            verbose: Print which source was used.
        """
        if not offline:
            try:
                df = self.load_api()
                if verbose:
                    print(f"[DataLoader] Downloaded {len(df)} rows for {self.stock_symbol} "
                          f"from Yahoo Finance (cache updated).")
                return df
            except ConnectionError as exc:
                if verbose:
                    print(f"[DataLoader] Online download failed ({exc}). Using offline cache.")
        df = self.load_csv(self.price_cache_path)
        if verbose:
            print(f"[DataLoader] Loaded {len(df)} cached rows for {self.stock_symbol} "
                  f"from {os.path.relpath(self.price_cache_path, PROJECT_ROOT)}.")
        return df

    def load_headlines(self, path: str | None = None) -> pd.DataFrame:
        """Load the pre-downloaded headline dataset (CSV or CSV.GZ).

        Returns a DataFrame with a ``date`` (datetime64) and ``headline`` (str) column.
        """
        path = path or self.headline_cache_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No headline file at {path}. Run: python scripts/fetch_headlines.py {self.stock_symbol}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No headline file at {path}")
        df = pd.read_csv(path)
        self.validate_headlines(df)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["headline"] = df["headline"].astype(str)
        df = df.dropna(subset=["date"])
        self.headlines = df[HEADLINE_COLUMNS].reset_index(drop=True)
        return self.headlines

    def cache_info(self) -> dict:
        """Return the JSON metadata of the cached price file (empty dict if none)."""
        try:
            with open(self.meta_cache_path, encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
