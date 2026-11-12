"""One-time download of the financial-headline dataset into the offline cache.

Source: "Tech-Stocks-News" by David-Chew-HL on Hugging Face (Apache-2.0)
        https://huggingface.co/datasets/David-Chew-HL/Tech-Stocks-News
        News for AAPL, AMZN, TSLA, GOOG, MSFT, META, BABA, NVDA (2015-01-01 to
        2024-01-30), originally pulled from the Alpaca Markets news API.

The raw per-ticker files are large (50-180 MB) because they contain article
summaries. This script keeps only ``date`` and the first ~300 characters of each
item (the headline-like opening), de-duplicates, and writes a small gzip CSV to
``data/cache/<SYMBOL>_headlines.csv.gz`` that the application reads offline.

Usage:
    python scripts/fetch_headlines.py AAPL                 # download + trim
    python scripts/fetch_headlines.py AAPL --raw file.csv  # trim an existing download
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
CACHE_DIR = os.path.join(ROOT, "data", "cache")
URL = "https://huggingface.co/datasets/David-Chew-HL/Tech-Stocks-News/resolve/main/{sym}_Alpaca_NLP.csv"
SUPPORTED = {"AAPL", "AMZN", "TSLA", "GOOG", "MSFT", "META", "BABA", "NVDA"}
MAX_CHARS = 300


def download(symbol: str) -> str:
    """Download the raw CSV for ``symbol`` into data/raw and return its path."""
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, f"{symbol}_Alpaca_NLP.csv")
    if os.path.exists(dest):
        print(f"Using existing download {dest}")
        return dest
    url = URL.format(sym=symbol)
    print(f"Downloading {url} (this file is large; it may take a while)...")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def shorten(text: str, limit: int = MAX_CHARS) -> str:
    """Collapse whitespace and cut at a word boundary near ``limit`` characters."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut


def trim(raw_path: str, symbol: str) -> str:
    """Reduce the raw file to (date, headline) and save it to the cache."""
    df = pd.read_csv(raw_path, usecols=["headline", "date"])
    rows_in = len(df)
    df = df.dropna(subset=["headline", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    df["headline"] = df["headline"].map(shorten)
    df = df[df["headline"].str.len() >= 15]
    df = df.drop_duplicates(subset=["date", "headline"]).sort_values("date")

    os.makedirs(CACHE_DIR, exist_ok=True)
    out = os.path.join(CACHE_DIR, f"{symbol}_headlines.csv.gz")
    df[["date", "headline"]].to_csv(out, index=False, compression="gzip")
    size_mb = os.path.getsize(out) / 1e6
    print(f"{rows_in} raw rows -> {len(df)} headlines "
          f"({df['date'].min()} to {df['date'].max()}), saved {out} ({size_mb:.1f} MB)")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("symbol", nargs="?", default="AAPL")
    parser.add_argument("--raw", help="path to an already downloaded raw CSV")
    args = parser.parse_args(argv)
    symbol = args.symbol.upper()
    if symbol not in SUPPORTED and not args.raw:
        print(f"{symbol} is not in the dataset. Choose one of: {sorted(SUPPORTED)}")
        return 1
    raw = args.raw or download(symbol)
    trim(raw, symbol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
