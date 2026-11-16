"""Run the full pipeline for every cached ticker and summarise the comparison.

This answers "does sentiment help?" per stock instead of only for AAPL, which
is what the proposal promised with "selected stocks".

Usage:
    python scripts/run_all_stocks.py                  # every cached ticker
    python scripts/run_all_stocks.py AAPL MSFT NVDA   # only these
    python scripts/run_all_stocks.py --relevant-only  # keyword-filtered headlines

The per-ticker pipeline is the same code the menu runs (``main.App``), so the
numbers here match ``python main.py --auto --offline --symbol <TICKER>``.
Results are printed as a Markdown table and saved to
``outputs/multi_stock_summary.csv``.
"""

from __future__ import annotations

import argparse
import os
import sys

# This script only saves charts, never opens windows: use the headless backend so
# it also behaves in CI and in background shells.
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from main import App, BASELINE, ENHANCED  # noqa: E402  (path set above)

OUTPUT_DIR = os.path.join(ROOT, "outputs")
CACHE_DIR = os.path.join(ROOT, "data", "cache")


def available_tickers() -> list[str]:
    """Tickers with a cached price file, sorted alphabetically."""
    suffix = "_prices.csv"
    return sorted(f[: -len(suffix)].upper() for f in os.listdir(CACHE_DIR)
                  if f.lower().endswith(suffix))


def summarise(app: App) -> dict:
    """One summary row: the best model of each feature set plus the naive baseline."""
    res = app.comparison
    naive = res.results[res.results["feature_set"] == "-"].iloc[0]
    base = res.best(BASELINE)
    row = {
        "ticker": app.symbol,
        "test_days": int(len(res.actual)),
        "naive_RMSE": naive["RMSE"],
        "baseline_model": base["model"],
        "baseline_RMSE": base["RMSE"],
        "baseline_DirAcc_%": base["Direction_Acc"] * 100,
    }
    if ENHANCED in set(res.results["feature_set"]):
        enh = res.best(ENHANCED)
        row.update({
            "enhanced_model": enh["model"],
            "enhanced_RMSE": enh["RMSE"],
            "enhanced_DirAcc_%": enh["Direction_Acc"] * 100,
            "RMSE_change_%": (enh["RMSE"] / base["RMSE"] - 1) * 100,
            "DirAcc_change_pts": (enh["Direction_Acc"] - base["Direction_Acc"]) * 100,
        })
    return row


def run_ticker(ticker: str, relevant_only: bool = False, verbose: bool = True) -> dict:
    """Run load -> clean/features -> sentiment -> train for one ticker (offline)."""
    app = App(ticker, offline=True, relevant_only=relevant_only)
    app.step_load()
    app.step_features()
    app.step_sentiment()
    app.step_train()
    if app.comparison is None:
        raise RuntimeError(f"{ticker}: no comparison was produced (is the price cache readable?)")
    return summarise(app)


def to_markdown(rows: list[dict]) -> str:
    """Render the summary as the Markdown table pasted into the README."""
    header = ("| Stock | Test days | Naive RMSE ($) | Best price-only model | RMSE ($) | Direction | "
              "Best +sentiment model | RMSE ($) | Direction | RMSE change | Direction change |")
    lines = [header, "|" + "---|" * 11]
    for r in rows:
        if "enhanced_RMSE" in r:
            lines.append(
                f"| {r['ticker']} | {r['test_days']} | {r['naive_RMSE']:.3f} | {r['baseline_model']} | "
                f"{r['baseline_RMSE']:.3f} | {r['baseline_DirAcc_%']:.1f}% | {r['enhanced_model']} | "
                f"{r['enhanced_RMSE']:.3f} | {r['enhanced_DirAcc_%']:.1f}% | "
                f"{r['RMSE_change_%']:+.2f}% | {r['DirAcc_change_pts']:+.1f} pts |")
        else:
            lines.append(f"| {r['ticker']} | {r['test_days']} | {r['naive_RMSE']:.3f} | "
                         f"{r['baseline_model']} | {r['baseline_RMSE']:.3f} | "
                         f"{r['baseline_DirAcc_%']:.1f}% | – | – | – | – | – |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*", help="tickers to run (default: every cached ticker)")
    parser.add_argument("--relevant-only", action="store_true",
                        help="score only headlines that mention the company")
    args = parser.parse_args(argv)

    tickers = [t.upper() for t in args.tickers] or available_tickers()
    if not tickers:
        print("No cached price files found. Run python scripts/fetch_headlines.py <SYM> first.")
        return 1

    rows, failed = [], []
    for ticker in tickers:
        print("\n" + "#" * 72 + f"\n## {ticker}\n" + "#" * 72)
        try:
            rows.append(run_ticker(ticker, relevant_only=args.relevant_only))
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"   {ticker} skipped: {exc}")
            failed.append(ticker)

    if not rows:
        return 1
    summary = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "multi_stock_summary.csv")
    summary.to_csv(csv_path, index=False, float_format="%.6f")
    print("\n" + to_markdown(rows))
    print(f"\nSaved {os.path.relpath(csv_path, ROOT)}")
    if failed:
        print(f"Skipped: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
