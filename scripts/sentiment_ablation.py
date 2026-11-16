"""Sentiment ablation: does our finance lexicon beat plain VADER?

Runs the same price + sentiment pipeline with three sentiment engines and
reports the best enhanced model for each, next to the price-only baseline:

    * VADER + finance lexicon (what the project uses)
    * plain VADER (no finance words)
    * the built-in lexicon fallback (no vaderSentiment install)

Usage:
    python scripts/sentiment_ablation.py
    python scripts/sentiment_ablation.py --symbol MSFT --relevant-only

Results are printed as a Markdown table and saved to
``outputs/sentiment_ablation.csv``.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_loader import DataLoader  # noqa: E402
from src.features import FeatureBuilder  # noqa: E402
from src.predictor import StockPredictor  # noqa: E402
from src.preprocessor import Preprocessor  # noqa: E402
from src.sentiment import RELEVANCE_KEYWORDS, SentimentAnalyzer  # noqa: E402

OUTPUT_DIR = os.path.join(ROOT, "outputs")
BASELINE = "price only"
ENHANCED = "price + sentiment"

VARIANTS = (
    ("VADER + finance lexicon", {"use_vader": True, "finance_lexicon": True}),
    ("Plain VADER", {"use_vader": True, "finance_lexicon": False}),
    ("Built-in lexicon", {"use_vader": False, "finance_lexicon": True}),
)


def enhanced_rows(prices: pd.DataFrame, daily: pd.DataFrame) -> tuple[FeatureBuilder, pd.DataFrame]:
    """Build features for the period the headlined days cover."""
    builder = FeatureBuilder(prices, daily)
    rows = builder.training_rows()
    first_news = rows.index[rows["news_count"] > 0].min()
    return builder, rows[rows.index >= first_news]


def run_variant(prices: pd.DataFrame, headlines: pd.DataFrame, symbol: str, params: dict,
                test_size: float, relevant_only: bool) -> dict:
    """Train the baseline and enhanced models for one sentiment engine."""
    analyzer = SentimentAnalyzer(headlines.copy(), **params)
    if relevant_only and symbol in RELEVANCE_KEYWORDS:
        analyzer.filter_relevant(RELEVANCE_KEYWORDS[symbol])
    analyzer.analyze()
    builder, rows = enhanced_rows(prices, analyzer.aggregate_by_date())
    train_df, test_df = Preprocessor.chronological_split(rows, test_size)
    feature_sets = {BASELINE: builder.feature_columns(False),
                    ENHANCED: builder.feature_columns(True)}
    res = StockPredictor.compare_models(train_df, test_df, feature_sets, include_naive=False)
    best_base, best_enh = res.best(BASELINE), res.best(ENHANCED)
    return {
        "variant": analyzer.engine,
        "label": next(name for name, p in VARIANTS if p == params),
        "best_base_model": best_base["model"],
        "base_RMSE": best_base["RMSE"],
        "base_DirAcc_%": best_base["Direction_Acc"] * 100,
        "best_model": best_enh["model"],
        "RMSE": best_enh["RMSE"],
        "DirAcc_%": best_enh["Direction_Acc"] * 100,
        "RMSE_change_%": (best_enh["RMSE"] / best_base["RMSE"] - 1) * 100,
        "DirAcc_change_pts": (best_enh["Direction_Acc"] - best_base["Direction_Acc"]) * 100,
    }


def to_markdown(frame: pd.DataFrame) -> str:
    header = ("| Sentiment method | Engine | Best enhanced model | RMSE ($) | Direction | "
              "Δ RMSE vs price only | Δ Direction |")
    lines = [header, "|" + "---|" * 7]
    for r in frame.itertuples():
        lines.append(f"| {r.label} | `{r.variant}` | {r.best_model} | {r.RMSE:.3f} | "
                     f"{r.DirAcc_:.1f}% | {getattr(r, '_8'):+.2f}% | "
                     f"{getattr(r, '_9'):+.1f} pts |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="AAPL", help="ticker to analyse (default AAPL)")
    parser.add_argument("--test-size", type=float, default=0.2, help="share of days used for testing")
    parser.add_argument("--relevant-only", action="store_true",
                        help="score only headlines that mention the company")
    args = parser.parse_args(argv)

    symbol = args.symbol.upper()
    loader = DataLoader(symbol)
    prices = Preprocessor(loader.load_prices(offline=True, verbose=False)).run()
    headlines = loader.load_headlines()

    rows = []
    for label, params in VARIANTS:
        row = run_variant(prices, headlines, symbol, params, args.test_size, args.relevant_only)
        rows.append(row)
        print(f"   {label:<26} best {row['best_model']:<18} RMSE ${row['RMSE']:.3f}  "
              f"direction {row['DirAcc_%']:.1f}%", flush=True)

    summary = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "sentiment_ablation.csv")
    summary.to_csv(csv_path, index=False, float_format="%.6f")

    view = summary[["label", "variant", "best_model", "RMSE", "DirAcc_%",
                    "RMSE_change_%", "DirAcc_change_pts"]]
    print("\n| Sentiment method | Engine | Best enhanced model | RMSE ($) | Direction | "
          "Δ RMSE vs price only | Δ Direction |")
    print("|" + "---|" * 7)
    for r in view.itertuples(index=False):
        print(f"| {r.label} | `{r.variant}` | {r.best_model} | {r.RMSE:.3f} | "
              f"{getattr(r, '_4'):.1f}% | {getattr(r, '_5'):+.2f}% | "
              f"{getattr(r, '_6'):+.1f} pts |")
    print(f"\nSaved {os.path.relpath(csv_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
