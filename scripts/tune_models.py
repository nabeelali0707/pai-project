"""Tune Ridge, Random Forest and Gradient Boosting on the training years only.

Search strategy (documented in README):
    * Only the chronological training portion is used (default: the first 80%
      of the labelled days). The test period is never touched.
    * Cross-validation is scikit-learn ``TimeSeriesSplit`` with 5 expanding
      folds, so every validation fold comes after its training fold.
    * Scoring is ``neg_root_mean_squared_error`` on next-day returns; the table
      reports the same number as a percentage.

Usage:
    python scripts/tune_models.py                  # AAPL, enhanced features
    python scripts/tune_models.py --symbol MSFT
    python scripts/tune_models.py --price-only

Best parameters are saved to ``outputs/tuning_results.csv``. Copy them into
``src/predictor.py`` defaults when they change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_loader import DataLoader  # noqa: E402
from src.features import FeatureBuilder, TARGET_RETURN  # noqa: E402
from src.preprocessor import Preprocessor  # noqa: E402
from src.sentiment import SentimentAnalyzer  # noqa: E402

OUTPUT_DIR = os.path.join(ROOT, "outputs")
CACHE_DIR = os.path.join(ROOT, "data", "cache")
RANDOM_STATE = 42

# name -> (unfitted estimator, parameter grid). Ranges are deliberately small:
# daily returns are noisy and a wide search would only fit noise.
SEARCHES: dict[str, tuple[object, dict]] = {
    "Ridge Regression": (
        Ridge(),
        {"reg__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0, 30000.0]},
    ),
    "Random Forest": (
        RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        {"reg__max_depth": [3, 5, 8, None], "reg__min_samples_leaf": [5, 15, 25, 50]},
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, subsample=0.8,
                                  random_state=RANDOM_STATE),
        {"reg__max_depth": [1, 2, 3], "reg__min_samples_leaf": [5, 15, 25]},
    ),
}


def load_daily_sentiment(loader: DataLoader) -> pd.DataFrame:
    """Reuse the cached daily sentiment if step 3 has run, otherwise score headlines."""
    path = os.path.join(CACHE_DIR, f"{loader.stock_symbol}_sentiment_daily.csv")
    if os.path.exists(path):
        daily = pd.read_csv(path, index_col="date", parse_dates=["date"])
        print(f"   Using cached sentiment {os.path.relpath(path, ROOT)}")
        return daily
    analyzer = SentimentAnalyzer(loader.load_headlines())
    analyzer.analyze()
    return analyzer.aggregate_by_date()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--price-only", action="store_true", help="tune on price features only")
    args = parser.parse_args(argv)

    symbol = args.symbol.upper()
    loader = DataLoader(symbol)
    prices = Preprocessor(loader.load_prices(offline=True, verbose=False)).run()
    daily = None if args.price_only else load_daily_sentiment(loader)
    builder = FeatureBuilder(prices, daily)
    rows = builder.training_rows()
    if daily is not None:
        first_news = rows.index[rows["news_count"] > 0].min()
        rows = rows[rows.index >= first_news]
        print(f"   Headline coverage starts {first_news.date()}")
    train_df, test_df = Preprocessor.chronological_split(rows, args.test_size)
    columns = builder.feature_columns(daily is not None)
    print(f"   Tuning {symbol} on {len(train_df)} days ({train_df.index.min().date()} to "
          f"{train_df.index.max().date()}); test period {test_df.index.min().date()} to "
          f"{test_df.index.max().date()} is untouched.\n")

    X = train_df[columns].to_numpy(dtype=float)
    y = train_df[TARGET_RETURN].to_numpy(dtype=float)
    cv = TimeSeriesSplit(n_splits=args.folds)
    results = []
    for name, (estimator, grid) in SEARCHES.items():
        search = GridSearchCV(Pipeline([("scale", StandardScaler()), ("reg", estimator)]),
                              grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        search.fit(X, y)
        rmse = -search.best_score_
        best = json.dumps({k.replace("reg__", ""): v for k, v in search.best_params_.items()})
        results.append({"model": name, "best_params": best, "cv_RMSE_%": rmse * 100,
                        "candidates": len(search.cv_results_["params"]), "folds": args.folds,
                        "train_days": len(train_df)})
        print(f"   {name:<18} CV RMSE {rmse * 100:.4f}%  best {best}", flush=True)

    summary = pd.DataFrame(results)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "tuning_results.csv")
    summary.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSaved {os.path.relpath(csv_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
