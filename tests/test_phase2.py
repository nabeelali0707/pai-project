"""Unit tests for Phase 2 deliverables:
- Walk-forward validation (TimeSeriesSplit)
- Statistical significance tests (Binomial and Diebold-Mariano)
- Hyperparameter tuning on training set only
- Sentiment ablation (plain VADER vs finance lexicon)
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluator import ModelEvaluator, binomial_test_direction, diebold_mariano_test
from src.features import FeatureBuilder, PRICE_FEATURES, SENTIMENT_FEATURES, TARGET_RETURN
from src.predictor import (
    MODEL_FACTORIES,
    ComparisonResult,
    GradientBoostingPredictor,
    LinearRegressionPredictor,
    NaivePredictor,
    RandomForestPredictor,
    RidgePredictor,
    StockPredictor,
    tune_hyperparameters,
)
from src.preprocessor import Preprocessor
from src.sentiment import SentimentAnalyzer, run_sentiment_ablation


# ------------------------------------------------------------- 2.1 Walk-Forward Validation
def test_walk_forward_validation_runs(prices):
    feats = FeatureBuilder(prices).build().dropna(subset=[TARGET_RETURN])
    feature_sets = {"price_only": PRICE_FEATURES[:3]}
    factories = {"Ridge": RidgePredictor}

    res = StockPredictor.compare_models(
        feats,
        feature_sets=feature_sets,
        predictor_factories=factories,
        walk_forward=True,
        n_splits=3,
    )

    assert isinstance(res, ComparisonResult)
    assert res.is_walk_forward is True
    assert "RMSE_std" in res.results.columns
    assert "MAE_std" in res.results.columns
    assert "Direction_Acc_std" in res.results.columns
    assert len(res.results) == 2  # Naive + Ridge

    fmt_df = res.formatted_table()
    assert "+/-" in fmt_df["RMSE ($)"].iloc[0]


# ------------------------------------------------------------- 2.2 Significance Checks
def test_binomial_test_direction():
    # 50 out of 100 should have p-value ~ 0.5 (or > 0.4)
    res_null = binomial_test_direction(50, 100, p0=0.5, alternative="greater")
    assert pytest.approx(res_null["accuracy"]) == 0.5
    assert res_null["p_value"] >= 0.4
    assert res_null["significant_05"] is False

    # 90 out of 100 is highly significant
    res_sig = binomial_test_direction(90, 100, p0=0.5, alternative="greater")
    assert res_sig["accuracy"] == 0.9
    assert res_sig["p_value"] < 0.001
    assert res_sig["significant_05"] is True


def test_diebold_mariano_test():
    rng = np.random.default_rng(42)
    # Identical errors -> DM stat 0, p-val 1.0
    errors_base = rng.normal(0, 1, 100)
    res_same = diebold_mariano_test(errors_base, errors_base)
    assert res_same["stat"] == 0.0
    assert res_same["p_value"] == 1.0
    assert res_same["significant_05"] is False

    # Enhanced model has much smaller errors -> positive stat, significant p-value
    errors_enh = errors_base * 0.1
    res_better = diebold_mariano_test(errors_base, errors_enh)
    assert res_better["stat"] > 0
    assert res_better["p_value"] < 0.05
    assert res_better["significant_05"] is True


def test_evaluator_direction_test():
    prev = np.array([100, 100, 100, 100, 100])
    actual = np.array([102, 103, 98, 105, 97])
    # Predicted got 4 out of 5 directions right
    predicted = np.array([101, 102, 99, 104, 101])
    ev = ModelEvaluator(actual, predicted)
    k, n = ev.direction_counts(prev)
    assert k == 4
    assert n == 5
    dtest = ev.direction_test(prev)
    assert dtest["k"] == 4
    assert dtest["n"] == 5


# ------------------------------------------------------------- 2.3 Hyperparameter Tuning
def test_tune_hyperparameters_train_only(prices):
    feats = FeatureBuilder(prices).build().dropna(subset=[TARGET_RETURN])
    train_df, test_df = Preprocessor.chronological_split(feats, 0.2)

    # Pass train_df ONLY
    cols = PRICE_FEATURES[:3]
    tuning = tune_hyperparameters(train_df, cols, n_splits=3, verbose=False)

    assert "Ridge Regression" in tuning
    assert "Random Forest" in tuning
    assert "Gradient Boosting" in tuning
    assert "alpha" in tuning["Ridge Regression"]["best_params"]
    assert "max_depth" in tuning["Random Forest"]["best_params"]
    assert "min_samples_leaf" in tuning["Random Forest"]["best_params"]
    assert "val_rmse" in tuning["Ridge Regression"]
    assert tuning["Ridge Regression"]["val_rmse"] > 0


# ------------------------------------------------------------- 2.4 Sentiment Ablation
def test_sentiment_analyzer_without_finance_lexicon(headlines):
    sa_fin = SentimentAnalyzer(headlines, use_vader=True, include_finance_lexicon=True)
    sa_plain = SentimentAnalyzer(headlines, use_vader=True, include_finance_lexicon=False)

    assert sa_fin.engine == "vader+finance"
    assert sa_plain.engine == "vader_plain"

    # A finance word like "outperform" should score higher with finance lexicon
    score_fin = sa_fin.score_text("analysts outperform target")
    score_plain = sa_plain.score_text("analysts outperform target")
    assert score_fin["compound"] >= score_plain["compound"]


def test_run_sentiment_ablation(prices, headlines):
    sa = SentimentAnalyzer(headlines)
    sa.analyze()
    ablation_df = run_sentiment_ablation(headlines, prices.iloc[:120], test_size=0.3,
                                         models=["Ridge Regression"])
    assert len(ablation_df) >= 3
    assert set(ablation_df["Sentiment Configuration"]) == {
        "Price only (no sentiment)",
        "Plain VADER (no finance words)",
        "VADER + Finance Lexicon (ours)",
    }
