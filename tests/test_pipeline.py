"""Unit tests for every pipeline module (run with: python -m pytest)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import DataLoader, DataValidationError
from src.evaluator import ModelEvaluator
from src.features import FeatureBuilder, PRICE_FEATURES, SENTIMENT_FEATURES, TARGET_PRICE, TARGET_RETURN
from src.predictor import (BasePredictor, MODEL_FACTORIES, NaivePredictor, RidgePredictor,
                           StockPredictor)
from src.preprocessor import Preprocessor
from src.sentiment import SentimentAnalyzer


# ------------------------------------------------------------------ DataLoader
def test_offline_fallback_uses_cache(tmp_path, prices, monkeypatch):
    loader = DataLoader("TEST", cache_dir=str(tmp_path))
    loader.data = prices
    loader.save_cache()
    meta = json.loads((tmp_path / "TEST_prices_meta.json").read_text())
    assert meta["rows"] == len(prices)

    def fail(*args, **kwargs):
        raise ConnectionError("network down")
    monkeypatch.setattr(DataLoader, "load_api", fail)

    fresh = DataLoader("TEST", cache_dir=str(tmp_path))
    df = fresh.load_prices(verbose=False)
    assert fresh.source == "cache"
    assert len(df) == len(prices)
    np.testing.assert_allclose(df["Close"].to_numpy(), prices["Close"].to_numpy(), rtol=1e-6)


def test_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataLoader("NOPE", cache_dir=str(tmp_path)).load_prices(offline=True, verbose=False)


def test_validation_rejects_missing_columns(prices):
    with pytest.raises(DataValidationError):
        DataLoader.validate_prices(prices.drop(columns=["Close"]))
    with pytest.raises(DataValidationError):
        DataLoader.validate_headlines(pd.DataFrame({"date": ["2020-01-01"]}))


def test_normalise_multiindex_columns(prices):
    raw = prices.copy()
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["TEST"]], names=["Price", "Ticker"])
    df = DataLoader._normalise_api_frame(raw)
    assert "Close" in df.columns and not isinstance(df.columns, pd.MultiIndex)


# ---------------------------------------------------------------- Preprocessor
def test_clean_and_missing(prices):
    dirty = pd.concat([prices, prices.iloc[[5]]])          # duplicate date
    dirty.iloc[10, dirty.columns.get_loc("Close")] = -1     # invalid price
    dirty.iloc[20, dirty.columns.get_loc("Volume")] = np.nan
    pre = Preprocessor(dirty)
    out = pre.run()
    assert out.index.is_unique and out.index.is_monotonic_increasing
    assert (out["Close"] > 0).all() and not out.isna().any().any()
    assert pre.report["duplicate_dates_removed"] == 1
    assert pre.report["invalid_price_rows"] == 1


def test_chronological_split_has_no_overlap(prices):
    train, test = Preprocessor.chronological_split(prices, 0.25)
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(prices)
    with pytest.raises(ValueError):
        Preprocessor.chronological_split(prices, 1.5)


# ------------------------------------------------------------ SentimentAnalyzer
def test_preprocess_text_removes_noise():
    text = "Apple (NASDAQ: AAPL) jumps!  See https://example.com/x"
    assert SentimentAnalyzer.preprocess_text(text) == "apple jumps! see"
    assert SentimentAnalyzer.preprocess_text(None) == ""


@pytest.mark.parametrize("use_vader", [True, False])
def test_sentiment_polarity(headlines, use_vader):
    sa = SentimentAnalyzer(headlines, use_vader=use_vader)
    scores = sa.analyze()
    assert scores.loc[0, "compound"] > 0.05            # surge / record
    assert scores.loc[1, "label"] == "positive"        # upgrade / bullish
    assert scores.loc[2, "label"] == "negative"        # downgraded / slump
    daily = sa.aggregate_by_date()
    assert daily.loc[pd.Timestamp("2020-01-04"), "news_count"] == 2
    assert set(daily.columns) >= {"sentiment_mean", "positive_ratio", "negative_ratio"}


def test_finance_lexicon_negation():
    sa = SentimentAnalyzer(use_vader=False)
    assert sa.score_text("shares did not surge")["compound"] < 0


# ------------------------------------------------------------- FeatureBuilder
def test_features_and_target_alignment(prices):
    fb = FeatureBuilder(prices)
    feats = fb.build()
    assert not feats[PRICE_FEATURES].isna().any().any()
    # return_1d is computed with NumPy; check it against pandas
    expected = prices["Close"].pct_change().loc[feats.index]
    np.testing.assert_allclose(feats["return_1d"], expected, rtol=1e-10)
    # target on day t is the close on day t+1
    nxt = prices["Close"].shift(-1).loc[feats.index]
    pd.testing.assert_series_equal(feats[TARGET_PRICE], nxt, check_names=False)
    assert pd.isna(feats[TARGET_RETURN].iloc[-1])          # latest day: unknown future
    assert len(fb.training_rows()) == len(feats) - 1


def test_weekend_news_moves_to_next_trading_day(prices, headlines):
    sa = SentimentAnalyzer(headlines)
    sa.analyze()
    daily = sa.aggregate_by_date()
    fb = FeatureBuilder(prices.loc[:"2020-03-31"], daily)
    fb.add_returns(); fb.add_moving_averages(); fb.merge_features()
    df = fb.price_data
    # 2020-01-04 is a Saturday and 2020-01-06 a Monday: 2 + 1 headlines land on Monday
    assert pd.Timestamp("2020-01-04") not in df.index
    assert df.loc["2020-01-06", "news_count"] == 3
    assert df.loc["2020-01-03", "news_count"] == 0          # no news before -> neutral
    assert df.loc["2020-01-03", "sentiment_mean"] == 0
    assert all(c in df for c in SENTIMENT_FEATURES)


# ------------------------------------------------------------- Predictor
def test_polymorphic_predictors(prices):
    feats = FeatureBuilder(prices).build().dropna(subset=[TARGET_RETURN])
    train, test = Preprocessor.chronological_split(feats)
    models = [NaivePredictor()] + [factory() for factory in MODEL_FACTORIES.values()]
    for model in models:
        assert isinstance(model, BasePredictor)
        model.train(train[PRICE_FEATURES], train[TARGET_RETURN])
        pred = model.predict_price(test[PRICE_FEATURES], test["Close"])
        assert pred.shape == (len(test),) and np.isfinite(pred).all()
    np.testing.assert_allclose(NaivePredictor().predict_price(test[PRICE_FEATURES], test["Close"]),
                               test["Close"])


def test_untrained_model_raises(prices):
    with pytest.raises(RuntimeError):
        RidgePredictor().predict(prices)


def test_compare_models(prices, headlines):
    feats = FeatureBuilder(prices).build().dropna(subset=[TARGET_RETURN])
    train, test = Preprocessor.chronological_split(feats)
    res = StockPredictor.compare_models(train, test, {"price": PRICE_FEATURES},
                                        {"Ridge": RidgePredictor})
    assert set(res.results["model"]) == {"Ridge", "Naive (no change)"}
    assert {"MAE", "RMSE", "R2", "Direction_Acc"} <= set(res.results.columns)
    assert len(res.predictions[("Ridge", "price")]) == len(test)


# ------------------------------------------------------------- Evaluator
def test_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    actual = rng.normal(100, 5, 200)
    predicted = actual + rng.normal(0, 1, 200)
    ev = ModelEvaluator(actual, predicted)
    assert ev.mae() == pytest.approx(mean_absolute_error(actual, predicted))
    assert ev.rmse() == pytest.approx(np.sqrt(mean_squared_error(actual, predicted)))
    assert ev.r2() == pytest.approx(r2_score(actual, predicted))


def test_direction_accuracy():
    prev = np.array([10, 10, 10, 10.0])
    actual = np.array([11, 9, 11, 9.0])
    ev = ModelEvaluator(actual, np.array([12, 8, 9, 11.0]))
    assert ev.direction_accuracy(prev) == 0.5
    assert np.isnan(ModelEvaluator(actual, prev).direction_accuracy(prev))  # never moves


def test_evaluator_rejects_bad_input():
    with pytest.raises(ValueError):
        ModelEvaluator([1, 2], [1])


def test_plots_are_saved(tmp_path, monkeypatch, prices):
    import src.evaluator as evaluator
    monkeypatch.setattr(evaluator, "FIGURE_DIR", str(tmp_path))
    ev = ModelEvaluator(prices["Close"], prices["Close"] * 1.01)
    path = ev.plot_results(prices.index, filename="t.png")
    assert (tmp_path / "t.png").exists() and path.endswith("t.png")
