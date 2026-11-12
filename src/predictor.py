"""BasePredictor / StockPredictor: train, predict and compare regression models.

Owner: Ali Abbas

Design notes:
    * ``BasePredictor`` is an abstract class; every model exposes the same
      ``train()`` / ``predict()`` interface (polymorphism), so the comparison
      loop treats a naive baseline, a linear model and a random forest alike.
    * Models learn the next-day *return* and we convert it to a price:
      ``predicted_close(t+1) = Close(t) * (1 + predicted_return)``.
      Prices trend upward over the years (AAPL ~$27 in 2015 vs ~$190 in 2023),
      so a model trained on old price levels cannot extrapolate them; returns
      are comparable across years. Metrics are still reported on prices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluator import ModelEvaluator
from .features import TARGET_PRICE, TARGET_RETURN

RANDOM_STATE = 42


class BasePredictor(ABC):
    """Common interface for every predictor."""

    name: str = "Base"

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> "BasePredictor":
        """Fit on features ``X`` and next-day returns ``y``."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted next-day returns for each row of ``X``."""

    def predict_price(self, X: pd.DataFrame, current_close: pd.Series) -> np.ndarray:
        """Convert predicted returns into next-day closing prices."""
        return np.asarray(current_close, dtype=float) * (1.0 + self.predict(X))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class NaivePredictor(BasePredictor):
    """Persistence baseline: tomorrow's close equals today's close (return 0).

    Any useful model has to beat this; for daily stock prices it is hard to.
    """

    name = "Naive (no change)"

    def train(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class StockPredictor(BasePredictor):
    """A scikit-learn regressor wrapped with feature scaling.

    Attributes:
        model: The fitted sklearn ``Pipeline`` (scaler + regressor).
        X_train / y_train: The data the model was last trained on.
        feature_columns: Column order used for training (reused at predict time).
    """

    name = "Stock model"

    def __init__(self, estimator=None, name: str | None = None):
        self.estimator = estimator if estimator is not None else self.default_estimator()
        if name:
            self.name = name
        self.model: Pipeline | None = None
        self.X_train: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.feature_columns: list[str] = []

    def default_estimator(self):
        return LinearRegression()

    def train(self, X: pd.DataFrame, y: pd.Series) -> "StockPredictor":
        self.feature_columns = list(X.columns)
        self.X_train, self.y_train = X, y
        self.model = Pipeline([("scale", StandardScaler()), ("reg", self.estimator)])
        self.model.fit(X.to_numpy(dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} has not been trained yet.")
        X = X[self.feature_columns]
        return self.model.predict(X.to_numpy(dtype=float))

    def feature_importance(self) -> pd.Series:
        """Coefficient magnitudes (linear) or impurity importances (trees)."""
        if self.model is None:
            raise RuntimeError("Train the model first.")
        reg = self.model.named_steps["reg"]
        if hasattr(reg, "feature_importances_"):
            values = reg.feature_importances_
        elif hasattr(reg, "coef_"):
            values = np.abs(np.ravel(reg.coef_))
        else:
            return pd.Series(dtype=float)
        return pd.Series(values, index=self.feature_columns).sort_values(ascending=False)

    # ----------------------------------------------------------- comparison
    @staticmethod
    def compare_models(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       feature_sets: dict[str, list[str]],
                       predictor_factories: dict[str, callable] | None = None,
                       include_naive: bool = True) -> "ComparisonResult":
        """Train every model on every feature set and evaluate on the test period.

        Args:
            train_df / test_df: Chronologically split feature tables.
            feature_sets: e.g. ``{"Baseline (price only)": [...], "Enhanced (+ sentiment)": [...]}``.
            predictor_factories: name -> zero-argument callable returning a
                fresh ``BasePredictor``. Defaults to :data:`MODEL_FACTORIES`.
            include_naive: Also evaluate the no-change baseline.
        """
        factories = predictor_factories or MODEL_FACTORIES
        actual = test_df[TARGET_PRICE].to_numpy(dtype=float)
        prev_close = test_df["Close"].to_numpy(dtype=float)
        rows, predictions, fitted = [], {}, {}

        def record(model_name, set_name, predictor, cols):
            pred_price = predictor.predict_price(test_df[cols], test_df["Close"])
            ev = ModelEvaluator(actual, pred_price, label=f"{model_name} | {set_name}")
            rows.append({"model": model_name, "feature_set": set_name, "n_features": len(cols),
                         **ev.summary(previous=prev_close)})
            predictions[(model_name, set_name)] = pd.Series(pred_price, index=test_df.index)
            fitted[(model_name, set_name)] = predictor

        if include_naive:
            naive = NaivePredictor()
            record(naive.name, "-", naive, [])

        for set_name, cols in feature_sets.items():
            for model_name, factory in factories.items():
                predictor = factory()
                predictor.train(train_df[cols], train_df[TARGET_RETURN])
                record(model_name, set_name, predictor, cols)

        results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
        return ComparisonResult(results=results, predictions=predictions,
                                actual=pd.Series(actual, index=test_df.index), models=fitted)


class LinearRegressionPredictor(StockPredictor):
    name = "Linear Regression"

    def default_estimator(self):
        return LinearRegression()


class RidgePredictor(StockPredictor):
    name = "Ridge Regression"

    def default_estimator(self):
        return Ridge(alpha=10.0)


class RandomForestPredictor(StockPredictor):
    name = "Random Forest"

    def default_estimator(self):
        # Shallow trees + large leaves: daily returns are noisy, deep trees overfit.
        return RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=25,
                                     random_state=RANDOM_STATE, n_jobs=-1)


class GradientBoostingPredictor(StockPredictor):
    name = "Gradient Boosting"

    def default_estimator(self):
        return GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03,
                                         subsample=0.8, random_state=RANDOM_STATE)


MODEL_FACTORIES: dict[str, callable] = {
    cls.name: cls for cls in (LinearRegressionPredictor, RidgePredictor,
                              RandomForestPredictor, GradientBoostingPredictor)
}


@dataclass
class ComparisonResult:
    """Everything produced by :meth:`StockPredictor.compare_models`."""

    results: pd.DataFrame
    predictions: dict = field(default_factory=dict)
    actual: pd.Series | None = None
    models: dict = field(default_factory=dict)

    def best(self, feature_set: str | None = None) -> pd.Series:
        """Lowest-RMSE row, optionally restricted to one feature set."""
        df = self.results
        if feature_set is not None:
            df = df[df["feature_set"] == feature_set]
        return df.sort_values("RMSE").iloc[0]

    def sentiment_effect(self, baseline: str, enhanced: str) -> pd.DataFrame:
        """Per-model RMSE/MAE change when sentiment features are added (negative = better)."""
        base = self.results[self.results["feature_set"] == baseline].set_index("model")
        enh = self.results[self.results["feature_set"] == enhanced].set_index("model")
        common = base.index.intersection(enh.index)
        out = pd.DataFrame({
            "RMSE_baseline": base.loc[common, "RMSE"],
            "RMSE_enhanced": enh.loc[common, "RMSE"],
            "MAE_baseline": base.loc[common, "MAE"],
            "MAE_enhanced": enh.loc[common, "MAE"],
            "DirAcc_baseline": base.loc[common, "Direction_Acc"],
            "DirAcc_enhanced": enh.loc[common, "Direction_Acc"],
        })
        out["RMSE_change_%"] = (out["RMSE_enhanced"] / out["RMSE_baseline"] - 1) * 100
        return out
