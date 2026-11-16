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
from sklearn.model_selection import TimeSeriesSplit
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
    def compare_models(train_df: pd.DataFrame, test_df: pd.DataFrame | None = None,
                       feature_sets: dict[str, list[str]] | None = None,
                       predictor_factories: dict[str, callable] | None = None,
                       include_naive: bool = True,
                       walk_forward: bool = False,
                       n_splits: int = 5) -> "ComparisonResult":
        """Train every model on every feature set and evaluate on test period or rolling folds.

        Args:
            train_df: Chronological training dataframe (or full table if walk_forward=True and test_df is None).
            test_df: Chronological test dataframe (optional if walk_forward=True).
            feature_sets: e.g. ``{"Baseline (price only)": [...], "Enhanced (price + sentiment)": [...]}``.
            predictor_factories: name -> zero-argument callable returning a fresh ``BasePredictor``.
            include_naive: Also evaluate the no-change baseline.
            walk_forward: If True, evaluates models across rolling folds using TimeSeriesSplit (Phase 2 Task 2.1).
            n_splits: Number of rolling folds (default 5).
        """
        feature_sets = feature_sets or {}
        factories = predictor_factories or MODEL_FACTORIES

        if walk_forward:
            combined = pd.concat([train_df, test_df]) if test_df is not None else train_df
            return StockPredictor._walk_forward_compare(
                data=combined,
                feature_sets=feature_sets,
                factories=factories,
                include_naive=include_naive,
                n_splits=n_splits,
            )

        if test_df is None:
            raise ValueError("test_df must be provided when walk_forward=False.")

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
                                actual=pd.Series(actual, index=test_df.index), models=fitted,
                                is_walk_forward=False)

    @staticmethod
    def _walk_forward_compare(data: pd.DataFrame,
                              feature_sets: dict[str, list[str]],
                              factories: dict[str, callable],
                              include_naive: bool = True,
                              n_splits: int = 5) -> "ComparisonResult":
        """Evaluate rolling performance across n_splits TimeSeriesSplit folds."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_rows = []
        oof_predictions: dict[tuple[str, str], list[pd.Series]] = {}
        oof_actual: list[pd.Series] = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(data), 1):
            train_fold = data.iloc[train_idx]
            test_fold = data.iloc[test_idx]
            actual = test_fold[TARGET_PRICE].to_numpy(dtype=float)
            prev_close = test_fold["Close"].to_numpy(dtype=float)

            if include_naive:
                naive = NaivePredictor()
                pred_price = naive.predict_price(test_fold[[]], test_fold["Close"])
                ev = ModelEvaluator(actual, pred_price, label=naive.name)
                s = ev.summary(previous=prev_close)
                fold_rows.append({"fold": fold, "model": naive.name, "feature_set": "-",
                                 "n_features": 0, **s})
                oof_predictions.setdefault((naive.name, "-"), []).append(
                    pd.Series(pred_price, index=test_fold.index)
                )

            for set_name, cols in feature_sets.items():
                for model_name, factory in factories.items():
                    predictor = factory()
                    predictor.train(train_fold[cols], train_fold[TARGET_RETURN])
                    pred_price = predictor.predict_price(test_fold[cols], test_fold["Close"])
                    ev = ModelEvaluator(actual, pred_price, label=f"{model_name} | {set_name}")
                    s = ev.summary(previous=prev_close)
                    fold_rows.append({"fold": fold, "model": model_name, "feature_set": set_name,
                                     "n_features": len(cols), **s})
                    oof_predictions.setdefault((model_name, set_name), []).append(
                        pd.Series(pred_price, index=test_fold.index)
                    )

            oof_actual.append(pd.Series(actual, index=test_fold.index))

        df_folds = pd.DataFrame(fold_rows)
        agg_rows = []
        for (m, fset), g in df_folds.groupby(["model", "feature_set"], sort=False):
            n_feat = int(g["n_features"].iloc[0])
            agg_rows.append({
                "model": m,
                "feature_set": fset,
                "n_features": n_feat,
                "MAE": float(g["MAE"].mean()),
                "MAE_std": float(g["MAE"].std(ddof=1) if len(g) > 1 else 0.0),
                "RMSE": float(g["RMSE"].mean()),
                "RMSE_std": float(g["RMSE"].std(ddof=1) if len(g) > 1 else 0.0),
                "R2": float(g["R2"].mean()),
                "R2_std": float(g["R2"].std(ddof=1) if len(g) > 1 else 0.0),
                "MAPE_%": float(g["MAPE_%"].mean()),
                "MAPE_%_std": float(g["MAPE_%"].std(ddof=1) if len(g) > 1 else 0.0),
                "Direction_Acc": float(g["Direction_Acc"].mean()),
                "Direction_Acc_std": float(g["Direction_Acc"].std(ddof=1) if len(g) > 1 else 0.0),
            })

        results = pd.DataFrame(agg_rows).sort_values("RMSE").reset_index(drop=True)
        combined_preds = {k: pd.concat(v) for k, v in oof_predictions.items()}
        combined_actual = pd.concat(oof_actual)
        return ComparisonResult(
            results=results,
            predictions=combined_preds,
            actual=combined_actual,
            models={},
            is_walk_forward=True,
            fold_results=df_folds,
            n_splits=n_splits
        )


class LinearRegressionPredictor(StockPredictor):
    name = "Linear Regression"

    def default_estimator(self):
        return LinearRegression()


class RidgePredictor(StockPredictor):
    name = "Ridge Regression"

    def __init__(self, alpha: float = 10.0, **kwargs):
        self.alpha = float(alpha)
        super().__init__(estimator=Ridge(alpha=self.alpha), name="Ridge Regression")

    def default_estimator(self):
        return Ridge(alpha=getattr(self, "alpha", 10.0))


class RandomForestPredictor(StockPredictor):
    name = "Random Forest"

    def __init__(self, n_estimators: int = 300, max_depth: int | None = 5,
                 min_samples_leaf: int = 25, random_state: int = RANDOM_STATE, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        super().__init__(estimator=estimator, name="Random Forest")

    def default_estimator(self):
        return RandomForestRegressor(
            n_estimators=getattr(self, "n_estimators", 300),
            max_depth=getattr(self, "max_depth", 5),
            min_samples_leaf=getattr(self, "min_samples_leaf", 25),
            random_state=getattr(self, "random_state", RANDOM_STATE),
            n_jobs=-1,
        )


class GradientBoostingPredictor(StockPredictor):
    name = "Gradient Boosting"

    def __init__(self, n_estimators: int = 200, max_depth: int = 2,
                 min_samples_leaf: int = 1, learning_rate: float = 0.03,
                 subsample: float = 0.8, random_state: int = RANDOM_STATE, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        estimator = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        super().__init__(estimator=estimator, name="Gradient Boosting")

    def default_estimator(self):
        return GradientBoostingRegressor(
            n_estimators=getattr(self, "n_estimators", 200),
            max_depth=getattr(self, "max_depth", 2),
            min_samples_leaf=getattr(self, "min_samples_leaf", 1),
            learning_rate=getattr(self, "learning_rate", 0.03),
            subsample=getattr(self, "subsample", 0.8),
            random_state=getattr(self, "random_state", RANDOM_STATE),
        )


MODEL_FACTORIES: dict[str, callable] = {
    cls.name: cls for cls in (LinearRegressionPredictor, RidgePredictor,
                              RandomForestPredictor, GradientBoostingPredictor)
}


def tune_hyperparameters(train_df: pd.DataFrame,
                         feature_cols: list[str],
                         n_splits: int = 5,
                         random_state: int = RANDOM_STATE,
                         verbose: bool = False) -> dict[str, dict]:
    """Tune Ridge, Random Forest, and Gradient Boosting hyperparameters using TimeSeriesSplit.

    STRICT COMPLIANCE WITH PHASE 2 TASK 2.3:
    Tuning is strictly performed using TimeSeriesSplit on the training set (train_df) only.
    The test dataset is never accessed or evaluated during tuning.

    Args:
        train_df: Chronological training dataframe only.
        feature_cols: Features to train on.
        n_splits: Number of rolling folds (default 5).
        random_state: Reproducibility seed.
        verbose: Print progress if True.

    Returns:
        dict with best parameters, validation RMSE, and search range per model.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 1. Ridge Regression: tune alpha
    ridge_alphas = [0.01, 0.1, 1.0, 10.0, 25.0, 50.0, 100.0, 200.0]
    best_ridge_alpha, best_ridge_rmse = None, float("inf")
    for alpha in ridge_alphas:
        rmses = []
        for tr_idx, val_idx in tscv.split(train_df):
            tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            model = RidgePredictor(alpha=alpha)
            model.train(tr[feature_cols], tr[TARGET_RETURN])
            pred_price = model.predict_price(val[feature_cols], val["Close"])
            rmses.append(float(np.sqrt(np.mean((val[TARGET_PRICE].to_numpy() - pred_price) ** 2))))
        mean_rmse = float(np.mean(rmses))
        if mean_rmse < best_ridge_rmse:
            best_ridge_rmse = mean_rmse
            best_ridge_alpha = alpha

    # 2. Random Forest: tune max_depth and min_samples_leaf
    rf_depths = [3, 5, 8]
    rf_leaves = [15, 25, 50]
    best_rf_params, best_rf_rmse = None, float("inf")
    for depth in rf_depths:
        for leaf in rf_leaves:
            rmses = []
            for tr_idx, val_idx in tscv.split(train_df):
                tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
                model = RandomForestPredictor(n_estimators=100, max_depth=depth,
                                             min_samples_leaf=leaf, random_state=random_state)
                model.train(tr[feature_cols], tr[TARGET_RETURN])
                pred_price = model.predict_price(val[feature_cols], val["Close"])
                rmses.append(float(np.sqrt(np.mean((val[TARGET_PRICE].to_numpy() - pred_price) ** 2))))
            mean_rmse = float(np.mean(rmses))
            if mean_rmse < best_rf_rmse:
                best_rf_rmse = mean_rmse
                best_rf_params = {"max_depth": depth, "min_samples_leaf": leaf}

    # 3. Gradient Boosting: tune max_depth, min_samples_leaf, learning_rate
    gb_depths = [2, 3, 4]
    gb_leaves = [10, 25, 50]
    gb_lrs = [0.02, 0.03, 0.05]
    best_gb_params, best_gb_rmse = None, float("inf")
    for depth in gb_depths:
        for leaf in gb_leaves:
            for lr in gb_lrs:
                rmses = []
                for tr_idx, val_idx in tscv.split(train_df):
                    tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
                    model = GradientBoostingPredictor(n_estimators=120, max_depth=depth,
                                                     min_samples_leaf=leaf, learning_rate=lr,
                                                     random_state=random_state)
                    model.train(tr[feature_cols], tr[TARGET_RETURN])
                    pred_price = model.predict_price(val[feature_cols], val["Close"])
                    rmses.append(float(np.sqrt(np.mean((val[TARGET_PRICE].to_numpy() - pred_price) ** 2))))
                mean_rmse = float(np.mean(rmses))
                if mean_rmse < best_gb_rmse:
                    best_gb_rmse = mean_rmse
                    best_gb_params = {"max_depth": depth, "min_samples_leaf": leaf, "learning_rate": lr}

    return {
        "Ridge Regression": {
            "search_range": {"alpha": ridge_alphas},
            "best_params": {"alpha": best_ridge_alpha},
            "val_rmse": round(best_ridge_rmse, 4),
        },
        "Random Forest": {
            "search_range": {"max_depth": rf_depths, "min_samples_leaf": rf_leaves},
            "best_params": best_rf_params,
            "val_rmse": round(best_rf_rmse, 4),
        },
        "Gradient Boosting": {
            "search_range": {"max_depth": gb_depths, "min_samples_leaf": gb_leaves, "learning_rate": gb_lrs},
            "best_params": best_gb_params,
            "val_rmse": round(best_gb_rmse, 4),
        },
    }


@dataclass
class ComparisonResult:
    """Everything produced by :meth:`StockPredictor.compare_models`."""

    results: pd.DataFrame
    predictions: dict = field(default_factory=dict)
    actual: pd.Series | None = None
    models: dict = field(default_factory=dict)
    is_walk_forward: bool = False
    fold_results: pd.DataFrame | None = None
    n_splits: int = 5

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

    def formatted_table(self) -> pd.DataFrame:
        """Return a formatted summary table (with mean +/- std if walk-forward)."""
        df = self.results.copy()
        if self.is_walk_forward and "RMSE_std" in df.columns:
            return pd.DataFrame({
                "model": df["model"],
                "feature_set": df["feature_set"],
                "MAE ($)": df.apply(lambda r: f"{r['MAE']:.3f} +/- {r['MAE_std']:.3f}", axis=1),
                "RMSE ($)": df.apply(lambda r: f"{r['RMSE']:.3f} +/- {r['RMSE_std']:.3f}", axis=1),
                "R2": df.apply(lambda r: f"{r['R2']:.4f} +/- {r['R2_std']:.4f}", axis=1),
                "Direction Acc": df.apply(
                    lambda r: f"{r['Direction_Acc']*100:.1f}% +/- {r['Direction_Acc_std']*100:.1f}%"
                    if pd.notna(r["Direction_Acc"]) else "-", axis=1
                ),
            })
        return pd.DataFrame({
            "model": df["model"],
            "feature_set": df["feature_set"],
            "MAE ($)": df["MAE"].map("{:.3f}".format),
            "RMSE ($)": df["RMSE"].map("{:.3f}".format),
            "R2": df["R2"].map("{:.4f}".format),
            "Direction Acc": df["Direction_Acc"].map(
                lambda v: f"{v*100:.1f}%" if pd.notna(v) else "-"
            ),
        })
