"""Menu-driven entry point for the stock-prediction-with-sentiment project.

Run ``python main.py`` for the interactive menu, or ``python main.py --auto``
to run every step once (useful for the demo and for checking the project works).
Add ``--offline`` to skip the network and use only the cached data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from src.data_loader import DataLoader
from src.evaluator import FIGURE_DIR, ModelEvaluator
from src.features import FeatureBuilder, PRICE_FEATURES, SENTIMENT_FEATURES, TARGET_RETURN
from src.predictor import MODEL_FACTORIES, ComparisonResult, StockPredictor
from src.preprocessor import Preprocessor
from src.sentiment import RELEVANCE_KEYWORDS, SentimentAnalyzer

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "outputs")
BASELINE = "Baseline (price only)"
ENHANCED = "Enhanced (price + sentiment)"
DISCLAIMER = ("Educational project only: these predictions are NOT investment advice. "
              "Daily stock prices are mostly unpredictable.")

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def heading(text: str) -> None:
    print("\n" + "=" * 72 + f"\n  {text}\n" + "=" * 72)


class App:
    """Holds the pipeline state between menu choices.

    Each ``step_*`` method runs one stage; if an earlier stage has not been
    run yet it is run automatically first, so any menu option is safe to pick.
    """

    def __init__(self, symbol: str = "AAPL", offline: bool = False, show_plots: bool = False,
                 test_size: float = 0.2, relevant_only: bool = False,
                 price_csv: str | None = None, headlines_csv: str | None = None):
        self.loader = DataLoader(symbol)
        self.symbol = self.loader.stock_symbol
        self.price_csv = price_csv
        self.headlines_csv = headlines_csv
        if price_csv:
            self.loader.file_path = price_csv
        self.offline = offline
        self.show_plots = show_plots
        self.test_size = test_size
        self.relevant_only = relevant_only

        self.prices: pd.DataFrame | None = None
        self.clean_prices: pd.DataFrame | None = None
        self.headlines: pd.DataFrame | None = None
        self.analyzer: SentimentAnalyzer | None = None
        self.daily_sentiment: pd.DataFrame | None = None
        self.builder: FeatureBuilder | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.comparison: ComparisonResult | None = None
        self.sentiment_attempted = False

    @property
    def sentiment_cache_path(self) -> str:
        return os.path.join(self.loader.cache_dir, f"{self.symbol}_sentiment_daily.csv")

    # ------------------------------------------------------------- 1. load
    def step_load(self) -> None:
        heading(f"1. Load data for {self.symbol}")
        if self.price_csv:
            self.prices = self.loader.load_csv(self.price_csv)
            print(f"   [DataLoader] Loaded {len(self.prices)} rows from custom CSV {self.price_csv}.")
        else:
            self.prices = self.loader.load_prices(offline=self.offline)
        p = self.prices
        print(f"   Prices:    {len(p)} trading days, {p.index.min().date()} to {p.index.max().date()} "
              f"(source: {self.loader.source})")
        try:
            self.headlines = self.loader.load_headlines(self.headlines_csv)
            h = self.headlines
            print(f"   Headlines: {len(h)} items, {h['date'].min().date()} to {h['date'].max().date()}")
        except FileNotFoundError as exc:
            self.headlines = None
            print(f"   Headlines: not available ({exc})")
        print("\n   Last 3 price rows:")
        print(p.tail(3).round(2).to_string())

    # -------------------------------------------------- 2. clean + features
    def step_features(self) -> None:
        if self.prices is None:
            self.step_load()
        heading("2. Clean data and engineer price features")
        pre = Preprocessor(self.prices)
        self.clean_prices = pre.run()
        for key, value in pre.report.items():
            print(f"   {key.replace('_', ' '):<28} {value}")
        self.builder = FeatureBuilder(self.clean_prices, self.daily_sentiment)
        feats = self.builder.build()
        print(f"\n   Feature table: {len(feats)} rows x {len(feats.columns)} columns")
        print(f"   Price features: {', '.join(PRICE_FEATURES)}")
        print("\n   Feature summary:")
        print(feats[PRICE_FEATURES].describe().T[["mean", "std", "min", "max"]].round(4).to_string())

        paths = [
            ModelEvaluator.plot_price_history(feats, self.symbol, show=self.show_plots),
            ModelEvaluator.plot_returns_distribution(feats, show=self.show_plots),
            ModelEvaluator.plot_correlation(feats, PRICE_FEATURES + [TARGET_RETURN], show=self.show_plots),
        ]
        self._print_figures(paths)

    # -------------------------------------------------------- 3. sentiment
    def step_sentiment(self) -> None:
        if self.clean_prices is None:
            self.step_features()
        heading("3. Headline sentiment analysis (NLP)")
        self.sentiment_attempted = True
        if self.headlines is None:
            print("   No headline file found, so the enhanced model cannot be built.")
            print(f"   Run: python scripts/fetch_headlines.py {self.symbol}")
            return

        self.analyzer = SentimentAnalyzer(self.headlines)
        if self.relevant_only and self.symbol in RELEVANCE_KEYWORDS:
            kept = self.analyzer.filter_relevant(RELEVANCE_KEYWORDS[self.symbol])
            print(f"   Relevance filter: kept {len(kept)} of {len(self.headlines)} headlines "
                  f"that mention {self.symbol} keywords")
        print(f"   Scoring {len(self.analyzer.headlines)} headlines with {self.analyzer.engine} ...")
        scores = self.analyzer.analyze()
        self.daily_sentiment = self.analyzer.aggregate_by_date()
        self.daily_sentiment.to_csv(self.sentiment_cache_path, float_format="%.5f")
        summary = self.analyzer.summary()
        with open(os.path.join(self.loader.cache_dir, f"{self.symbol}_sentiment_summary.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        for key, value in summary.items():
            print(f"   {key:<15} {value}")
        print("\n   Most positive and most negative headlines:")
        ranked = scores.sort_values("compound")
        for _, row in pd.concat([ranked.tail(3).iloc[::-1], ranked.head(3)]).iterrows():
            print(f"   {row['compound']:+.3f}  {row['date'].date()}  {row['clean_text'][:90]}")

        # Rebuild features with sentiment merged in by trading date.
        self.builder = FeatureBuilder(self.clean_prices, self.daily_sentiment)
        feats = self.builder.build()
        covered = (feats["news_count"] > 0).mean()
        print(f"\n   Trading days with at least one headline: {covered:.1%}")
        corr = feats[SENTIMENT_FEATURES + [TARGET_RETURN]].corr()[TARGET_RETURN].drop(TARGET_RETURN)
        print("   Correlation of sentiment features with next-day return:")
        print(corr.round(4).to_string())

        paths = [
            ModelEvaluator.plot_sentiment_distribution(scores, self.daily_sentiment, show=self.show_plots),
            ModelEvaluator.plot_correlation(feats, PRICE_FEATURES + SENTIMENT_FEATURES + [TARGET_RETURN],
                                            filename="feature_correlation_with_sentiment.png",
                                            show=self.show_plots),
        ]
        self._print_figures(paths)

    # ----------------------------------------------------------- 4. train
    def step_train(self) -> None:
        if self.builder is None or (not self.builder.sentiment_available and not self.sentiment_attempted):
            self.step_sentiment()
        if self.builder is None:
            return
        heading("4. Train and compare models")
        rows = self.builder.training_rows()
        if self.builder.sentiment_available:
            # Only evaluate the period the headline dataset covers.
            first_news = rows.index[rows["news_count"] > 0].min()
            rows = rows[rows.index >= first_news]
        self.train_df, self.test_df = Preprocessor.chronological_split(rows, self.test_size)
        print(f"   Train: {self.train_df.index.min().date()} to {self.train_df.index.max().date()} "
              f"({len(self.train_df)} days)")
        print(f"   Test:  {self.test_df.index.min().date()} to {self.test_df.index.max().date()} "
              f"({len(self.test_df)} days)")

        feature_sets = {BASELINE: self.builder.feature_columns(False)}
        if self.builder.sentiment_available:
            feature_sets[ENHANCED] = self.builder.feature_columns(True)
        print(f"   Models: {', '.join(MODEL_FACTORIES)}  x  {len(feature_sets)} feature sets\n")

        self.comparison = StockPredictor.compare_models(self.train_df, self.test_df, feature_sets)
        print(self.comparison.results.round(4).to_string(index=False))
        self._save_results()

    # -------------------------------------------------------- 5. evaluate
    def step_evaluate(self) -> None:
        if self.comparison is None:
            self.step_train()
        if self.comparison is None:
            return
        heading("5. Metrics and charts")
        res = self.comparison
        naive = res.results[res.results["feature_set"] == "-"].iloc[0]
        best_base = res.best(BASELINE)
        print(f"   Naive 'no change' baseline: RMSE ${naive['RMSE']:.3f}, MAE ${naive['MAE']:.3f}")
        print(f"   Best price-only model:      {best_base['model']} (RMSE ${best_base['RMSE']:.3f}, "
              f"direction {best_base['Direction_Acc']:.1%})")

        best = best_base
        if ENHANCED in set(res.results["feature_set"]):
            best_enh = res.best(ENHANCED)
            best = best_enh
            print(f"   Best price+sentiment model: {best_enh['model']} (RMSE ${best_enh['RMSE']:.3f}, "
                  f"direction {best_enh['Direction_Acc']:.1%})")
            effect = res.sentiment_effect(BASELINE, ENHANCED)
            print("\n   Effect of adding sentiment (RMSE change < 0 means sentiment helped):")
            print(effect.round(4).to_string())
            helped = int((effect["RMSE_change_%"] < 0).sum())
            dir_up = int((effect["DirAcc_enhanced"] > effect["DirAcc_baseline"]).sum())
            dir_gain = (effect["DirAcc_enhanced"] - effect["DirAcc_baseline"]).mean() * 100
            # ~1 standard error of an accuracy near 50% on this many test days
            noise = 100 * 0.5 / len(res.actual) ** 0.5
            print("\n   Verdict:")
            print(f"   - RMSE: sentiment lowered it for {helped} of {len(effect)} models "
                  f"(mean change {effect['RMSE_change_%'].mean():+.2f}%).")
            print(f"   - Direction: sentiment improved it for {dir_up} of {len(effect)} models "
                  f"(mean {dir_gain:+.1f} points; one standard error is about {noise:.1f} points).")
            margin = naive["RMSE"] - best_enh["RMSE"]
            print(f"   - The best enhanced model {'beats' if margin > 0 else 'does NOT beat'} the naive "
                  f"'no change' forecast on RMSE (margin ${margin:+.4f}).")

        key = (best["model"], best["feature_set"])
        model = res.models[key]
        if hasattr(model, "feature_importance"):
            print(f"\n   Top features for {best['model']} ({best['feature_set']}):")
            print(model.feature_importance().head(8).round(4).to_string())

        ev = ModelEvaluator(res.actual, res.predictions[key], label=f"{best['model']}, {best['feature_set']}")
        base_key = (best_base["model"], BASELINE)
        paths = [
            ev.plot_results(res.actual.index, show=self.show_plots,
                            baseline=res.predictions[base_key] if base_key != key else None,
                            baseline_label=f"{best_base['model']}, price only"),
            ModelEvaluator.plot_comparison(res.results, show=self.show_plots),
        ]
        last = res.actual.index[-60:]
        zoom = ModelEvaluator(res.actual.loc[last], res.predictions[key].loc[last], label=ev.label)
        paths.append(zoom.plot_results(last, filename="actual_vs_predicted_last60.png", show=self.show_plots))
        self._print_figures(paths)

    # ---------------------------------------------------------- 6. predict
    def step_predict(self) -> None:
        if self.comparison is None:
            self.step_train()
        if self.comparison is None:
            return
        heading("6. Next-day price prediction")
        latest = self.builder.latest_row()
        date, close = latest.index[0].date(), float(latest["Close"].iloc[0])
        print(f"   Latest trading day in data: {date}, close ${close:.2f}")
        all_rows = pd.concat([self.train_df, self.test_df])
        print("   (each model below is retrained on all labelled days before predicting)\n")
        for set_name in self.comparison.results["feature_set"].unique():
            if set_name == "-":
                continue
            best = self.comparison.best(set_name)
            cols = self.builder.feature_columns(set_name == ENHANCED)
            model = MODEL_FACTORIES[best["model"]]()
            model.train(all_rows[cols], all_rows[TARGET_RETURN])
            pred = float(model.predict_price(latest[cols], latest["Close"])[0])
            change = pred / close - 1
            print(f"   {set_name:<30} {best['model']:<18} -> ${pred:.2f} ({change:+.2%})")
        print(f"\n   {DISCLAIMER}")

    # ---------------------------------------------------------- helpers
    def _print_figures(self, paths) -> None:
        paths = [p for p in paths if p]
        if paths:
            print("\n   Saved figures:")
            for p in paths:
                print(f"     - {os.path.relpath(p, ROOT)}")

    def _save_results(self) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
        self.comparison.results.to_csv(csv_path, index=False, float_format="%.6f")
        preds = pd.DataFrame({"actual": self.comparison.actual})
        for (model, fset), series in self.comparison.predictions.items():
            preds[f"{model} | {fset}"] = series
        preds.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), float_format="%.4f")
        print(f"\n   Results saved to {os.path.relpath(csv_path, ROOT)} and outputs/test_predictions.csv")

    def run_all(self) -> None:
        for step in (self.step_load, self.step_features, self.step_sentiment,
                     self.step_train, self.step_evaluate, self.step_predict):
            step()

    def reconfigure_data(self) -> None:
        """Select another ticker or custom CSVs, then discard stale pipeline state."""
        print("\n   Enter a ticker symbol, or C to load custom CSV files.")
        selection = input("   Selection: ").strip()
        if not selection:
            print("   No change.")
            return

        if selection.lower() == "c":
            price_path = input("   Custom price CSV path (Enter to use this ticker's prices): ").strip()
            headlines_path = input(
                "   Custom headline CSV path (Enter to use this ticker's headlines): ").strip()
            self.price_csv = price_path or None
            self.headlines_csv = headlines_path or None
        else:
            self.symbol = selection.upper()
            self.price_csv = None
            self.headlines_csv = None

        self.loader = DataLoader(self.symbol)
        if self.price_csv:
            self.loader.file_path = self.price_csv
        self.prices = None
        self.clean_prices = None
        self.headlines = None
        self.analyzer = None
        self.daily_sentiment = None
        self.builder = None
        self.train_df = None
        self.test_df = None
        self.comparison = None
        self.sentiment_attempted = False
        print(f"   Data selection updated for {self.symbol}; previous pipeline results cleared.")

    # ------------------------------------------------------------ menu
    def menu(self) -> None:
        options = {
            "1": ("Load data (API or offline cache)", self.step_load),
            "2": ("Clean data and engineer features", self.step_features),
            "3": ("Run sentiment analysis on headlines", self.step_sentiment),
            "4": ("Train and compare models", self.step_train),
            "5": ("View metrics and charts", self.step_evaluate),
            "6": ("Predict next-day closing price", self.step_predict),
            "7": ("Run everything (1-6)", self.run_all),
            "8": ("Change stock or load custom CSV files", self.reconfigure_data),
            "0": ("Exit", None),
        }
        while True:
            heading(f"Stock Price Prediction with Market Sentiment  |  {self.symbol}"
                    f"{'  |  offline' if self.offline else ''}")
            for key, (label, _) in options.items():
                print(f"   {key}. {label}")
            choice = input("\n   Choose an option: ").strip()
            if choice == "0":
                print("   Goodbye.")
                return
            if choice not in options:
                print("   Invalid choice, please enter a number from the menu.")
                continue
            try:
                options[choice][1]()
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                print(f"\n   Error: {exc}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AI-based stock price prediction using market sentiment")
    parser.add_argument("--symbol", default="AAPL", help="ticker symbol (default AAPL)")
    parser.add_argument("--offline", action="store_true", help="do not use the network; read cached data only")
    parser.add_argument("--auto", action="store_true", help="run every step once without the menu")
    parser.add_argument("--show", action="store_true", help="open chart windows as well as saving PNGs")
    parser.add_argument("--test-size", type=float, default=0.2, help="share of days used for testing")
    parser.add_argument("--relevant-only", action="store_true",
                        help="score only headlines that mention the company (keyword filter)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    app = App(args.symbol, offline=args.offline, show_plots=args.show, test_size=args.test_size,
              relevant_only=args.relevant_only)
    try:
        if args.auto:
            app.run_all()
        else:
            app.menu()
    except KeyboardInterrupt:
        print("\n   Interrupted.")
    except FileNotFoundError as exc:
        print(f"\n   Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
