"""SentimentAnalyzer: turns financial headlines into daily sentiment features.

Owner: Siraj Udin

Pipeline:
    preprocess_text()  -> normalise each headline (lower-case, strip URLs,
                          exchange tags such as "(nasdaq: aapl)", noise)
    analyze()          -> score each headline with VADER, extended with a small
                          finance-specific lexicon (e.g. "downgrade", "bullish")
    aggregate_by_date()-> one row per calendar date: mean score, counts, and the
                          share of positive / negative headlines

If the ``vaderSentiment`` package is not installed, a built-in lexicon scorer
with the same output columns is used instead, so the program still runs.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

# Finance-specific word scores (VADER scale: about -4 .. +4). General-purpose
# lexicons miss or misjudge many market terms ("bearish", "downgrade", "miss").
FINANCE_LEXICON: dict[str, float] = {
    # positive
    "bullish": 2.2, "upgrade": 1.8, "upgraded": 1.8, "upgrades": 1.8,
    "outperform": 1.8, "outperformed": 1.8, "outperforms": 1.8, "overweight": 1.2,
    "surge": 1.9, "surged": 1.9, "surges": 1.9, "soar": 2.0, "soared": 2.0, "soars": 2.0,
    "rally": 1.7, "rallied": 1.7, "rallies": 1.7, "jump": 1.3, "jumped": 1.3, "jumps": 1.3,
    "gain": 1.3, "gained": 1.3, "gains": 1.3, "rise": 1.0, "rises": 1.0, "rose": 1.0,
    "beat": 1.5, "tops": 1.2, "record": 1.2, "profit": 1.3, "profits": 1.3,
    "profitable": 1.6, "growth": 1.4, "buyback": 1.3, "dividend": 0.9, "upside": 1.5,
    "rebound": 1.4, "rebounds": 1.4, "strong": 1.5, "stronger": 1.5, "buy": 1.0,
    # negative
    "bearish": -2.2, "downgrade": -1.8, "downgraded": -1.8, "downgrades": -1.8,
    "underperform": -1.8, "underperformed": -1.8, "underweight": -1.2,
    "plunge": -2.2, "plunged": -2.2, "plunges": -2.2, "tumble": -1.9, "tumbled": -1.9,
    "tumbles": -1.9, "slump": -1.8, "slumped": -1.8, "slumps": -1.8, "selloff": -1.9,
    "sell-off": -1.9, "drop": -1.3, "dropped": -1.3, "drops": -1.3, "fall": -1.2,
    "fell": -1.2, "falls": -1.2, "decline": -1.3, "declined": -1.3, "declines": -1.3,
    "miss": -1.5, "missed": -1.5, "misses": -1.5, "loss": -1.6, "losses": -1.6,
    "lawsuit": -1.6, "sued": -1.6, "probe": -1.3, "antitrust": -1.1, "fine": -0.6,
    "fined": -1.5, "recall": -1.5, "layoffs": -1.6, "downside": -1.5, "weak": -1.5,
    "weaker": -1.5, "sell": -1.0, "shortage": -1.3, "slowdown": -1.5, "warning": -1.3,
    "volatile": -0.8, "volatility": -0.6, "risk": -0.7, "risks": -0.7,
}

_GENERIC_LEXICON: dict[str, float] = {
    "good": 1.9, "great": 3.1, "best": 3.2, "positive": 2.6, "success": 2.7,
    "successful": 2.8, "win": 2.8, "wins": 2.7, "happy": 2.7, "improve": 1.9,
    "improved": 2.1, "optimistic": 1.3, "boost": 1.7, "boosts": 1.7,
    "bad": -2.5, "worst": -3.1, "negative": -2.7, "fail": -2.5, "fails": -2.4,
    "failure": -2.6, "crash": -2.7, "crisis": -3.1, "fear": -2.2, "fears": -2.2,
    "concern": -1.4, "concerns": -1.4, "worry": -1.9, "worries": -1.9, "problem": -1.7,
    "problems": -1.7, "cut": -1.1, "cuts": -1.1, "pessimistic": -1.6,
}
_NEGATIONS = {"not", "no", "never", "without", "isn't", "wasn't", "doesn't", "didn't", "won't", "cannot"}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EXCHANGE_TAG_RE = re.compile(r"\(\s*(?:nasdaq|nyse|amex|otc|nysearca|otcmkts)\s*:\s*[a-z.\s]+\)", re.I)
_NON_TEXT_RE = re.compile(r"[^a-z0-9$%.,!?'\-\s]")
_SPACE_RE = re.compile(r"\s+")

RELEVANCE_KEYWORDS: dict[str, list[str]] = {
    "AAPL": ["apple", "aapl", "iphone", "ipad", "mac", "macbook", "ios", "app store",
             "tim cook", "airpods", "apple watch", "vision pro"],
    "MSFT": ["microsoft", "msft", "windows", "azure", "xbox", "satya nadella", "openai"],
    "AMZN": ["amazon", "amzn", "aws", "prime", "alexa", "jeff bezos", "andy jassy"],
    "GOOG": ["google", "goog", "googl", "alphabet", "youtube", "android", "sundar pichai"],
    "META": ["meta", "facebook", "instagram", "whatsapp", "zuckerberg", "fb"],
    "TSLA": ["tesla", "tsla", "elon musk", "model 3", "model y", "cybertruck"],
    "NVDA": ["nvidia", "nvda", "geforce", "jensen huang", "gpu", "gpus"],
    "BABA": ["alibaba", "baba", "taobao", "alipay", "jack ma"],
}

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def _load_vader(include_finance_lexicon: bool = True):
    """Return a VADER analyser, optionally extended with FINANCE_LEXICON, or None if unavailable."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return None
    analyser = SentimentIntensityAnalyzer()
    if include_finance_lexicon:
        analyser.lexicon.update(FINANCE_LEXICON)
    return analyser


class SentimentAnalyzer:
    """Score financial headlines and aggregate the scores per date.

    Attributes:
        headlines: DataFrame with ``date`` and ``headline`` columns.
        sentiment_scores: Per-headline scores after :meth:`analyze`.
        daily_sentiment: Per-date aggregates after :meth:`aggregate_by_date`.
        engine: ``"vader+finance"``, ``"vader_plain"``, or ``"lexicon"`` (fallback).
    """

    def __init__(self, headlines: pd.DataFrame | None = None, use_vader: bool = True,
                 include_finance_lexicon: bool = True):
        self.headlines = headlines
        self.include_finance_lexicon = include_finance_lexicon
        self.sentiment_scores: pd.DataFrame | None = None
        self.daily_sentiment: pd.DataFrame | None = None
        self._vader = _load_vader(include_finance_lexicon=include_finance_lexicon) if use_vader else None
        if self._vader:
            self.engine = "vader+finance" if include_finance_lexicon else "vader_plain"
        else:
            self.engine = "lexicon"

    # ------------------------------------------------------------- text prep
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Normalise a headline for scoring.

        Lower-cases, removes URLs and exchange tags like "(nasdaq: aapl)",
        drops unusual symbols and collapses whitespace. Sentiment-bearing
        punctuation (! ?) and negations are kept because VADER uses them.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
        text = _URL_RE.sub(" ", text)
        text = _EXCHANGE_TAG_RE.sub(" ", text)
        text = _NON_TEXT_RE.sub(" ", text)
        return _SPACE_RE.sub(" ", text).strip()

    def filter_relevant(self, keywords: list[str]) -> pd.DataFrame:
        """Keep only headlines that mention at least one keyword (whole word, any case).

        News feeds tag an article with every ticker it touches, so many items
        filed under one stock are really about another company.
        """
        if self.headlines is None:
            raise ValueError("No headlines loaded.")
        pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, keywords)) + r")\b", re.I)
        is_relevant = lambda text: bool(pattern.search(str(text)))  # noqa: E731 (lab 04: lambda)
        keep = list(filter(lambda pair: is_relevant(pair[1]), enumerate(self.headlines["headline"])))
        self.headlines = self.headlines.iloc[[i for i, _ in keep]].reset_index(drop=True)
        self.sentiment_scores = self.daily_sentiment = None
        return self.headlines

    # --------------------------------------------------------------- scoring
    def score_text(self, text: str) -> dict[str, float]:
        """Return ``{"compound", "pos", "neg", "neu"}`` for one cleaned headline."""
        if not text:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        if self._vader is not None:
            return self._vader.polarity_scores(text)
        return self._lexicon_score(text, include_finance=self.include_finance_lexicon)

    @staticmethod
    def _lexicon_score(text: str, include_finance: bool = True) -> dict[str, float]:
        """Fallback scorer: sums word valences (flipping after a negation)."""
        lexicon = {**_GENERIC_LEXICON, **(FINANCE_LEXICON if include_finance else {})}
        tokens = re.findall(r"[a-z][a-z'\-]*", text)
        valences = []
        for i, tok in enumerate(tokens):
            if tok in lexicon:
                value = lexicon[tok]
                if any(t in _NEGATIONS for t in tokens[max(0, i - 3):i]):
                    value *= -0.74  # same damping VADER applies to negated words
                valences.append(value)
        total = sum(valences)
        compound = total / math.sqrt(total * total + 15) if valences else 0.0
        pos = sum(v for v in valences if v > 0)
        neg = -sum(v for v in valences if v < 0)
        neutral_words = max(len(tokens) - len(valences), 0)
        denom = pos + neg + neutral_words or 1
        return {"compound": round(compound, 4), "pos": pos / denom,
                "neg": neg / denom, "neu": neutral_words / denom}

    @staticmethod
    def label(compound: float) -> str:
        """Map a compound score to ``positive`` / ``negative`` / ``neutral``."""
        if compound >= POSITIVE_THRESHOLD:
            return "positive"
        if compound <= NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"

    def analyze(self, headlines: pd.DataFrame | None = None) -> pd.DataFrame:
        """Clean and score every headline.

        Returns a DataFrame with ``date, headline, clean_text, compound, pos,
        neg, neu, label`` and stores it in :attr:`sentiment_scores`.
        """
        if headlines is not None:
            self.headlines = headlines
        if self.headlines is None or self.headlines.empty:
            raise ValueError("No headlines to analyse.")

        df = self.headlines[["date", "headline"]].copy()
        df["clean_text"] = list(map(self.preprocess_text, df["headline"]))
        df = df[df["clean_text"].map(len) > 0]  # drop rows that were only noise

        scores = pd.DataFrame(list(map(self.score_text, df["clean_text"])), index=df.index)
        df = df.join(scores[["compound", "pos", "neg", "neu"]])
        df["label"] = df["compound"].map(self.label)
        self.sentiment_scores = df.reset_index(drop=True)
        return self.sentiment_scores

    def aggregate_by_date(self) -> pd.DataFrame:
        """Aggregate per-headline scores into one row per calendar date.

        Columns: ``sentiment_mean`` (mean compound), ``sentiment_std``,
        ``news_count``, ``positive_ratio``, ``negative_ratio``.
        """
        if self.sentiment_scores is None:
            self.analyze()
        df = self.sentiment_scores.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["is_pos"] = (df["label"] == "positive").astype(float)
        df["is_neg"] = (df["label"] == "negative").astype(float)

        daily = df.groupby("date").agg(
            sentiment_mean=("compound", "mean"),
            sentiment_std=("compound", "std"),
            news_count=("compound", "size"),
            positive_ratio=("is_pos", "mean"),
            negative_ratio=("is_neg", "mean"),
        )
        daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
        daily.index.name = "date"
        self.daily_sentiment = daily
        return daily

    def summary(self) -> dict:
        """Headline-level summary statistics for printing in the menu."""
        if self.sentiment_scores is None:
            return {}
        s = self.sentiment_scores
        counts = s["label"].value_counts()
        return {
            "engine": self.engine,
            "headlines": int(len(s)),
            "days": int(s["date"].nunique()),
            "mean_compound": round(float(np.mean(s["compound"])), 4),
            "positive": int(counts.get("positive", 0)),
            "neutral": int(counts.get("neutral", 0)),
            "negative": int(counts.get("negative", 0)),
        }


def run_sentiment_ablation(headlines: pd.DataFrame, clean_prices: pd.DataFrame,
                           test_size: float = 0.2,
                           models: list[str] | None = None) -> pd.DataFrame:
    """Compare performance across Plain VADER vs VADER + Finance Lexicon vs Price-Only.

    Demonstrates whether extending VADER with domain-specific financial terms
    actually improves predictive performance.

    Args:
        headlines: DataFrame containing raw headlines.
        clean_prices: Preprocessed OHLCV price DataFrame.
        test_size: Chronological test fraction (default 0.2).
        models: List of model names to test (default Ridge and Gradient Boosting).

    Returns:
        pd.DataFrame comparing the sentiment configurations by RMSE and Direction Accuracy.
    """
    from .features import FeatureBuilder
    from .preprocessor import Preprocessor
    from .predictor import RidgePredictor, GradientBoostingPredictor, NaivePredictor

    target_models = {
        "Ridge Regression": RidgePredictor,
        "Gradient Boosting": GradientBoostingPredictor,
    }
    if models is not None:
        target_models = {k: v for k, v in target_models.items() if k in models}

    configs = [
        ("Price only (no sentiment)", None),
        ("Plain VADER (no finance words)", False),
        ("VADER + Finance Lexicon (ours)", True),
    ]

    rows = []
    for label, fin_lex in configs:
        if fin_lex is None:
            daily_sent = None
            features_with_sentiment = False
        else:
            sa = SentimentAnalyzer(headlines.copy(), use_vader=True, include_finance_lexicon=fin_lex)
            sa.analyze()
            daily_sent = sa.aggregate_by_date()
            features_with_sentiment = True

        fb = FeatureBuilder(clean_prices, daily_sent)
        feats = fb.build()
        train_rows = fb.training_rows()
        if daily_sent is not None:
            first_news = train_rows.index[train_rows["news_count"] > 0].min()
            train_rows = train_rows[train_rows.index >= first_news]

        train_df, test_df = Preprocessor.chronological_split(train_rows, test_size)
        cols = fb.feature_columns(features_with_sentiment)

        from .evaluator import ModelEvaluator
        from .features import TARGET_PRICE, TARGET_RETURN
        actual = test_df[TARGET_PRICE].to_numpy(dtype=float)
        prev_close = test_df["Close"].to_numpy(dtype=float)

        for m_name, m_cls in target_models.items():
            model = m_cls()
            model.train(train_df[cols], train_df[TARGET_RETURN])
            pred = model.predict_price(test_df[cols], test_df["Close"])
            ev = ModelEvaluator(actual, pred)
            s = ev.summary(previous=prev_close)
            rows.append({
                "Sentiment Configuration": label,
                "Model": m_name,
                "MAE ($)": round(s["MAE"], 4),
                "RMSE ($)": round(s["RMSE"], 4),
                "R2": round(s["R2"], 4),
                "Direction Acc (%)": round(s["Direction_Acc"] * 100, 2),
            })

    return pd.DataFrame(rows)
