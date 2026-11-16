# Project Guide: Stock Price Prediction Using Market Sentiment

> **Simple, plain-English guide to understanding, navigating, and running the project.**

---

## 1. What Is This Project in Simple Terms?

Imagine you want to guess whether a company's stock (like Apple, Microsoft, or NVIDIA) will go up or down tomorrow, and what its price might be at the end of the day.

Most traditional computer models look only at **numbers**:
- Past closing prices.
- How much the price changed over the last 5, 10, or 20 days.
- How volatile (jumpy) the stock has been.
- How many shares were traded.

However, humans don't just look at charts—they read the **news**. If Apple announces a record-breaking earnings report or unveils a breakthrough AI chip, people get excited and buy. If there's news about supply chain delays or lawsuits, people get nervous and sell.

### The Big Question This Project Answers:
> **"Does feeding news sentiment into machine learning models actually help them predict tomorrow's stock price better than using historical prices alone?"**

To answer this honestly and scientifically, the project builds two sets of models:
1. **Baseline models:** Look **only at price and volume history**.
2. **Enhanced models:** Look at **price history PLUS daily news sentiment scores**.

Then it compares both against a third reality check: the **Naive Benchmark** (which assumes tomorrow's price will simply be the same as today's price).

---

## 2. The Parts of the Project (How It Works Under the Hood)

The application is structured into modular Python files inside the `src/` directory, coordinated by `main.py`:

```
Yahoo Finance ──┐                     ┌──> Preprocessor ──> FeatureBuilder ──┐
                ├──> DataLoader ─(cache)                      ▲              ├──> StockPredictor ──> ModelEvaluator
Alpaca News  ───┘                     └──> SentimentAnalyzer ─┘              │    (Baseline vs        (MAE, RMSE, R²,
Dataset                                                                      │     Enhanced)           Charts, Tests)
```

Here is what each component does in plain language:

### 1. Data Loader (`src/data_loader.py`)
- **What it does:** Downloads and caches market data.
- **Where data comes from:**
  - **Stock prices:** Fetched from Yahoo Finance using `yfinance` (daily Open, High, Low, Close, Volume from 2015 to 2024).
  - **Headlines:** Over 14,000 financial news items from the Alpaca Markets news dataset on Hugging Face.
- **Key design feature:** **Offline-first fallback**. If the internet goes down or Yahoo rate-limits requests, it automatically loads saved data from `data/cache/` without crashing.

### 2. Preprocessor (`src/preprocessor.py`)
- **What it does:** Cleans messy raw numbers and prepares the timeline.
- **Tasks performed:**
  - Removes duplicate dates and invalid prices ($\le 0$).
  - Fills small missing gaps using forward-fill (carrying yesterday's known price forward).
  - Performs a **chronological split** (e.g., training on 2015–2022, testing on 2022–2024). It never shuffles data randomly, preventing "future leakage" (cheating by peeking ahead).

### 3. Sentiment Analyzer (`src/sentiment.py`)
- **What it does:** Reads English headlines and converts them into positive or negative numbers between $-1.0$ (very bad news) and $+1.0$ (very good news).
- **The Brain:** Uses **VADER** (Valence Aware Dictionary and sEntiment Reasoner) enhanced with a custom **Financial Lexicon**.
  - Words like `"bullish"`, `"soared"`, `"rally"`, and `"upgrade"` get positive boosts.
  - Words like `"downgrade"`, `"slump"`, `"plunge"`, and `"investigation"` get negative penalties.
- **The Weekend Rule:** Stock markets are closed on Saturdays and Sundays. If major news breaks on a Sunday, the sentiment is automatically shifted to **Monday morning**, right when traders can react.

### 4. Feature Builder (`src/features.py`)
- **What it does:** Converts raw prices and daily sentiment into numeric clues (features) that machine learning algorithms can understand.
- **Price Features:**
  - `return_1d`: Daily price percentage change.
  - `return_lag1`, `return_lag2`, `return_lag3`: Momentum over previous days.
  - `ma_ratio_5`, `ma_ratio_10`, `ma_ratio_20`: Ratio of today's price to 5, 10, and 20-day moving averages (trend indicator).
  - `volatility_10`: 10-day standard deviation of returns (risk indicator).
  - `volume_change`: Spike or drop in trading volume.
  - `hl_range`: Intraday price swing `(High - Low) / Close`.
- **Sentiment Features:**
  - `sentiment_mean`: Average compound sentiment score of the day's news.
  - `sentiment_ma3`: 3-day smoothed sentiment trend.
  - `sentiment_change`: Today's sentiment minus yesterday's.
  - `positive_ratio` & `negative_ratio`: Proportion of bullish vs. bearish articles.
  - `news_count_log`: Log-scaled headline volume.
- **The Target:** Predicts next-day **percentage return** rather than raw dollar price, avoiding distortion from long-term inflation or stock splits.

### 5. Predictors (`src/predictor.py`)
- **What it does:** Trains machine learning models to predict tomorrow's return, then converts that back into a forecasted dollar price:
  $$\text{Price}_{t+1} = \text{Price}_t \times (1 + \text{Return}_{t+1})$$
- **Models tested:**
  1. `NaivePredictor`: The baseline benchmark ($\text{Price}_{t+1} = \text{Price}_t$).
  2. `LinearRegressionPredictor`: Simple linear relationships.
  3. `RidgePredictor`: Linear regression with $L_2$ regularization to prevent overfitting.
  4. `RandomForestPredictor`: Ensemble of decision trees capturing non-linear interactions.
  5. `GradientBoostingPredictor`: Iterative tree boosting that corrects previous mistakes.
- **Advanced Modes:**
  - **Walk-Forward Validation:** Evaluates models across 5 expanding time windows (`TimeSeriesSplit`) to test real-world consistency.
  - **Hyperparameter Tuning:** Tunes model depth, leaf sizes, and regularization strictly on training years.

### 6. Evaluator (`src/evaluator.py`)
- **What it does:** Measures model performance, runs statistical tests, and draws visual charts.
- **Key Metrics:**
  - **RMSE (Root Mean Squared Error):** Average dollar mistake penalty.
  - **MAE (Mean Absolute Error):** Average dollar distance from actual price.
  - **Directional Accuracy:** Percentage of days the model correctly guessed if the price would go **UP** or **DOWN**.
- **Statistical Significance:**
  - **Binomial Test:** Is directional accuracy statistically better than flipping a 50/50 coin?
  - **Diebold–Mariano Test:** Is the difference in daily errors between the baseline and enhanced models statistically meaningful or just random chance?

### 7. Main Interface (`main.py`)
- **What it does:** The user interface. Provides both an **interactive numbered menu** and **command-line automation flags**.

---

## 3. How to Run It

### Setup (One-time)
Open PowerShell or your terminal in the project folder and run:
```bash
# Option A: Using uv (fastest)
uv pip install -r requirements.txt

# Option B: Standard pip with virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

---

### Running the Application

#### A. Interactive Menu (Explore step by step)
```bash
python main.py
```
This opens the main menu:
```
========================================================================
  Stock Price Prediction with Market Sentiment  |  AAPL
========================================================================
    1. Load data (API or offline cache)
    2. Clean data and engineer features
    3. Run sentiment analysis on headlines
    4. Train and compare models
    5. View metrics and charts
    6. Predict next-day closing price
    7. Run walk-forward validation (TimeSeriesSplit 5 folds)
    8. Run sentiment ablation (Plain VADER vs Finance Lexicon)
    9. Run hyperparameter tuning (training set only)
   10. Run everything (1-6)
   11. Change stock or load custom CSV files
    0. Exit
```

#### B. Full Automatic Offline Run (Recommended for grading / demo)
Runs the entire pipeline end-to-end using cached data without needing internet:
```bash
python main.py --auto --offline
```

#### C. Run on Different Stocks
You can run the full offline pipeline for Microsoft (`MSFT`) or NVIDIA (`NVDA`):
```bash
# Microsoft
python main.py --symbol MSFT --auto --offline

# NVIDIA
python main.py --symbol NVDA --auto --offline
```

#### D. Run the Automated Unit Test Suite
Verifies all 31 tests across all modules:
```bash
python -m pytest
```

#### E. Specialized Experiments
```bash
# 5-fold Walk-Forward Rolling Evaluation
python main.py --walk-forward

# Sentiment Ablation Study (Plain VADER vs VADER + Finance Lexicon)
python main.py --ablation

# Hyperparameter Grid Search (on training data only)
python main.py --tune
```

---

## 4. When to Run What (Quick Cheat-Sheet)

| Scenario | Command to Run | Why |
|---|---|---|
| **Class Presentation / Live Viva** | `python main.py --auto --offline` | Fast, reliable, requires zero internet, generates every metric and figure on the spot. |
| **Exploring the Code Step-by-Step** | `python main.py` | Walk through step 1 (loading), step 2 (cleaning), step 3 (NLP sentiment), step 4 (training), etc. |
| **Demonstrating Other Stocks** | `python main.py --symbol MSFT --auto --offline` or `NVDA` | Proves the system is generalizable across multiple companies and not hardcoded to Apple. |
| **Testing User-Provided CSV Data** | Run `python main.py`, pick Option `11`, enter `C` | Lets an evaluator load custom external price and headline files. |
| **Defending Model Stability (Viva Question)** | `python main.py --walk-forward` | Shows the model tested across 5 rolling time slices, not just a single lucky test split. |
| **Defending the NLP Lexicon (Viva Question)** | `python main.py --ablation` | Demonstrates that the custom financial lexicon outperforms vanilla off-the-shelf VADER. |
| **Code Verification / Before Submitting** | `python -m pytest` | Ensures all unit tests pass with zero errors. |

---

## 5. What Were the Final Findings? (Does Sentiment Help?)

The project reached an honest, scientifically grounded conclusion that aligns with the **Efficient Market Hypothesis (EMH)**:

1. **Directional Accuracy (Up vs Down):**
   - **Yes, sentiment helps slightly.** Adding financial news sentiment improved up/down directional accuracy across models (reaching **53.2%** on AAPL and **50.2%** on NVDA, up from ~49.4% baseline).
   - Our ablation test proved that adding domain-specific words (`surge`, `rally`, `downgrade`, `plunge`) added an extra **+1.1% to +1.4%** directional accuracy over standard VADER.
   - However, binomial hypothesis testing showed $p \ge 0.05$, meaning this gain is an encouraging weak signal rather than an infallible crystal ball.

2. **Price Level (Dollar Error / RMSE):**
   - **No meaningful dollar reduction.** Machine learning models predicting next-day price level do not beat the **Naive Benchmark** ($\text{Price}_{t+1} = \text{Price}_t$) by more than a fraction of a cent.
   - Public headlines are digested and priced in within minutes by high-frequency institutional traders, so end-of-day news rarely creates large daily dollar pricing lag.

3. **Bottom Line:**
   - Combining price momentum indicators with market sentiment gives a modest, consistent directional edge, but cannot predict exact future stock prices with certainty. This represents realistic, academically honest AI research.

---

## 6. Generated Figures and Outputs

Whenever you run the pipeline, charts and result tables are saved locally:

- **Saved Charts:** `outputs/figures/`
  - `price_history.png`: 10-year OHLCV price history and 20/50-day moving averages.
  - `returns_distribution.png`: Histogram and KDE curve of daily percentage returns.
  - `feature_correlation.png`: Heatmap showing how price features relate to one another.
  - `sentiment_distribution.png`: Breakdown of positive, negative, and neutral headlines.
  - `feature_correlation_with_sentiment.png`: Correlation matrix including sentiment signals.
  - `actual_vs_predicted.png`: Actual stock prices vs. predictions over the test period.
  - `actual_vs_predicted_last60.png`: Zoomed-in look at the last 60 trading days.
  - `model_comparison.png`: Bar chart comparing MAE, RMSE, and Direction Accuracy across all models.

- **Saved Data Tables:** `outputs/`
  - `model_comparison.csv`: Complete metrics table for all models.
  - `test_predictions.csv`: Day-by-day actual prices alongside every model's predictions.
