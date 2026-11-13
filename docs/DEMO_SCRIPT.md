# 5-Minute Final Demo Script

**Project:** AI-Based Stock Price Prediction Using Market Sentiment  
**Course:** AL-2001 Programming for Artificial Intelligence (Lab), Fall 2026  
**Format:** Offline live walkthrough using the interactive CLI menu (`python main.py --offline`).  
**Total Time:** 5:00 minutes (split evenly across 4 team members).

---

## Setup Before Presenting
1. Disconnect Wi-Fi / disable network to prove strict offline capability.
2. In the terminal:
   ```bash
   cd pai-project
   python main.py --offline
   ```
3. Keep `outputs/figures/` open in an image viewer or slide deck for visual demonstration.

---

## Minute-by-Minute Walkthrough

### 0:00 - 1:15 | Zain ul Abideen (Data Loading, Cleaning & Numerical Features)
* **Action:** Choose Option `1` (Load data), then Option `2` (Clean data & engineer features).
* **Talking Points:**
  - *"Our goal is to predict next-day stock price movements and evaluate whether financial news sentiment genuinely improves forecast accuracy."*
  - *"In Step 1, our `DataLoader` loads 2,284 cached trading days of AAPL OHLCV data from Yahoo Finance and 14,400 financial news headlines covering 2015–2024. If the internet fails, it automatically falls back to our local cache."*
  - *"In Step 2, our `Preprocessor` enforces data integrity: checking date monotonicity, stripping duplicate timestamps, validating prices (> 0), and forward-filling minor gaps."*
  - *"Next, `FeatureBuilder` computes 10 price-based technical indicators using vectorised NumPy operations: 1-day, 2-day, 3-day return lags, moving average ratios (MA5, MA10, MA20), 10-day rolling volatility, volume changes, and high-low daily spreads."*
  - *"Crucially, to eliminate look-ahead bias, rolling windows strictly look backwards, and prices are split chronologically (80% train, 20% test)."*

---

### 1:15 - 2:30 | Siraj Udin (NLP & Sentiment Analysis)
* **Action:** Choose Option `3` (Run sentiment analysis on headlines).
* **Talking Points:**
  - *"In Step 3, `SentimentAnalyzer` processes all 14,400 financial headlines. General-purpose sentiment tools fail in financial domains where words like 'bearish', 'downgrade', or 'slump' have precise market implications."*
  - *"We extended VADER with a dedicated finance lexicon of over 70 domain-specific terms and calibrated weights, while maintaining negation handling (e.g. 'shares did not surge')."*
  - *"A critical design choice: markets are closed on weekends, but news keeps breaking. `FeatureBuilder` aligns news timestamps so that all weekend and holiday news items roll forward to Monday's opening session."*
  - *"We aggregate sentiment by calendar date into daily mean compound score, 3-day smoothed sentiment, daily sentiment velocity, log headline count, and positive/negative ratios."*
  - *"Notice the correlation of daily sentiment with next-day returns is modest (|r| <= 0.04), perfectly aligned with semi-strong Efficient Market Hypothesis: publicly available news is priced in rapidly."*

---

### 2:30 - 3:45 | Ali Abbas (Model Training, Walk-Forward Validation & Tuning)
* **Action:** Choose Option `4` (Train and compare models), then Option `7` (Walk-forward validation).
* **Talking Points:**
  - *"In Step 4, we evaluate four regressors: Linear Regression, L2-regularized Ridge Regression, Random Forest, and Gradient Boosting across two distinct feature sets: Baseline (price-only) vs Enhanced (price + sentiment). All implement our polymorphic `BasePredictor` interface."*
  - *"Key modeling decision: we train models to predict next-day percentage returns: $Close_{t+1} = Close_t \times (1 + r_{t+1})$. AAPL traded at \$27 in 2015 and \$190 in 2024; raw price levels cannot be extrapolated by tree-based models, whereas returns are stationary across market regimes."*
  - *"In Phase 2, we added 5-fold rolling Walk-Forward Validation using `TimeSeriesSplit`. Single train/test splits can be noisy; walk-forward testing validates models across multiple chronological expansion periods, showing mean +/- std."*
  - *"For hyperparameter tuning (Option 9), we tuned Ridge alpha and tree depth/leaves strictly on the training period using `TimeSeriesSplit`. Test data was never touched during tuning."*

---

### 3:45 - 5:00 | Nabeel Ali (Evaluation, Significance Checks, & Limitations)
* **Action:** Choose Option `5` (View metrics & charts), then Option `6` (Predict next-day price).
* **Talking Points:**
  - *"In Step 5, we evaluate MAE, RMSE, R², and Up/Down Direction Accuracy. Crucially, we compare every model against the Naive 'No Change' baseline ($Close_{t+1} = Close_t$)."*
  - *"The Naive baseline achieves an RMSE of \$2.813 with an R² of 0.978. While models achieve a high R², that is simply because daily stock prices follow a random walk with drift; beating the naive baseline on RMSE is notoriously difficult in liquid markets."*
  - *"However, sentiment does provide an edge in Direction Accuracy: adding sentiment features improved directional accuracy across all 4 models, boosting Ridge from 51.2% to 53.2% (+2.0%)."*
  - *"In Phase 2, we added rigorous statistical significance testing:*
    - *A binomial test vs 50% coin-flip ($p = 0.1292$ on 451 test days): the directional gain is a positive signal, but not yet statistically significant at $\alpha = 0.05$.*
    - *A Diebold-Mariano paired error test ($p = 0.7933$): confirming that price RMSE changes remain within noise.*
  - *"Finally, Step 6 generates tomorrow's prediction, accompanied by our clear educational disclaimer: stock markets are non-stationary and daily prices are largely unpredictable."*

---

## Viva & Defense Q&A Cheatsheet

| Question | Strong Answer |
|---|---|
| **Why did weekend news move to Monday?** | Financial markets are closed on Saturday and Sunday. Any news breaking over the weekend cannot be traded on until market open on Monday. Assigning weekend news to Monday aligns news with the earliest trading session that can react to it. |
| **Why predict returns instead of price levels?** | Stock prices have strong upward trends over a 10-year span (non-stationary). Tree models (Random Forest, Gradient Boosting) cannot predict values outside their training range. Percentage returns are stationary and comparable across 2015 and 2024. |
| **Why does Naive baseline have high R² (~0.98)?** | Daily price changes are typically small (~1-2%). Yesterday's price is already a 98% accurate predictor of today's price level. Therefore, R² on price levels is deceptive; RMSE vs Naive and Direction Accuracy are the true tests of model skill. |
| **Why tune hyperparameters only on training data?** | Tuning hyperparameters on test data causes data leakage and optimistic bias. We used `TimeSeriesSplit` cross-validation strictly on training years, leaving the test set completely untouched. |
