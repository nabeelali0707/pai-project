# AI-Based Stock Price Prediction Using Market Sentiment
## Presentation Slides (6–8 Slides)
**Course:** AL-2001 Programming for Artificial Intelligence (Lab), Fall 2026, Track A (Text / NLP)  
**Institution:** FAST NUCES Karachi, Department of AI & Data Science  
**Team:** Ali Abbas (Lead), Zain ul Abideen, Siraj Udin, Nabeel Ali  

---

<!-- slide -->
### Slide 1: Title & Overview
#### AI-Based Stock Price Prediction Using Market Sentiment

* **The Core Question:** Does public financial news sentiment contain measurable predictive signal for forecasting next-day stock price movements?
* **Our Approach:**
  * Combine 9 years of historical OHLCV prices (2015–2024) with 14,400 financial headlines.
  * Extract domain-adapted sentiment features (VADER + Finance Lexicon).
  * Train and contrast 4 regression architectures against a Naive persistence benchmark.
  * Rigorously evaluate performance with 5-fold Walk-Forward Validation and Statistical Significance Testing.
* **Team Roles:**
  * Zain ul Abideen: Data Pipeline, Caching & Numerical Feature Engineering
  * Siraj Udin: NLP Sentiment Extraction, Lexicon Extension & Alignment
  * Ali Abbas: Polymorphic Architecture, Walk-Forward Validation & Tuning
  * Nabeel Ali: Evaluation Metrics, Hypothesis Testing & Visualization

---

<!-- slide -->
### Slide 2: Problem Statement & Avoiding Look-Ahead Bias

* **The Random Walk Challenge:**
  * Daily stock prices in liquid markets closely approximate a random walk with drift.
  * Most models claim ~98% $R^2$ simply by predicting yesterday's price.
* **Strict Anti-Leakage Protocol:**
  * **Stationary Returns Target:** Predict $Close_{t+1} = Close_t \times (1 + Return_{t+1})$. Eliminates regime drift across 2015–2024.
  * **Chronological Split:** Test data (2022–2024) occurs strictly after training data (2015–2022).
  * **Backward-Looking Windows:** Technical indicators (MA5, MA10, MA20, Volatility) look only backwards from close $t$.
  * **Weekend News Realignment:** News published on weekends/holidays rolls into Monday's trading session before market open.

```
Raw Prices & News  ──>  Data Cleaning  ──>  Return Engineering  ──>  Chronological Train/Test
   (2015 - 2024)         (Zero Gaps)         (Backward-Only)           (No Future Leakage)
```

---

<!-- slide -->
### Slide 3: System Architecture & Object-Oriented Design

* **Modular OOP Architecture (Lab 05 Compliance):**
  * `DataLoader`: Network retries, data validation, and seamless automatic offline cache fallback.
  * `Preprocessor`: Monotonic sorting, duplicate filtering, forward filling.
  * `FeatureBuilder`: NumPy-vectorised returns, rolling ratios, and sentiment merger.
  * `BasePredictor`: Abstract base class implementing polymorphism across all regressors:
    * `NaivePredictor` ($Return_{t+1} = 0$)
    * `LinearRegressionPredictor`
    * `RidgePredictor` (L2 regularization)
    * `RandomForestPredictor` (Ensemble bagging)
    * `GradientBoostingPredictor` (Ensemble boosting)
  * `ModelEvaluator`: Vectorised metric calculations (MAE, RMSE, $R^2$, Direction Accuracy, Diebold-Mariano).

---

<!-- slide -->
### Slide 4: NLP Pipeline & Financial Sentiment Lexicon

* **Why General NLP Fails in Finance:**
  * General sentiment lexicons misclassify or miss standard market vernacular ("bearish", "downgraded", "missed estimates", "plunge", "surged", "rally").
* **Custom Financial Lexicon:**
  * Over 70 domain-calibrated tokens integrated directly into VADER's intensity engine.
  * Full negation handling (e.g., *"shares did not surge"* -> negative compound score).
* **Sentiment Ablation Results:**
  * Comparing Price-Only vs Plain VADER vs VADER + Finance Lexicon:
  * Domain-adapted finance lexicon improves headline polarity discrimination and produces consistent directional gains (+2.0% over baseline).

---

<!-- slide -->
### Slide 5: Machine Learning & Hyperparameter Tuning

* **Predictors & Feature Sets:**
  * **Baseline (10 features):** 1-day, 2-day, 3-day returns, MA ratios (5, 10, 20), 10-day volatility, volume change, high-low spread.
  * **Enhanced (16 features):** Baseline + daily mean sentiment, 3-day sentiment MA, sentiment change, log headline count, positive/negative headline ratios.
* **Light Hyperparameter Tuning (Training Period Only):**
  * `TimeSeriesSplit` (5 folds) strictly on 2015–2022 training data (test period untouched).
  * **Ridge:** $\alpha \in [0.01, \dots, 200.0] \rightarrow \text{best } \alpha = 10.0$.
  * **Random Forest:** Depth $\in [3, 5, 8]$, Min Leaves $\in [15, 25, 50] \rightarrow \text{best depth}=5, \text{leaf}=25$.
  * **Gradient Boosting:** Depth $\in [2, 3, 4]$, Learning Rate $\in [0.02, 0.05] \rightarrow \text{best depth}=2, \text{lr}=0.03$.

---

<!-- slide -->
### Slide 6: Results & Walk-Forward Validation

* **Test Period Results (453 days, 2022-04-08 to 2024-01-29):**
  * **Naive Benchmark:** RMSE \$2.813 | MAE \$2.107 | Direction: N/A
  * **Gradient Boosting (Price):** RMSE \$2.809 | MAE \$2.116 | Direction: 49.4%
  * **Gradient Boosting (Price + Sentiment):** RMSE \$2.811 | MAE \$2.112 | **Direction: 52.8% (+3.4%)**
  * **Ridge Regression (Price + Sentiment):** RMSE \$2.826 | MAE \$2.123 | **Direction: 53.2% (+2.0%)**
* **5-Fold Walk-Forward Rolling Evaluation (TimeSeriesSplit):**
  * Evaluated across expanding chronological windows:
  * Direction Accuracy: Enhanced models maintain **$51.3\% \pm 1.7\%$** to **$51.5\% \pm 2.9\%$** across rolling regimes.
  * Demonstrates that directional edge is persistent rather than a single-period fluke.

---

<!-- slide -->
### Slide 7: Statistical Significance: Does Sentiment Help?

* **Hypothesis Testing:**
  * **Direction Accuracy (Binomial Test vs 50% Coin-Flip):**
    * Enhanced Direction: 52.8% (238/451 days) $\rightarrow p\text{-value} = 0.1292$.
    * Result: Noticeable upward signal, but not statistically significant at $\alpha = 0.05$ threshold.
  * **Daily Error Test (Diebold-Mariano Paired Test):**
    * Squared Error Loss (MSE): $DM = -0.262, p\text{-value} = 0.7933$.
    * Absolute Error Loss (MAE): $DM = +0.609, p\text{-value} = 0.5426$.
    * Result: Error differences are well within noise margins.
* **Honest Scientific Conclusion:**
  * Sentiment provides a consistent directional tilt (+2.0% to +3.4%), but does not beat price persistence on dollar error.

---

<!-- slide -->
### Slide 8: The "Naive Baseline" Lesson & Limitations

* **The Naive Baseline Lesson:**
  * In financial time series, high $R^2$ (~0.98) is an illusion of autocorrelation.
  * Always compare against $Close_{t+1} = Close_t$. If a model cannot beat Naive, it provides no genuine price forecast skill.
* **Limitations & Ethical Considerations:**
  * **Not Investment Advice:** Daily market prices are non-stationary and predominantly noise.
  * **Execution Realities:** Slippage, transaction fees, and intraday spread would eliminate marginal directional edges.
  * **Lexicon Boundaries:** Misses subtle sarcasm and complex geopolitical macro context.
  * **Data Timestamps:** Daily-resolution news timestamps capture post-market close news as part of calendar day.
