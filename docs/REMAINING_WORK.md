# Remaining Work

Status as of 12 Nov 2026. The pipeline from the proposal runs end to end.
`python main.py --auto --offline` loads the cached data, cleans it, scores
sentiment, trains and compares the models, and draws the charts. All 20 tests
pass. What remains splits into two phases.

- **Phase 1:** finish what the proposal promised and make sure every member can
  run and explain the project.
- **Phase 2:** make the "does sentiment help?" result more reliable and prepare the demo.

## Already done

| Proposal item | Where |
|---|---|
| Data loading with an offline cache (milestone 1) | `src/data_loader.py`, `data/cache/` |
| Cleaning, indicators, exploratory plots (milestone 2) | `src/preprocessor.py`, `src/features.py`, `src/evaluator.py` |
| Daily sentiment features merged with prices (milestone 3) | `src/sentiment.py`, `src/features.py` |
| Baseline vs sentiment-enhanced models with MAE/RMSE/R² (milestone 4) | `src/predictor.py`, `main.py` |
| README, architecture document, unit tests | `README.md`, `docs/ARCHITECTURE.md`, `tests/` |

---

## Phase 1: Complete the scope and make it ours

**Target:** by the end of week 8 (proposal milestone 4).

| # | Task | Owner | Done when |
|---|---|---|---|
| 1.1 | **Review your own module and be able to explain it.** Read it line by line, run its tests, and note anything unclear in an issue. The viva will ask each member about their module. | All (Zain: `data_loader`, `preprocessor`; Siraj: `sentiment`; Ali: `predictor`, `main`; Nabeel: `evaluator`, `tests`) | Each member can explain their module's purpose, inputs, outputs and one design choice, such as why weekend news moves to Monday. |
| 1.2 | **Install and run on every member's laptop.** It has only been tested on Python 3.14 with pandas 3.0. Check Python 3.10–3.12 and pandas 2.x, then pin the tested versions in `requirements.txt`. | Nabeel | `pip install -r requirements.txt`, `python -m pytest` and `python main.py --auto --offline` all work on 4 machines. |
| 1.3 | **Let the user choose the stock and the data file from the menu.** The proposal says the user selects a stock dataset, but the menu currently uses AAPL unless `--symbol` is passed. Add a menu option to change the symbol and to load a custom price or headline CSV. `DataLoader.load_csv()` already accepts a path. | Zain | From the menu you can switch to MSFT or load your own CSV, and invalid files show a clear error instead of crashing. |
| 1.4 | **Cache more than one stock.** The proposal says "selected stocks", but only AAPL is cached. Add at least 2 more from the dataset, for example MSFT and NVDA, using `scripts/fetch_headlines.py <SYM> --first-pass`. | Zain / Siraj | `data/cache/` holds price and headline files for 3 or more tickers, and each runs offline. |
| 1.5 | **Results table across stocks.** Run the comparison for every cached ticker and add one summary table (RMSE and direction accuracy, baseline vs enhanced) to the README. | Ali | The README answers whether sentiment helps for each stock, not only AAPL. |
| 1.6 | **Use the full headline file if the network allows.** The AAPL cache uses only the first chronological sweep, 14,400 of the items, because the download was too slow. Run `python scripts/fetch_headlines.py AAPL` on a faster connection and rerun. | Siraj | Either the README numbers are updated using the full file, or the README states why the subset was kept. |
| 1.7 | **Proposal admin.** Fill in the section numbers (currently "TBD") and get instructor approval. Also note that the code lives at `nabeelali0707/pai-project`, not the URL in the proposal. | Ali | The approved proposal is on file and the repo URL is confirmed with the instructor. |

---

## Phase 2: Strengthen the evaluation and prepare the demo

**Target:** by one week before the lab final (proposal milestone 5).

| # | Task | Owner | Done when |
|---|---|---|---|
| 2.1 | **Walk-forward validation.** Results now come from a single train/test split, so differences of 1–2 points are within noise. Add a rolling evaluation using scikit-learn's `TimeSeriesSplit` (for example 5 folds) and report each metric as mean ± standard deviation. | Ali | `compare_models` can run in walk-forward mode, and the README table shows mean ± std. |
| 2.2 | **Significance check.** Add a binomial test for direction accuracy against 50%, and a paired test on the daily errors (Diebold–Mariano or a paired t-test) for baseline vs enhanced. | Nabeel | Step 5 prints a p-value next to the sentiment verdict, so the README can say whether the effect is significant. |
| 2.3 | **Light hyperparameter tuning.** Tune Ridge `alpha` and the Random Forest and Gradient Boosting depth and leaf size, using `TimeSeriesSplit` on the training years only. Never tune on the test period. | Ali | The chosen settings and the search ranges are documented, and the test data is untouched during tuning. |
| 2.4 | **Sentiment ablation.** Compare the current VADER + finance lexicon with plain VADER, without the finance words. FinBERT is an optional stretch goal and only if it runs offline in reasonable time. This shows whether our finance lexicon matters. | Siraj | A small table compares the sentiment methods by direction accuracy and RMSE. |
| 2.5 | **Continuous integration.** Add a GitHub Actions workflow that runs `pytest` and `python main.py --auto --offline` on every push. | Nabeel | A green check appears on GitHub for every commit. |
| 2.6 | **Demo script and slides.** Prepare a 5-minute walkthrough using the menu in `--offline` mode: load, clean, sentiment, train, charts, prediction. Prepare 6–8 slides covering the problem, architecture, NLP, results, the "naive baseline" lesson and limitations. Rehearse once with the network switched off. | All (Nabeel: slides) | A full rehearsal runs without errors and each member presents their own module. |
| 2.7 | **Final cleanup.** Update the README results after 2.1–2.4, remove anything unused, and tag the release `v1.0`. | All | The `v1.0` tag is on GitHub, and the README numbers match `python main.py --auto --offline`. |

---

## Out of scope (from the proposal)

Real-money trading, broker integration, intraday prediction, paid APIs and
training a language model from scratch. Do not add these.
