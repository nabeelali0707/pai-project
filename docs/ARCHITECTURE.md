# AI-Based Stock Price Prediction Using Market Sentiment: Project and Architecture

## 1. System Architecture

The system has three layers: data sources, the processing pipeline, and the output.

```mermaid
flowchart LR
    subgraph Sources["Data Sources"]
        YF["Yahoo Finance<br/>(yfinance API)"]
        HL["Financial Headlines<br/>(downloaded dataset)"]
    end

    subgraph Cache["Offline Cache"]
        C["data/cache/<br/>CSV / JSON"]
    end

    subgraph Pipeline["Processing Pipeline"]
        DL["DataLoader"]
        PP["Preprocessor"]
        SA["SentimentAnalyzer<br/>(NLP)"]
        FB["FeatureBuilder"]
        SP["StockPredictor<br/>(ML)"]
        ME["ModelEvaluator"]
    end

    subgraph Output["Output"]
        M["MAE / RMSE / R²"]
        G["Actual vs Predicted<br/>Charts"]
        P["Next-Day<br/>Price Prediction"]
    end

    YF -- "online" --> DL
    HL --> DL
    DL -- "save first download" --> C
    C -- "offline fallback" --> DL
    DL --> PP
    PP -- "price data" --> FB
    DL -- "headlines" --> SA
    SA -- "daily sentiment scores" --> FB
    FB -- "combined features" --> SP
    SP --> ME
    ME --> M
    ME --> G
    SP --> P
```

## 2. Module Responsibilities

| Module | Responsibility | Owner |
|---|---|---|
| DataLoader | Loads API/CSV data, validates it, manages the offline cache | Zain |
| Preprocessor | Cleans data, handles missing values, does chronological split | Zain |
| SentimentAnalyzer | Preprocesses headline text, scores sentiment, aggregates by date | Siraj |
| FeatureBuilder | Adds returns and moving averages, merges price and sentiment features | Zain / Siraj |
| StockPredictor | Trains and predicts with scikit-learn models, compares configurations | Ali |
| ModelEvaluator | Calculates MAE, RMSE, R², and plots results | Nabeel |

## 3. Class Diagram

```mermaid
classDiagram
    class DataLoader {
        +stock_symbol
        +file_path
        +data
        +load_api()
        +load_csv()
        +save_cache()
    }

    class Preprocessor {
        +dataframe
        +clean()
        +handle_missing()
        +chronological_split()
    }

    class SentimentAnalyzer {
        +headlines
        +sentiment_scores
        +preprocess_text()
        +analyze()
        +aggregate_by_date()
    }

    class FeatureBuilder {
        +price_data
        +sentiment_data
        +add_returns()
        +add_moving_averages()
        +merge_features()
    }

    class StockPredictor {
        +model
        +X_train
        +y_train
        +train()
        +predict()
        +compare_models()
    }

    class ModelEvaluator {
        +actual
        +predicted
        +mae()
        +rmse()
        +r2()
        +plot_results()
    }

    class BasePredictor {
        <<abstract>>
        +train()
        +predict()
    }

    DataLoader --> Preprocessor : feeds
    Preprocessor --> FeatureBuilder : price data
    SentimentAnalyzer --> FeatureBuilder : sentiment data
    FeatureBuilder --> StockPredictor : feature set
    StockPredictor --> ModelEvaluator : predictions
    BasePredictor <|-- StockPredictor : polymorphism
```

## 4. Workflow (Menu-Driven Execution)

```mermaid
flowchart TD
    Start([Start]) --> Menu{Main Menu}
    Menu --> Load["1. Load data<br/>(API or cache)"]
    Menu --> Clean["2. Clean and<br/>engineer features"]
    Menu --> Sent["3. Run sentiment<br/>analysis"]
    Menu --> Train["4. Train and compare<br/>models"]
    Menu --> Eval["5. View metrics<br/>and charts"]
    Menu --> Exit([Exit])

    Load --> Clean
    Clean --> Sent
    Sent --> Train
    Train --> Eval
    Eval --> Menu
```

## 5. Model Comparison Design

```mermaid
flowchart LR
    D["Merged Dataset<br/>(prices + indicators)"] --> Split["Chronological<br/>Train / Test Split"]
    S["Daily Sentiment<br/>Scores"] --> Merge["Merge by<br/>Trading Date"]
    D --> Merge

    Split --> B["Baseline Model<br/>price features only"]
    Split --> E["Enhanced Model<br/>price + sentiment"]

    B --> Metrics["MAE, RMSE, R²"]
    E --> Metrics
    Metrics --> Verdict{"Does sentiment<br/>improve accuracy?"}
```

## 6. Folder Structure

```text
pai-project/
├── data/
│   └── cache/              # offline CSV/JSON copies (committed)
├── docs/
│   └── ARCHITECTURE.md     # this file
├── scripts/
│   └── fetch_headlines.py  # one-time headline dataset download + trim
├── src/
│   ├── data_loader.py      # DataLoader
│   ├── preprocessor.py     # Preprocessor
│   ├── sentiment.py        # SentimentAnalyzer
│   ├── features.py         # FeatureBuilder
│   ├── predictor.py        # BasePredictor, StockPredictor (+ model subclasses)
│   └── evaluator.py        # ModelEvaluator
├── tests/                  # pytest unit tests
├── main.py                 # menu-driven entry point
├── requirements.txt
└── README.md
```
