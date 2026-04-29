# Stock Price Predictor — LSTM on Azure

An end-to-end ML system that predicts the next 30 days of stock closing prices using stacked LSTM neural networks, deployed on Azure with a live Streamlit frontend.

> ⚠️ Backend Azure Functions are currently paused to avoid cloud costs. The full codebase — training pipeline, prediction logic, and CI/CD configuration — is available in this repository.

---

## Live Demo

🔗 **https://stock-sage.streamlit.app/**

---

## Architecture

```
yfinance API
     │
     ▼
┌──────────────────┐
│  Data Collector  │  Azure Function — fetches & stores daily stock data to ADLS
│  (Azure Function)│
└──────────────────┘
     │
     ▼
┌──────────────────┐
│  Azure Data Lake │  Stores raw CSVs and trained model weights (.h5), versioned by date
│  Storage (ADLS)  │
└──────────────────┘
     │
     ▼
┌──────────────────┐
│  Train Function  │  Azure Function — trains LSTM on 5 years of data, saves weights to ADLS
│  (HTTP Trigger)  │
└──────────────────┘
     │
     ▼
┌──────────────────┐
│ Predict Function │  Azure Function — loads latest weights, returns 30-day forecast as JSON
│  (HTTP Trigger)  │
└──────────────────┘
     │
     ▼
┌──────────────────┐
│    Streamlit     │  Frontend — calls predict API, visualizes historical + predicted prices
│    Frontend      │
└──────────────────┘
```

---

## Tech Stack

- **Python** — core language
- **TensorFlow / Keras** — LSTM model training and inference
- **Azure Functions** — serverless compute for train and predict pipelines
- **Azure Data Lake Storage Gen2 (ADLS)** — data and model weight storage
- **Azure Identity (DefaultAzureCredential)** — secure, passwordless authentication
- **Streamlit** — interactive frontend hosted on Streamlit Cloud
- **GitHub Actions** — CI/CD pipeline with automated deployment to Azure

---

## Supported Stocks

| Ticker | Exchange |
|---|---|
| AAPL | NASDAQ |
| GOOG | NASDAQ |
| META | NASDAQ |
| PAYTM | NSE |
| TCS | NSE |

---

## Model Details

| Parameter | Value |
|---|---|
| Architecture | 3-layer stacked LSTM |
| Units per layer | 50 |
| Training epochs | 150 |
| Batch size | 64 |
| Input window | 5–15 days (ticker-specific) |
| Training data | 5 years of daily close prices |
| Prediction horizon | 30 days |
| Loss function | Mean Squared Error |
| Optimizer | Adam |

---

## Key Features

- Automated daily data ingestion via Azure Function timer trigger
- Model weights versioned by date in ADLS — latest weights auto-selected at inference
- Separate train and predict function architecture for clean separation of concerns
- Secure cloud access using `DefaultAzureCredential` (no hardcoded secrets)
- CI/CD pipeline via GitHub Actions — 35+ production deployments
- Interactive frontend with historical vs predicted price visualization

---

## Repository Structure

```
major-project/
├── .github/workflows/       # GitHub Actions CI/CD pipeline
├── data_collector/          # Azure Function — daily data ingestion from yfinance
├── train_AAPL/              # Azure Function — LSTM training for AAPL
├── train_GOOG/              # Azure Function — LSTM training for GOOG
├── train_META/              # Azure Function — LSTM training for META
├── train_PAYTM/             # Azure Function — LSTM training for PAYTM
├── train_TCS/               # Azure Function — LSTM training for TCS
├── pred_AAPL/               # Azure Function — 30-day prediction for AAPL
├── pred_GOOG/               # Azure Function — 30-day prediction for GOOG
├── pred_META/               # Azure Function — 30-day prediction for META
├── pred_PAYTM/              # Azure Function — 30-day prediction for PAYTM
├── pred_TCS/                # Azure Function — 30-day prediction for TCS
├── host.json                # Azure Functions host configuration
├── requirements.txt         # Python dependencies
└── README.md
```

Frontend repository: [mazor-project-frontend](https://github.com/aman-krsingh/mazor-project-frontend)
