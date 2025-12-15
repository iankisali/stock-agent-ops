# 📈 MLOps Stock Prediction Pipeline

A production-ready MLOps system for predicting stock prices using **LSTM** networks and **Transfer Learning**. This project demonstrates an end-to-end machine learning lifecycle, from real-time data ingestion to scalable deployment on Kubernetes.

---

## 🎯 1. Problem Statement
Stock market prediction is traditionally resource-intensive. Training deep learning models from scratch for every single ticker symbol is computationally expensive and slow.

**Our Solution**:
We implement a **Parent-Child Transfer Learning Strategy**:
1.  **Parent Model**: A robust base model trained on a major index (e.g., `^GSPC` - S&P 500) to learn general market dynamics.
2.  **Child Models**: Lightweight models for specific stocks (e.g., `GOOG`, `TSLA`) that "inherit" knowledge from the parent. This drastically reduces training time (by ~10-15%) and improves convergence on smaller datasets.

---

## 🛠 2. Technology Stack

### Core ML & Backend
*   **Framework**: PyTorch (LSTM Models)
*   **API**: FastAPI (Async, High-performance)
*   **Data Processing**: Pandas, NumPy, Scikit-Learn

### MLOps Infrastructure
*   **Feature Store**: Feast (Offline/Online feature management)
*   **Experiment Tracking**: MLflow & DagsHub
*   **Vector Database**: Qdrant (Semantic memory for AI agents)
*   **Caching**: Redis (Rate limiting & API caching)
*   **Monitoring**: Prometheus & Grafana

### Deployment
*   **Containerization**: Docker & Docker Compose
*   **Orchestration**: Kubernetes (EKS - AWS)
*   **CI/CD**: GitHub Actions (planned)

---

## 🔄 3. Project Lifecycle & Architecture

The system operates in three main stages:

### A. Data Ingestion & Feature Engineering
*   Raw OHLCV data is fetched via `yfinance`.
*   Technical indicators (RSI, MACD, Volume) are computed.
*   **Feast** creates a point-in-time correct dataset to prevent data leakage.

### B. Training Pipeline (The "Intelligence")
1.  **Parent Training**: The system checks if the base model exists. If not, it automatically trains on the S&P 500.
2.  **Transfer Learning**: When a prediction is requested for a new stock (e.g., Google), the system leverages the pre-trained Parent Model weights.
3.  **Evaluation**: Models are evaluated on MSE, RMSE, and R² scores. Plots (Predictions vs Actuals) are automatically generated and logged to MLflow.

### C. Inference Pipeline (The "Service")
*   **Endpoint**: `/predict-child`
*   **Logic**:
    1.  Check Redis cache for recent predictions.
    2.  Check if a model exists for the requested ticker.
    3.  If missing, **trigger background training** immediately.
    4.  Run inference and return forecast.

---

## 🛡 4. Edge Cases & Reliability

We handled several critical production scenarios:

| Scenario | Handling Strategy |
| :--- | :--- |
| **Missing Model** | **Auto-Healing**: The API detects the missing model, returns a "Training Started" status, and launches a background job. |
| **Cold Start (No Parent)** | If the Parent model is missing, the system recursively trains the Parent first, then the Child. |
| **Data Gaps** | `yfinance` failures are caught, and the system falls back to cached features if available. |
| **Rate Limiting** | Implemented customizable Rate Limiting (e.g., 5 requests/minute) using Redis to protect the API. |
| **Concurrency** | Training runs are locked per-ticker to prevent multiple requests triggering duplicate training jobs. |

---

## 🚀 5. Usage Guide

### Option A: Local Deployment (Docker Compose)
The easiest way to stand up the full stack (API, Redis, Qdrant, Grafana, MLflow).

```bash
# 1. Clone & Setup
git clone <repo-url>
cd mlops-pipeline

# 2. Run the Stack
docker-compose up --build

# 3. Access Interfaces
# - API Docs: http://localhost:8000/docs
# - Frontend: http://localhost:8501
# - Grafana:  http://localhost:3000
```

### Option B: Cloud Deployment (AWS EKS)
We support full scalable deployment on AWS.
*   See **[`AWS.md`](./AWS.md)** for the detailed Kubernetes manifest and setup instructions.

---

## 🔌 6. API Reference (Curl Commands)

### 1. Check System Status
Monitor the health of background training jobs.
```bash
curl -X 'GET' \
  'http://localhost:8000/status' \
  -H 'accept: application/json'
```

### 2. Predict / Train (Child Model)
If the model exists, returns prediction. If not, triggers training.
```bash
curl -X 'POST' \
  'http://localhost:8000/predict-child' \
  -H 'Content-Type: application/json' \
  -d '{
  "ticker": "GOOG",
  "days": 7
}'
```

### 3. Predict Parent (Base Model)
Direct access to the S&P 500 model.
```bash
curl -X 'POST' \
  'http://localhost:8000/predict-parent' \
  -H 'Content-Type: application/json' \
  -d '{
  "days": 5
}'
```

### 4. Interactive AI Agent
Chat with the market analyst agent (requires Qdrant memory).
```bash
curl -X 'POST' \
  'http://localhost:8000/agent/chat' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What is the sentiment for Google based on recent performance?"
}'
```
