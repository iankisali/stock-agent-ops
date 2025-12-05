# Low Level Design (LLD) - AI Stock Analyst System

## 1. System Overview

This project is an **Agentic AI-driven Stock Analysis System** that combines traditional Machine Learning (LSTM Time-series forecasting) with Generative AI (LLM Agents) to produce comprehensive financial reports. It follows a Microservices-based architecture containerized with Docker.

### High-Level Architecture

```mermaid
graph TD
    User[User] -->|HTTP| Frontend[Streamlit Dashboard\n(frontend:8501)]
    Frontend -->|HTTP /analyze| Backend[FastAPI Backend\n(fastapi:8000)]
    
    subgraph "Backend Services"
        Backend -->|Train/Predict| ML[ML Pipeline\n(PyTorch LSTM)]
        Backend -->|Orchestrate| Agents[LangGraph Agents\n(Trader, Quant, News)]
        
        ML -->|Read/Write| FS[FileSystem\n(Model Artifacts)]
        
        Agents -->|Reference| ML
        Agents -->|Cache/RateLimit| Redis[Redis Stack\n(redis:6379)]
        Agents -->|RAG/Memory| Qdrant[Qdrant Vector DB\n(qdrant:6333)]
    end
    
    subgraph "Observability"
        Prometheus[Prometheus] -->|Scrape| Backend
        Grafana[Grafana] -->|Query| Prometheus
    end
```

---

## 2. Infrastructure Layer (Docker)

The system is composed of the following containerized services defined in `docker-compose.yml`:

| Service | Image | Internal Port | Exposed Port | specific Command | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **fastapi** | `python:3.11` | 8000 | 8000 | `uvicorn --workers 4` | Core API, ML Training, Agent Orchestration. Workers enabled for concurrency. |
| **frontend** | `python:3.9-slim` | 8501 | 8501 | `streamlit run` | User Interface for triggering analysis and viewing reports. |
| **redis** | `redis/redis-stack:latest` | 6379, 8001 | 6379 | - | **Semantic Cache**: Caches Tool/LLM outputs.<br>**Rate Limiting**: Throttles API requests.<br>**Vector Store**: (Optional) For high-speed similarity search. |
| **qdrant** | `qdrant/qdrant:latest` | 6333 | 6333 | - | **Episodic Memory**: Stores embeddings of past successful analyses to enable RAG (Retrieval Augmented Generation). |
| **prometheus** | `prom/prometheus` | 9090 | 9090 | - | Scrapes metrics from FastAPI for monitoring CPU, RAM, and Latency. |
| **grafana** | `grafana/grafana` | 3000 | 3000 | - | Visualizes metrics in dashboards. |

---

## 3. Application Layer - Backend (`src/`)

### 3.1. API Endpoints (`main.py`)

| Method | Endpoint | Description | Flow |
| :--- | :--- | :--- | :--- |
| `POST` | `/analyze` | **Main Entrypoint**. Triggers full analysis. | 1. Check/Train Model<br>2. Predict (7-day)<br>3. Run Agent Graph<br>4. Return JSON Report |
| `POST` | `/train-child` | Manually trigger child model training. | Uses Transfer Learning (Fine-tune Parent). |
| `POST` | `/predict-child` | Get raw numerical predictions. | Checks Cache -> Runs Inference. |
| `POST` | `/agent/chat` | Direct chat with the Agent. | For conversational follow-ups. |
| `GET` | `/metrics` | Prometheus metrics endpoint. | Exposes system & custom metrics. |

### 3.2. Machine Learning Pipeline (`src/pipelines/`)

The system uses a **Parent-Child Transfer Learning** strategy for Time Series Forecasting.

1.  **Parent Model (`train_parent`)**:
    *   **Data**: S&P 500 (`^GSPC`) generic market data.
    *   **Architecture**: LSTM (Long Short-Term Memory).
    *   **Goal**: Learn general market features (Volatility, RSI/MACD patterns).
    *   **Output**: Pre-trained weights saved to `outputs/parent/`.

2.  **Child Model (`train_child`)**:
    *   **Data**: Specific Ticker (e.g., `NVDA`).
    *   **Transfer Strategy**: Configurable via `src/config.py`.
        *   `freeze`: Lock LSTM weights, train only Head (Prevent overfitting).
        *   `fine_tune`: Low LR training of all layers (Adapt to specific volatility).
    *   **Trigger**: Auto-trained on-demand if missing during `/analyze`.

### 3.3. Agentic Architecture (`src/agents/`)

We use **LangGraph** to orchestrate a team of specialized AI agents.

#### The Graph (`src/agents/graph.py`)
State Graph: `Trader -> Quant -> News -> Supervisor`

1.  **Trader Agent (`trader_node`)**:
    *   **Role**: Technical Analyst.
    *   **Input**: Raw price predictions + Current Price.
    *   **Logic**: "Is the trend Bullish/Bearish based on momentum?"
2.  **Quant Agent (`quant_node`)**:
    *   **Role**: Risk Manager.
    *   **Input**: Volatility, Prediction Confidence.
    *   **Logic**: "Is the Risk/Reward ratio attractive?"
3.  **News Agent (`news_node`)**:
    *   **Role**: Sentiment Analyst.
    *   **Tool**: `YahooFinanceNewsTool`.
    *   **Logic**: Fetches last 5 news items -> Sentiment Analysis.
4.  **Supervisor (`supervisor_node`)**:
    *   **Role**: Chief Editor.
    *   **Input**: Outputs from Trader, Quant, News.
    *   **Logic**: Synthesizes all inputs into a structured Markdown Financial Report. Persists result to Episodic Memory.

#### Memory Modules (`src/memory/`)
*   **Semantic Cache (`cache.py`)**:
    *   Wraps Tool calls.
    *   Hashes inputs (`ticker + tool_name`).
    *   If cache hit in Redis, returns stored result immediately (Speed optimization).
*   **Episodic Memory (`episodic.py`)**:
    *   Interface to **Qdrant**.
    *   Stores "Episodes" (Ticker, Date, Summary, Outcome).
    *   Allows Agents to query: "What happened last time NVDA had this RSI pattern?"

---

## 4. Application Layer - Frontend (`frontend/`)

A lightweight **Streamlit** application isolated in its own container.

*   **File**: `app.py`
*   **Logic**:
    1.  Accepts User Input (Ticker).
    2.  Sends `POST` request to `http://fastapi:8000/analyze`.
    3.  **Visualization**:
        *   Renders Interactive Line Chart (Matplotlib/Streamlit Native) of `predictions`.
        *   Renders Markdown Report from `report` field.
*   **Decoupling**: Does **not** share code with backend. Communicates strictly over HTTP.

---

## 5. Deployment & Configuration

*   **Configuration**: `src/config.py` uses `dataclasses` to manage hyperparameters (Epochs, Learning Rate, Batch Size, device).
*   **Dependencies**: managed in `requirements.txt`.
*   **Logging**: Centralized logger in `src/logger.py`.

## 6. Execution Flow (Example: "Analyze NVDA")

1.  **User** types "NVDA" in Dashboard -> Click "Generate".
2.  **Frontend** POSTs to `fastapi:8000/analyze`.
3.  **Backend** checks `outputs/NVDA/child_model.pt`.
    *   *If missing*: Triggers `train_child("NVDA")`.
        *   Loads `outputs/parent/parent_model.pt`.
        *   Fine-tunes on NVDA data.
        *   Saves artifacts.
4.  **Backend** runs `predict_child("NVDA")`.
    *   Returns 7-day forecast array.
5.  **Backend** initializes **LangGraph**.
    *   **Trader**: Reviews forecast.
    *   **Quant**: Checks risk.
    *   **News**: Fetches live news (External API).
    *   **Supervisor**: Compiles Report.
6.  **Backend** returns JSON `{ "predictions": [...], "report": "..." }`.
7.  **Frontend** displays Chart + Report.
