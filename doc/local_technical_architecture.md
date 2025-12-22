# Local Technical Architecture

This document maps the application's logical components to the local Docker Compose infrastructure. It aligns strictly with the modular breakdown of the system.

---

## Master Architecture Diagram

```mermaid
---
config:
  layout: dagre
---
flowchart TB
 subgraph Block1_UI["Frontend"]
        UI[("Streamlit Container
        (Port: 8501)")]
  end
 subgraph Block2_Backend["Backend & AI Orchestration"]
        API[("FastAPI Router")]
        ML_Engine["PyTorch
        (Transfer Learning)"]
        Agent_Engine["Agentic AI
        (LangGraph)"]
  end
 subgraph Block3_Storage["Data & Storage"]
        Redis[("Redis
        (Cache)")]
        Qdrant[("Qdrant
        (Vector Memory)")]
  end
 subgraph Block4_Obs["Observability"]
        Prom[("Prometheus")]
        Graf[("Grafana")]
  end
    API -- Trigger --> ML_Engine
    ML_Engine -- "1. Output Predictions" --> Agent_Engine
    Prom --> Graf
    User(("User")) -- Browser --> Block1_UI
    UI -- HTTP --> Block2_Backend
    Internet(("Internet / News")) -- "2. Input News" --> Agent_Engine
    Agent_Engine -- "3. Save Analysis" --> Qdrant
    Agent_Engine -- "4. Return Report" --> API
    API <-- Rate Limit --> Redis
    Prom -. Metrics .-> API

    style Agent_Engine fill:#ffccbc,stroke:#bf360c,stroke-width:1px
    style Block1_UI fill:#e1f5fe,stroke:#01579b
    style Block2_Backend fill:#fff3e0,stroke:#ff6f00
    style Block3_Storage fill:#e8f5e9,stroke:#2e7d32
    style Block4_Obs fill:#f3e5f5,stroke:#7b1fa2
```

---

## Component Details

### 1. Streamlit UI
Interacts with the User and visualizes results. Polling mechanism allows asynchronous feedback for long-running training jobs.

### 2. FastAPI Backend
The central nervous system.
- **Rate Limit**: Uses Redis to count requests per minute/day.
- **Task Runner**: Offloads `train_child` to background threads to keep the API responsive.

### 3. Feature Store (Feast)
- **Offline**: Stores historical data for training (Parquet).
- **Online**: Serves low-latency feature vectors for inference.

### 4. Training Pipeline
Implements **Transfer Learning**:
- Loads generic market knowledge from the Parent Model.
- Freezes LSTM layers.
- Trains only the specific ticker's behavior in the details.

### 5. Inference Pipeline
Robust Serving:
- Checks disk for models.
- If missing, auto-triggers training.
- Uses Feast for point-in-time correct features.

### 6. Tracking (MLflow)
Every single training run is logged. We use DagsHub as the remote server to store artifacts and visualize loss curves.

### 7. Agentic AI
A **LangGraph** workflow powered by **Ollama**:
- **Analyst**: Looks at numbers.
- **Expert**: Looks at news.
- **Critic**: Ensures quality.
- **Qdrant**: Acts as the "Long Term Memory" to recall past analyses.

### 8. Storage Services
- **Redis**: Ephemeral data (Cache, Locks, Counters).
- **Qdrant**: Persistent Semantic Vectors.

### 9. Observability
- **Prometheus**: Scrapes metrics.
- **Grafana**: Visualizes health.
- **Agent Eval**: Quality assurance for the AI responses.
