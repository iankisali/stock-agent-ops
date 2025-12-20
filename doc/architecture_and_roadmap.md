# Architecture & Roadmap Decisions

## 1. Load Balancer (LB)
### **Usage & Meaning**
A Load Balancer distributes incoming network traffic across multiple servers (instances) to ensure no single server bears too much load.

### **Do you need it?**
*   **For Development (Docker Compose)**: **No**, not strictly. You typically map ports directly (e.g., `8000:8000`). However, if you add a frontend (Streamlit/React) + Backend (FastAPI) + Tools (Grafana, MLflow), a *Reverse Proxy* (like Nginx) acts similarly to an LB to route `/api` to backend and `/` to frontend.
*   **For Production (Kubernetes)**: **Yes**.
    *   **Scalability**: If you set `replicas: 3` for your FastAPI app, K8s needs an internal Service (ClusterIP) to balance traffic between them.
    *   **Exposure**: To access the app from the internet, you use an `Ingress` or `LoadBalancer` service type that routes external traffic to your internal pods.

### **Implementation Plan**
1.  **Kubernetes**: We will use an **Ingress Controller** (like Nginx Ingress) defined in `k8s/ingress.yaml` to route traffic to the `fastapi-service`.
2.  **Docker Compose**: We can add an `nginx` container to handle routing if we want to simulate production, or stick to port mapping for simplicity.

---

## 2. LLM Evaluation (LangSmith)
### **Usage & Meaning**
As your Stock Agent uses Chains/Graphs (LangGraph), debugging "why did it say that?" is hard.
*   **LangSmith**: A platform by LangChain to trace, debug, and evaluate LLM apps.
*   **Evaluation**: Checking if the output is "correct". For stocks, is the reasoning sound? Did it cite sources?

### **How to Set it Up**
1.  **Tracing**:
    *   Sign up for LangSmith.
    *   Set environment variables:
        ```bash
        export LANGCHAIN_TRACING_V2=true
        export LANGCHAIN_API_KEY=<your-key>
        export LANGCHAIN_PROJECT="stock-prediction-agent"
        ```
    *   *Result*: Every API call, token count, and latency is logged automatically.

2.  **Evaluation (Testing)**:
    *   **Dataset**: Create a dataset of "Golden Questions" and "Ideal Answers".
        *   *Example*: Input: "Analyze NVDA", Expected Output: Must contain "Bullish/Bearish" and "Price Target".
    *   **Evaluator LLM**: Use a "Judge" model (e.g., GPT-4) to grade your agent's response against the ideal answer using LangSmith's testing framework.

---

## 3. Data & Model Drift
### **Usage & Meaning**
*   **Data Drift (Input changes)**: The shape of market data changes.
    *   *Example*: Volatility in 2020 vs 2024. If your model trained on 2020 data receives 2024 data, the distribution of features (e.g., `daily_return`) might be "drifted".
*   **Model Drift (Performance decays)**: The model's predictions differ from reality over time.
    *   *Example*: Model predicts \$100, actual is \$105. If this error (MSE) grows over weeks, the model has drifted.

### **Implementation Plan**
We will implement a **Drift Detection Pipeline**:

1.  **Data Drift (Input)**:
    *   **Tool**: **Evidently AI** (Open source, works with Pandas).
    *   **Action**:
        *   Save a "Reference Dataset" (Training data from `features.parquet`).
        *   Every day, take the "Current Batch" (Live data).
        *   Run `evidently_profile`.
        *   If drift detected -> **Trigger Retraining** (Auto-train capability we already have!).

2.  **Model Drift (Output)**:
    *   **Metric**: MSE / MAE.
    *   **Action**: We already track `TRAINING_MSE` in Prometheus.
    *   **Extension**: We need a "Ground Truth Job". 24 hours after a prediction, we fetch the *actual* close price, compare it to the *predicted* price, calculate the error, and log it to Prometheus (`prediction_error_last_24h`).

---

## Summary of Next Steps
1.  **LangSmith**: Add env vars to `.env` and `k8s/deployment.yaml`.
2.  **Drift**: Create a `monitor_drift.py` script using `evidently` to check features in `feature_repo`.
3.  **K8s**: Ensure `ingress.yaml` is set up for load balancing.
