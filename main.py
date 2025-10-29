from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import redis
import src.utils as utils
import src.pipelines.training_pipeline as training
import src.pipelines.inference_pipeline as inference
from src.config import Config
from src.logger import get_logger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import uvicorn

# ------------------------------
# Setup
# ------------------------------
logger = get_logger()
utils.initialize_dirs()

# Initialize DagsHub MLflow tracking (with automatic fallback to local)
utils.setup_dagshub_mlflow()

app = FastAPI(title="Stock Prediction MLOps API")
config = Config()

# Prometheus Metrics
REQUEST_COUNT = Counter("api_requests_total", "Total number of API requests")
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency in seconds")

# ------------------------------
# Redis
# ------------------------------
try:
    redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("✓ Connected to Redis successfully.")
except Exception as e:
    logger.error(f"✗ Failed to connect to Redis: {e}")
    redis_client = None  # Continue gracefully even if Redis isn't available

# ------------------------------
# Middleware for Prometheus
# ------------------------------
@app.middleware("http")
async def prometheus_metrics(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(latency)
    return response

# ------------------------------
# Schemas
# ------------------------------
class ChildRequest(BaseModel):
    ticker: str

# ------------------------------
# Parent Training
# ------------------------------
@app.get("/train-parent")
def train_parent():
    parent_model_path = os.path.join(config.parent_dir, "model.pt")
    parent_scaler_path = os.path.join(config.parent_dir, "parent_scaler.pkl")
    parent_onnx_path = os.path.join(config.parent_dir, "model.onnx")

    # if os.path.exists(parent_model_path) and os.path.exists(parent_scaler_path) and os.path.exists(parent_onnx_path):
    #     msg = f"✓ Using existing parent model at: {config.parent_dir}"
    #     logger.info(msg)
    #     return {"message": msg}

    try:
        parent_summary = training.train_parent(
            ticker=config.parent_ticker,
            start=config.start_date,
            epochs=config.parent_epochs,
            out_dir=config.parent_dir
        )
        logger.info(f"✓ Parent model trained and saved to: {config.parent_dir}")
        return {"message": "Parent model trained successfully", "summary": parent_summary}
    except Exception as e:
        logger.error(f"✗ Error training parent model: {e}")
        if os.path.exists(parent_model_path) and os.path.exists(parent_scaler_path):
            msg = f"✓ Found existing parent model at: {config.parent_dir}. Continuing..."
            logger.info(msg)
            return {"message": msg}
        raise HTTPException(status_code=500, detail=f"Parent model training failed: {e}")

# ------------------------------
# Child Training + Caching
# ------------------------------
@app.post("/train-child")
def train_child(request: ChildRequest):
    ticker = request.ticker.upper()
    cache_key = f"predictions:{ticker}"

    if redis_client:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            logger.info(f"✓ Cache hit for {ticker}")
            return json.loads(cached_data)

    try:
        summary = training.train_child(
            ticker=ticker,
            start=config.start_date,
            epochs=config.child_epochs,
            parent_dir=config.parent_dir,
            workdir=config.workdir
        )
        logger.info(f"✓ {ticker} model trained and saved to: {summary['checkpoint']}")
        logger.info(f"✓ Predictions saved to: {summary['json']}")

        preds = inference.predict_child(ticker=ticker, parent_dir=config.parent_dir, workdir=config.workdir)
        if "error" in preds:
            raise HTTPException(status_code=500, detail=preds["error"])

        response = {"ticker": ticker, "summary": summary, "predictions": preds.get("predictions", {})}

        if redis_client:
            redis_client.setex(cache_key, 6 * 3600, json.dumps(response))
            logger.info(f"✓ Cached results for {ticker} in Redis")

        return response

    except Exception as e:
        logger.error(f"✗ Error training or predicting for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------
# Health Check
# ------------------------------
@app.get("/health")
def health_check():
    status = {"api": "ok"}
    try:
        if redis_client and redis_client.ping():
            status["redis"] = "connected"
        else:
            status["redis"] = "not connected"
    except Exception as e:
        status["redis"] = f"error: {e}"
    return status

# ------------------------------
# Prometheus Metrics Endpoint
# ------------------------------
@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
