
import json
import asyncio
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional
import uvicorn
import redis
import psutil
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, Counter, Histogram

from src.rate_limiter import rate_limit
from src.pipelines.training_pipeline import train_parent, train_child
from src.pipelines.inference_pipeline import predict_parent, predict_child
from src.utils import setup_dagshub_mlflow, initialize_dirs
from src.logger import get_logger
from src.exception import PipelineError

# Agent Imports
from src.agents.graph import analyze_stock

# Monitoring Imports
from src.monitoring.drift import check_drift
from src.monitoring.agent_eval import AgentEvaluator

# =========================================================
# SETUP
# =========================================================
setup_dagshub_mlflow()
logger = get_logger()
BASE_PATH = "outputs"
app = FastAPI(title="MLOps Stock Pipeline", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = None
executor = ThreadPoolExecutor(max_workers=4)

# =========================================================
# Prometheus Metrics
# =========================================================
registry = CollectorRegistry()
SYSTEM_CPU = Gauge("system_cpu_percent", "CPU percent", registry=registry)
SYSTEM_RAM = Gauge("system_ram_used_mb", "RAM MB", registry=registry)
SYSTEM_DISK = Gauge("system_disk_used_mb", "Disk Used MB", registry=registry)
REDIS_STATUS = Gauge("redis_up", "Redis up=1/down=0", registry=registry)
REDIS_KEYS = Gauge("redis_keys_total", "Number of keys in Redis", registry=registry)
TRAINING_STATUS = Gauge("training_status", "0=idle 1=running 2=completed", ["task_id"], registry=registry)
TRAINING_MSE = Gauge("training_mse_last", "Last training MSE", registry=registry)
TRAINING_DURATION = Histogram("training_duration_seconds", "Training duration in seconds", ["task_id"], registry=registry)
PREDICTION_COUNTER = Counter("prediction_total", "Total predictions", ["type"], registry=registry)
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", ["type"], registry=registry)
CACHE_HIT = Counter("redis_cache_hit_total", "Cache hits", ["key"], registry=registry)
CACHE_MISS = Counter("redis_cache_miss_total", "Cache misses", ["key"], registry=registry)

Instrumentator(registry=registry).instrument(app)

def refresh_system_metrics():
    SYSTEM_CPU.set(psutil.cpu_percent())
    SYSTEM_RAM.set(psutil.virtual_memory().used / (1024**2))
    SYSTEM_DISK.set(psutil.disk_usage('/').used / (1024**2))
    if redis_client:
        try:
            REDIS_KEYS.set(redis_client.dbsize())
        except:
            pass

# Task Status Tracking (Redis)
def get_task_key(task_id: str) -> str:
    return f"task_status:{task_id.lower()}"

def save_task_status(task_id: str, status_data: Dict[str, Any], ttl: int = 3600):
    """Save task status to Redis with TTL."""
    try:
        if redis_client:
            redis_client.set(get_task_key(task_id), json.dumps(status_data), ex=ttl)
    except Exception as e:
        logger.error(f"Failed to save task status for {task_id}: {e}")

def get_task_status_redis(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status from Redis."""
    try:
        if redis_client:
            val = redis_client.get(get_task_key(task_id))
            if val:
                return json.loads(val)
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
    return None

async def run_training_worker(task_id: str, fn, *args):
    """Actual training worker (runs in thread pool)."""
    loop = asyncio.get_event_loop()
    start_time = time.time()
    try:
        result = await loop.run_in_executor(executor, fn, *args)
        duration = time.time() - start_time
        TRAINING_DURATION.labels(task_id).observe(duration)
        
        status_data = {"status": "completed", "result": result, "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        save_task_status(task_id, status_data, ttl=3600) # Keep completed status for 1 hour
        
        TRAINING_STATUS.labels(task_id).set(2)
        
        # Update metrics if MSE is available in result
        if isinstance(result, dict) and "mse" in result:
             TRAINING_MSE.set(result["mse"])
    except Exception as e:
        status_data = {"status": "failed", "error": str(e), "failed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        save_task_status(task_id, status_data, ttl=3600)
        
        TRAINING_STATUS.labels(task_id).set(0)
        logger.error(f"Training failed: {e}")

async def run_blocking_fn(fn, *args):
    """Run a blocking function in the thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fn, *args)

async def run_training(task_id: str, fn, *args):
    """Start training in background and return immediately."""
    task_id = task_id.lower()
    
    # Check if already running using Redis
    current_status = get_task_status_redis(task_id)
    if current_status and current_status.get("status") == "running":
        return
        
    # Set initial status
    status_data = {"status": "running", "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    save_task_status(task_id, status_data, ttl=7200) # 2 hours max run time assumption
    
    TRAINING_STATUS.labels(task_id).set(1)
    
    # Spawn background task
    asyncio.create_task(run_training_worker(task_id, fn, *args))

# Dataclass
class AnalyzeRequest(BaseModel):
    ticker: str
    use_fmi: bool = False
    thread_id: str | None = None

# =========================================================
# Startup
# =========================================================
@app.on_event("startup")
async def startup():
    global redis_client
    initialize_dirs()
    
    # Retry logic for Redis connection
    for i in range(10):
        try:
            redis_client = redis.Redis(host="redis", port=6379, db=0)
            redis_client.ping()
            REDIS_STATUS.set(1)
            logger.info("✅ Systems online (Redis, MLflow, Agents)")
            return
        except Exception as e:
            logger.warning(f"⏳ Waiting for Redis... attempt {i+1}/10")
            await asyncio.sleep(5)
            
    REDIS_STATUS.set(0)
    logger.error("❌ Failed to connect to Redis after multiple attempts.")


# =========================================================
# Core Routes
# =========================================================
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.ticker:
        raise HTTPException(400, "Ticker required")

    try:
        return analyze_stock(req.ticker, thread_id=req.thread_id)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")

# =========================================================
# Granular Endpoints (Restored)
# =========================================================

@app.post("/train-parent")
@rate_limit(limit=5, window_sec=3600, key_prefix="train_parent")
async def train_parent_endpoint():
    """Trigger parent model training."""
    task_id = "parent_training"
    if get_task_status_redis(task_id) and get_task_status_redis(task_id).get("status") == "running":
         return {"status": "already running", "task_id": task_id}
         
    await run_training(task_id, train_parent)
    return {"status": "started", "task_id": task_id}

@app.post("/train-child")
@rate_limit(limit=5, window_sec=3600, key_prefix="train_child")
async def train_child_endpoint(request: Request):
    """Trigger child model training."""
    data = await request.json()
    ticker = data.get("ticker", "").strip().upper()
    if not ticker:
        raise HTTPException(400, "ticker is required")
        
    task_id = ticker.lower()
    
    # Logic Fix: Check if Parent Model exists
    # Assuming standard path from Config
    from src.config import Config
    cfg = Config()
    parent_path = os.path.join(cfg.parent_dir, f"{cfg.parent_ticker}_parent_model.pt")
    
    if not os.path.exists(parent_path):
        logger.warning("Parent model missing. Triggering parent training first.")
        # Trigger parent training (Background)
        
        parent_status = get_task_status_redis("parent_training")
        if not parent_status or parent_status.get("status") != "completed":
             await run_training("parent_training", train_parent)
             # Check if it started running
             parent_status = get_task_status_redis("parent_training")
             if parent_status and parent_status.get("status") == "running":
                 return {"status": "started_parent", "task_id": "parent_training", "detail": "Parent model missing. Training parent first."}
    
    # Check if already running (Redis)
    curr_status = get_task_status_redis(task_id)
    if curr_status and curr_status.get("status") == "running":
        return {"status": "running", "task_id": task_id, "detail": "Training already in progress"}

    await run_training(task_id, train_child, ticker)
    return {"status": "started", "task_id": task_id}

# =========================================================
# Helpers
# =========================================================
def get_or_set_cache(key: str, compute_fn, expire: int = 86400):
    """Helper to check Redis cache or compute and cache."""
    refresh_system_metrics()
    try:
        if redis_client:
            val = redis_client.get(key)
            if val:
                CACHE_HIT.labels(key).inc()
                return json.loads(val), True
        
        result = compute_fn()
        
        if redis_client:
            redis_client.set(key, json.dumps(result), ex=expire)
            CACHE_MISS.labels(key).inc()
        return result, False
    except Exception as e:
        logger.error(f"Redis cache error: {e}")
        return compute_fn(), False

@app.post("/predict-parent")
@rate_limit(limit=40, window_sec=3600, key_prefix="predict_parent")
async def predict_parent_endpoint():
    """Get parent model predictions."""
    PREDICTION_COUNTER.labels(type="parent").inc()
    start_time = time.time()
    try:
        result = await run_blocking_fn(predict_parent)
        PREDICTION_LATENCY.labels(type="parent").observe(time.time() - start_time)
        return {"result": result}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict-child")
@rate_limit(limit=40, window_sec=3600, key_prefix="predict_child")
async def predict_child_endpoint(request: Request, response: Response):
    """Get child model predictions."""
    data = await request.json()
    ticker = data.get("ticker", "").strip().upper()
    if not ticker:
        raise HTTPException(400, "ticker is required")
        
    task_id = ticker.lower()
    PREDICTION_COUNTER.labels(type="child").inc()

    start_time = time.time()
    try:
        # Cache layer for raw predictions
        def get_preds():
            return get_or_set_cache(f"predict_child_{ticker.lower()}", lambda: predict_child(ticker), expire=1800)
            
        preds, _ = await run_blocking_fn(get_preds)
        PREDICTION_LATENCY.labels(type="child").observe(time.time() - start_time)
        return {"result": preds}
    except (FileNotFoundError, PipelineError) as e:
        # Specific handling for missing models -> Auto-train
        if "Missing" in str(e) or "not found" in str(e):
            logger.info(f"Model missing for {ticker}, triggering auto-training.")
            
            # Check if already running
            status = get_task_status_redis(task_id)
            if status and status.get("status") == "running":
                 response.status_code = 202
                 return {"status": "training", "detail": "Training in progress. Please retry later."}
            
            # Trigger training
            await run_training(task_id, train_child, ticker)
            response.status_code = 202
            return {"status": "training_started", "detail": f"Model for {ticker} missing. Training started."}
            
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Check status of background training task (Async)."""
    tid = task_id.lower()
    if tid == "parent":
        tid = "parent_training"
        
    status = get_task_status_redis(tid)
    if not status:
        raise HTTPException(404, f"Task '{task_id}' not found.")
    
    # Calculate elapsed time if running
    response = status.copy()
    if response.get("status") == "running" and "start_time" in response:
        try:
            start_dt = datetime.strptime(response["start_time"], "%Y-%m-%d %H:%M:%S")
            response["elapsed_seconds"] = int((datetime.now() - start_dt).total_seconds())
        except Exception:
            pass
    
    # Remove start_time for cleaner output
    response.pop("start_time", None)
    return response
@app.get("/system/logs")
async def get_logs(lines: int = 100):
    """Retrieve the latest log lines."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        return {"logs": "Log directory not found."}
    
    # Get latest log file
    files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    if not files:
        return {"logs": "No log files found."}
    
    latest_file = sorted(files)[-1]
    path = os.path.join(log_dir, latest_file)
    
    try:
        with open(path, "r") as f:
            # Read last N lines
            content = f.readlines()
            last_lines = content[-lines:]
            return {"logs": "".join(last_lines), "filename": latest_file}
    except Exception as e:
        return {"error": f"Failed to read logs: {e}"}

@app.post("/monitor/parent")
async def monitor_parent():
    """
    Monitor ONLY the Parent Model (^GSPC).
    Triggers Drift & wrapper evaluation for the market index.
    """
    from src.config import Config
    cfg = Config()
    ticker = cfg.parent_ticker # ^GSPC
    
    # Run Drift
    try:
        drift_res = check_drift(ticker, BASE_PATH)
    except Exception as e:
        drift_res = {"status": "failed", "error": str(e)}

    # Run Agent Eval (Market Analysis Agent)
    try:
        evaluator = AgentEvaluator(BASE_PATH)
        eval_res = evaluator.evaluate_live(ticker)
    except Exception as e:
        eval_res = {"status": "failed", "error": str(e)}

    return {
        "ticker": ticker,
        "type": "Parent Model (Market Index)",
        "drift": drift_res,
        "agent_eval": eval_res,
        "links": {
            "get_drift_json": f"/monitor/{ticker}/drift",
            "get_eval_json": f"/monitor/{ticker}/eval"
        }
    }

@app.post("/monitor/{ticker}")
async def trigger_monitoring(ticker: str):
    """
    Trigger live monitoring for a ticker.
    Drift is ONLY calculated if ticker is the parent ticker.
    """
    from src.config import Config
    cfg = Config()
    clean_ticker = ticker.strip().upper()
    is_parent = (clean_ticker == cfg.parent_ticker)
    
    # Run Drift ONLY for parent
    drift_res = {"status": "skipped", "detail": "Drift calculation reserved for parent model."}
    if is_parent:
        try:
            drift_res = check_drift(clean_ticker, BASE_PATH)
        except Exception as e:
            drift_res = {"status": "failed", "error": str(e)}
        
    # Run Agent Eval (Always)
    try:
        evaluator = AgentEvaluator(BASE_PATH)
        eval_res = evaluator.evaluate_live(clean_ticker)
    except Exception as e:
        eval_res = {"status": "failed", "error": str(e)}
            
    return {
        "ticker": clean_ticker,
        "is_parent": is_parent,
        "drift": drift_res,
        "agent_eval": eval_res
    }

@app.get("/monitor/{ticker}/drift")
def get_drift_result(ticker: str):
    """Get the latest drift result JSON (if we save it, otherwise HTML path)."""
    # Currently drift.py only saves HTML. We can return the path or status.
    # Ideally should save JSON too.
    # For now, rerun check or checking file existence? 
    # Let's just return file info.
    t = ticker.lower()
    drift_dir = os.path.join(BASE_PATH, t, "drift")
    
    if not os.path.exists(drift_dir):
        raise HTTPException(404, "No drift report found")
    
    json_path = os.path.join(drift_dir, "latest_drift.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
            
    files = os.listdir(drift_dir)
    return {"files": files, "message": "Access HTML report in outputs/", "detail": "JSON summary missing."}

@app.get("/monitor/{ticker}/eval")
def get_eval_result(ticker: str):
    """Get the latest agent eval result JSON."""
    t = ticker.lower()
    path = os.path.join(BASE_PATH, t, "agent_eval", "latest_eval.json")
    
    if not os.path.exists(path):
        raise HTTPException(404, "No evaluation found. Run POST /monitor/{ticker} first.")
    
    with open(path, "r") as f:
        return json.load(f)

@app.delete("/system/reset")
def reset_system():
    """Wipe all system data (Redis Cache, Qdrant Memory)."""
    try:
        # 1. Wipe Redis
        if redis_client:
            redis_client.flushall()
            logger.info("✅ Redis flushed")
            
        # 2. Wipe Qdrant
        # We need to instantiate client here or reuse if available
        # Simple approach: use the helper class
        from src.memory.semantic_cache import SemanticCache
        mem = SemanticCache(host="qdrant", port=6333)
        mem.client.delete_collection(mem.collection_name)
        mem._ensure_collection() # Recreate empty
        logger.info("✅ Qdrant wiped")

        # 3. Wipe Feast (Filesystem)
        repo_path = os.path.join(os.getcwd(), "feature_repo")
        data_path = os.path.join(repo_path, "data")
        
        # Delete registry and parquet to ensure clean slate
        # Note: 'src/data/ingestion.py' recreates these
        for file in ["registry.db", "features.parquet"]:
             p = os.path.join(data_path, file)
             if os.path.exists(p):
                 os.remove(p)
                 logger.info(f"✅ Deleted Feast file: {p}")

        # 4. Wipe Outputs Directory
        import shutil
        if os.path.exists(BASE_PATH):
            shutil.rmtree(BASE_PATH)
            os.makedirs(BASE_PATH)
            logger.info(f"✅ Wiped and recreated outputs directory: {BASE_PATH}")

        return {"status": "System Reset Complete", "details": "Redis flushed, Qdrant emptied, Feast registry/data wiped, Outputs directory cleared."}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(500, f"Reset failed: {e}")

@app.get("/system/cache")
def inspect_cache(ticker: Optional[str] = None):
    """
    Inspect Redis Cache.
    - No params: List all tickers with cached predictions.
    - ?ticker=XYZ: Get cached data for XYZ.
    """
    if not redis_client:
        raise HTTPException(503, "Redis not connected")

    # Pattern for prediction cache
    pattern = "predict_child_*"
    # scan_iter is safer than keys for production, but keys is fine here
    try:
        keys = [k.decode("utf-8") for k in redis_client.keys(pattern)]
    except Exception as e:
         logger.error(f"Redis scan failed: {e}")
         return {"error": str(e)}

    # Extract ticker names
    # Key format: predict_child_nvda
    cached_map = {k.replace("predict_child_", "").upper(): k for k in keys}

    if not ticker:
        return {
            "cached_tickers": list(cached_map.keys()),
            "count": len(cached_map),
            "detail": "Pass ?ticker=SYMBOL to see data."
        }
    
    # Fetch specific
    target_ticker = ticker.strip().upper()
    target_key = cached_map.get(target_ticker)
    
    if not target_key:
        raise HTTPException(404, f"No cache found for {target_ticker}")

    val = redis_client.get(target_key)
    return json.loads(val)

@app.get("/metrics")
async def prometheus_metrics():
    refresh_system_metrics()
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)