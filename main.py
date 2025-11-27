import json
import asyncio
import time
import os
from typing import Dict, Any
import uvicorn
import redis
import psutil
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import (
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry,
    Gauge, Counter, Histogram, PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR
)

from src.pipelines.training_pipeline import train_parent, train_child
from src.pipelines.inference_pipeline import predict_parent, predict_child
from src.utils import setup_dagshub_mlflow, initialize_dirs
from src.logger import get_logger
from src.exception import PipelineError

# =========================================================
# SETUP
# =========================================================
setup_dagshub_mlflow()
logger = get_logger()
BASE_PATH = "outputs"
app = FastAPI(title="MLOps Stock Pipeline", version="3.0")

redis_client = None
executor = ThreadPoolExecutor(max_workers=4)

# Individual task status — clean & scalable
task_status: Dict[str, Dict[str, Any]] = {}

# =========================================================
# Prometheus Metrics
# =========================================================
registry = CollectorRegistry()
registry.register(GC_COLLECTOR)
registry.register(PLATFORM_COLLECTOR)
registry.register(PROCESS_COLLECTOR)

SYSTEM_CPU = Gauge("system_cpu_percent", "CPU percent", registry=registry)
SYSTEM_RAM = Gauge("system_ram_used_mb", "RAM MB", registry=registry)
SYSTEM_DISK = Gauge("system_disk_used_mb", "Disk MB", registry=registry)
REDIS_STATUS = Gauge("redis_up", "Redis up=1/down=0", registry=registry)

TRAINING_STATUS = Gauge("training_status", "0=idle 1=running 2=completed", ["task_id"], registry=registry)
TRAINING_DURATION = Histogram("training_duration_seconds", "Training duration", ["task_id"], registry=registry)

PREDICTION_COUNTER = Counter("prediction_total", "Total predictions", ["type"], registry=registry)
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", ["type"], registry=registry)

CACHE_HIT = Counter("redis_cache_hit_total", "Cache hits", ["key"], registry=registry)
CACHE_MISS = Counter("redis_cache_miss_total", "Cache misses", ["key"], registry=registry)

Instrumentator(registry=registry).instrument(app)

def refresh_system_metrics():
    SYSTEM_CPU.set(psutil.cpu_percent())
    SYSTEM_RAM.set(psutil.virtual_memory().used / (1024**2))
    SYSTEM_DISK.set(psutil.disk_usage("/").used / (1024**2))

# =========================================================
# Redis Helper
# =========================================================
def get_or_set_cache(key: str, compute_fn, expire: int = 86400):
    refresh_system_metrics()
    try:
        val = redis_client.get(key)
        if val is not None:
            CACHE_HIT.labels(key).inc()
            return json.loads(val), True
        result = compute_fn()
        redis_client.set(key, json.dumps(result), ex=expire)
        CACHE_MISS.labels(key).inc()
        return result, False
    except Exception as e:
        logger.error(f"Redis error: {e}")
        return compute_fn(), False

# =========================================================
# Auto-cleanup old tasks (10 minutes)
# =========================================================
def schedule_cleanup(task_id: str, delay: int = 600):
    async def cleanup():
        await asyncio.sleep(delay)
        task_status.pop(task_id, None)
        TRAINING_STATUS.labels(task_id).set(0)
    asyncio.create_task(cleanup())

# =========================================================
# Training Runner
# =========================================================
async def run_training(task_id: str, fn, *args):
    task_id = task_id.lower()
    task_status[task_id] = {
        "status": "running",
        "progress": 0,
        "start_time": time.time()
    }
    TRAINING_STATUS.labels(task_id).set(1)

    loop = asyncio.get_event_loop()
    start = time.time()

    # Background progress updater
    def update_progress():
        for p in range(10, 96, 10):
            if task_status.get(task_id, {}).get("status") == "running":
                task_status[task_id]["progress"] = p
            time.sleep(3)
        if task_status.get(task_id, {}).get("status") == "running":
            task_status[task_id]["progress"] = 95

    loop.run_in_executor(executor, update_progress)

    try:
        result = await loop.run_in_executor(executor, fn, *args)
        duration = time.time() - start

        task_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "duration": round(duration, 2),
            "completed_at": time.time(),
            "result": result or "success"
        }
        TRAINING_STATUS.labels(task_id).set(2)
        TRAINING_DURATION.labels(task_id).observe(duration)
        schedule_cleanup(task_id)

    except Exception as e:
        task_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "error": str(e)
        }
        TRAINING_STATUS.labels(task_id).set(0)
        logger.error(f"Training failed for {task_id}: {e}")
        schedule_cleanup(task_id)

# =========================================================
# Startup
# =========================================================
@app.on_event("startup")
def startup():
    global redis_client
    try:
        setup_dagshub_mlflow()
        initialize_dirs()
        redis_client = redis.Redis(host="redis", port=6379, db=0)
        redis_client.ping()
        REDIS_STATUS.set(1)
        logger.info("Redis connected")
    except Exception as e:
        REDIS_STATUS.set(0)
        logger.error(f"Redis failed: {e}")

# =========================================================
# Routes
# =========================================================
@app.get("/health")
async def health():
    refresh_system_metrics()
    return {"status": "healthy"}

@app.post("/train-parent")
async def train_parent_api():
    task_id = "parent"
    if task_status.get(task_id, {}).get("status") == "running":
        return JSONResponse({"status": "already_running", "task_id": task_id}, status_code=409)
    asyncio.create_task(run_training(task_id, train_parent))
    return {"status": "started", "task_id": task_id}

@app.post("/train-child")
async def train_child_api(request: Request):
    data = await request.json()
    ticker = data.get("ticker", "").strip()
    if not ticker:
        raise HTTPException(400, "ticker is required")
    
    task_id = ticker.lower()
    if task_status.get(task_id, {}).get("status") == "running":
        return JSONResponse({"status": "already_running", "task_id": task_id}, status_code=409)
    
    asyncio.create_task(run_training(task_id, train_child, ticker))
    return {"status": "started", "task_id": task_id}

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    status = task_status.get(task_id.lower())
    if not status:
        raise HTTPException(404, "Task not found")
    return status

@app.post("/predict-parent")
async def predict_parent_api():
    with PREDICTION_LATENCY.labels("parent").time():
        result, cached = get_or_set_cache("predict_parent", predict_parent)
    PREDICTION_COUNTER.labels("parent").inc()
    return {"cached": cached, "result": result}

@app.post("/predict-child")
async def predict_child_api(request: Request):
    data = await request.json()
    ticker = data.get("ticker", "").strip()
    if not ticker:
        raise HTTPException(400, "ticker is required")

    cache_key = f"predict_child_{ticker.lower()}"

    with PREDICTION_LATENCY.labels("child").time():
        try:
            result, cached = get_or_set_cache(cache_key, lambda: predict_child(ticker))
        except PipelineError:
            logger.warning(f"Model missing for {ticker}, training on-demand...")
            await run_training(ticker.lower(), train_child, ticker)
            result, cached = get_or_set_cache(cache_key, lambda: predict_child(ticker))

    PREDICTION_COUNTER.labels("child").inc()
    return {"cached": cached, "result": result}

@app.post("/metrics")
async def get_metrics(request: Request):
    data = await request.json()
    ticker = data.get("ticker", "").strip().lower()
    if not ticker:
        raise HTTPException(400, "ticker is required")

    if ticker == "parent":
        path = os.path.join(BASE_PATH, "parent_parent_metrics.json")
    else:
        path = os.path.join(BASE_PATH, ticker, f"{ticker}_child_metrics.json")

    if not os.path.exists(path):
        raise HTTPException(404, "Metrics file not found")

    with open(path) as f:
        return json.load(f)

@app.get("/metrics")
async def prometheus_metrics():
    refresh_system_metrics()
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)