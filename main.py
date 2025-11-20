import json
import asyncio
import time
import uvicorn
import redis
import psutil
from fastapi import FastAPI, Body
from fastapi.responses import Response

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Gauge,
    Counter,
    Histogram,
    PROCESS_COLLECTOR,
    PLATFORM_COLLECTOR,
    GC_COLLECTOR,
)

from src.pipelines.training_pipeline import train_parent, train_child
from src.pipelines.inference_pipeline import predict_parent, predict_child
from src.utils import setup_dagshub_mlflow, initialize_dirs
from src.logger import get_logger
from src.exception import PipelineError


# ---------------------------------------------------------
# INITIAL SETUP
# ---------------------------------------------------------
setup_dagshub_mlflow()
initialize_dirs()
logger = get_logger()

app = FastAPI(title="MLOps Stock Pipeline", version="3.0")

redis_client = None
task_status = {}  # track background tasks: { "parent": {...}, ticker: {...} }


# ---------------------------------------------------------
# PROMETHEUS REGISTRY + METRICS
# ---------------------------------------------------------
registry = CollectorRegistry()
registry.register(GC_COLLECTOR)
registry.register(PLATFORM_COLLECTOR)
registry.register(PROCESS_COLLECTOR)

# System metrics
SYSTEM_CPU = Gauge("system_cpu_percent", "CPU percent", registry=registry)
SYSTEM_RAM = Gauge("system_ram_used_mb", "RAM in MB", registry=registry)
SYSTEM_DISK = Gauge("system_disk_used_mb", "Disk in MB", registry=registry)

# Redis
REDIS_STATUS = Gauge("redis_up", "Redis up=1/down=0", registry=registry)

# Training metrics
TRAINING_STATUS = Gauge("training_status", "Training status (0=idle,1=running,2=done)", ["task"], registry=registry)
TRAINING_DURATION = Histogram("training_duration_seconds", "Training duration", ["task"], registry=registry)

# Prediction metrics
PREDICTION_COUNTER = Counter("prediction_total", "Total predictions", ["type"], registry=registry)
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", ["type"], registry=registry)

# Cache metrics
CACHE_HIT = Counter("redis_cache_hit_total", "Cache hit", ["key"], registry=registry)
CACHE_MISS = Counter("redis_cache_miss_total", "Cache miss", ["key"], registry=registry)

Instrumentator(registry=registry).instrument(app)


# ---------------------------------------------------------
# SYSTEM METRICS
# ---------------------------------------------------------
def refresh_system_metrics():
    SYSTEM_CPU.set(psutil.cpu_percent())
    SYSTEM_RAM.set(psutil.virtual_memory().used / (1024 * 1024))
    SYSTEM_DISK.set(psutil.disk_usage("/").used / (1024 * 1024))


# ---------------------------------------------------------
# REDIS CACHE
# ---------------------------------------------------------
def get_or_set_cache(key, compute_fn, expire=86400):
    refresh_system_metrics()

    try:
        val = redis_client.get(key)
        if val:
            CACHE_HIT.labels(key).inc()
            return json.loads(val), True

        CACHE_MISS.labels(key).inc()
        result = compute_fn()
        redis_client.set(key, json.dumps(result), ex=expire)
        return result, False

    except Exception as e:
        logger.error(f"Redis cache error: {e}")
        return compute_fn(), False


# ---------------------------------------------------------
# STARTUP EVENT
# ---------------------------------------------------------
@app.on_event("startup")
def startup():
    global redis_client
    try:
        import platform
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Machine: {platform.machine()}")
        logger.info(f"Processor: {platform.processor()}")
        
        redis_client = redis.Redis(host="redis", port=6379, db=0)
        redis_client.ping()
        REDIS_STATUS.set(1)
        logger.info("Redis connected")
    except:
        REDIS_STATUS.set(0)
        logger.error("Redis connection failed")


# ---------------------------------------------------------
# TRAINING TASK HANDLER
# ---------------------------------------------------------
async def run_training(task_name, fn, *args):
    """Run training asynchronously and record metrics."""
    task_status[task_name] = {"status": "running", "start": time.time()}
    TRAINING_STATUS.labels(task_name).set(1)

    start_time = time.time()
    try:
        result = await asyncio.to_thread(fn, *args)
        duration = time.time() - start_time

        task_status[task_name] = {"status": "completed", "duration": duration}
        TRAINING_STATUS.labels(task_name).set(2)
        TRAINING_DURATION.labels(task_name).observe(duration)

        return result

    except Exception as e:
        task_status[task_name] = {"status": "failed", "error": str(e)}
        TRAINING_STATUS.labels(task_name).set(0)
        logger.error(f"{task_name} failed: {e}")


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------
@app.get("/health")
async def health():
    refresh_system_metrics()
    return {"status": "healthy"}


# ---------------------------------------------------------
# TRAINING ENDPOINTS (ASYNC)
# ---------------------------------------------------------
@app.post("/train-parent")
async def train_parent_api():
    task_name = "parent"
    asyncio.create_task(run_training(task_name, train_parent))
    return {"status": "started", "task": task_name}


@app.post("/train-child")
async def train_child_api(data: dict = Body(...)):
    ticker = data.get("ticker")
    if not ticker:
        return {"error": "ticker is required"}

    task_name = ticker.lower()
    asyncio.create_task(run_training(task_name, train_child, ticker))
    return {"status": "started", "task": task_name}


# ---------------------------------------------------------
# CHECK TASK STATUS
# ---------------------------------------------------------
@app.get("/status/{task_id}")
async def status(task_id: str):
    return task_status.get(task_id, {"status": "not_found"})


# ---------------------------------------------------------
# PREDICTION (ASYNC)
# ---------------------------------------------------------
@app.post("/predict-parent")
async def predict_parent_api():
    with PREDICTION_LATENCY.labels("parent").time():
        result, cached = get_or_set_cache("predict_parent", predict_parent)

    PREDICTION_COUNTER.labels("parent").inc()
    return {"cached": cached, "result": result}


@app.post("/predict-child")
async def predict_child_api(data: dict = Body(...)):
    ticker = data.get("ticker")
    if not ticker:
        return {"error": "ticker required"}

    cache_key = f"predict_child_{ticker}"

    with PREDICTION_LATENCY.labels("child").time():
        try:
            result, cached = get_or_set_cache(cache_key, lambda: predict_child(ticker))

        except PipelineError:
            logger.warning(f"Model missing for {ticker}. Training now...")
            await run_training(ticker, train_child, ticker)
            result, cached = get_or_set_cache(cache_key, lambda: predict_child(ticker))

    PREDICTION_COUNTER.labels("child").inc()
    return {"cached": cached, "result": result}


# ---------------------------------------------------------
# METRICS ENDPOINT
# ---------------------------------------------------------
@app.get("/metrics")
async def metrics():
    refresh_system_metrics()
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
