import json
import asyncio
import time
import uvicorn
import redis
import psutil
from concurrent.futures import ThreadPoolExecutor
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
logger = get_logger()

app = FastAPI(title="MLOps Stock Pipeline", version="3.0")

redis_client = None
task_status = {}  # {"parent": {...}, "ticker": {...}}

executor = ThreadPoolExecutor(max_workers=4)   # ⭐ Improved concurrency


# ---------------------------------------------------------
# PROMETHEUS REGISTRY + METRICS
# ---------------------------------------------------------
registry = CollectorRegistry()
registry.register(GC_COLLECTOR)
registry.register(PLATFORM_COLLECTOR)
registry.register(PROCESS_COLLECTOR)

SYSTEM_CPU = Gauge("system_cpu_percent", "CPU percent", registry=registry)
SYSTEM_RAM = Gauge("system_ram_used_mb", "RAM MB", registry=registry)
SYSTEM_DISK = Gauge("system_disk_used_mb", "Disk MB", registry=registry)

REDIS_STATUS = Gauge("redis_up", "Redis up=1/down=0", registry=registry)

TRAINING_STATUS = Gauge("training_status", "Training status", ["task"], registry=registry)
TRAINING_DURATION = Histogram("training_duration_seconds", "Training duration", ["task"], registry=registry)

PREDICTION_COUNTER = Counter("prediction_total", "Prediction count", ["type"], registry=registry)
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", ["type"], registry=registry)

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
        logger.error(f"Redis error: {e}")
        return compute_fn(), False


# ---------------------------------------------------------
# STARTUP
# ---------------------------------------------------------
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
    except:
        REDIS_STATUS.set(0)
        logger.error("Redis connection failed")


# ---------------------------------------------------------
# TRAINING TASK EXECUTOR WITH PROGRESS
# ---------------------------------------------------------
async def run_training(task_name, fn, *args):
    """
    Run training asynchronously using ThreadPoolExecutor,
    track status, progress, errors and duration.
    """
    task_status[task_name] = {
        "status": "running",
        "progress": 0,
        "start": time.time()
    }
    TRAINING_STATUS.labels(task_name).set(1)

    loop = asyncio.get_event_loop()
    start_time = time.time()

    try:
        future = loop.run_in_executor(executor, fn, *args)

        # Simple progress simulation (real progress can be integrated later)
        while not future.done():
            await asyncio.sleep(2)
            if task_status[task_name]["progress"] < 90:
                task_status[task_name]["progress"] += 10

        result = await future
        duration = time.time() - start_time

        task_status[task_name] = {
            "status": "completed",
            "duration": duration,
            "result": result,
            "progress": 100
        }

        TRAINING_STATUS.labels(task_name).set(2)
        TRAINING_DURATION.labels(task_name).observe(duration)

        return result

    except Exception as e:
        task_status[task_name] = {
            "status": "failed",
            "error": str(e),
            "progress": 0
        }
        TRAINING_STATUS.labels(task_name).set(0)
        logger.error(f"Training failed for {task_name}: {e}")


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------
@app.get("/health")
async def health():
    refresh_system_metrics()
    return {"status": "healthy"}


# ---------------------------------------------------------
# TRAINING ENDPOINTS
# ---------------------------------------------------------
@app.post("/train-parent")
async def train_parent_api():
    task_name = "parent"

    if task_name in task_status and task_status[task_name]["status"] == "running":
        return {"status": "already_running"}

    asyncio.create_task(run_training(task_name, train_parent))
    return {"status": "started", "task": task_name}


@app.post("/train-child")
async def train_child_api(data: dict = Body(...)):
    ticker = data.get("ticker")
    if not ticker:
        return {"error": "ticker is required"}

    task_name = ticker.lower()

    if task_name in task_status and task_status[task_name]["status"] == "running":
        return {"status": "already_running"}

    asyncio.create_task(run_training(task_name, train_child, ticker))
    return {"status": "started", "task": task_name}


# ---------------------------------------------------------
# STATUS ENDPOINTS
# ---------------------------------------------------------
@app.get("/status/{task_id}")
async def status(task_id: str):
    return task_status.get(task_id, {"status": "not_found"})


@app.get("/status-training")
async def status_training():
    return task_status


# ---------------------------------------------------------
# PREDICTION ENDPOINTS
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
            logger.warning(f"Model missing for {ticker}, training now...")
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