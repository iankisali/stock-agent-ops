from fastapi import FastAPI
from .routes.health import router as health_router
from .routes.training import router as training_router
from .routes.inference import router as inference_router
from src.logger import get_logger

logger = get_logger()

app = FastAPI(
    title="Stock Prediction MLOps API",
    description="API for training and predicting stock prices using an MLOps pipeline.",
    version="0.0.1",
    docs_url="/docs",
    openapi_tags=[
        {"name": "Health", "description": "Health check endpoints"},
        {"name": "Training", "description": "Endpoints for training parent and child models"},
        {"name": "Inference", "description": "Endpoints for stock price predictions and plots"}
    ]
)

# Include routers
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(training_router, prefix="/train", tags=["Training"])
app.include_router(inference_router, prefix="/predict", tags=["Inference"])

@app.on_event("startup")
async def startup_event():
    logger.info("Stock Prediction API started")