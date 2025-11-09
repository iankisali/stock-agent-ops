import uvicorn
from fastapi import FastAPI, Body
from src.pipelines.training_pipeline import train_parent, train_child
from src.pipelines.inference_pipeline import predict_parent, predict_child
from src.utils import setup_dagshub_mlflow, initialize_dirs
from src.logger import get_logger

setup_dagshub_mlflow()
initialize_dirs()
logger = get_logger()

app = FastAPI(title="MLOps Stock Pipeline", version="1.0")


@app.post("/train-parent")
def train_parent_api():
    logger.info("🚀 Training Parent Model (^GSPC)...")
    result = train_parent()
    return {"status": "ok", "result": result}


@app.post("/train-child")
def train_child_api(data: dict = Body(...)):
    ticker = data.get("ticker")
    if not ticker:
        return {"status": "error", "message": "ticker is required"}
    logger.info(f"🚀 Training Child Model for {ticker}...")
    result = train_child(ticker)
    return {"status": "ok", "result": result}


@app.post("/predict-parent")
def predict_parent_api():
    logger.info("🔮 Predicting Parent (^GSPC)...")
    preds = predict_parent()
    return {"status": "ok", "predictions": preds}


@app.post("/predict-child")
def predict_child_api(data: dict = Body(...)):
    ticker = data.get("ticker")
    if not ticker:
        return {"status": "error", "message": "ticker is required"}
    logger.info(f"🔮 Predicting Child {ticker}...")
    preds = predict_child(ticker)
    return {"status": "ok", "predictions": preds}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
