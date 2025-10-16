from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import src.utils as utils
import src.pipelines.training_pipeline as training
import src.pipelines.inference_pipeline as inference
from src.config import Config
from src.logger import get_logger

# Initialize logger and directories
logger = get_logger()
utils.initialize_dirs()

app = FastAPI(title="Stock Prediction MLOps API")

config = Config()

class ChildRequest(BaseModel):
    ticker: str

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/train-parent")
def train_parent():
    """Train the parent model for S&P 500"""
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
        raise HTTPException(status_code=500, detail=f"Parent model training failed: {e}")

@app.post("/train-child")
def train_child(request: ChildRequest):
    """Train child model for a given ticker and return predictions"""
    ticker = request.ticker.upper()
    
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

        return {"ticker": ticker, "summary": summary, "predictions": preds.get("predictions", {})}
    except Exception as e:
        logger.error(f"✗ Error training or predicting for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
