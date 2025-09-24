from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import src.utils as utils
import src.pipelines.training_pipeline as training
import src.pipelines.inference_pipeline as inference
from src.config import Config
from src.logger import get_logger
from typing import Dict
import mlflow
from mlflow.tracking import MlflowClient
import os

app = FastAPI()
logger = get_logger()
config = Config()

utils.initialize_dirs()

class ChildTicker(BaseModel):
    ticker: str

def get_production_model(ticker: str) -> tuple[any, any, str] | None:
    """Check MLflow Model Registry for a Production model for the given ticker."""
    try:
        client = MlflowClient()
        model_name = f"ChildModel_{ticker}"
        versions = client.search_model_versions(f"name='{model_name}'")
        for version in versions:
            if version.current_stage == "Production":
                model_uri = f"models:/{model_name}/{version.version}"
                model = mlflow.pytorch.load_model(model_uri)
                run_id = version.run_id
                scaler_path = client.download_artifacts(run_id, f"scalers/{ticker}/{ticker}_child_scaler.pkl")
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                logger.info(f"Found Production model for {ticker}: {model_uri}")
                return model, scaler, version.version
        logger.info(f"No Production model found for {ticker} in MLflow Model Registry.")
        return None
    except Exception as e:
        logger.error(f"Error checking MLflow Model Registry for {ticker}: {e}")
        return None

@app.get("/parent")
async def train_parent_model() -> Dict:
    """Train the parent model for S&P 500."""
    parent_model_path = os.path.join(config.parent_dir, "model.pt")
    parent_scaler_path = os.path.join(config.parent_dir, f"{config.parent_ticker}_parent_scaler.pkl")
    parent_onnx_path = os.path.join(config.parent_dir, "model.onnx")

    try:
        if os.path.exists(parent_model_path) and os.path.exists(parent_scaler_path) and os.path.exists(parent_onnx_path):
            logger.info(f"Using existing parent model at: {config.parent_dir}")
            return {
                "status": "success",
                "message": f"Using existing parent model at: {config.parent_dir}",
                "checkpoint": config.parent_dir
            }
        
        parent_summary = training.train_parent(
            ticker=config.parent_ticker,
            start=config.start_date,
            epochs=config.parent_epochs,
            out_dir=config.parent_dir
        )
        logger.info(f"Parent model trained and saved to: {parent_summary['checkpoint']}")
        return {
            "status": "success",
            "message": f"Parent model trained successfully for {config.parent_ticker}",
            "summary": parent_summary
        }
    except Exception as e:
        logger.error(f"Error training parent model: {e}")
        if os.path.exists(parent_model_path) and os.path.exists(parent_scaler_path):
            logger.info(f"Found existing parent model at: {config.parent_dir}. Returning existing model info.")
            return {
                "status": "success",
                "message": f"Found existing parent model at: {config.parent_dir} despite training failure",
                "checkpoint": config.parent_dir
            }
        logger.error("No existing parent model found. Cannot proceed without parent model.")
        raise HTTPException(status_code=500, detail=f"Error training parent model: {str(e)}. No existing model found.")

@app.post("/child/train")
async def train_child_model(child: ChildTicker) -> Dict:
    """Train a child model for the specified ticker if no Production model exists."""
    ticker = child.ticker
    if ticker not in config.child_tickers:
        logger.error(f"Invalid ticker: {ticker}. Must be one of {config.child_tickers}")
        raise HTTPException(status_code=400, detail=f"Invalid ticker: {ticker}. Must be one of {config.child_tickers}")

    try:
        # Check if a Production model exists in MLflow
        production_model = get_production_model(ticker)
        if production_model:
            logger.info(f"Production model already exists for {ticker}. Skipping training.")
            return {
                "status": "skipped",
                "message": f"Production model already exists for {ticker} in MLflow Model Registry",
                "model_name": f"ChildModel_{ticker}",
                "model_version": production_model[2]
            }

        # Train the child model
        summary = training.train_child(
            ticker=ticker,
            start=config.start_date,
            epochs=config.child_epochs,
            parent_dir=config.parent_dir,
            workdir=config.workdir
        )
        logger.info(f"Child model trained and saved to: {summary['checkpoint']}")
        logger.info(f"Predictions saved to: {summary['json']}")
        logger.info(f"Metrics saved to: {summary['checkpoint']}/{ticker}_child_metrics.json")
        
        return {
            "status": "success",
            "message": f"Child model trained successfully for {ticker}",
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error training child model for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Error training child model for {ticker}: {str(e)}")

@app.post("/child/predict")
async def predict_child_model(child: ChildTicker, return_base64_plot: bool = False) -> Dict:
    """Perform inference using the Production model for the specified ticker, training if none exists."""
    ticker = child.ticker
    if ticker not in config.child_tickers:
        logger.error(f"Invalid ticker: {ticker}. Must be one of {config.child_tickers}")
        raise HTTPException(status_code=400, detail=f"Invalid ticker: {ticker}. Must be one of {config.child_tickers}")

    try:
        # Use infer_child_stock for prediction, which handles MLflow registry check and training
        result = inference.infer_child_stock(
            ticker=ticker,
            start=config.start_date,
            epochs=config.child_epochs,
            parent_dir=config.parent_dir,
            workdir=config.workdir,
            train_if_not_exists=True,
            return_base64_plot=return_base64_plot
        )

        predictions = result.get('predictions', {})
        if "error" in predictions:
            logger.error(f"Prediction failed for {ticker}: {predictions['error']}")
            raise HTTPException(status_code=500, detail=f"Prediction failed for {ticker}: {predictions['error']}")

        response = {
            "status": "success",
            "message": f"Predictions generated for {ticker}",
            "predictions": {
                "ticker": ticker,
                "next_business_days": predictions.get('next_business_days', []),
                "next_day": {
                    "open": f"${predictions.get('next_day', {}).get('open', 'N/A'):.2f}",
                    "high": f"${predictions.get('next_day', {}).get('high', 'N/A'):.2f}",
                    "low": f"${predictions.get('next_day', {}).get('low', 'N/A'):.2f}",
                    "close": f"${predictions.get('next_day', {}).get('close', 'N/A'):.2f}"
                },
                "next_week": {
                    "high": f"${predictions.get('next_week', {}).get('high', 'N/A'):.2f}",
                    "low": f"${predictions.get('next_week', {}).get('low', 'N/A'):.2f}"
                }
            },
            "model_version": result.get('model_version', 'N/A'),
            "model_source": f"models:/ChildModel_{ticker}/{result.get('model_version', 'N/A')}" if result.get('model_version') != 'N/A' else f"{os.path.join(config.workdir, ticker)}"
        }

        if return_base64_plot and "plot_base64" in result:
            response["plot_base64"] = result["plot_base64"]

        return response
    except Exception as e:
        logger.error(f"Error predicting for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting for {ticker}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)