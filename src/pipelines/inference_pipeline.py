import os
from dotenv import load_dotenv
import pickle
import pandas as pd
from typing import Dict
from src.config import Config
from src.exception import PipelineError
from src.data.ingestion import fetch_ohlcv
from src.model.saving import load_model
from src.inference import predict_one_step_and_week
from src.utils import save_json, setup_dagshub_mlflow
from src.pipelines.training_pipeline import train_child
from src.logger import get_logger
import mlflow
from mlflow.tracking import MlflowClient

# Load environment variables from .env file
load_dotenv()

# Initialize DagsHub MLflow tracking (with automatic fallback to local)
setup_dagshub_mlflow()

logger = get_logger()


def load_model_from_registry(model_name: str, stage: str = "Production"):
    """Load model and scaler from MLflow model registry for the specified stage."""
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        for version in versions:
            if version.current_stage == stage:
                model_uri = f"models:/{model_name}/{version.version}"
                model = mlflow.pytorch.load_model(model_uri)
                run_id = version.run_id
                scaler_path = client.download_artifacts(
                    run_id,
                    f"scalers/{model_name.split('_')[-1]}/{model_name.split('_')[-1]}_child_scaler.pkl"
                )
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                logger.info(f"Loaded {stage} model {model_name} version {version.version} from MLflow registry")
                return model, scaler, version.version
        raise FileNotFoundError(f"No {stage} model found for {model_name} in MLflow registry")
    except Exception as e:
        logger.error(f"Failed to load model {model_name} from MLflow registry: {e}")
        raise FileNotFoundError(f"Failed to load model {model_name} from MLflow registry: {e}")


def predict_child(ticker: str, parent_dir: str = Config().parent_dir, workdir: str = Config().workdir) -> Dict:
    """Predict using child model (Inference Stage) without plots."""
    try:
        child_dir = os.path.join(workdir, ticker)
        # Load model from MLflow registry
        try:
            session, scaler, version = load_model_from_registry(f"ChildModel_{ticker}", stage="Production")
            logger.info(f"Loaded Production model for {ticker} from MLflow registry")
        except FileNotFoundError:
            # Fallback to local filesystem
            session, scaler = load_model(path=child_dir, model_type="child", ticker=ticker)
            logger.info(f"Loaded model for {ticker} from filesystem: {child_dir}")
            version = "N/A"

        df = fetch_ohlcv(ticker)
        payload = predict_one_step_and_week(session, df, scaler, ticker)
        json_filename = f"{ticker}_inference_forecast.json"
        save_json(payload, os.path.join(child_dir, json_filename))

        return {
            "ticker": ticker,
            "predictions": payload,
            "model_version": version
        }
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


def infer_child_stock(
    ticker: str,
    start: str = Config().start_date,
    epochs: int = Config().child_epochs,
    parent_dir: str = Config().parent_dir,
    workdir: str = Config().workdir,
    train_if_not_exists: bool = True
) -> Dict:
    """Infer stock predictions, training if necessary, without plots."""
    child_dir = os.path.join(workdir, ticker)

    try:
        session, scaler, version = load_model_from_registry(f"ChildModel_{ticker}", stage="Production")
        logger.info(f"Loaded Production model for {ticker} from MLflow registry")
    except FileNotFoundError as e:
        if train_if_not_exists:
            logger.info(f"No Production model found for {ticker}. Training now...")
            try:
                train_summary = train_child(ticker, start, epochs, parent_dir, workdir)
                child_dir = train_summary["checkpoint"]
                session, scaler = load_model(path=child_dir, model_type="child", ticker=ticker)
                version = train_summary.get("model_version", "N/A")
            except Exception as e:
                raise PipelineError(f"Failed to train child model for {ticker}: {e}")
        else:
            raise PipelineError(f"Child model for {ticker} not found in MLflow registry or {child_dir}.") from e

    logger.info(f"Running inference for {ticker}...")
    df = fetch_ohlcv(ticker, start)
    predictions = predict_one_step_and_week(session, df, scaler, ticker)

    return {
        "ticker": ticker,
        "predictions": predictions,
        "model_version": version
    }
