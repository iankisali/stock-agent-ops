import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict
import mlflow
from src.config import Config
from src.data.ingestion import fetch_ohlcv
# from src.utils import save_json
# from src.saving import load_model
from src.logger import get_logger
import onnxruntime as ort
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = get_logger()

def evaluate_model_temp(session: ort.InferenceSession, df: pd.DataFrame, scaler: StandardScaler, temp_dir: str, ticker: str) -> Dict:
    """Evaluate model performance and save metrics directly to MLflow without local persistence."""
    try:
        config = Config()
        vals = scaler.transform(df[config.features]).astype("float32")
        X, Y = [], []
        for t in range(config.context_len, len(vals) - config.pred_len):
            past = vals[t - config.context_len:t]
            fut = vals[t:t + config.pred_len]
            if past.shape == (config.context_len, config.input_size) and fut.shape == (config.pred_len, config.input_size):
                X.append(past)
                Y.append(fut)
            else:
                logger.error(f"Skipping invalid evaluation sample at index {t}: past shape {past.shape}, fut shape {fut.shape}")

        if not X:
            logger.error(f"No valid samples for evaluation for {ticker}")
            return {}

        X, Y = np.array(X), np.array(Y)
        preds = [session.run(None, {'input': x.reshape(1, config.context_len, config.input_size)})[0] for x in X]
        preds = np.array(preds)
        Y_ohlcv = Y.reshape(-1, config.input_size)[:, :5]
        preds_ohlcv = preds.reshape(-1, config.input_size)[:, :5]

        mse = mean_squared_error(Y_ohlcv, preds_ohlcv)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_ohlcv, preds_ohlcv)

        metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}
        
        # Save metrics to temporary file and log to MLflow
        metrics_filename = f"{ticker}_metrics.json"
        metrics_path = os.path.join(temp_dir, metrics_filename)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"{ticker} → MSE: {mse:.5f}, RMSE: {rmse:.5f}, R²: {r2:.5f}")

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_path, f"metrics/{ticker}")

        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed for {ticker}: {e}")
        return {}

def evaluate_model(session: ort.InferenceSession, df: pd.DataFrame, scaler: StandardScaler, out_dir: str, ticker: str) -> Dict:
    """Evaluate model performance (Model Evaluation Stage) - Legacy function for backward compatibility."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        config = Config()
        vals = scaler.transform(df[config.features]).astype("float32")
        X, Y = [], []
        for t in range(config.context_len, len(vals) - config.pred_len):
            past = vals[t - config.context_len:t]
            fut = vals[t:t + config.pred_len]
            if past.shape == (config.context_len, config.input_size) and fut.shape == (config.pred_len, config.input_size):
                X.append(past)
                Y.append(fut)
            else:
                logger.error(f"Skipping invalid evaluation sample at index {t}: past shape {past.shape}, fut shape {fut.shape}")

        if not X:
            logger.error(f"No valid samples for evaluation for {ticker}")
            return {}

        X, Y = np.array(X), np.array(Y)
        preds = [session.run(None, {'input': x.reshape(1, config.context_len, config.input_size)})[0] for x in X]
        preds = np.array(preds)
        Y_ohlcv = Y.reshape(-1, config.input_size)[:, :5]
        preds_ohlcv = preds.reshape(-1, config.input_size)[:, :5]

        mse = mean_squared_error(Y_ohlcv, preds_ohlcv)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_ohlcv, preds_ohlcv)

        metrics = {"MSE": mse, "RMSE": rmse, "R2": r2}
        metrics_filename = f"{ticker}_parent_metrics.json" if "parent" in out_dir else f"{ticker}_child_metrics.json"
        metrics_path = os.path.join(out_dir, metrics_filename)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"{ticker} → MSE: {mse:.5f}, RMSE: {rmse:.5f}, R²: {r2:.5f}")

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_path)

        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed for {ticker}: {e}")
        return {}