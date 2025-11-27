import os
import pickle
import joblib
import torch
import base64
import io
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from feast import FeatureStore
from src.model.definition import LSTMModel
from src.config import Config
from src.data.ingestion import fetch_ohlcv
from src.inference import predict_one_step_and_week
from src.logger import get_logger
from src.exception import PipelineError

logger = get_logger()
cfg = Config()

# Initialize Feast Feature Store
try:
    store = FeatureStore(repo_path="feature_repo")
except Exception as e:
    logger.warning(f"Feast Feature Store not initialized: {e}")
    store = None


# =============================================================
# 🔧 SAFE LOADERS
# =============================================================

def _safe_load_scaler(path: str):
    """Try loading scaler via pickle → fallback to joblib."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            return joblib.load(path)
        except Exception as e:
            raise PipelineError(f"Scaler load failed ({path}): {e}")


def _load_local_model(ticker: str, model_type: str):
    """Load PyTorch model + scaler directly from local filesystem."""
    try:
        if model_type == "parent":
            base_dir = cfg.parent_dir
            pt_path = os.path.join(base_dir, f"{cfg.parent_ticker}_parent_model.pt")
            scaler_path = os.path.join(base_dir, f"{cfg.parent_ticker}_parent_scaler.pkl")
        else:
            base_dir = os.path.join(cfg.workdir, ticker)
            pt_path = os.path.join(base_dir, f"{ticker}_child_model.pt")
            scaler_path = os.path.join(base_dir, f"{ticker}_child_scaler.pkl")

        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Missing PyTorch file for {ticker}: {pt_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler file for {ticker}: {scaler_path}")

        # Load PyTorch Model
        model = LSTMModel().to(cfg.device)
        model.load_state_dict(torch.load(pt_path, map_location=cfg.device))
        model.eval()
        
        scaler = _safe_load_scaler(scaler_path)
        logger.info(f"✅ Loaded {model_type} model for {ticker}")
        return model, scaler

    except Exception as e:
        raise PipelineError(f"Local model load failed for {ticker}: {e}")


# =============================================================
# 📊 PLOTTING HELPER
# =============================================================

def _generate_plot(history_df, forecast_data, ticker):
    """
    Generates a plot of historical data and forecast, returns base64 string.
    Also logs the figure to MLflow.
    """
    try:
        # Prepare data
        last_30_days = history_df.tail(30)
        dates = pd.to_datetime(last_30_days["date"])
        closes = last_30_days["close"]
        
        forecast_dates = [pd.to_datetime(d["date"]) for d in forecast_data]
        forecast_closes = [d["close"] for d in forecast_data]

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(dates, closes, label="Historical (Last 30 Days)", color="blue")
        plt.plot(forecast_dates, forecast_closes, label="Forecast", color="red", linestyle="--")
        
        plt.title(f"Price Prediction for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # Log to MLflow
        try:
            mlflow.log_figure(plt.gcf(), f"{ticker}_prediction_plot.png")
        except Exception as e:
            logger.warning(f"Failed to log figure to MLflow: {e}")

        plt.close()
        return plot_base64

    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        return None


# =============================================================
# 🧠 PREDICTION FUNCTIONS
# =============================================================

def predict_parent():
    """Predict using locally saved parent model."""
    try:
        ticker = cfg.parent_ticker
        model, scaler = _load_local_model(ticker, "parent")
        df = fetch_ohlcv(ticker)
        preds = predict_one_step_and_week(model, df, scaler, ticker)

        logger.info(f"✅ Parent prediction completed for {ticker}")
        return preds

    except Exception as e:
        logger.error(f"Parent prediction failed: {e}")
        raise PipelineError(f"Parent prediction failed: {e}")


def predict_child(ticker: str):
    """Predict using locally saved child model."""
    try:
        # Fetch features from Feast (Demonstration)
        if store:
            try:
                feature_vector = store.get_online_features(
                    features=[
                        "stock_stats:Open",
                        "stock_stats:High",
                        "stock_stats:Low",
                        "stock_stats:Close",
                        "stock_stats:Volume",
                        "stock_stats:RSI14",
                        "stock_stats:MACD",
                    ],
                    entity_rows=[{"ticker": ticker}]
                ).to_dict()
                logger.info(f"✅ Fetched online features from Feast for {ticker}: {feature_vector}")
            except Exception as e:
                logger.warning(f"Failed to fetch from Feast: {e}")

        model, scaler = _load_local_model(ticker, "child")
        df = fetch_ohlcv(ticker)
        preds = predict_one_step_and_week(model, df, scaler, ticker)

        # Generate plot
        plot_b64 = _generate_plot(df, preds["predictions"]["full_forecast"], ticker)
        if plot_b64:
            preds["plot_base64"] = plot_b64

        logger.info(f"✅ Child prediction completed for {ticker}")
        return preds

    except Exception as e:
        logger.error(f"Child prediction failed: {e}")
        raise PipelineError(f"Child prediction failed: {e}")
