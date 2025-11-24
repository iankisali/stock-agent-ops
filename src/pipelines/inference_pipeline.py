import os
import pickle
import joblib
import torch
from src.model.definition import LSTMModel
from src.config import Config
from src.data.ingestion import fetch_ohlcv
from src.inference import predict_one_step_and_week
from src.logger import get_logger
from src.exception import PipelineError

logger = get_logger()
cfg = Config()


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
        model, scaler = _load_local_model(ticker, "child")
        df = fetch_ohlcv(ticker)
        preds = predict_one_step_and_week(model, df, scaler, ticker)

        logger.info(f"✅ Child prediction completed for {ticker}")
        return preds

    except Exception as e:
        logger.error(f"Child prediction failed: {e}")
        raise PipelineError(f"Child prediction failed: {e}")
