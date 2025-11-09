import os
import pickle
import joblib
import onnxruntime as ort
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
    """Load ONNX + scaler directly from local filesystem."""
    try:
        if model_type == "parent":
            base_dir = cfg.parent_dir
            onnx_path = os.path.join(base_dir, f"{cfg.parent_ticker}_parent_model.onnx")
            scaler_path = os.path.join(base_dir, f"{cfg.parent_ticker}_parent_scaler.pkl")
        else:
            base_dir = os.path.join(cfg.workdir, ticker)
            onnx_path = os.path.join(base_dir, f"{ticker}_child_model.onnx")
            scaler_path = os.path.join(base_dir, f"{ticker}_child_scaler.pkl")

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Missing ONNX file for {ticker}: {onnx_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler file for {ticker}: {scaler_path}")

        session = ort.InferenceSession(onnx_path)
        scaler = _safe_load_scaler(scaler_path)
        logger.info(f"✅ Loaded {model_type} model for {ticker}")
        return session, scaler

    except Exception as e:
        raise PipelineError(f"Local model load failed for {ticker}: {e}")


# =============================================================
# 🧠 PREDICTION FUNCTIONS
# =============================================================

def predict_parent():
    """Predict using locally saved parent model."""
    try:
        ticker = cfg.parent_ticker
        session, scaler = _load_local_model(ticker, "parent")
        df = fetch_ohlcv(ticker)
        preds = predict_one_step_and_week(session, df, scaler, ticker)

        logger.info(f"✅ Parent prediction completed for {ticker}")
        return preds

    except Exception as e:
        logger.error(f"Parent prediction failed: {e}")
        raise PipelineError(f"Parent prediction failed: {e}")


def predict_child(ticker: str):
    """Predict using locally saved child model."""
    try:
        session, scaler = _load_local_model(ticker, "child")
        df = fetch_ohlcv(ticker)
        preds = predict_one_step_and_week(session, df, scaler, ticker)

        logger.info(f"✅ Child prediction completed for {ticker}")
        return preds

    except Exception as e:
        logger.error(f"Child prediction failed: {e}")
        raise PipelineError(f"Child prediction failed: {e}")
