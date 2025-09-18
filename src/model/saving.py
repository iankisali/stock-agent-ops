import os
import joblib
import torch
import mlflow
import mlflow.pytorch
import mlflow.onnx
import onnxruntime as ort
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler
from src.exception import PipelineError
from src.model.export import model_to_onnx      
from src.logger import get_logger
import onnxruntime as ort
import onnx

logger = get_logger()

def save_model(model: torch.nn.Module, scaler: StandardScaler, path: str, model_type: str = "parent", ticker: Optional[str] = None):
    """Save model, scaler, and ONNX model locally and log to MLflow (Model Registry Stage)."""
    try:
        os.makedirs(path, exist_ok=True)
        torch_save_path = os.path.join(path, "model.pt")
        scaler_filename = "parent_scaler.pkl" if model_type == "parent" else f"{ticker}_child_scaler.pkl"
        scaler_path = os.path.join(path, scaler_filename)
        onnx_path = os.path.join(path, "model.onnx")

        torch.save(model.state_dict(), torch_save_path)
        joblib.dump(scaler, scaler_path)
        model_to_onnx(model, onnx_path)
        logger.info(f"Model and scaler saved locally at {path}")

        # Log to MLflow
        mlflow.pytorch.log_model(model, "pytorch_model")
        onnx_model = onnx.load(onnx_path)
        mlflow.onnx.log_model(onnx_model, "onnx_model")
        mlflow.log_artifact(scaler_path)
    except Exception as e:
        logger.error(f"Failed to save model at {path}: {e}")
        raise PipelineError(f"Failed to save model at {path}: {e}")

def load_model(path: str, model_type: str = "parent", ticker: Optional[str] = None) -> Tuple[ort.InferenceSession, StandardScaler]:
    """Load ONNX model and scaler from path (Inference Stage)."""
    if model_type == "child" and not ticker:
        raise ValueError("Ticker must be provided for child model")
    
    scaler_filename = "parent_scaler.pkl" if model_type == "parent" else f"{ticker}_child_scaler.pkl"
    onnx_path = os.path.join(path, "model.onnx")
    scaler_path = os.path.join(path, scaler_filename)
    
    if os.path.exists(onnx_path) and os.path.exists(scaler_path):
        return ort.InferenceSession(onnx_path), joblib.load(scaler_path)
    logger.error(f"ONNX model or scaler not found at {path}")
    raise FileNotFoundError(f"ONNX model or scaler not found at {path}")