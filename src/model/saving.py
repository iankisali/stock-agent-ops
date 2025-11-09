import os
import joblib
import torch
import mlflow
from sklearn.preprocessing import StandardScaler
from src.exception import PipelineError
from src.model.export import model_to_onnx
from src.logger import get_logger

logger = get_logger()


def save_model(model, scaler: StandardScaler, path: str, model_type="parent", ticker=None):
    """Save PyTorch model, ONNX model, and scaler locally and log to MLflow."""
    try:
        os.makedirs(path, exist_ok=True)

        torch_path = os.path.join(path, "model.pt")
        scaler_filename = "parent_scaler.pkl" if model_type == "parent" else f"{ticker}_child_scaler.pkl"
        scaler_path = os.path.join(path, scaler_filename)
        onnx_path = os.path.join(path, "model.onnx")

        # Save locally
        torch.save(model.state_dict(), torch_path)
        joblib.dump(scaler, scaler_path)
        model_to_onnx(model, onnx_path)

        # Log to MLflow
        mlflow.log_artifact(torch_path, "model")
        mlflow.log_artifact(onnx_path, "model")
        mlflow.log_artifact(scaler_path, "model")
        logger.info(f"✅ Model artifacts logged to MLflow for {model_type.upper()} model")

        return torch_path, scaler_path, onnx_path
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise PipelineError(f"Failed to save model: {e}")
