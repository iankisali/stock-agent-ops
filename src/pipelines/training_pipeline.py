import os
import torch
import mlflow
import onnxruntime as ort
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from typing import Dict

from src.config import Config
from src.data.ingestion import fetch_ohlcv
from src.data.preparation import StockDataset
from src.model.definition import LSTMModel
from src.model.training import fit_model
from src.model.evaluation import evaluate_model_temp
from src.utils import setup_dagshub_mlflow
from src.logger import get_logger
from src.exception import PipelineError
from mlflow.tracking import MlflowClient


logger = get_logger()
client = MlflowClient()


# =============================================================
# ✅ HELPER FUNCTIONS
# =============================================================

def _safe_promote_to_production(model_name: str, version: int):
    """Promote model version to Production (safe for DagsHub)."""
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"✅ Promoted {model_name} v{version} → Production")
    except Exception as e:
        logger.warning(f"⚠️ Registry not supported: {e}")


def _register_model_in_registry(onnx_path: str, model_name: str):
    """Try to register ONNX model to MLflow registry."""
    try:
        model_info = mlflow.onnx.log_model(
            onnx_model=onnx_path,
            artifact_path="onnx_model",
            registered_model_name=model_name
        )
        version = model_info.version
        _safe_promote_to_production(model_name, version)
        return version
    except Exception as e:
        logger.warning(f"⚠️ Could not register model: {e}")
        return None


def _get_output_paths(base_dir: str, ticker: str, model_type: str):
    os.makedirs(base_dir, exist_ok=True)
    prefix = f"{ticker}_{model_type}"
    return (
        os.path.join(base_dir, f"{prefix}_model.pt"),
        os.path.join(base_dir, f"{prefix}_model.onnx"),
        os.path.join(base_dir, f"{prefix}_scaler.pkl"),
    )


def _export_to_onnx(model, example_input, onnx_path):
    """Export trained model to ONNX format."""
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    try:
        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17
        )
        logger.info(f"✅ Exported model to ONNX: {onnx_path}")
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        raise PipelineError(f"ONNX export failed: {e}")


# =============================================================
# 🧠 PARENT MODEL TRAINING
# =============================================================

def train_parent() -> Dict:
    """Train fixed parent model (^GSPC)."""
    cfg = Config()
    ticker = cfg.parent_ticker
    start = cfg.start_date
    epochs = cfg.parent_epochs
    out_dir = cfg.parent_dir

    with mlflow.start_run(run_name=f"Parent_Training_{ticker}") as run:
        try:
            df = fetch_ohlcv(ticker, start)
            scaler = StandardScaler().fit(df[cfg.features])
            dataset = StockDataset(df, scaler)

            train_size = int(0.8 * len(dataset))
            train_ds, val_ds = torch.utils.data.random_split(
                dataset, [train_size, len(dataset) - train_size]
            )
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

            model = LSTMModel().to(cfg.device)
            model = fit_model(model, train_loader, val_loader, epochs=epochs, lr=1e-3)

            torch_path, onnx_path, scaler_path = _get_output_paths(out_dir, ticker, "parent")
            torch.save(model.state_dict(), torch_path)
            joblib.dump(scaler, scaler_path)

            # 🧠 Export to ONNX
            example_input = torch.randn(1, cfg.context_len, cfg.input_size).to(cfg.device)
            _export_to_onnx(model, example_input, onnx_path)

            # ✅ Evaluate
            session = ort.InferenceSession(onnx_path)
            metrics = evaluate_model_temp(session, df, scaler, out_dir, ticker)

            # Log to MLflow
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(torch_path, "torch_model")
            mlflow.log_artifact(onnx_path, "onnx_model")
            mlflow.log_artifact(scaler_path, "scaler")

            _register_model_in_registry(onnx_path, f"ParentModel_{ticker}")

            logger.info(f"✅ Parent {ticker} trained successfully")
            return {"ticker": ticker, "run_id": run.info.run_id, "metrics": metrics}
        except Exception as e:
            logger.error(f"Parent training failed: {e}")
            raise PipelineError(f"Parent training failed: {e}")


# =============================================================
# 🧬 CHILD MODEL TRAINING (TRANSFER LEARNING)
# =============================================================

def train_child(ticker: str) -> Dict:
    """Train child model using parent weights (transfer learning)."""
    cfg = Config()
    start = cfg.start_date
    epochs = cfg.child_epochs
    workdir = cfg.workdir
    parent_dir = cfg.parent_dir

    with mlflow.start_run(run_name=f"Child_Training_{ticker}") as run:
        try:
            df = fetch_ohlcv(ticker, start)
            scaler = StandardScaler().fit(df[cfg.features])
            dataset = StockDataset(df, scaler)

            train_size = int(0.8 * len(dataset))
            train_ds, val_ds = torch.utils.data.random_split(
                dataset, [train_size, len(dataset) - train_size]
            )
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

            parent_model_path = os.path.join(parent_dir, f"{cfg.parent_ticker}_parent_model.pt")
            if not os.path.exists(parent_model_path):
                raise FileNotFoundError(f"Parent model missing at {parent_model_path}")

            parent_model = LSTMModel().to(cfg.device)
            parent_model.load_state_dict(torch.load(parent_model_path, map_location=cfg.device))
            logger.info(f"🔁 Loaded parent weights from {parent_model_path}")

            # Freeze LSTM layers for transfer learning
            for name, param in parent_model.named_parameters():
                if "lstm" in name:
                    param.requires_grad = False

            model = fit_model(parent_model, train_loader, val_loader, epochs=epochs, lr=3e-4)

            child_dir = os.path.join(workdir, ticker)
            torch_path, onnx_path, scaler_path = _get_output_paths(child_dir, ticker, "child")
            torch.save(model.state_dict(), torch_path)
            joblib.dump(scaler, scaler_path)

            # 🧠 Export to ONNX
            example_input = torch.randn(1, cfg.context_len, cfg.input_size).to(cfg.device)
            _export_to_onnx(model, example_input, onnx_path)

            # ✅ Evaluate
            session = ort.InferenceSession(onnx_path)
            metrics = evaluate_model_temp(session, df, scaler, child_dir, ticker)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(torch_path, "torch_model")
            mlflow.log_artifact(onnx_path, "onnx_model")
            mlflow.log_artifact(scaler_path, "scaler")

            _register_model_in_registry(onnx_path, f"ChildModel_{ticker}")

            logger.info(f"✅ Child {ticker} trained successfully")
            return {"ticker": ticker, "run_id": run.info.run_id, "metrics": metrics}
        except Exception as e:
            logger.error(f"Child training failed: {e}")
            raise PipelineError(f"Child training failed: {e}")
