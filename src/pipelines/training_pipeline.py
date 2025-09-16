import torch
from typing import Dict
import mlflow
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.config import Config
from src.exception import PipelineError
from src.data.ingestion import fetch_ohlcv
from src.data.preparation import StockDataset
from src.model.definition import LSTMModel
from src.model.training import fit_model
from src.model.saving import save_model, load_model
from src.model.evaluation import evaluate_model
from src.inference import predict_one_step_and_week
from src.utils import save_json, plot_outputs
from src.logger import get_logger

logger = get_logger()

def train_parent(ticker: str = Config().parent_ticker, start: str = Config().start_date, 
                 epochs: int = Config().parent_epochs, out_dir: str = Config().parent_dir) -> Dict:
    """Train parent model (Full Lifecycle: Ingestion, Prep, Train, Eval, Registry)."""
    with mlflow.start_run(run_name=f"Parent_Training_{ticker}") as run:
        config = Config()
        mlflow.log_params({
            "ticker": ticker,
            "start": start,
            "epochs": epochs,
            "context_len": config.context_len,
            "pred_len": config.pred_len,
            "input_size": config.input_size,
            "batch_size": config.batch_size,
            "device": config.device
        })
        
        try:
            df = fetch_ohlcv(ticker, start)
            scaler = StandardScaler().fit(df[config.features])
            dataset = StockDataset(df, scaler)
            train_size = int(0.8 * len(dataset))
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

            model = LSTMModel()
            model = fit_model(model, train_loader, val_loader, epochs=epochs, lr=1e-3)
            save_model(model, scaler, out_dir, model_type="parent", ticker=ticker)
            session = load_model(out_dir, "parent")[0]
            metrics = evaluate_model(session, df, scaler, out_dir, ticker.replace("^", ""))
            payload = predict_one_step_and_week(session, df, scaler, ticker)
            json_filename = f"{ticker}_parent_forecast.json"
            json_path = save_json(payload, os.path.join(out_dir, json_filename))
            plot_outputs(df, payload, out_dir, ticker)
            logger.info(f"Parent model trained successfully for {ticker}")
            return {"checkpoint": out_dir, "json": json_path}
        except Exception as e:
            mlflow.log_param("error", str(e))
            logger.error(f"Parent model training failed for {ticker}: {e}")
            raise PipelineError(f"Parent model training failed for {ticker}: {e}")

def train_child(ticker: str, start: str = Config().start_date, epochs: int = Config().child_epochs, 
                parent_dir: str = Config().parent_dir, workdir: str = Config().workdir) -> Dict:
    """Train child model using parent model weights (Full Lifecycle: Ingestion, Prep, Train, Eval, Registry, Infer, Viz)."""
    with mlflow.start_run(run_name=f"Child_Training_{ticker}") as run:
        config = Config()
        mlflow.log_params({
            "ticker": ticker,
            "start": start,
            "epochs": epochs,
            "context_len": config.context_len,
            "pred_len": config.pred_len,
            "input_size": config.input_size,
            "batch_size": config.batch_size,
            "device": config.device,
            "parent_dir": parent_dir
        })
        
        try:
            df = fetch_ohlcv(ticker, start)
            parent_model = LSTMModel()
            parent_model_path = os.path.join(parent_dir, "model.pt")
            if not os.path.exists(parent_model_path):
                raise FileNotFoundError(f"Parent model not found at {parent_model_path}")
            parent_model.load_state_dict(torch.load(parent_model_path, map_location=config.device))

            for name, param in parent_model.named_parameters():
                if "lstm" in name:
                    param.requires_grad = False

            scaler = StandardScaler().fit(df[config.features])
            if (df[config.features].std() == 0).any():
                raise PipelineError(f"Zero variance in features for {ticker}")
            dataset = StockDataset(df, scaler)
            train_size = int(0.8 * len(dataset))
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

            child_model = fit_model(parent_model, train_loader, val_loader, epochs=epochs, lr=3e-4)
            child_dir = os.path.join(workdir, ticker)
            save_model(child_model, scaler, child_dir, model_type="child", ticker=ticker)

            session = load_model(child_dir, "child", ticker)[0]
            payload = predict_one_step_and_week(session, df, scaler, ticker)
            json_filename = f"{ticker}_child_forecast.json"
            json_path = save_json(payload, os.path.join(child_dir, json_filename))
            plot_outputs(df, payload, child_dir, ticker)
            evaluate_model(session, df, scaler, child_dir, ticker)
            logger.info(f"Child model trained successfully for {ticker}")
            return {"checkpoint": child_dir, "json": json_path}
        except Exception as e:
            mlflow.log_param("error", str(e))
            logger.error(f"Child model training failed for {ticker}: {e}")
            raise PipelineError(f"Child model training failed for {ticker}: {e}")