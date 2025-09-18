import os
from dotenv import load_dotenv
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from typing import Dict
from src.config import Config
from src.exception import PipelineError
from src.data.ingestion import fetch_ohlcv
from src.model.saving import load_model
from src.inference import predict_one_step_and_week
from src.utils import save_json
from src.pipelines.training_pipeline import train_child
from src.logger import get_logger
import mlflow

# Load environment variables from .env file
load_dotenv()

# Set MLflow tracking URI from .env (needed for model registry access)
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not mlflow_tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI not set in .env file")
mlflow.set_tracking_uri(mlflow_tracking_uri)

logger = get_logger()

def load_model_from_registry(model_name: str, version: str = "latest"):
    """Load model and scaler from MLflow model registry."""
    try:
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.onnx.load_model(model_uri)
        # Load scaler from artifacts
        client = mlflow.tracking.MlflowClient()
        run_id = client.get_latest_versions(model_name, stages=["None", "Production", "Staging"])[0].run_id
        scaler_path = client.download_artifacts(run_id, f"scalers/{model_name.split('_')[-1]}/{model_name.split('_')[-1]}_child_scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model {model_name} from MLflow registry: {e}")

def plot_outputs(df: pd.DataFrame, payload: Dict, out_dir: str, ticker: str, return_base64: bool = False):
    """Plot the last 14 days of historical prices and forecasted prices with a continuous prediction line."""
    try:
        if "error" in payload:
            logger.error(f"Cannot plot for {ticker}: prediction failed with error {payload['error']}")
            raise PipelineError(f"Cannot plot for {ticker}: prediction failed with error {payload['error']}")

        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(12, 5))
        
        # Ensure the 'date' column is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Slice the DataFrame to the last 14 days
        last_date = df['date'].max()
        start_date = last_date - pd.Timedelta(days=13)  # 14 days including start and end
        df_last_14 = df[df['date'] >= start_date].copy()
        
        # Plot historical closes for the last 14 days
        plt.plot(df_last_14["date"], df_last_14["Close"], label="Historical Close (Last 14 Days)", color='blue')
        
        # Extract forecast closes for continuous line
        forecast_closes = []
        for p in payload["predictions"]["full_forecast"]:
            close = p.get("close")
            if not isinstance(close, (int, float)) or pd.isna(close):
                logger.error(f"Invalid close value in full_forecast for {ticker}: {close}")
                raise PipelineError(f"Invalid close value in full_forecast for {ticker}: {close}")
            forecast_closes.append(float(close))
        print(f"Forecast closes for {ticker}: {forecast_closes}")
        
        # Last historical close
        last_close = float(df_last_14["Close"].iloc[-1])
        print(f"Last historical close for {ticker}: {last_close}")
        
        # Dates for forecast
        last_date = pd.to_datetime(payload["last_date"])
        next_dates = [pd.to_datetime(d) for d in payload["next_business_days"]]
        
        # Continuous forecast line starting from last historical point
        plot_dates = [last_date] + next_dates
        plot_closes = [last_close] + forecast_closes
        plt.plot(plot_dates, plot_closes, 'm--', label="Forecast Close (Next 5 Days)")
        
        plt.legend()
        plt.title(f"{ticker} Historical (Last 14 Days) and Forecasted Close Prices (Next 5 Days)")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.grid(True)
        
        if return_base64:
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close()
            return img_base64
        
        plot_filename = f"{ticker}_history_forecast_14days.png"
        plot_path = os.path.join(out_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved for {ticker} at {plot_path}")
        return plot_path
    except Exception as e:
        logger.error(f"Plotting failed for {ticker}: {e}")
        raise PipelineError(f"Plotting failed for {ticker}: {e}")

def predict_child(ticker: str, parent_dir: str = Config().parent_dir, workdir: str = Config().workdir, return_base64_plot: bool = False) -> Dict:
    """Predict using child model (Inference and Viz Stage)."""
    try:
        child_dir = os.path.join(workdir, ticker)
        # Try loading from MLflow model registry
        try:
            session, scaler = load_model_from_registry(f"ChildModel_{ticker}")
            print(f"Loaded model for {ticker} from MLflow registry")
        except FileNotFoundError:
            # Fallback to filesystem
            session, scaler = load_model(path=child_dir, model_type="child", ticker=ticker)
            print(f"Loaded model for {ticker} from filesystem: {child_dir}")
            
        df = fetch_ohlcv(ticker)
        payload = predict_one_step_and_week(session, df, scaler, ticker)
        json_filename = f"{ticker}_inference_forecast.json"
        json_path = save_json(payload, os.path.join(child_dir, json_filename))
        
        if return_base64_plot:
            plot_base64 = plot_outputs(df, payload, child_dir, ticker, return_base64=True)
            payload["plot_base64"] = plot_base64
        else:
            plot_path = plot_outputs(df, payload, child_dir, ticker)
        return payload
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}

def infer_child_stock(
    ticker: str,
    start: str = Config().start_date,
    epochs: int = Config().child_epochs,
    parent_dir: str = Config().parent_dir,
    workdir: str = Config().workdir,
    train_if_not_exists: bool = True,
    return_base64_plot: bool = False
) -> Dict:
    """Infer stock predictions, training if necessary (Inference with Fallback to Training)."""
    child_dir = os.path.join(workdir, ticker)
    
    try:
        # Try loading from MLflow model registry
        session, scaler = load_model_from_registry(f"ChildModel_{ticker}")
        print(f"Loaded model for {ticker} from MLflow registry")
    except FileNotFoundError as e:
        if train_if_not_exists:
            print(f"Model for {ticker} not found in MLflow registry or filesystem. Training now...")
            try:
                train_summary = train_child(ticker, start, epochs, parent_dir, workdir)
                child_dir = train_summary["checkpoint"]
                session, scaler = load_model(path=child_dir, model_type="child", ticker=ticker)
            except Exception as e:
                raise PipelineError(f"Failed to train child model for {ticker}: {e}")
        else:
            raise PipelineError(f"Child model for {ticker} not found in MLflow registry or {child_dir}.") from e
    
    print(f"Running inference for {ticker}...")
    df = fetch_ohlcv(ticker, start)
    predictions = predict_one_step_and_week(session, df, scaler, ticker)
    
    output = {"ticker": ticker, "predictions": predictions}
    
    if return_base64_plot:
        plot_base64 = plot_outputs(df, predictions, child_dir, ticker, return_base64=True)
        output["plot_base64"] = plot_base64
    else:
        plot_path = plot_outputs(df, predictions, child_dir, ticker)
    
    return output