import os
import mlflow
from typing import Dict
from src.config import Config
from src.exception import PipelineError
from src.data.ingestion import fetch_ohlcv
from src.model.saving import load_model
from src.inference import predict_one_step_and_week
from src.utils import save_json, plot_outputs
from src.pipelines.training_pipeline import train_child

def predict_child(ticker: str, parent_dir: str = Config().parent_dir, workdir: str = Config().workdir, return_base64_plot: bool = False) -> Dict:
    """Predict using child model (Inference and Viz Stage)."""
    with mlflow.start_run(run_name=f"Child_Inference_{ticker}") as run:
        try:
            child_dir = os.path.join(workdir, ticker)
            df = fetch_ohlcv(ticker)
            session, scaler = load_model(path=child_dir, model_type="child", ticker=ticker)
            payload = predict_one_step_and_week(session, df, scaler, ticker)
            json_filename = f"{ticker}_inference_forecast.json"
            json_path = save_json(payload, os.path.join(child_dir, json_filename))
            if return_base64_plot:
                plot_base64 = plot_outputs(df, payload, child_dir, ticker, return_base64=True)
                payload["plot_base64"] = plot_base64
            else:
                plot_outputs(df, payload, child_dir, ticker)
            return payload
        except Exception as e:
            mlflow.log_param("error", str(e))
            print(f"Prediction failed for {ticker}: {e}")
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
    onnx_path = os.path.join(child_dir, "model.onnx")
    scaler_path = os.path.join(child_dir, f"{ticker}_child_scaler.pkl")
    
    try:
        session, scaler = load_model(path=child_dir, model_type="child", ticker=ticker)
        print(f"Loaded model for {ticker} from {child_dir}")
    except FileNotFoundError as e:
        if train_if_not_exists:
            print(f"Model for {ticker} not found. Training now...")
            try:
                train_summary = train_child(ticker, start, epochs, parent_dir, workdir)
                child_dir = train_summary["checkpoint"]
                session, scaler = load_model(path=child_dir, model_type="child", ticker=ticker)
            except Exception as e:
                raise PipelineError(f"Failed to train child model for {ticker}: {e}")
        else:
            raise PipelineError(f"Child model for {ticker} not found in {child_dir}.") from e
    
    print(f"Running inference for {ticker}...")
    df = fetch_ohlcv(ticker, start)
    predictions = predict_one_step_and_week(session, df, scaler, ticker)
    
    output = {"ticker": ticker, "predictions": predictions}
    
    if return_base64_plot:
        plot_base64 = plot_outputs(df, predictions, child_dir, ticker, return_base64=True)
        output["plot_base64"] = plot_base64
    
    return output