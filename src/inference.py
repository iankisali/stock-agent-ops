import json
import numpy as np
import pandas as pd
from typing import Dict
from src.config import Config
from src.exception import PipelineError
from src.logger import get_logger
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler

logger = get_logger()

def predict_one_step_and_week(session: ort.InferenceSession, df: pd.DataFrame, scaler: StandardScaler, ticker: str) -> Dict:
    """Predict next day and week OHLCV values (Inference Stage)."""
    try:
        config = Config()
        vals = scaler.transform(df[config.features]).astype("float32")
        if vals.shape[0] < config.context_len:
            logger.error(f"Insufficient data: {vals.shape[0]} rows, need at least {config.context_len}")
            raise PipelineError(f"Insufficient data: {vals.shape[0]} rows, need at least {config.context_len}")
        X = vals[-config.context_len:].reshape(1, config.context_len, config.input_size)
        logger.info(f"Input shape for {ticker}: {X.shape}")

        pred = session.run(None, {'input': X})[0]
        logger.info(f"Prediction shape for {ticker}: {pred.shape}")
        if pred.shape != (1, config.pred_len, config.input_size):
            raise PipelineError(f"Unexpected prediction shape for {ticker}: {pred.shape}")

        pred_full = scaler.inverse_transform(pred.reshape(-1, config.input_size))
        logger.info(f"Inverse transformed shape for {ticker}: {pred_full.shape}")
        pred_full = pred_full.reshape(config.pred_len, config.input_size)
        pred = pred_full[:, :5]  # OHLCV only

        # Validate predictions
        if np.any(np.isnan(pred)):
            raise PipelineError(f"NaN values in predictions for {ticker}")

        last_date = pd.to_datetime(df["date"].iloc[-1])
        next_business_days = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=config.pred_len)
        next_business_days_str = [str(d.date()) for d in next_business_days]

        full_forecast = []
        for i, d in enumerate(next_business_days):
            forecast_entry = {
                "date": str(d.date()),
                "open": round(float(pred[i, 0]), 2),
                "high": round(float(pred[i, 1]), 2),
                "low": round(float(pred[i, 2]), 2),
                "close": round(float(pred[i, 3]), 2),
                "volume": int(pred[i, 4])
            }
            logger.info(f"Forecast entry {i} for {ticker}: {forecast_entry}")
            full_forecast.append(forecast_entry)

        output = {
            "ticker": ticker,
            "last_date": str(last_date.date()),
            "future_window_days": config.pred_len,
            "next_business_days": next_business_days_str,
            "predictions": {
                "next_day": {
                    "open": round(float(pred[0, 0]), 2),
                    "high": round(float(pred[0, 1]), 2),
                    "low": round(float(pred[0, 2]), 2),
                    "close": round(float(pred[0, 3]), 2),
                    "volume": int(pred[0, 4])
                },
                "next_week": {
                    "high": round(float(np.max(pred[:, 1])), 2),
                    "low": round(float(np.min(pred[:, 2])), 2)
                },
                "full_forecast": full_forecast
            }
        }
        logger.info(f"Prediction output for {ticker}: {json.dumps(output, indent=2)}")
        return output
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}