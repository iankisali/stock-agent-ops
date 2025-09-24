import os
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from typing import Dict
from src.config import Config
from src.exception import PipelineError
from src.logger import get_logger


logger = get_logger()

def initialize_dirs():
    """Initialize output directories."""
    config = Config()
    os.makedirs(config.parent_dir, exist_ok=True)
    for ticker in config.child_tickers:
        os.makedirs(os.path.join(config.workdir, ticker), exist_ok=True)

def save_json(data: Dict, path: str) -> str:
    """Save dictionary to JSON file."""
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path

def plot_outputs(df: pd.DataFrame, payload: Dict, out_dir: str, ticker: str, return_base64: bool = False):
    """Save plot of last 14 days of historical prices and forecasted prices to outputs/ directory."""
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
        logger.info(f"Forecast closes for {ticker}: {forecast_closes}")
        
        # Last historical close
        last_close = float(df_last_14["Close"].iloc[-1])
        logger.info(f"Last historical close for {ticker}: {last_close}")
        
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
        
        # Save plot to filesystem
        plot_filename = f"{ticker}_history_forecast_14days.png"
        plot_path = os.path.join(out_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Plot saved for {ticker} at {plot_path}")
        
        if return_base64:
            with open(plot_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            return img_base64
        
        return plot_path
    except Exception as e:
        logger.error(f"Plotting failed for {ticker}: {e}")
        raise PipelineError(f"Plotting failed for {ticker}: {e}")