import os
import json
import base64
from io import BytesIO
# import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from typing import Dict
from src.config import Config
from src.exception import PipelineError
from src.logger import get_logger
from dotenv import load_dotenv


logger = get_logger()

def setup_dagshub_mlflow():
    """Initialize DagsHub for remote MLflow tracking with authentication."""
    load_dotenv()
    
    dagshub_user = os.getenv("DAGSHUB_USER_NAME")
    dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    # Check if DagsHub credentials are provided
    if dagshub_user and dagshub_repo:
        try:
            import dagshub
            # Initialize DagsHub
            dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
            
            # Set MLflow tracking URI from .env
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                logger.info(f"✓ DagsHub MLflow tracking initialized: {mlflow_tracking_uri}")
            else:
                # Fallback to constructed URI
                dagshub_mlflow_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
                mlflow.set_tracking_uri(dagshub_mlflow_uri)
                logger.info(f"✓ DagsHub MLflow tracking initialized: {dagshub_mlflow_uri}")
            
            # Ensure Model Registry URI points to the same backend
            try:
                registry_uri = mlflow.get_tracking_uri()
                mlflow.set_registry_uri(registry_uri)
                logger.info(f"✓ MLflow Model Registry initialized: {registry_uri}")
            except Exception as e:
                logger.warning(f"Failed setting MLflow registry URI: {e}")
            
            # Set authentication credentials for MLflow
            if dagshub_token:
                os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_user
                os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
                logger.info("✓ DagsHub authentication configured")
            else:
                logger.warning("DAGSHUB_TOKEN not set - you may have read-only access")
            
            return True
        except ImportError:
            logger.warning("dagshub package not installed. Install with: pip install dagshub")
            logger.info("Please install: pip install dagshub")
        except Exception as e:
            logger.warning(f"Failed to initialize DagsHub: {e}")
    else:
        logger.warning("DAGSHUB_USER_NAME or DAGSHUB_REPO_NAME not set in .env file")
    
    return False

def initialize_dirs():
    """Initialize output directories."""
    config = Config()
    os.makedirs(config.parent_dir, exist_ok=True)
    # for ticker in config.child_tickers:
    #     os.makedirs(os.path.join(config.workdir, ticker), exist_ok=True)

def save_json(data: Dict, path: str) -> str:
    """Save dictionary to JSON file."""
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path

# def plot_outputs(df: pd.DataFrame, payload: Dict, out_dir: str, ticker: str, return_base64: bool = False):
#     """Save plot of last 14 days of historical prices and forecasted prices to outputs/ directory."""
#     try:
#         if "error" in payload:
#             logger.error(f"Cannot plot for {ticker}: prediction failed with error {payload['error']}")
#             raise PipelineError(f"Cannot plot for {ticker}: prediction failed with error {payload['error']}")

#         os.makedirs(out_dir, exist_ok=True)
#         plt.figure(figsize=(12, 5))
        
#         # Ensure the 'date' column is in datetime format
#         df['date'] = pd.to_datetime(df['date'])
        
#         # Slice the DataFrame to the last 14 days
#         last_date = df['date'].max()
#         start_date = last_date - pd.Timedelta(days=13)  # 14 days including start and end
#         df_last_14 = df[df['date'] >= start_date].copy()
        
#         # Plot historical closes for the last 14 days
#         plt.plot(df_last_14["date"], df_last_14["Close"], label="Historical Close (Last 14 Days)", color='blue')
        
#         # Extract forecast closes for continuous line
#         forecast_closes = []
#         for p in payload["predictions"]["full_forecast"]:
#             close = p.get("close")
#             if not isinstance(close, (int, float)) or pd.isna(close):
#                 logger.error(f"Invalid close value in full_forecast for {ticker}: {close}")
#                 raise PipelineError(f"Invalid close value in full_forecast for {ticker}: {close}")
#             forecast_closes.append(float(close))
#         logger.info(f"Forecast closes for {ticker}: {forecast_closes}")
        
#         # Last historical close
#         last_close = float(df_last_14["Close"].iloc[-1])
#         logger.info(f"Last historical close for {ticker}: {last_close}")
        
#         # Dates for forecast
#         last_date = pd.to_datetime(payload["last_date"])
#         next_dates = [pd.to_datetime(d) for d in payload["next_business_days"]]
        
#         # Continuous forecast line starting from last historical point
#         plot_dates = [last_date] + next_dates
#         plot_closes = [last_close] + forecast_closes
#         plt.plot(plot_dates, plot_closes, 'm--', label="Forecast Close (Next 5 Days)")
        
#         plt.legend()
#         plt.title(f"{ticker} Historical (Last 14 Days) and Forecasted Close Prices (Next 5 Days)")
#         plt.xlabel("Date")
#         plt.ylabel("Close Price")
#         plt.grid(True)
        
#         # Save plot to filesystem
#         plot_filename = f"{ticker}_history_forecast_14days.png"
#         plot_path = os.path.join(out_dir, plot_filename)
#         plt.savefig(plot_path)
#         plt.close()
#         logger.info(f"Plot saved for {ticker} at {plot_path}")
        
#         if return_base64:
#             with open(plot_path, "rb") as f:
#                 img_base64 = base64.b64encode(f.read()).decode("utf-8")
#             return img_base64
        
#         return plot_path
#     except Exception as e:
#         logger.error(f"Plotting failed for {ticker}: {e}")
#         raise PipelineError(f"Plotting failed for {ticker}: {e}")