from fastapi import APIRouter, Depends, HTTPException, FileResponse
from src.logger import get_logger
from src.config import Config
from src.pipelines.inference_pipeline import infer_child_stock
from src.models import InferenceResponse, ErrorResponse
import os

logger = get_logger()

router = APIRouter()

@router.get("/{ticker}", response_model=InferenceResponse, responses={500: {"model": ErrorResponse}})
async def api_predict(ticker: str, train_if_not_exists: bool = True, 
                     config: Config = Depends(get_config)):
    """Predict next 5 days for a ticker, returning JSON with predictions and base64 plot."""
    try:
        if ticker not in config.child_tickers:
            logger.warning(f"Invalid ticker {ticker} requested for prediction")
            raise HTTPException(status_code=400, detail=f"Ticker {ticker} not in configured child tickers")
        logger.info(f"Starting prediction for ticker {ticker}")
        output = infer_child_stock(ticker=ticker, start=config.start_date, epochs=config.child_epochs, 
                                  parent_dir=config.parent_dir, workdir=config.workdir, 
                                  train_if_not_exists=train_if_not_exists, return_base64_plot=True)
        logger.info(f"Prediction completed for {ticker}")
        return InferenceResponse(**output)
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/plot/{ticker}", response_model=None, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def api_get_plot(ticker: str, config: Config = Depends(get_config)):
    """Get the prediction plot as PNG for a ticker."""
    try:
        if ticker not in config.child_tickers:
            logger.warning(f"Invalid ticker {ticker} requested for plot")
            raise HTTPException(status_code=400, detail=f"Ticker {ticker} not in configured child tickers")
        child_dir = os.path.join(config.workdir, ticker)
        plot_path = os.path.join(child_dir, f"{ticker}_history_forecast.png")
        if os.path.exists(plot_path):
            logger.info(f"Serving plot for {ticker} from {plot_path}")
            return FileResponse(plot_path, media_type="image/png", filename=f"{ticker}_plot.png")
        else:
            logger.warning(f"Plot not found for {ticker} at {plot_path}")
            raise HTTPException(status_code=404, detail="Plot not found. Run prediction first.")
    except Exception as e:
        logger.error(f"Failed to serve plot for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))