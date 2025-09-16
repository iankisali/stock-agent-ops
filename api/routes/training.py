from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from src.logger import get_logger
from src.config import Config
from src.pipelines.training_pipeline import train_parent, train_child
from src.models import TrainingResponse, ErrorResponse

logger = get_logger()

router = APIRouter()

@router.post("/parent", response_model=TrainingResponse, responses={500: {"model": ErrorResponse}})
async def api_train_parent(config: Config = Depends(get_config)):
    """Train the parent model for the configured ticker."""
    try:
        logger.info(f"Starting parent model training for ticker {config.parent_ticker}")
        summary = train_parent(ticker=config.parent_ticker, start=config.start_date, 
                              epochs=config.parent_epochs, out_dir=config.parent_dir)
        logger.info(f"Parent model training completed for {config.parent_ticker}")
        return TrainingResponse(message="Parent model trained", details=summary)
    except Exception as e:
        logger.error(f"Parent model training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/child/{ticker}", response_model=TrainingResponse, responses={500: {"model": ErrorResponse}})
async def api_train_child(ticker: str, config: Config = Depends(get_config)):
    """Train a child model for a specific ticker."""
    try:
        if ticker not in config.child_tickers:
            logger.warning(f"Invalid ticker {ticker} requested for child model training")
            raise HTTPException(status_code=400, detail=f"Ticker {ticker} not in configured child tickers")
        logger.info(f"Starting child model training for ticker {ticker}")
        summary = train_child(ticker=ticker, start=config.start_date, epochs=config.child_epochs, 
                             parent_dir=config.parent_dir, workdir=config.workdir)
        logger.info(f"Child model training completed for {ticker}")
        return TrainingResponse(message=f"Child model trained for {ticker}", details=summary)
    except Exception as e:
        logger.error(f"Child model training failed for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))