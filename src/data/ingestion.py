import os
import pandas as pd
import yfinance as yf
from typing import Optional
from dotenv import load_dotenv
from src.config import Config
from src.exception import PipelineError

load_dotenv()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI for a given series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def macd(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Calculate MACD for a given series."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def fetch_ohlcv(ticker: str, start: str = Config().start_date, end: Optional[str] = None) -> pd.DataFrame:
    """Fetch OHLCV data with technical indicators (Data Ingestion Stage)."""
    config = Config()
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            raise PipelineError(f"No data downloaded for {ticker}")
        df = df.reset_index().rename(columns={"Date": "date"})
        df = df[["date", "Open", "High", "Low", "Close", "Volume"]].dropna()
        df["RSI14"] = rsi(df["Close"])
        df["MACD"] = macd(df["Close"])
        df = df[["date"] + config.features].dropna()
        
        # Validate data
        if len(df) < config.context_len + config.pred_len:
            raise PipelineError(f"Insufficient data for {ticker}: {len(df)} rows, need at least {config.context_len + config.pred_len}")
        if df[config.features].isnull().any().any():
            raise PipelineError(f"NaN values found in features for {ticker}")
        if not df[config.features].apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise PipelineError(f"Non-numeric values found in features for {ticker}")
        print(f"Fetched {len(df)} rows for {ticker}")
        return df
    except Exception as e:
        raise PipelineError(f"Failed to fetch data for {ticker}: {e}")