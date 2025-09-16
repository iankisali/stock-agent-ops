from pydantic import BaseModel
from typing import Dict, List, Optional

class ForecastEntry(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class PredictionResponse(BaseModel):
    ticker: str
    last_date: str
    future_window_days: int
    next_business_days: List[str]
    predictions: Dict[str, Dict[str, float] | List[ForecastEntry]]

class InferenceResponse(BaseModel):
    ticker: str
    predictions: Optional[PredictionResponse]
    plot_base64: Optional[str]
    error: Optional[str]

class TrainingResponse(BaseModel):
    message: str
    details: Dict[str, str]

class ErrorResponse(BaseModel):
    detail: str