# from fastapi import FastAPI
# from api.routes.health import router as health_router  # Fixed import (use 'router')
# from api.routes.training import router as training_router  # Fixed import (use 'router')
# from api.routes.inference import router as inference_router  # Fixed import (use 'router')
# from src.logger import get_logger  # Import from src.logger

# logger = get_logger()

# app = FastAPI(
#     title="Stock Prediction MLOps API",
#     description="API for training and predicting stock prices using an MLOps pipeline.",
#     version="0.0.1",
#     docs_url="/docs",
#     openapi_tags=[
#         {"name": "Health", "description": "Health check endpoints"},
#         {"name": "Training", "description": "Endpoints for training parent and child models"},
#         {"name": "Inference", "description": "Endpoints for stock price predictions and plots"}
#     ]
# )

# # Include routers
# app.include_router(health_router, prefix="/health", tags=["Health"])
# app.include_router(training_router, prefix="/train", tags=["Training"])
# app.include_router(inference_router, prefix="/predict", tags=["Inference"])

# @app.on_event("startup")
# async def startup_event():
#     logger.info("Stock Prediction API started")

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port="8000")


import yfinance as yf
ticker = yf.Ticker("GOOG")
data = ticker.history(period="1d")
print(data)