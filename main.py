import sys
import src.utils as utils
import src.pipelines.training_pipeline as training
import src.pipelines.inference_pipeline as inference
from src.config import Config
import os   
from src.logger import get_logger

logger = get_logger()

utils.initialize_dirs()

def main():
    """Run the stock prediction MLOps pipeline, demonstrating the full lifecycle with MLflow tracking."""
    config = Config()
    
    # Train parent model
    print("1. Training parent model for S&P 500...")
    parent_model_path = os.path.join(config.parent_dir, "model.pt")
    parent_scaler_path = os.path.join(config.parent_dir, "parent_scaler.pkl")
    parent_onnx_path = os.path.join(config.parent_dir, "model.onnx")
    if os.path.exists(parent_model_path) and os.path.exists(parent_scaler_path) and os.path.exists(parent_onnx_path):
        print(f"✓ Using existing parent model at: {config.parent_dir}")
    else:
        try:
            parent_summary = training.train_parent(ticker=config.parent_ticker, start=config.start_date, 
                                          epochs=config.parent_epochs, out_dir=config.parent_dir)
            logger.info(f"✓ Parent model trained and saved to: {config.parent_dir}")
        except Exception as e:
            logger.error(f"✗ Error training parent model: {e}")
            if os.path.exists(parent_model_path) and os.path.exists(parent_scaler_path):
                logger.info(f"✓ Found existing parent model at: {config.parent_dir}. Continuing...")
            else:
                logger.error("✗ No existing parent model found. Cannot proceed without parent model. Exiting.")
                sys.exit(1)

    # Train child models
    results = {}
    print("\n2. Training child models sequentially...")
    for ticker in config.child_tickers:
        print(f"Training child model for {ticker}...")
        try:
            summary = training.train_child(ticker=ticker, start=config.start_date, epochs=config.child_epochs, 
                                          parent_dir=config.parent_dir, workdir=config.workdir)
            results[ticker] = summary
            logger.info(f"✓ {ticker} model trained and saved to: {summary['checkpoint']}")
            logger.info(f"✓ Predictions saved to: {summary['json']}")
            logger.info(f"✓ Metrics saved to: {summary['checkpoint']}/{ticker}_child_metrics.json")
        except Exception as e:
            logger.error(f"✗ Error training {ticker}: {e}")

    # Generate predictions
    print("\n3. Generating fresh predictions...")
    for ticker in config.child_tickers:
        try:
            preds = inference.predict_child(ticker=ticker, parent_dir=config.parent_dir, workdir=config.workdir)
            if "error" in preds:
                logger.error(f"✗ Error predicting {ticker}: {preds['error']}")
                continue
            predictions = preds.get('predictions', {})
            next_business_days = predictions.get('next_business_days', [])
            next_day = predictions.get('next_day', {})
            next_week = predictions.get('next_week', {})
            logger.info(f"✓ {ticker} predictions for {next_business_days}:")
            logger.info(f"  Next-day open: ${next_day.get('open', 'N/A'):.2f}")
            logger.info(f"  Next-day high: ${next_day.get('high', 'N/A'):.2f}")
            logger.info(f"  Next-day low: ${next_day.get('low', 'N/A'):.2f}")
            logger.info(f"  Next-day close: ${next_day.get('close', 'N/A'):.2f}")
            logger.info(f"  Next-week high: ${next_week.get('high', 'N/A'):.2f}")
            logger.info(f"  Next-week low: ${next_week.get('low', 'N/A'):.2f}")
        except Exception as e:
            logger.error(f"✗ Error predicting {ticker}: {e}")

    logger.info(f"\n{'=' * 50}")
    logger.info("Pipeline completed! Check 'outputs/' directory for models, scalers, predictions, metrics, and plots.")
    print("MLflow tracking: Run 'mlflow ui' to view experiments, params, metrics, and artifacts.")
    print("\nFile structure:")
    print("outputs/")
    print("├── parent/")
    print(f"│   ├── model.pt")
    logger.info(f"│   ├── model.onnx")
    logger.info(f"│   ├── parent_scaler.pkl")
    logger.info(f"│   └── {config.parent_ticker.replace('^', '')}_parent_metrics.json")
    for ticker in config.child_tickers:
        if ticker in results:
            logger.info(f"├── {ticker}/")
            logger.info(f"│   ├── model.pt")
            logger.info(f"│   ├── model.onnx")
            logger.info(f"│   ├── {ticker}_child_scaler.pkl")
            logger.info(f"│   ├── {ticker}_child_forecast.json")
            logger.info(f"│   ├── {ticker}_child_metrics.json")
            logger.info(f"│   └── {ticker}_history_forecast.png")

if __name__ == '__main__':
    main()