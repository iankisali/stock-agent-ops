# import sys
# import os
# import mlflow
# from mlflow.tracking import MlflowClient

# import src.utils as utils
# import src.pipelines.training_pipeline as training
# import src.pipelines.inference_pipeline as inference
# from src.config import Config
# from src.logger import get_logger

# logger = get_logger()

# # Initialize DagsHub MLflow and directories
# utils.setup_dagshub_mlflow()
# utils.initialize_dirs()


# def check_dagshub_connection():
#     """
#     Verify DagsHub MLflow connection before starting pipeline.
    
#     Returns:
#         bool: True if connection successful, False otherwise
#     """
#     try:
#         client = MlflowClient()
#         experiments = client.search_experiments(max_results=1)
#         tracking_uri = mlflow.get_tracking_uri()
        
#         logger.info(f"✓ Successfully connected to MLflow: {tracking_uri}")
#         logger.info(f"✓ Found {len(experiments)} experiment(s)")
#         return True
        
#     except Exception as e:
#         logger.error(f"✗ Failed to connect to DagsHub MLflow: {e}")
#         logger.error("Please check your .env file and ensure DAGSHUB_TOKEN is set correctly")
#         return False


# def train_parent_model(config):
#     """
#     Train the parent model (S&P 500) and promote to Production.
    
#     Args:
#         config: Configuration object with training parameters
        
#     Returns:
#         dict: Training summary with model info and metrics
#     """
#     print("\n" + "="*60)
#     print("1. Training Parent Model (S&P 500)")
#     print("="*60)
    
#     try:
#         parent_summary = training.train_parent(
#             ticker=config.parent_ticker,
#             start=config.start_date,
#             epochs=config.parent_epochs,
#             out_dir=config.parent_dir
#         )
        
#         logger.info(f"✓ Parent model trained: {parent_summary['model_name']}")
#         logger.info(f"✓ Model path: {parent_summary['model_path']}")
#         logger.info(f"✓ Run ID: {parent_summary['run_id']}")
#         logger.info(f"✓ Metrics: {parent_summary['metrics']}")
        
#         # Auto-promote to Production
#         inference.promote_model_to_production(
#             model_name=parent_summary['model_name'],
#             version=parent_summary['model_version']
#         )
#         logger.info(f"✓ Promoted parent model to Production")
        
#         return parent_summary
        
#     except Exception as e:
#         logger.error(f"✗ Error training parent model: {e}")
#         logger.error("Cannot proceed without parent model. Exiting.")
#         sys.exit(1)


# def train_child_models(config):
#     """
#     Train child models for each ticker and promote to Production.
    
#     Args:
#         config: Configuration object with training parameters
        
#     Returns:
#         dict: Dictionary mapping tickers to their training summaries
#     """
#     print("\n" + "="*60)
#     print("2. Training Child Models")
#     print("="*60)
    
#     results = {}
    
#     for ticker in config.child_tickers:
#         print(f"\nTraining child model for {ticker}...")
        
#         try:
#             summary = training.train_child(
#                 ticker=ticker,
#                 start=config.start_date,
#                 epochs=config.child_epochs,
#                 parent_dir=config.parent_dir,
#                 workdir=config.workdir
#             )
            
#             results[ticker] = summary
            
#             logger.info(f"✓ {ticker} model trained: {summary['model_name']}")
#             logger.info(f"✓ Model path: {summary['model_path']}")
#             logger.info(f"✓ Run ID: {summary['run_id']}")
#             logger.info(f"✓ Metrics: {summary['metrics']}")
            
#             # Auto-promote to Production
#             inference.promote_model_to_production(
#                 model_name=summary['model_name'],
#                 version=summary['model_version']
#             )
#             logger.info(f"✓ Promoted {ticker} model to Production")
            
#         except Exception as e:
#             logger.error(f"✗ Error training {ticker}: {e}")
    
#     return results


# def generate_predictions(config):
#     """
#     Generate predictions from Production models for all tickers.
    
#     Args:
#         config: Configuration object with prediction parameters
#     """
#     print("\n" + "="*60)
#     print("3. Generating Predictions from Production Models")
#     print("="*60)
    
#     for ticker in config.child_tickers:
#         print(f"\nPredicting for {ticker}...")
        
#         try:
#             preds = inference.predict_child(ticker=ticker, workdir=config.workdir)
            
#             predictions = preds.get('predictions', {})
#             model_path = preds.get('model_path', 'N/A')
#             stage = preds.get('stage', 'N/A')
#             next_business_days = predictions.get('next_business_days', [])
#             next_day = predictions.get('next_day', {})
#             next_week = predictions.get('next_week', {})
            
#             logger.info(f"✓ {ticker} predictions ({stage}):")
#             logger.info(f"  Model path: {model_path}")
#             logger.info(f"  Forecast dates: {next_business_days}")
#             logger.info(f"  Next-day open: ${next_day.get('open', 0):.2f}")
#             logger.info(f"  Next-day high: ${next_day.get('high', 0):.2f}")
#             logger.info(f"  Next-day low: ${next_day.get('low', 0):.2f}")
#             logger.info(f"  Next-day close: ${next_day.get('close', 0):.2f}")
#             logger.info(f"  Next-week high: ${next_week.get('high', 0):.2f}")
#             logger.info(f"  Next-week low: ${next_week.get('low', 0):.2f}")
            
#         except Exception as e:
#             logger.error(f"✗ Error predicting {ticker}: {e}")


# def print_summary(config, results):
#     """
#     Print pipeline completion summary with links and model info.
    
#     Args:
#         config: Configuration object
#         results: Dictionary of training results
#     """
#     print("\n" + "="*60)
#     print("Pipeline Completed Successfully!")
#     print("="*60)
    
#     logger.info(f"✓ All models saved locally and logged to MLflow")
#     logger.info(f"✓ All models marked as Production")
#     logger.info(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
#     dagshub_user = os.getenv('DAGSHUB_USER_NAME')
#     dagshub_repo = os.getenv('DAGSHUB_REPO_NAME')
    
#     print("\n📊 View your experiments:")
#     print(f"  🔗 DagsHub: https://dagshub.com/{dagshub_user}/{dagshub_repo}")
#     print(f"  🔗 MLflow UI: https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow")
    
#     print("\n📦 Models in Production:")
#     print(f"  • ParentModel_{config.parent_ticker} -> {config.parent_dir}")
    
#     for ticker in results.keys():
#         print(f"  • ChildModel_{ticker} -> {results[ticker]['model_path']}")
    
#     print("\n✨ Models saved locally + artifacts logged to MLflow")
#     print("="*60)


# def main():
#     """Run the stock prediction MLOps pipeline with DagsHub MLflow tracking."""
#     config = Config()
    
#     # Step 0: Check DagsHub connection
#     print("\n" + "="*60)
#     print("Checking DagsHub MLflow Connection...")
#     print("="*60)
    
#     if not check_dagshub_connection():
#         logger.error("Cannot proceed without DagsHub connection. Exiting.")
#         sys.exit(1)
    
#     # Step 1: Train parent model
#     parent_summary = train_parent_model(config)
    
#     # Step 2: Train child models
#     results = train_child_models(config)
    
#     # Step 3: Generate predictions
#     generate_predictions(config)
    
#     # Step 4: Print summary
#     print_summary(config, results)


# if __name__ == '__main__':
#     try:
#         main()
#     except KeyboardInterrupt:
#         logger.info("\n\n⚠️  Pipeline interrupted by user")
#         sys.exit(0)
#     except Exception as e:
#         logger.error(f"\n\n❌ Pipeline failed: {e}")
#         sys.exit(1)

# delete_models.py
from mlflow.tracking import MlflowClient

client = MlflowClient()

# ✅ Compatible across MLflow versions
try:
    models = client.list_registered_models()
except AttributeError:
    # Fallback for older MLflow versions
    models = client.search_registered_models()

if not models:
    print("No registered models found.")
else:
    print(f"Found {len(models)} registered models:")
    for m in models:
        name = getattr(m, "name", None) or getattr(m, "model", {}).get("name", "Unknown")
        print(" -", name)

# List of models you want to delete
TO_DELETE = [
    "ChildModel_AAPL",
    "ChildModel_GOOG",
    "ChildModel_META",
    "ChildModel_TSLA",
    "ChildModel_AMZN",
    "ParentModel_^GSPC"
]

for name in TO_DELETE:
    try:
        client.delete_registered_model(name=name)
        print(f"✅ Deleted model: {name}")
    except Exception as e:
        print(f"⚠️ Could not delete {name}: {e}")
