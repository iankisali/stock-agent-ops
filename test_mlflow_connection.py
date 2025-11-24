import os
import mlflow
import sys

# Add current directory to path so we can import src
sys.path.append(os.getcwd())

from src.utils import setup_dagshub_mlflow

print("--- Environment Variables ---")
print(f"DAGSHUB_USER_NAME: {os.getenv('DAGSHUB_USER_NAME')}")
print(f"DAGSHUB_REPO_NAME: {os.getenv('DAGSHUB_REPO_NAME')}")
print(f"MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI')}")
print("---------------------------")

print("Initializing DagsHub/MLflow...")
success = setup_dagshub_mlflow()
print(f"Setup success: {success}")

print(f"Current Tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLFLOW_TRACKING_USERNAME: {os.environ.get('MLFLOW_TRACKING_USERNAME')}")
# Don't print password

print("Attempting to start a run...")
try:
    with mlflow.start_run(run_name="Docker_Connectivity_Test"):
        mlflow.log_param("test_param", "test_value")
    print("✅ Successfully created and logged to MLflow run!")
except Exception as e:
    print(f"❌ Failed to start run: {e}")
    import traceback
    traceback.print_exc()
