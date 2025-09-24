import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # Adjust as needed
mlflow.set_experiment("test_experiment")
with mlflow.start_run():
    mlflow.log_param("test_param", "value")
    mlflow.log_metric("test_metric", 1.0)
print("MLflow test run created successfully!")