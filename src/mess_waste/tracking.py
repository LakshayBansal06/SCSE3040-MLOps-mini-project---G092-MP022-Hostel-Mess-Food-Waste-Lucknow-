"""MLflow experiment tracking configuration."""

import mlflow

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "hostel-mess-waste"


def setup_mlflow():
    """Configure MLflow and create or select the project experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment
