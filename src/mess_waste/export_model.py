from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

MODEL_NAME = "hostel-mess-waste-regressor"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "models" / "model.joblib"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, OUTPUT_PATH)

print(f"Exported champion model to: {OUTPUT_PATH}")
