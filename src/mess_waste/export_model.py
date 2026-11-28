"""Save the @champion model from the MLflow registry to models/model.joblib.

The FastAPI service (app/main.py) loads that file.
"""

import joblib
import mlflow.sklearn

from mess_waste.config import ROOT
from mess_waste.tracking import setup_mlflow

MODEL_NAME = "hostel-mess-waste-regressor"
OUTPUT_PATH = ROOT / "models" / "model.joblib"


def main():
    setup_mlflow()
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_PATH)

    print(f"Exported champion model to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
