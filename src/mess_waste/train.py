"""Train and track v1 hostel mess waste regression models."""

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mess_waste.config import load_params, resolve, file_sha256
from mess_waste.tracking import setup_mlflow, TRACKING_URI

MODEL_NAME = "hostel-mess-waste-regressor"
RANDOM_STATE = 42


def build_pipeline(numeric_features, categorical_features, model):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])


def main():
    params = load_params()
    data_config = params["data"]

    original_hash = file_sha256(data_config["train"])
    cleaned_path = resolve(params["cleaning"]["output"])

    df = pd.read_csv(cleaned_path)

    numeric_features = data_config["numeric_features"]
    categorical_features = data_config["categorical_features"]
    target = data_config["target"]

    features = numeric_features + categorical_features
    X = df[features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    experiment = setup_mlflow()

    configurations = [
        {
            "name": "random_forest_200",
            "model": RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "params": {
                "model_type": "RandomForestRegressor",
                "n_estimators": 200,
                "max_depth": 8
            }
        },
        {
            "name": "random_forest_300",
            "model": RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "params": {
                "model_type": "RandomForestRegressor",
                "n_estimators": 300,
                "max_depth": -1,
                "min_samples_leaf": 2
            }
        },
        {
            "name": "extra_trees_300",
            "model": ExtraTreesRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "params": {
                "model_type": "ExtraTreesRegressor",
                "n_estimators": 300,
                "max_depth": -1,
                "min_samples_leaf": 2
            }
        }
    ]

    best = None

    for config in configurations:
        pipeline = build_pipeline(
            numeric_features,
            categorical_features,
            config["model"]
        )

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=config["name"]
        ) as run:

            mlflow.log_params(config["params"])
            mlflow.log_param("data_sha256", original_hash)
            mlflow.log_param("validation_size", 0.20)
            mlflow.log_param("random_state", RANDOM_STATE)

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_val)
            mae = mean_absolute_error(y_val, predictions)

            mlflow.log_metric("mae_validation", mae)

            print(f"{config['name']} | Validation MAE: {mae:.4f}")

            if best is None or mae < best["mae"]:
                best = {
                    "mae": mae,
                    "run_id": run.info.run_id,
                    "pipeline": pipeline,
                    "params": config["params"],
                    "name": config["name"]
                }

    # Log the selected model into its best experiment run.
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.sklearn.log_model(
    sk_model=best["pipeline"],
    name="model",
    serialization_format="cloudpickle"
)

    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri, MODEL_NAME)

    client = MlflowClient(tracking_uri=TRACKING_URI)
    client.set_registered_model_alias(
        MODEL_NAME, "champion", registered.version
    )

    print("\nV1 TRAINING COMPLETE")
    print("Best configuration:", best["name"])
    print("Best validation MAE:", round(best["mae"], 4))
    print("Best run ID:", best["run_id"])
    print("Registered model:", MODEL_NAME)
    print("Model version:", registered.version)
    print("Champion alias: champion")


if __name__ == "__main__":
    main()
