"""Monitor data drift, evaluate V1, train V2 and generate holdout predictions."""

import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from mess_waste.config import load_params, resolve, file_sha256
from mess_waste.data_cleaning import clean_train
from mess_waste.data_checks import normalise_label
from mess_waste.tracking import setup_mlflow


MODEL_NAME = "hostel-mess-waste-regressor"
EXPERIMENT_NAME = "hostel-mess-waste"


def calculate_psi(reference, current):
    """Calculate PSI using quantile bins from the reference data."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))

    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), 1e-6)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), 1e-6)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def main():
    params = load_params()
    data = params["data"]

    numeric = data["numeric_features"]
    categorical = data["categorical_features"]
    features = numeric + categorical
    target = data["target"]
    date_col = data["date_column"]
    group_col = data["group_column"]

    setup_mlflow()
    client = MlflowClient()

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load datasets.
    train_raw = pd.read_csv(resolve(data["train"]))
    later_raw = pd.read_csv(resolve(data["later"]))
    holdout = pd.read_csv(resolve(data["holdout"]))

    cleaned_train = pd.read_csv(
        resolve(params["cleaning"]["output"])
    )

    data_hash = file_sha256(data["train"])

    # Sort later data chronologically and split 70/30.
    later_raw = later_raw.sort_values(date_col).reset_index(drop=True)
    split = int(len(later_raw) * 0.70)

    later_train_raw = later_raw.iloc[:split].copy()
    later_test = later_raw.iloc[split:].copy()

    # Clean the V2 training portion using the project's cleaning function.
    later_train = clean_train(
        later_train_raw,
        later_test,
        params,
    )

    # Normalize categories in the evaluation portion.
    for col in categorical:
        later_test[col] = later_test[col].map(normalise_label)

    # Calculate PSI for every numeric feature.
    psi_results = {}

    for col in numeric:
        psi_results[col] = calculate_psi(
            cleaned_train[col],
            later_raw[col],
        )

    # Evaluate registered V1 on the final 30% of later data.
    v1 = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/1")

    X_test = later_test[features]
    y_test = later_test[target]

    v1_predictions = v1.predict(X_test)
    v1_mae = mean_absolute_error(y_test, v1_predictions)

    hostel_mae = {}

    for hostel, group in later_test.groupby(group_col):
        predictions = v1.predict(group[features])
        hostel_mae[str(hostel)] = float(
            mean_absolute_error(group[target], predictions)
        )

    # Train V2 on cleaned train + first 70% of later data.
    combined = pd.concat(
        [cleaned_train, later_train],
        ignore_index=True,
    )

    X_v2 = combined[features]
    y_v2 = combined[target]

    v2 = clone(v1)

    with mlflow.start_run(run_name="v2_extra_trees_300") as run:
        v2.fit(X_v2, y_v2)

        v2_predictions = v2.predict(X_test)
        v2_mae = mean_absolute_error(y_test, v2_predictions)

        mlflow.log_params({
            "model_version": "v2",
            "training_rows": len(combined),
            "later_training_rows": len(later_train),
            "later_test_rows": len(later_test),
            "data_sha256": data_hash,
            "training_split": "first_70_percent_later",
        })

        mlflow.log_metric("mae_later", float(v2_mae))
        mlflow.log_metric("v1_mae_later", float(v1_mae))

        mlflow.sklearn.log_model(
            sk_model=v2,
            name="model",
            serialization_format="cloudpickle",
        )

        v2_run_id = run.info.run_id

    # Register V2. Promote only if MAE is at least 10% lower.
    model_uri = f"runs:/{v2_run_id}/model"

    registered = mlflow.register_model(model_uri, MODEL_NAME)

    required_mae = v1_mae * 0.90
    promoted = v2_mae <= required_mae

    if promoted:
        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            registered.version,
        )

    # Load the champion model and generate holdout predictions.
    champion = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@champion"
    )

    for col in categorical:
        holdout[col] = holdout[col].map(normalise_label)

    holdout_predictions = champion.predict(holdout[features])

    prediction_output = pd.DataFrame({
        data["id_column"]: holdout[data["id_column"]],
        target: holdout_predictions,
    })

    output_path = resolve("reports/holdout_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_output.to_csv(output_path, index=False)

    # Save monitoring results.
    results = {
        "v1_mae_later": float(v1_mae),
        "v2_mae_later": float(v2_mae),
        "required_v2_mae": float(required_mae),
        "v2_promoted": bool(promoted),
        "v2_registered_version": str(registered.version),
        "champion_version": str(
            registered.version if promoted else 1
        ),
        "psi": psi_results,
        "v1_mae_by_hostel": hostel_mae,
        "v2_run_id": v2_run_id,
        "holdout_predictions": str(output_path),
    }

    report_path = resolve("reports/monitoring.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2))

    # Update findings.yaml tracking section.
    findings_path = resolve("reports/findings.yaml")

    if findings_path.exists():
        findings = yaml.safe_load(findings_path.read_text()) or {}
    else:
        findings = {}

    findings["tracking"] = {
        "experiment_name": EXPERIMENT_NAME,
        "best_run_id": v2_run_id if promoted else "96a1a07d2a444b77be67ddf9194d9c1e",
        "registered_model_name": MODEL_NAME,
        "champion_version": int(registered.version if promoted else 1),
    }

    findings_path.parent.mkdir(parents=True, exist_ok=True)
    findings_path.write_text(
        yaml.safe_dump(findings, sort_keys=False)
    )

    print("\nMONITORING AND V2 COMPLETE")
    print("V1 MAE:", round(v1_mae, 4))
    print("V2 MAE:", round(v2_mae, 4))
    print("Required V2 MAE:", round(required_mae, 4))
    print("V2 promoted:", promoted)
    print("Registered V2 version:", registered.version)
    print("Champion version:", registered.version if promoted else 1)
    print("\nPSI:")
    for col, value in psi_results.items():
        print(f"  {col}: {value:.4f}")

    print("\nV1 MAE by hostel:")
    for hostel, value in hostel_mae.items():
        print(f"  {hostel}: {value:.4f}")

    print("\nHoldout predictions:", output_path)
    print("Monitoring report:", report_path)
    print("Findings updated:", findings_path)


if __name__ == "__main__":
    main()
