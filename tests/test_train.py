"""Tests for the training pipeline and the MLflow tracking / registry flow."""

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from mess_waste import monitor, train
from mess_waste.train import build_pipeline

from conftest import make_mess_frame


def _split(params, df):
    d = params["data"]
    features = d["numeric_features"] + d["categorical_features"]
    return df[features], df[d["target"]], d


def _model():
    return RandomForestRegressor(n_estimators=30, random_state=0)


def test_pipeline_fits_and_predicts_one_number_per_row(params, mess_frame):
    X, y, d = _split(params, mess_frame)
    pipe = build_pipeline(d["numeric_features"], d["categorical_features"], _model())
    pipe.fit(X, y)

    pred = pipe.predict(X)
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()


def test_pipeline_handles_blanks_and_unseen_categories(params, mess_frame):
    X, y, d = _split(params, mess_frame)
    pipe = build_pipeline(d["numeric_features"], d["categorical_features"], _model())
    pipe.fit(X, y)

    odd = X.head(3).copy()
    odd.loc[odd.index[0], "temp_c"] = np.nan
    odd.loc[odd.index[1], "hostel"] = "brand new hostel"
    odd.loc[odd.index[2], "meal"] = np.nan

    pred = pipe.predict(odd)
    assert np.isfinite(pred).all()


def test_pipeline_beats_predicting_the_average(params):
    train_df = make_mess_frame(150, seed=1)
    val_df = make_mess_frame(60, seed=2)
    X_tr, y_tr, d = _split(params, train_df)
    X_val, y_val, _ = _split(params, val_df)

    pipe = build_pipeline(d["numeric_features"], d["categorical_features"], _model())
    pipe.fit(X_tr, y_tr)

    model_mae = mean_absolute_error(y_val, pipe.predict(X_val))
    baseline_mae = mean_absolute_error(y_val, np.full(len(y_val), y_tr.mean()))
    assert model_mae < baseline_mae


def test_train_and_monitor_use_the_same_registered_model_name():
    assert train.MODEL_NAME == monitor.MODEL_NAME


@pytest.fixture
def temp_mlflow(tmp_path, monkeypatch):
    """A throwaway MLflow store so tests never touch the real mlflow.db."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'test_mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test-experiment")
    yield uri
    mlflow.set_tracking_uri(None)


def test_run_is_tracked_registered_and_loadable_as_champion(params, mess_frame, temp_mlflow):
    X, y, d = _split(params, mess_frame)
    pipe = build_pipeline(d["numeric_features"], d["categorical_features"], _model())
    pipe.fit(X, y)
    mae = mean_absolute_error(y, pipe.predict(X))

    with mlflow.start_run() as run:
        mlflow.log_param("data_sha256", "0" * 64)
        mlflow.log_metric("mae_validation", mae)
        # same format train.py uses for the real champion model
        mlflow.sklearn.log_model(
            sk_model=pipe, name="model", serialization_format="cloudpickle"
        )

    client = MlflowClient(tracking_uri=temp_mlflow)
    logged = client.get_run(run.info.run_id)
    assert logged.data.params["data_sha256"] == "0" * 64
    assert logged.data.metrics["mae_validation"] == pytest.approx(mae)

    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", train.MODEL_NAME)
    client.set_registered_model_alias(train.MODEL_NAME, "champion", version.version)

    champion = mlflow.sklearn.load_model(f"models:/{train.MODEL_NAME}@champion")
    np.testing.assert_allclose(champion.predict(X.head(5)), pipe.predict(X.head(5)))
