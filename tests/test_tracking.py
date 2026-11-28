"""The committed MLflow registry must work on every member's laptop.

mlflow.db once stored absolute paths from one laptop (/Users/...), so
training crashed everywhere else with "Access is denied". These tests read
the committed mlflow.db and artifacts/mlruns.zip; no CSV is needed.
"""

import re
import sqlite3
import zipfile

import pytest

from mess_waste import tracking
from mess_waste.config import ROOT

ARTIFACT_COLUMNS = [
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("model_versions", "storage_location"),
    ("logged_models", "artifact_location"),
]

ABSOLUTE = re.compile(r"^(/|[A-Za-z]:[\/]|file:)")


@pytest.fixture(scope="module")
def registry():
    connection = sqlite3.connect(ROOT / "mlflow.db")
    yield connection
    connection.close()


@pytest.mark.parametrize("table,column", ARTIFACT_COLUMNS)
def test_artifact_paths_are_relative_to_the_repository(registry, table, column):
    paths = [row[0] for row in registry.execute(f"select {column} from {table}")]

    machine_specific = [p for p in paths if p and ABSOLUTE.match(p)]

    assert paths
    assert machine_specific == []


def test_archive_holds_the_files_of_every_registered_version(registry):
    locations = [
        row[0].removeprefix("./")
        for row in registry.execute("select storage_location from model_versions")
    ]
    with zipfile.ZipFile(tracking.ARTIFACT_ARCHIVE) as archive:
        names = archive.namelist()

    for location in locations:
        assert any(name.startswith(location + "/") for name in names), location


def test_restore_artifacts_unpacks_missing_files_only(tmp_path, monkeypatch):
    archive_path = tmp_path / "artifacts" / "mlruns.zip"
    archive_path.parent.mkdir()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("mlruns/1/models/m-1/artifacts/MLmodel", "packed")
        archive.writestr("mlruns/1/models/m-2/artifacts/MLmodel", "packed")
    kept = tmp_path / "mlruns/1/models/m-1/artifacts/MLmodel"
    kept.parent.mkdir(parents=True)
    kept.write_text("already here")
    monkeypatch.setattr(tracking, "ROOT", tmp_path)
    monkeypatch.setattr(tracking, "ARTIFACT_ARCHIVE", archive_path)

    assert tracking.restore_artifacts() is True
    assert kept.read_text() == "already here"
    assert (tmp_path / "mlruns/1/models/m-2/artifacts/MLmodel").read_text() == "packed"
    assert tracking.restore_artifacts() is False


def test_restore_artifacts_refuses_paths_outside_the_repository(tmp_path, monkeypatch):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "no")
    monkeypatch.setattr(tracking, "ROOT", tmp_path / "repo")
    monkeypatch.setattr(tracking, "ARTIFACT_ARCHIVE", archive_path)

    with pytest.raises(ValueError):
        tracking.restore_artifacts()
