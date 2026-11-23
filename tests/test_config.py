"""Tests for params.yaml loading and the small file helpers."""

import hashlib
from pathlib import Path

from mess_waste.config import ROOT, file_sha256, resolve


def test_params_yaml_has_the_settings_the_code_reads(params):
    data = params["data"]
    for key in ["train", "later", "holdout", "id_column", "date_column",
                "target", "group_column", "numeric_features",
                "categorical_features"]:
        assert key in data, f"params.yaml data section is missing {key!r}"
    assert "output" in params["cleaning"]
    assert params["checks"]["unit_factors"]


def test_target_and_id_are_not_used_as_features(params):
    data = params["data"]
    features = data["numeric_features"] + data["categorical_features"]
    assert data["target"] not in features
    assert data["id_column"] not in features
    assert len(features) == len(set(features)), "a feature is listed twice"


def test_resolve_makes_relative_paths_absolute_under_repo_root(tmp_path):
    assert resolve("data/raw/train.csv") == ROOT / "data" / "raw" / "train.csv"
    assert resolve(tmp_path) == tmp_path


def test_file_sha256_matches_hashlib_and_changes_with_content(tmp_path):
    f = tmp_path / "sample.csv"
    f.write_bytes(b"a,b\n1,2\n")
    assert file_sha256(f) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()

    first = file_sha256(f)
    f.write_bytes(b"a,b\n1,3\n")
    assert file_sha256(f) != first
    assert isinstance(Path(f), Path)
