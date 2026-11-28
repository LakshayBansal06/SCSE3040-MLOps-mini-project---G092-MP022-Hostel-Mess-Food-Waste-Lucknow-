"""Tests for the FastAPI prediction service.

These build their own request payloads: no CSV is read, so they run in CI
where data/raw/*.csv is not committed.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def valid_payload(**overrides):
    """A request the client would call normal, with optional overrides."""
    payload = {
        "students_registered": 420,
        "attendance_pct": 78.5,
        "menu_rating": 3.8,
        "temp_c": 29.4,
        "cooked_kg": 340.0,
        "plate_size_cm": 26.0,
        "meal": "lunch",
        "hostel": "Boys Hostel 1",
    }
    payload.update(overrides)
    return payload


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_a_number_for_a_valid_request():
    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["prediction"], float)
    assert body["unit"] == "kg"
    # Waste is a weight: never negative, and never more than what was cooked.
    assert 0 <= body["prediction"] <= valid_payload()["cooked_kg"]


def test_rule_one_rejects_temperature_above_75():
    """PROBLEM.md section 5: temp_c above 75 is impossible for us."""
    response = client.post("/predict", json=valid_payload(temp_c=75.1))

    assert response.status_code == 422
    assert any(error["loc"][-1] == "temp_c" for error in response.json()["detail"])


def test_rule_two_rejects_cooked_kg_below_25():
    """PROBLEM.md section 5: cooked_kg below 25 is always a typing mistake."""
    response = client.post("/predict", json=valid_payload(cooked_kg=24.9))

    assert response.status_code == 422
    assert any(error["loc"][-1] == "cooked_kg" for error in response.json()["detail"])


@pytest.mark.parametrize("field,value", [("temp_c", 75), ("cooked_kg", 25)])
def test_the_boundary_values_are_still_accepted(field, value):
    """The rules are 'above 75' and 'below 25', so 75 and 25 themselves pass."""
    response = client.post("/predict", json=valid_payload(**{field: value}))

    assert response.status_code == 200


def test_a_missing_field_is_rejected():
    payload = valid_payload()
    del payload["hostel"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_an_unexpected_field_is_rejected():
    """The body must have exactly the eight fields the client listed."""
    response = client.post("/predict", json=valid_payload(waste_kg=12.0))

    assert response.status_code == 422


def test_hostel_label_spelling_does_not_change_the_prediction():
    """The model was trained on normalised labels, so the API normalises too."""
    as_written = client.post("/predict", json=valid_payload(hostel="Boys Hostel 2"))
    as_cleaned = client.post("/predict", json=valid_payload(hostel="boys hostel 2"))

    assert as_written.status_code == 200
    assert as_cleaned.status_code == 200
    # The 300 trees are summed in parallel, so the order of the additions,
    # and with it the last floating-point digit, can differ between calls.
    assert as_written.json()["prediction"] == pytest.approx(
        as_cleaned.json()["prediction"], abs=1e-9
    )


def test_each_prediction_writes_one_json_line_to_stdout(capsys):
    """Handbook section 9: one JSON line per prediction, on stdout."""
    capsys.readouterr()

    client.post("/predict", json=valid_payload())

    printed = capsys.readouterr().out.strip().splitlines()
    records = [json.loads(line) for line in printed if line.startswith("{")]
    predictions = [r for r in records if r.get("event") == "prediction"]

    assert len(predictions) == 1
    assert predictions[0]["level"] == "INFO"
    assert "prediction_kg" in predictions[0]
