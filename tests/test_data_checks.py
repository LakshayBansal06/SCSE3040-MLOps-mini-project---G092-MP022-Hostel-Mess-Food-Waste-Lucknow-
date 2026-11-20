"""Tests for the seven data checks.

The CSVs are not in git, so every test builds its own small DataFrame.
"""

import pandas as pd
import pytest

from mess_waste.config import load_params
from mess_waste.data_checks import (
    find_duplicate_rows,
    find_leaking_column,
    find_sentinel_value,
    find_spelling_variants,
    find_stuck_sensor,
    find_target_typos,
    find_unit_mixup,
    normalise_label,
)


@pytest.fixture
def clean_df():
    n = 60
    return pd.DataFrame(
        {
            "record_id": [f"R{i:05d}" for i in range(n)],
            "date": pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "students_registered": [300 + (i * 37) % 700 for i in range(n)],
            "menu_rating": [1 + (i * 7) % 40 / 10 for i in range(n)],
            "cooked_kg": [100 + (i * 13) % 400 for i in range(n)],
            "hostel": ["Boys Hostel 1", "Boys Hostel 2", "Girls Hostel"] * (n // 3),
            "meal": ["breakfast", "lunch", "dinner"] * (n // 3),
            "waste_kg": [20 + (i * 11) % 45 for i in range(n)],
        }
    )


def test_params_file_is_read():
    params = load_params()
    assert params["data"]["target"] == "waste_kg"
    assert len(params["data"]["numeric_features"]) == 6


def test_clean_data_has_no_problems(clean_df):
    holdout = clean_df.drop(columns="waste_kg")
    num = ["students_registered", "menu_rating", "cooked_kg"]
    assert not find_leaking_column(clean_df, holdout, "waste_kg", "record_id")["present"]
    assert not find_unit_mixup(clean_df, num, "hostel", [10, 100, 1000])["present"]
    assert not find_sentinel_value(clean_df, num, [-1, -999, 999])["present"]
    assert not find_duplicate_rows(clean_df, "record_id")["present"]
    assert not find_stuck_sensor(clean_df, num, "date", 20, 7)["present"]
    assert not find_target_typos(clean_df, "waste_kg")["present"]
    assert not find_spelling_variants(clean_df, clean_df, ["meal", "hostel"])["present"]


def test_leaking_column_found(clean_df):
    df = clean_df.assign(waste_logged=clean_df["waste_kg"] * 1.01)
    holdout = clean_df.drop(columns="waste_kg")
    res = find_leaking_column(df, holdout, "waste_kg", "record_id")
    assert res == {"present": True, "column": "waste_logged"}


def test_unit_mixup_found(clean_df):
    df = clean_df.copy()
    mask = df["hostel"] == "Boys Hostel 2"
    df.loc[mask, "cooked_kg"] = df.loc[mask, "cooked_kg"] * 1000
    res = find_unit_mixup(df, ["cooked_kg"], "hostel", [10, 12, 60, 100, 1000])
    assert res == {"present": True, "column": "cooked_kg", "group_value": "Boys Hostel 2", "factor": 1000}


def test_sentinel_found(clean_df):
    df = clean_df.copy()
    df.loc[:4, "students_registered"] = -999
    res = find_sentinel_value(df, ["students_registered"], [-1, -999, 999], min_count=3)
    assert res == {"present": True, "column": "students_registered", "value": -999, "count": 5}


def test_duplicates_counted(clean_df):
    df = pd.concat([clean_df, clean_df.iloc[:3]], ignore_index=True)
    assert find_duplicate_rows(df, "record_id") == {"present": True, "count": 3}


def test_duplicate_check_reports_missing_id_column(clean_df):
    with pytest.raises(
        ValueError,
        match="Cannot check duplicate rows: ID column 'record_id' is missing",
    ):
        find_duplicate_rows(clean_df.drop(columns="record_id"), "record_id")


def test_stuck_sensor_found(clean_df):
    df = clean_df.copy()
    df.loc[10:34, "menu_rating"] = 3.4  # 25 days in a row
    res = find_stuck_sensor(df, ["menu_rating"], "date", min_rows=20, min_days=7)
    assert res["present"] and res["column"] == "menu_rating"
    assert (res["start_date"], res["end_date"]) == ("2026-01-11", "2026-02-04")


def test_target_typos_found(clean_df):
    df = clean_df.copy()
    df.loc[[0, 5, 9], "waste_kg"] = df.loc[[0, 5, 9], "waste_kg"] * 10
    res = find_target_typos(df, "waste_kg", gap_ratio=3.0, max_share=0.1)
    assert res["present"] and res["count"] == 3


def test_spelling_variants_found(clean_df):
    df = clean_df.copy()
    df.loc[0, "meal"] = "Lunch "
    df.loc[1, "meal"] = "LUNCH."
    res = find_spelling_variants(df, clean_df, ["meal", "hostel"])
    assert res == {"present": True, "column": "meal", "true_levels": 3}
    assert normalise_label(" LUNCH. ") == "lunch"
