import pandas as pd

from mess_waste.data_cleaning import clean_train


def test_clean_train_applies_unit_sentinel_target_and_label_fixes():
    train = pd.DataFrame(
        {
            "record_id": [f"R{i}" for i in range(20)],
            "date": pd.date_range("2026-01-01", periods=20).strftime("%Y-%m-%d"),
            "students_registered": [300] * 15 + [-999] * 5,
            "attendance_pct": [80] * 20,
            "menu_rating": [3.5] * 20,
            "temp_c": [25] * 20,
            "cooked_kg": [300] * 20,
            "plate_size_cm": [24] * 20,
            "hostel": ["Boys Hostel 2"] * 20,
            "meal": ["Lunch "] * 20,
            "waste_kg": [30] * 19 + [300],
        }
    )
    later = train.copy()
    later["meal"] = "lunch"

    params = {
        "data": {
            "id_column": "record_id",
            "date_column": "date",
            "target": "waste_kg",
            "group_column": "hostel",
            "numeric_features": [
                "students_registered",
                "attendance_pct",
                "menu_rating",
                "temp_c",
                "cooked_kg",
                "plate_size_cm",
            ],
            "categorical_features": ["meal", "hostel"],
        },
        "checks": {
            "unit_factors": [10, 12, 60, 100, 1000],
            "unit_min_ratio": 5.0,
            "sentinel_candidates": [-1, -99, -999, 999, 9999],
            "sentinel_min_count": 3,
            "stuck_min_rows": 20,
            "stuck_min_days": 7,
            "typo_gap_ratio": 3.0,
            "typo_max_share": 0.05,
        },
    }

    cleaned = clean_train(train, later, params)

    assert -999 not in cleaned["students_registered"].values
    assert cleaned["waste_kg"].max() <= 67
    assert cleaned["meal"].eq("lunch").all()
    assert cleaned["cooked_kg"].eq(300).all()
