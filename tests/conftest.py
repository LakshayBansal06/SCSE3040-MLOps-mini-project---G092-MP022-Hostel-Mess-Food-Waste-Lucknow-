"""Shared test helpers.

The real CSVs are tracked by DVC and are not in git, so every test builds
its own small data frame here instead of reading data/raw/*.csv.
"""

import numpy as np
import pandas as pd
import pytest

from mess_waste.config import load_params

HOSTELS = ["boys hostel 1", "boys hostel 2", "girls hostel 1"]
MEALS = ["breakfast", "lunch", "dinner"]


def make_mess_frame(n_rows: int = 90, seed: int = 0) -> pd.DataFrame:
    """A clean, realistic-looking slice of mess data with a learnable target."""
    rng = np.random.default_rng(seed)
    cooked = rng.uniform(150, 450, n_rows).round(1)
    attendance = rng.uniform(55, 98, n_rows).round(1)
    rating = rng.uniform(1.5, 5.0, n_rows).round(2)
    waste = (cooked * (1 - attendance / 100) * 0.6 + (5 - rating) * 3
             + rng.normal(0, 1.5, n_rows)).round(2)
    return pd.DataFrame(
        {
            "record_id": [f"R{i:04d}" for i in range(n_rows)],
            "date": pd.date_range("2026-01-01", periods=n_rows).strftime("%Y-%m-%d"),
            "students_registered": rng.integers(200, 600, n_rows),
            "attendance_pct": attendance,
            "menu_rating": rating,
            "temp_c": rng.uniform(8, 40, n_rows).round(1),
            "cooked_kg": cooked,
            "plate_size_cm": rng.uniform(22, 28, n_rows).round(1),
            "meal": [MEALS[i % 3] for i in range(n_rows)],
            "hostel": [HOSTELS[i % 3] for i in range(n_rows)],
            "waste_kg": np.clip(waste, 1, None),
        }
    )


@pytest.fixture
def params() -> dict:
    """The project's real params.yaml (a small text file that is in git)."""
    return load_params()


@pytest.fixture
def mess_frame() -> pd.DataFrame:
    return make_mess_frame()
