"""More tests for clean_train: each data problem is fixed, nothing else breaks."""

import pandas as pd
import pandas.testing as pdt

from mess_waste.data_checks import longest_run
from mess_waste.data_cleaning import clean_data, clean_train


def test_cleaning_does_not_change_the_input_frame(mess_frame, params):
    before = mess_frame.copy()
    clean_train(mess_frame, params=params)
    pdt.assert_frame_equal(mess_frame, before)


def test_already_clean_data_keeps_every_row(mess_frame, params):
    cleaned = clean_train(mess_frame, params=params)
    assert len(cleaned) == len(mess_frame)
    assert list(cleaned.index) == list(range(len(cleaned)))
    assert cleaned["waste_kg"].notna().all()


def test_unit_mixup_in_one_hostel_is_converted_back(mess_frame, params):
    df = mess_frame.copy()
    bad = df["hostel"] == "girls hostel 1"
    df.loc[bad, "plate_size_cm"] = df.loc[bad, "plate_size_cm"] * 10  # cm -> mm

    cleaned = clean_train(df, params=params)

    medians = cleaned.groupby("hostel")["plate_size_cm"].median()
    assert medians.max() / medians.min() < 1.5
    assert cleaned["plate_size_cm"].max() < 30


def test_copies_of_a_record_with_a_new_id_are_dropped(mess_frame, params):
    copies = mess_frame.head(4).copy()
    copies["record_id"] = [f"DUP{i}" for i in range(4)]
    df = pd.concat([mess_frame, copies], ignore_index=True)

    cleaned = clean_train(df, params=params)

    assert len(cleaned) == len(mess_frame)
    assert not cleaned.drop(columns=["record_id"]).duplicated().any()


def test_stuck_sensor_run_is_replaced(mess_frame, params):
    df = mess_frame.copy()
    df.loc[30:59, "temp_c"] = 21.7  # 30 days of the same reading

    cleaned = clean_train(df, params=params)

    run = longest_run(cleaned, "temp_c", "date")
    assert run["rows"] < params["checks"]["stuck_min_rows"]
    assert cleaned["temp_c"].notna().all()


def test_labels_are_normalised_and_unknown_ones_blanked(mess_frame, params):
    df = mess_frame.copy()
    df.loc[0, "meal"] = " LUNCH."
    df.loc[1, "meal"] = "midnight snack"  # never seen in later.csv
    later = mess_frame.copy()

    cleaned = clean_train(df, later, params)

    assert cleaned.loc[0, "meal"] == "lunch"
    assert pd.isna(cleaned.loc[1, "meal"])
    assert set(cleaned["meal"].dropna()) <= {"breakfast", "lunch", "dinner"}


def test_clean_data_is_the_same_as_clean_train(mess_frame, params):
    pdt.assert_frame_equal(
        clean_data(mess_frame, params=params),
        clean_train(mess_frame, params=params),
    )
