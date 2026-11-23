"""Data-cleaning transformations for the training dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mess_waste.config import load_params, resolve
from mess_waste.data_checks import (
    find_sentinel_value,
    find_stuck_sensor,
    find_target_typos,
    find_unit_mixup,
    longest_run,
    normalise_label,
)


def clean_train(
    train: pd.DataFrame,
    later: pd.DataFrame | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """Apply the discovered cleaning transformations without mutating input."""
    if params is None:
        params = load_params()

    d = params["data"]
    c = params["checks"]
    df = train.copy()

    unit = find_unit_mixup(
        df,
        d["numeric_features"],
        d["group_column"],
        c["unit_factors"],
        c["unit_min_ratio"],
    )
    if unit["present"]:
        mask = df[d["group_column"]] == unit["group_value"]
        df.loc[mask, unit["column"]] = (
            df.loc[mask, unit["column"]] / unit["factor"]
        )

    sentinel = find_sentinel_value(
        df,
        d["numeric_features"],
        c["sentinel_candidates"],
        c["sentinel_min_count"],
    )
    if sentinel["present"]:
        df[sentinel["column"]] = df[sentinel["column"]].where(
            df[sentinel["column"]] != sentinel["value"],
        )
        df[sentinel["column"]] = df[sentinel["column"]].fillna(
            df[sentinel["column"]].median()
        )

    duplicate = df.drop(columns=[d["id_column"]], errors="ignore").duplicated()
    if d["id_column"] in df.columns:
        df = df.loc[~duplicate].copy()
    else:
        df = df.loc[~df.duplicated()].copy()

    stuck = find_stuck_sensor(
        df,
        d["numeric_features"],
        d["date_column"],
        c["stuck_min_rows"],
        c["stuck_min_days"],
    )
    if stuck["present"]:
        col = stuck["column"]
        ordered = df.sort_values(d["date_column"], kind="stable").copy()
        run = longest_run(ordered, col, d["date_column"])
        start = pd.Timestamp(run["start_date"])
        end = pd.Timestamp(run["end_date"])
        dates = pd.to_datetime(ordered[d["date_column"]])
        mask = (dates >= start) & (dates <= end)
        ordered.loc[mask, col] = pd.NA
        ordered[col] = pd.to_numeric(ordered[col], errors="coerce").interpolate(
            method="linear"
        )
        ordered[col] = ordered[col].bfill().ffill()
        df = ordered

    typo = find_target_typos(
        df,
        d["target"],
        c["typo_gap_ratio"],
        c["typo_max_share"],
    )
    if typo["present"]:
        threshold = typo["threshold"]
        df.loc[df[d["target"]] > threshold, d["target"]] /= 10

    for col in d["categorical_features"]:
        if col in df.columns:
            df[col] = df[col].map(normalise_label)

    if later is not None:
        for col in d["categorical_features"]:
            if col in df.columns and col in later.columns:
                allowed = set(later[col].dropna().map(normalise_label))
                df[col] = df[col].where(df[col].isin(allowed))

    return df.reset_index(drop=True)


def clean_data(
    train: pd.DataFrame,
    later: pd.DataFrame | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """Backward-compatible descriptive alias for clean_train."""
    return clean_train(train, later, params)


def main() -> Path:
    params = load_params()
    d = params["data"]
    train = pd.read_csv(resolve(d["train"]))
    later = pd.read_csv(resolve(d["later"]))
    cleaned = clean_train(train, later, params)

    output = resolve(params["cleaning"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)
    print(f"Saved cleaned training data to {output.relative_to(resolve('.'))}")
    print(f"Rows: {len(cleaned)}")
    return output


if __name__ == "__main__":
    main()
