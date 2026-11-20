"""The seven data-problem checks from the handbook (section 7).

Every function looks at the data and returns a dict shaped like the matching
block of reports/findings.yaml: ``present`` plus the details (or None when
the problem is absent).

Run all checks on our own files:

    python -m mess_waste.data_checks            # from inside src/
    PYTHONPATH=src python -m mess_waste.data_checks   # from the repo root

The results are printed and saved to reports/data_checks.json.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd

from mess_waste.config import load_csv, load_params, resolve


def _absent(*keys: str) -> dict:
    out = {"present": False}
    out.update({k: None for k in keys})
    return out


def _py(x):
    """numpy scalar -> plain Python (so json / yaml can write it)."""
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


# 1. leaking_column -------------------------------------------------------------
def find_leaking_column(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target: str,
    id_column: str,
    corr_threshold: float = 0.98,
) -> dict:
    """A column filled in only after the meal.

    Two signs: (a) it is in train but missing from the holdout features, or
    (b) it agrees almost exactly with the target (|correlation| >= threshold).
    """
    candidates = []
    missing = [c for c in train.columns if c not in holdout.columns and c != target]
    candidates.extend(missing)

    numeric = train.select_dtypes("number").columns
    for col in numeric:
        if col in (target, id_column) or col in candidates:
            continue
        corr = train[col].corr(train[target])
        if pd.notna(corr) and abs(corr) >= corr_threshold:
            candidates.append(col)

    if not candidates:
        return _absent("column")
    return {"present": True, "column": candidates[0]}


# 2. unit_mixup ----------------------------------------------------------------
def find_unit_mixup(
    df: pd.DataFrame,
    numeric_cols: list[str],
    group_col: str,
    factors: list[float],
    min_ratio: float = 5.0,
) -> dict:
    """One group recorded one column in another unit.

    For every numeric column, compare each group's median with the median of
    all other groups. If one group is far off (>= min_ratio times bigger or
    smaller), report the exact unit factor closest to that ratio (on a log
    scale).
    """
    best = None  # (log-distance to factor, column, group, factor)
    for col in numeric_cols:
        values = df[col]
        if (values <= 0).any():
            # negative / zero numbers (e.g. sentinels) would break the ratio
            values = values.where(values > 0)
        for grp in df[group_col].dropna().unique():
            in_grp = values[df[group_col] == grp].median()
            others = values[df[group_col] != grp].median()
            if not (in_grp and others) or pd.isna(in_grp) or pd.isna(others):
                continue
            ratio = in_grp / others
            big = max(ratio, 1 / ratio)
            if big < min_ratio:
                continue
            factor = min(factors, key=lambda f: abs(math.log(big) - math.log(f)))
            dist = abs(math.log(big) - math.log(factor))
            if best is None or dist < best[0]:
                best = (dist, col, grp, factor)

    if best is None:
        return _absent("column", "group_value", "factor")
    _, col, grp, factor = best
    return {"present": True, "column": col, "group_value": grp, "factor": _py(factor)}


# 3. sentinel_value ------------------------------------------------------------
def find_sentinel_value(
    df: pd.DataFrame,
    numeric_cols: list[str],
    candidates: list[float],
    min_count: int = 3,
) -> dict:
    """A special number (-1, -999, 999, 9999 ...) typed instead of a blank."""
    best = None  # (count, column, value)
    for col in numeric_cols:
        counts = df[col].value_counts()
        for value in candidates:
            n = int(counts.get(value, 0))
            if n >= min_count and (best is None or n > best[0]):
                best = (n, col, value)

    if best is None:
        return _absent("column", "value", "count")
    n, col, value = best
    return {"present": True, "column": col, "value": _py(value), "count": n}


# 4. duplicate_rows ------------------------------------------------------------
def find_duplicate_rows(df: pd.DataFrame, id_column: str) -> dict:
    """Extra copies of the same record.

    We count exact duplicate rows, and also rows that repeat every value
    except the record id (a re-typed record with a new id).
    """
    exact = int(df.duplicated().sum())
    without_id = int(df.drop(columns=[id_column]).duplicated().sum())
    count = max(exact, without_id)
    if count == 0:
        return _absent("count")
    return {"present": True, "count": count}


# 5. stuck_sensor --------------------------------------------------------------
def longest_run(df: pd.DataFrame, col: str, date_col: str) -> dict:
    """Longest block of consecutive rows (sorted by date) with one value."""
    d = df.sort_values(date_col, kind="stable").reset_index(drop=True)
    block = (d[col] != d[col].shift()).cumsum()
    runs = d.groupby(block).agg(
        value=(col, "first"),
        rows=(col, "size"),
        start=(date_col, "min"),
        end=(date_col, "max"),
    )
    top = runs.sort_values("rows", ascending=False).iloc[0]
    return {
        "column": col,
        "value": _py(top["value"]),
        "rows": int(top["rows"]),
        "start_date": str(top["start"]),
        "end_date": str(top["end"]),
    }


def find_stuck_sensor(
    df: pd.DataFrame,
    numeric_cols: list[str],
    date_col: str,
    min_rows: int = 20,
    min_days: int = 7,
) -> dict:
    """A device sent the same reading for weeks.

    Sort by date, find the longest run of identical values in every numeric
    column, and flag the longest run if it is long in rows and in days.
    """
    runs = [longest_run(df, c, date_col) for c in numeric_cols]
    top = max(runs, key=lambda r: r["rows"])
    days = (pd.Timestamp(top["end_date"]) - pd.Timestamp(top["start_date"])).days + 1
    if top["rows"] < min_rows or days < min_days:
        return _absent("column", "start_date", "end_date")
    return {
        "present": True,
        "column": top["column"],
        "start_date": top["start_date"],
        "end_date": top["end_date"],
    }


# 6. target_typos --------------------------------------------------------------
def find_target_typos(
    df: pd.DataFrame,
    target: str,
    gap_ratio: float = 3.0,
    max_share: float = 0.05,
) -> dict:
    """A few targets typed with an extra zero (10 times too big).

    Sort the target. Walk along the sorted values and find the biggest jump
    (ratio between neighbours). If that jump is at least ``gap_ratio`` and only
    a small share of rows sits above it, those rows are the typos.
    """
    y = df[target].dropna().sort_values().to_numpy()
    y = y[y > 0]
    if len(y) < 2:
        return _absent("count", "threshold")
    ratios = y[1:] / y[:-1]
    i = int(np.argmax(ratios))
    n_above = len(y) - (i + 1)
    if ratios[i] < gap_ratio or n_above > max_share * len(y):
        return _absent("count", "threshold")
    # threshold: middle of the empty gap (stored for the cleaning step)
    threshold = float(math.sqrt(y[i] * y[i + 1]))
    return {"present": True, "count": int(n_above), "threshold": round(threshold, 2)}


# 7. spelling_variants ---------------------------------------------------------
def normalise_label(s: str) -> str:
    """'Rain', ' rain ', 'RAIN.' -> 'rain'."""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def find_spelling_variants(
    train: pd.DataFrame,
    clean_reference: pd.DataFrame,
    text_cols: list[str],
) -> dict:
    """One text column has the same category written in different ways.

    A column has variants if train has more distinct spellings than real
    categories, or spellings that never appear in the clean later.csv.
    ``true_levels`` = number of categories in the clean reference.
    """
    for col in text_cols:
        raw = set(train[col].dropna().unique())
        clean = set(clean_reference[col].dropna().unique())
        norm = {normalise_label(v) for v in raw}
        if len(raw) > len(norm) or not raw <= clean:
            true_levels = len({normalise_label(v) for v in clean}) or len(norm)
            return {"present": True, "column": col, "true_levels": int(true_levels)}
    return _absent("column", "true_levels")


# Run everything ----------------------------------------------------------------
FINDINGS_KEYS = {
    "leaking_column": ["present", "column"],
    "unit_mixup": ["present", "column", "group_value", "factor"],
    "sentinel_value": ["present", "column", "value", "count"],
    "duplicate_rows": ["present", "count"],
    "stuck_sensor": ["present", "column", "start_date", "end_date"],
    "target_typos": ["present", "count"],
    "spelling_variants": ["present", "column", "true_levels"],
}


def run_all_checks(
    train: pd.DataFrame,
    later: pd.DataFrame,
    holdout: pd.DataFrame,
    params: dict,
) -> dict:
    d, c = params["data"], params["checks"]
    num, cat = d["numeric_features"], d["categorical_features"]
    return {
        "leaking_column": find_leaking_column(
            train, holdout, d["target"], d["id_column"], c["leak_corr_threshold"]
        ),
        "unit_mixup": find_unit_mixup(
            train, num, d["group_column"], c["unit_factors"], c["unit_min_ratio"]
        ),
        "sentinel_value": find_sentinel_value(
            train, num, c["sentinel_candidates"], c["sentinel_min_count"]
        ),
        "duplicate_rows": find_duplicate_rows(train, d["id_column"]),
        "stuck_sensor": find_stuck_sensor(
            train, num, d["date_column"], c["stuck_min_rows"], c["stuck_min_days"]
        ),
        "target_typos": find_target_typos(
            train, d["target"], c["typo_gap_ratio"], c["typo_max_share"]
        ),
        "spelling_variants": find_spelling_variants(train, later, cat),
    }


def to_findings(results: dict) -> dict:
    """Keep only the keys that reports/findings.yaml asks for."""
    return {k: {f: results[k].get(f) for f in keys} for k, keys in FINDINGS_KEYS.items()}


def main() -> dict:
    params = load_params()
    d = params["data"]
    train, later, holdout = load_csv(d["train"]), load_csv(d["later"]), load_csv(d["holdout"])
    results = run_all_checks(train, later, holdout, params)

    out = resolve(params["reports"]["data_checks"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(to_findings(results), indent=2))
    print(f"\nSaved full results to {out.relative_to(resolve('.'))}")
    return results


if __name__ == "__main__":
    main()
