# G092 · MP022: Hostel Mess Food Waste (Lucknow)

SCSE3040 MLOps mini project, Bennett University, Batch 58.

| | |
|---|---|
| **Group** | G092 |
| **Problem** | MP022, Hostel Mess Food Waste (Lucknow) |
| **Faculty** | Dr. Gaurav Tripathi |

## Members

| Enrolment no. | Name |
|---|---|
| S24CSEU0323 | Riddimasingh |
| S24CSEU0799 | EASHANGUPTA |
| S24CSEU0070 | LakshayBansal |

## The problem

The mess of the (made-up) Lucknow University Hostel Office wants to predict
`waste_kg`, the food thrown away after each meal, so the cooks can adjust
how much they cook. This is a **regression** problem, measured with **MAE** (kg).

Features: `students_registered`, `attendance_pct`, `menu_rating`, `temp_c`,
`cooked_kg`, `plate_size_cm`, `meal`, `hostel`.

## Repository layout

```
README.md                 this file
requirements.txt          pinned libraries
params.yaml               paths and settings, read with yaml.safe_load
src/mess_waste/           our Python package
  config.py               loads params.yaml, file hashing, CSV loading
  data_checks.py          the seven data-problem checks
tests/                    pytest tests (build their own small data, no CSVs needed)
data/raw/*.csv.dvc        DVC pointers for train.csv and later.csv (CSVs not in git)
data/raw/holdout_features.csv
.dvc/config               from dvc init
reports/findings.yaml     the marking form
reports/data_checks.json  full output of the data checks
docs/                     PROBLEM.md and STUDENT_HANDBOOK.md from the brief
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install dvc                    # data versioning
```

Put the original `train.csv` and `later.csv` from the zip into `data/raw/`
(do **not** open and save them in Excel). Check that they match the versions
tracked by DVC:

```bash
dvc status        # should say "Data and pipelines are up to date."
```

## Stage 1: the seven data problems in `train.csv`

Run the checks (all settings come from `params.yaml`):

```bash
PYTHONPATH=src python -m mess_waste.data_checks
```

This prints the result and saves it to `reports/data_checks.json`.

| Problem | Present | Details |
|---|---|---|
| `leaking_column` | no | Every train column except the target is also in `holdout_features.csv`, and no feature has a correlation of 0.98 or more with `waste_kg`. |
| `unit_mixup` | **yes** | `cooked_kg` for **Boys Hostel 2** is in grams: median about 325,650 against about 339 for the other hostels. Factor **1000**. |
| `sentinel_value` | **yes** | `students_registered` = **-999** in **45** rows. |
| `duplicate_rows` | no | 0 exact duplicate rows, and 0 duplicates when `record_id` is ignored. |
| `stuck_sensor` | **yes** | `menu_rating` is stuck at 3.4 on all 91 rows from **2026-03-29** to **2026-04-11** (sorted by date). |
| `target_typos` | **yes** | **10** rows of `waste_kg` between 278 and 636. The normal maximum is 67.6, so there is a big gap below them. Divided by 10 they fall in the normal range. |
| `spelling_variants` | no | `meal` has 3 clean levels and `hostel` has 3, all spelled exactly as in `later.csv`. |

SHA-256 of the original `train.csv`
(logged as `data_sha256` in every MLflow run):
`20b940086cd33d3aea1523bfab94b69bd4597fbcf1c9df442d45bf18e9eae369`

## Tests

```bash
pytest
```

## Progress

- [x] Stage 1: repo, DVC, the seven data problems, `findings.yaml` data part
- [ ] Stage 2: cleaning code, MLflow tracking and registry, tests
- [ ] Stage 3: FastAPI service, Dockerfile, GitHub Actions, Kubernetes, logging
- [ ] Stage 4: drift, retraining, holdout predictions, rest of `findings.yaml`
