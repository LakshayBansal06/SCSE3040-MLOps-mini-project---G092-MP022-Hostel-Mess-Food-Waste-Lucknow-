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
  data_cleaning.py        transformations for the cleaned training data
  tracking.py             MLflow setup (sqlite:///mlflow.db)
  train.py                trains v1 (3 settings), registers the best, sets @champion
  monitor.py              PSI drift, v1 MAE per hostel, v2 retrain, promotion rule
  export_model.py         saves the @champion model to models/model.joblib
app/                      the prediction service
  main.py                 FastAPI app: POST /predict and GET /health
  logging_config.py       one JSON line per prediction, to stdout
models/model.joblib       the champion model the service loads
artifacts/mlruns.zip      MLflow model files for v1-v3 (mlruns/ itself is gitignored)
tests/                    pytest tests (build their own small data, no CSVs needed)
Dockerfile                container image for the API
.github/workflows/ci.yml  test workflow for pushes
k8s/                      Kubernetes deployment and service
data/raw/*.csv.dvc        DVC pointers for train.csv and later.csv (CSVs not in git)
data/raw/holdout_features.csv
.dvc/config               from dvc init
reports/findings.yaml     the marking form
reports/data_checks.json  full output of the data checks
mlflow.db                 MLflow runs and model registry (committed)
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

## Stage 2: cleaning, MLflow tracking and registry

```bash
PYTHONPATH=src python -m mess_waste.data_cleaning   # writes data/clean/train.csv
PYTHONPATH=src python -m mess_waste.train           # 3 runs, registers the best as v1
```

Cleaning fixes every problem found in Stage 1: `cooked_kg` of Boys Hostel 2
divided by 1000, `-999` in `students_registered` replaced by the median, the
stuck `menu_rating` run interpolated, the 10 target typos divided by 10, and
labels normalised.

Training tracks to `sqlite:///mlflow.db` in the experiment `hostel-mess-waste`.
Every run logs `data_sha256` and `mae_validation`. Three settings were tried:

| Run | Model | Validation MAE |
|---|---|---|
| `random_forest_200` | RandomForest, 200 trees, max depth 8 | 3.601 |
| `random_forest_300` | RandomForest, 300 trees, min leaf 2 | 3.500 |
| `extra_trees_300` | ExtraTrees, 300 trees, min leaf 2 | **3.301** |

The best (`extra_trees_300`, run `96a1a07d2a444b77be67ddf9194d9c1e`) is
registered as `hostel-mess-waste-regressor` version 1 and got the alias
`champion`.

## Tests

```bash
pytest
```

## Stage 3: the service, the container, CI and Kubernetes

Export the `@champion` model once, then run the API:

```bash
PYTHONPATH=src python -m mess_waste.export_model    # writes models/model.joblib
PYTHONPATH="src;." uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`POST /predict` takes exactly the eight feature fields and returns the
predicted waste in kg:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json"   -d '{"students_registered":420,"attendance_pct":78.5,"menu_rating":3.8,
       "temp_c":29.4,"cooked_kg":340.0,"plate_size_cm":26.0,
       "meal":"lunch","hostel":"Boys Hostel 1"}'
# {"prediction":33.51,"waste_kg":33.51,"unit":"kg"}
```

`GET /health` returns `{"status": "ok", "model_loaded": true}`.

The client's two rules live in the pydantic schema, so a bad request is
rejected with **422** before the model is ever called:

| Rule | Field constraint | Example |
|---|---|---|
| `temp_c` above 75 is impossible | `Field(..., le=75)` | `temp_c: 80` -> 422 |
| `cooked_kg` below 25 is a typo | `Field(..., ge=25)` | `cooked_kg: 10` -> 422 |

The body must have exactly those eight fields: anything extra or missing is
also a 422. Text labels are normalised the same way the training data was
(`Boys Hostel 2` -> `boys hostel 2`), so the spelling a caller uses does not
change the prediction.

Every prediction writes one JSON line to stdout, never to a file:

```json
{"timestamp": "2026-11-26T16:12:53.398065+00:00", "level": "INFO",
 "logger": "mp022-api", "event": "prediction", "path": "/predict",
 "prediction_kg": 33.5104, "latency_ms": 84.348, "features": {...}}
```

Container and cluster:

```bash
docker build -t mp022-api:1.0 .
docker run --rm -p 8000:8000 mp022-api:1.0
kubectl apply -f k8s/
```

`k8s/deployment.yaml` runs **3 replicas** of `mp022-api` from `mp022-api:1.0`
on container port 8000, with readiness and liveness probes on `/health`
(liveness waits longer). `k8s/service.yaml` exposes `mp022-svc` on port 80.

`.github/workflows/ci.yml` runs `pytest` on every push.

## Stage 4: drift, retraining and the holdout

```bash
PYTHONPATH=src python -m mess_waste.monitor
```

| Question | Answer |
|---|---|
| Most drifted column (PSI, 10 bins) | **`temp_c`**, PSI 6.73 (every other column is under 0.07) |
| Worst-served group (v1 MAE on `later.csv`) | **Girls Hostel**, MAE 27.51 against 7.20 and 8.06 |
| v1 vs v2 on the last 30% | 10.52 against **4.34** |
| The client's rule (promote if 10% lower) | 4.34 is 58.8% lower, so **promote** |

`predictions/holdout_predictions.csv` has `record_id,prediction` for all 250
holdout rows, written by the `@champion` model.

`mlflow.db` stores the model files' locations as absolute paths from the
machine that trained them. To load v1-v3 from the registry on another
machine, unzip `artifacts/mlruns.zip` into the repository root and point those
paths at your copy (in a local copy of `mlflow.db`, not the committed one).

## Progress

- [x] Stage 1: repo, DVC, the seven data problems, `findings.yaml` data part
- [x] Stage 2: cleaning code, MLflow tracking and registry, tests
- [x] Stage 3: FastAPI service, Dockerfile, GitHub Actions, Kubernetes, logging
- [x] Stage 4: drift, retraining, holdout predictions, rest of `findings.yaml`
