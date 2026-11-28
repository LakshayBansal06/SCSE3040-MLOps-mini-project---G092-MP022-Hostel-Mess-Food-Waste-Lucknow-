"""MLflow experiment tracking configuration."""

import zipfile

import mlflow

from mess_waste.config import ROOT

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "hostel-mess-waste"

# Model files for the registered versions. mlruns/ is gitignored, so the
# files travel in this archive; mlflow.db refers to them as ./mlruns/...
ARTIFACT_ARCHIVE = ROOT / "artifacts" / "mlruns.zip"
ARTIFACT_DIR = ROOT / "mlruns"


def restore_artifacts() -> bool:
    """Unpack the committed model files into mlruns/ if they are not there.

    Returns True when files were unpacked. Existing files are never touched.
    """
    if not ARTIFACT_ARCHIVE.exists():
        return False
    with zipfile.ZipFile(ARTIFACT_ARCHIVE) as archive:
        missing = [
            name for name in archive.namelist()
            if not (ROOT / name).exists()
        ]
        for name in missing:
            target = (ROOT / name).resolve()
            if ROOT.resolve() not in target.parents:
                raise ValueError(f"unsafe path in {ARTIFACT_ARCHIVE.name}: {name}")
            archive.extract(name, ROOT)
    return bool(missing)


def setup_mlflow():
    """Configure MLflow and create or select the project experiment.

    Run the scripts from the repository root: mlflow.db and the artifact
    paths inside it (./mlruns/...) are relative to it.
    """
    restore_artifacts()
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment
