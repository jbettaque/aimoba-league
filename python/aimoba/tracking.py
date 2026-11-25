"""Optional MLflow experiment tracking for training and tuning scripts.

    from aimoba.tracking import start_run

    with start_run("aimoba-ppo", params=vars(args)) as run:   # no-op if MLflow is missing or enabled=False
        run.log_metrics({"episode/win_rate": 0.4}, step=1000)
        run.log_artifact("ppo_aimoba.pt", "checkpoints")

Runs go to ``$MLFLOW_TRACKING_URI`` when it is set (e.g. a shared ``mlflow server``), otherwise to a
local store in ``<repo>/mlruns`` (SQLite database + artifact folder). Browse it with

    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

_PKG_ROOT = Path(__file__).resolve().parents[2]
# Inside the repo checkout the store lives next to package.json; for an installed copy, use the cwd.
LOCAL_DIR = (_PKG_ROOT if (_PKG_ROOT / "package.json").exists() else Path.cwd()) / "mlruns"


def tracking_uri() -> str:
    """The tracking URI scripts log to: ``$MLFLOW_TRACKING_URI`` or the repo-local SQLite store."""
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(LOCAL_DIR / 'mlflow.db').as_posix()}"


class Run:
    """Thin wrapper around an active MLflow run. Every method is a no-op when tracking is off."""

    def __init__(self, mlflow: Any = None, run: Any = None):
        self.mlflow, self.run = mlflow, run

    @property
    def active(self) -> bool:
        return self.run is not None

    @property
    def run_id(self) -> Optional[str]:
        return self.run.info.run_id if self.run else None

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self.run:
            self.mlflow.log_params({k: _param(v) for k, v in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        if self.run:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        if self.run and Path(path).exists():
            self.mlflow.log_artifact(str(path), artifact_path)

    def log_dict(self, data: Any, artifact_file: str) -> None:
        if self.run:
            self.mlflow.log_dict(data, artifact_file)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if self.run:
            self.mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def child(self, run_name: str, params: Optional[Mapping[str, Any]] = None) -> "Run":
        """A nested run under this one (e.g. one tuning trial)."""
        if not self.run:
            return Run()
        child = Run(self.mlflow, self.mlflow.start_run(run_name=run_name, nested=True))
        if params:
            child.log_params(params)
        return child

    def end(self, status: str = "FINISHED") -> None:
        if self.run:
            self.mlflow.end_run(status=status)
            self.run = None

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end("FINISHED" if exc_type is None else "KILLED" if exc_type is KeyboardInterrupt else "FAILED")


def start_run(
    experiment: str,
    run_name: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, Any]] = None,
    enabled: bool = True,
) -> Run:
    """Start an MLflow run in ``experiment`` (created on first use). Returns a no-op ``Run`` when
    ``enabled`` is false or MLflow is not installed, so callers never need to check."""
    if not enabled:
        return Run()
    try:
        import mlflow
    except ImportError:
        print("[aimoba.tracking] mlflow not installed - run is not tracked (pip install mlflow)", flush=True)
        return Run()

    uri = tracking_uri()
    local = "MLFLOW_TRACKING_URI" not in os.environ  # check first: set_tracking_uri() exports the variable
    if local:
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        # Keep artifacts of the local store inside mlruns/ instead of wherever the script was started.
        mlflow.create_experiment(experiment, artifact_location=(LOCAL_DIR / "artifacts" / experiment).as_uri() if local else None)
    mlflow.set_experiment(experiment)

    run = Run(mlflow, mlflow.start_run(run_name=run_name, tags=dict(tags or {})))
    if params:
        run.log_params(params)
    print(f"[aimoba.tracking] MLflow run {run.run_id} in experiment '{experiment}' ({uri})", flush=True)
    return run


def _param(v: Any) -> Any:
    # MLflow params are strings capped at 6000 chars; lists/dicts are stored as their repr.
    return v if isinstance(v, (int, float, str, bool)) or v is None else str(v)[:6000]
