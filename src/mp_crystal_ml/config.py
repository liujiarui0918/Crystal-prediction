from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

MATBENCH_TASKS = {
    "matbench_mp_gap": {
        "display_name": "Matbench MP Gap",
        "task_type": "regression",
        "target_display": "PBE band gap",
        "unit": "eV",
        "sort_metric": "rmse",
        "sort_ascending": True,
    },
    "matbench_mp_e_form": {
        "display_name": "Matbench MP Formation Energy",
        "task_type": "regression",
        "target_display": "formation energy per atom",
        "unit": "eV/atom",
        "sort_metric": "rmse",
        "sort_ascending": True,
    },
    "matbench_mp_is_metal": {
        "display_name": "Matbench MP Metallicity",
        "task_type": "classification",
        "target_display": "is metal",
        "unit": "boolean",
        "sort_metric": "roc_auc",
        "sort_ascending": False,
    },
}

MODEL_NAMES = ("cgcnn", "alignn", "m3gnet")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = ROOT_DIR
    raw_dir: Path = ROOT_DIR / "data" / "raw"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    metrics_json: Path = ROOT_DIR / "reports" / "metrics_summary.json"
    report_md: Path = ROOT_DIR / "reports" / "summary.md"
    figures_dir: Path = ROOT_DIR / "reports" / "figures"
    models_dir: Path = ROOT_DIR / "models"

    def task_cache_file(self, task_name: str, sample_size: int, random_state: int) -> Path:
        return self.raw_dir / f"{task_name}_n{sample_size}_seed{random_state}.json.gz"

    def split_manifest_file(self, task_name: str) -> Path:
        return self.processed_dir / f"{task_name}_splits.json"

    def model_dir_for(self, task_name: str, model_name: str) -> Path:
        return self.models_dir / task_name / model_name

    def figure_path(self, filename: str) -> Path:
        return self.figures_dir / filename
