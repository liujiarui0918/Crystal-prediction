from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

from mp_crystal_ml.config import MATBENCH_TASKS, ProjectPaths


def _target_column(frame: pd.DataFrame) -> str:
    return next(column for column in frame.columns if column != "structure")


def _sample_frame(
    frame: pd.DataFrame,
    target_column: str,
    task_type: str,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    if sample_size >= len(frame):
        return frame.reset_index(drop=True)

    if task_type == "classification":
        sampled, _ = train_test_split(
            frame,
            train_size=sample_size,
            stratify=frame[target_column],
            random_state=random_state,
        )
        return sampled.reset_index(drop=True)

    return frame.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def _serialize_target(value: Any) -> float | bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return float(value)


def _record_from_row(task_name: str, index: int, row: pd.Series, target_column: str) -> dict[str, Any]:
    structure: Structure = row["structure"]
    return {
        "sample_id": f"{task_name}-{index:06d}",
        "formula_pretty": structure.composition.reduced_formula,
        "num_sites": int(len(structure)),
        "num_elements": int(len(structure.composition.elements)),
        "volume": float(structure.volume),
        "density": float(structure.density),
        "target": _serialize_target(row[target_column]),
        "structure": structure.as_dict(),
    }


def _write_records(cache_file: Path, records: list[dict[str, Any]]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_file, "wt", encoding="utf-8") as handle:
        json.dump(records, handle)


def _read_records(cache_file: Path) -> list[dict[str, Any]]:
    with gzip.open(cache_file, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_task_records(
    paths: ProjectPaths,
    task_name: str,
    sample_size: int,
    random_state: int,
    force_fetch: bool,
) -> list[dict[str, Any]]:
    cache_file = paths.task_cache_file(task_name, sample_size, random_state)
    if cache_file.exists() and not force_fetch:
        return _read_records(cache_file)

    if task_name not in MATBENCH_TASKS:
        raise KeyError(f"Unsupported Matbench task: {task_name}")

    frame = load_dataset(task_name)
    target_column = _target_column(frame)
    sampled = _sample_frame(
        frame=frame,
        target_column=target_column,
        task_type=MATBENCH_TASKS[task_name]["task_type"],
        sample_size=sample_size,
        random_state=random_state,
    )

    records = [
        _record_from_row(task_name=task_name, index=index, row=row, target_column=target_column)
        for index, (_, row) in enumerate(sampled.iterrows())
    ]
    _write_records(cache_file, records)
    return records


def split_records(
    records: list[dict[str, Any]],
    task_type: str,
    random_state: int,
) -> dict[str, list[dict[str, Any]]]:
    def can_stratify(values: list[int]) -> bool:
        if not values:
            return False
        unique, counts = np.unique(values, return_counts=True)
        return len(unique) > 1 and int(counts.min()) >= 2

    indices = np.arange(len(records))
    stratify = [int(record["target"]) for record in records] if task_type == "classification" else None
    if task_type == "classification" and not can_stratify(stratify):
        stratify = None

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )
    temp_stratify = None
    if task_type == "classification":
        temp_stratify = [int(records[index]["target"]) for index in temp_idx]
        if not can_stratify(temp_stratify):
            temp_stratify = None

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_stratify,
    )

    return {
        "train": [records[index] for index in train_idx],
        "val": [records[index] for index in val_idx],
        "test": [records[index] for index in test_idx],
    }


def write_split_manifest(paths: ProjectPaths, task_name: str, splits: dict[str, list[dict[str, Any]]]) -> None:
    manifest = {
        split_name: [record["sample_id"] for record in split_records]
        for split_name, split_records in splits.items()
    }
    output_file = paths.split_manifest_file(task_name)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def records_to_structures(records: list[dict[str, Any]]) -> list[Structure]:
    return [Structure.from_dict(record["structure"]) for record in records]


def records_to_targets(records: list[dict[str, Any]], task_type: str) -> np.ndarray:
    dtype = float if task_type == "regression" else int
    return np.asarray([record["target"] for record in records], dtype=dtype)


def records_to_alignn_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from jarvis.core.atoms import pmg_to_atoms

    samples = []
    for record in records:
        structure = Structure.from_dict(record["structure"])
        atoms = pmg_to_atoms(structure)
        samples.append(
            {
                "jid": record["sample_id"],
                "atoms": atoms.to_dict(),
                "target": float(record["target"]),
            }
        )
    return samples


def summarize_task_records(records: list[dict[str, Any]], task_type: str) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    summary = {
        "num_samples": int(len(frame)),
        "mean_num_sites": float(frame["num_sites"].mean()),
        "mean_num_elements": float(frame["num_elements"].mean()),
        "mean_density": float(frame["density"].mean()),
    }
    if task_type == "classification":
        summary["positive_fraction"] = float(frame["target"].astype(int).mean())
    else:
        summary["target_mean"] = float(frame["target"].astype(float).mean())
        summary["target_std"] = float(frame["target"].astype(float).std(ddof=0))
    return summary
