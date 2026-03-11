from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mp_crystal_ml.config import MATBENCH_TASKS, MODEL_NAMES, ProjectPaths
from mp_crystal_ml.data import (
    load_task_records,
    split_records,
    summarize_task_records,
    write_split_manifest,
)
from mp_crystal_ml.models import train_model_suite
from mp_crystal_ml.reporting import (
    save_confusion_matrix,
    save_metric_bar_chart,
    save_regression_plot,
    save_roc_curve,
    save_training_curve,
    write_metrics_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matbench crystal graph benchmark pipeline")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of structures to subsample per Matbench task for local retraining",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Discard cached sampled Matbench subsets and fetch them again",
    )
    return parser.parse_args()


def _ensure_directories(paths: ProjectPaths) -> None:
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_json.parent.mkdir(parents=True, exist_ok=True)


def _sorted_leaderboard(task_name: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    meta = MATBENCH_TASKS[task_name]
    metric = meta["sort_metric"]

    def sort_value(row: dict[str, object]) -> float:
        value = row.get(metric)
        if value is not None:
            return float(value)
        if meta["task_type"] == "classification":
            for fallback_metric in ("accuracy", "f1"):
                fallback_value = row.get(fallback_metric)
                if fallback_value is not None:
                    return float(fallback_value)
            return float("-inf")
        for fallback_metric in ("rmse", "mae"):
            fallback_value = row.get(fallback_metric)
            if fallback_value is not None:
                return float(fallback_value)
        return float("inf")

    return sorted(rows, key=sort_value, reverse=not meta["sort_ascending"])


def run_pipeline(sample_size: int, random_state: int, force_fetch: bool) -> dict[str, object]:
    paths = ProjectPaths()
    _ensure_directories(paths)

    tasks_payload: dict[str, object] = {}
    for task_name, task_meta in MATBENCH_TASKS.items():
        records = load_task_records(
            paths=paths,
            task_name=task_name,
            sample_size=sample_size,
            random_state=random_state,
            force_fetch=force_fetch,
        )
        splits = split_records(records, task_type=task_meta["task_type"], random_state=random_state)
        write_split_manifest(paths, task_name, splits)
        outcomes = train_model_suite(
            task_name=task_name,
            task_type=task_meta["task_type"],
            splits=splits,
            model_root=paths.models_dir / task_name,
            random_state=random_state,
        )

        leaderboard = []
        for model_name in MODEL_NAMES:
            outcome = outcomes[model_name]
            leaderboard.append({"model": model_name, **outcome.metrics})

            if task_meta["task_type"] == "regression":
                save_regression_plot(
                    y_true=outcome.targets,
                    predictions=outcome.predictions,
                    output_path=paths.figure_path(f"{task_name}_{model_name}_parity.png"),
                    title=f"{task_name} {model_name} parity",
                )
            else:
                save_confusion_matrix(
                    y_true=outcome.targets,
                    predictions=outcome.predictions,
                    output_path=paths.figure_path(f"{task_name}_{model_name}_confusion.png"),
                    title=f"{task_name} {model_name} confusion matrix",
                )
                save_roc_curve(
                    y_true=outcome.targets,
                    probabilities=outcome.probabilities,
                    output_path=paths.figure_path(f"{task_name}_{model_name}_roc.png"),
                    title=f"{task_name} {model_name} ROC",
                )

            save_training_curve(
                history=outcome.history,
                output_path=paths.figure_path(f"{task_name}_{model_name}_history.png"),
                title=f"{task_name} {model_name} training history",
            )

        leaderboard = _sorted_leaderboard(task_name, leaderboard)
        best_model = leaderboard[0]["model"]
        primary_metric = task_meta["sort_metric"]
        save_metric_bar_chart(
            leaderboard=leaderboard,
            metric_name=primary_metric,
            output_path=paths.figure_path(f"{task_name}_{primary_metric}_comparison.png"),
            title=f"{task_name} {primary_metric} comparison",
        )

        tasks_payload[task_name] = {
            "display_name": task_meta["display_name"],
            "task_type": task_meta["task_type"],
            "target_display": task_meta["target_display"],
            "unit": task_meta["unit"],
            "dataset_summary": summarize_task_records(records, task_type=task_meta["task_type"]),
            "split_sizes": {name: len(value) for name, value in splits.items()},
            "leaderboard": leaderboard,
            "best_model": best_model,
            "model_artifacts": {
                model_name: {
                    "metrics": outcomes[model_name].metrics,
                    "model_path": outcomes[model_name].model_path,
                }
                for model_name in MODEL_NAMES
            },
        }

    metrics_payload = {
        "data_source": {
            "provider": "Matbench v0.1 via Matminer / Materials Project",
            "tasks": list(MATBENCH_TASKS.keys()),
            "sample_size_per_task": sample_size,
            "random_state": random_state,
        },
        "tasks": tasks_payload,
    }
    write_metrics_report(metrics_payload, markdown_path=paths.report_md, json_path=paths.metrics_json)
    return metrics_payload


def main() -> None:
    args = parse_args()
    metrics = run_pipeline(
        sample_size=args.sample_size,
        random_state=args.random_state,
        force_fetch=args.force_fetch,
    )
    summary_rows = [
        {
            "task": task_name,
            "best_model": task_payload["best_model"],
            "best_metric": task_payload["leaderboard"][0][MATBENCH_TASKS[task_name]["sort_metric"]],
        }
        for task_name, task_payload in metrics["tasks"].items()
    ]
    print(pd.DataFrame(summary_rows))


if __name__ == "__main__":
    main()
