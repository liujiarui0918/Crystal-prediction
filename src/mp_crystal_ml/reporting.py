from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


sns.set_theme(style="whitegrid")


def _to_markdown_table(records: list[dict[str, object]]) -> str:
    return pd.DataFrame(records).round(4).to_markdown(index=False)


def save_regression_plot(y_true, predictions, output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(y_true, predictions, alpha=0.65, s=18, edgecolor="none")
    bounds = [min(np.min(y_true), np.min(predictions)), max(np.max(y_true), np.max(predictions))]
    axis.plot(bounds, bounds, linestyle="--", color="black", linewidth=1.2)
    axis.set_xlabel("True")
    axis.set_ylabel("Predicted")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def save_metric_bar_chart(
    leaderboard: list[dict[str, object]],
    metric_name: str,
    output_path: Path,
    title: str,
) -> None:
    frame = pd.DataFrame(leaderboard)
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.barplot(data=frame, x="model", y=metric_name, hue="model", legend=False, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel(metric_name.upper())
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def save_confusion_matrix(y_true, predictions, output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    display = ConfusionMatrixDisplay.from_predictions(y_true, predictions, ax=axis, colorbar=False)
    display.ax_.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def save_roc_curve(y_true, probabilities, output_path: Path, title: str) -> None:
    if len(np.unique(y_true)) < 2:
        return
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=axis)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def save_training_curve(history: list[dict[str, float]], output_path: Path, title: str) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    figure, axis = plt.subplots(figsize=(6.5, 4))
    if "train_loss" in frame:
        axis.plot(frame["epoch"], frame["train_loss"], label="train_loss")
    if "val_score" in frame:
        axis.plot(frame["epoch"], frame["val_score"], label="val_score")
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_metrics_report(
    metrics_payload: dict[str, object],
    markdown_path: Path,
    json_path: Path,
) -> None:
    json_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    lines = ["# Crystal Property Benchmark Report", ""]
    lines.extend(
        [
            "## Data Source",
            "",
            "- Source: Matbench v0.1 structure datasets hosted by the Materials Project / Matminer.",
            "- Tasks: matbench_mp_gap, matbench_mp_e_form, matbench_mp_is_metal.",
            "- Models: CGCNN, ALIGNN, M3GNet.",
            "",
            "## Task Summaries",
            "",
        ]
    )

    for task_name, task_payload in metrics_payload["tasks"].items():
        lines.append(f"### {task_name}")
        lines.append("")
        lines.append(f"- Task type: {task_payload['task_type']}")
        lines.append(f"- Samples used: {task_payload['dataset_summary']['num_samples']}")
        lines.append(f"- Mean sites per structure: {task_payload['dataset_summary']['mean_num_sites']:.2f}")
        lines.append(_to_markdown_table(task_payload["leaderboard"]))
        lines.append("")
        lines.append(f"Best model: {task_payload['best_model']}")
        lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
