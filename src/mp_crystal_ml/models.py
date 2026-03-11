from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from alignn.models.alignn import ALIGNN, ALIGNNConfig
import dgl
from dgl.dataloading import GraphDataLoader
from jarvis.core.atoms import pmg_to_atoms
from alignn.graphs import Graph
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataset, collate_fn_graph
from matgl.models._m3gnet import M3GNet
from pymatgen.core import Structure
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool

from mp_crystal_ml.config import TrainingRuntimeConfig
from mp_crystal_ml.data import records_to_structures


@dataclass
class ModelOutcome:
    model_name: str
    task_type: str
    metrics: dict[str, float | None]
    predictions: list[float]
    targets: list[float]
    probabilities: list[float] | None
    history: list[dict[str, float]]
    model_path: str


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dgl_device() -> torch.device:
    return _device()


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": None if len(y_true) < 2 else float(r2_score(y_true, y_pred)),
    }


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": None,
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


class _GaussianDistance:
    def __init__(self, centers: np.ndarray, width: float):
        self.centers = centers
        self.width = width

    def __call__(self, distances: np.ndarray) -> np.ndarray:
        return np.exp(-((distances[:, None] - self.centers[None, :]) ** 2) / (self.width**2))


class _CrystalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, Any]], cutoff: float = 6.0, max_neighbors: int = 12):
        self.records = records
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.distance_expansion = _GaussianDistance(
            centers=np.linspace(0.0, cutoff, 32, dtype=float),
            width=0.4,
        )

    def _build_graph(self, record: dict[str, Any]) -> Data:
        structure = Structure.from_dict(record["structure"])
        src: list[int] = []
        dst: list[int] = []
        distances: list[float] = []
        for index, neighbors in enumerate(structure.get_all_neighbors(self.cutoff, include_index=True)):
            ordered = sorted(neighbors, key=lambda item: item.nn_distance)[: self.max_neighbors]
            for neighbor in ordered:
                src.append(index)
                dst.append(int(neighbor.index))
                distances.append(float(neighbor.nn_distance))
        if not src:
            raise RuntimeError(f"Failed to construct graph for {record['sample_id']}")
        return Data(
            z=torch.tensor([site.specie.Z for site in structure], dtype=torch.long),
            num_nodes=len(structure),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_attr=torch.tensor(self.distance_expansion(np.asarray(distances)), dtype=torch.float32),
            y=torch.tensor([float(record["target"])], dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Data:
        return self._build_graph(self.records[index])


class _CGCNN(nn.Module):
    def __init__(self, hidden_dim: int = 64, edge_dim: int = 32, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_dim)
        self.convs = nn.ModuleList(
            [CGConv(channels=hidden_dim, dim=edge_dim, batch_norm=True) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Data) -> torch.Tensor:
        x = self.embedding(batch.z)
        for conv, norm in zip(self.convs, self.norms, strict=False):
            x = conv(x, batch.edge_index, batch.edge_attr)
            x = norm(x)
            x = F.softplus(x)
        pooled = global_mean_pool(x, batch.batch)
        return self.head(pooled).squeeze(-1)


class _ALIGNNGraphDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def _build_sample(self, record: dict[str, Any]):
        structure = Structure.from_dict(record["structure"])
        atoms = pmg_to_atoms(structure)
        graph, line_graph = Graph.atom_dgl_multigraph(
            atoms=atoms,
            neighbor_strategy="k-nearest",
            cutoff=8.0,
            max_neighbors=12,
            atom_features="cgcnn",
            use_canonize=True,
        )
        lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float32)
        target = torch.tensor(float(record["target"]), dtype=torch.float32)
        return graph, line_graph, lattice, target

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        return self._build_sample(self.records[index])


def _collate_alignn_graphs(batch):
    graphs, line_graphs, lattices, targets = zip(*batch, strict=False)
    return (
        dgl.batch(list(graphs)),
        dgl.batch(list(line_graphs)),
        torch.stack(list(lattices)),
        torch.stack(list(targets)),
    )


def _train_cgcnn(
    task_type: str,
    splits: dict[str, list[dict[str, Any]]],
    model_dir: Path,
    training_config: TrainingRuntimeConfig,
) -> ModelOutcome:
    model_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = _CrystalGraphDataset(splits["train"])
    val_dataset = _CrystalGraphDataset(splits["val"])
    test_dataset = _CrystalGraphDataset(splits["test"])
    train_loader = DataLoader(train_dataset, batch_size=training_config.cgcnn_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.cgcnn_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=training_config.cgcnn_batch_size, shuffle=False)

    device = _device()
    model = _CGCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.cgcnn_learning_rate, weight_decay=1e-5)

    train_targets = np.asarray([float(record["target"]) for record in splits["train"]], dtype=float)
    target_mean = float(train_targets.mean()) if task_type == "regression" else 0.0
    target_std = float(train_targets.std(ddof=0)) if task_type == "regression" else 1.0
    if target_std == 0:
        target_std = 1.0

    best_score = np.inf if task_type == "regression" else -np.inf
    best_path = model_dir / "cgcnn.pt"
    patience = training_config.cgcnn_patience
    patience_left = patience
    history: list[dict[str, float]] = []

    for epoch in range(1, training_config.cgcnn_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            target = batch.y.view(-1)
            if task_type == "regression":
                normalized = (target - target_mean) / target_std
                loss = F.mse_loss(output, normalized)
            else:
                loss = F.binary_cross_entropy_with_logits(output, target)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        val_target, val_pred, val_prob = _predict_cgcnn(
            model=model,
            loader=val_loader,
            device=device,
            task_type=task_type,
            target_mean=target_mean,
            target_std=target_std,
        )
        if task_type == "regression":
            score = _regression_metrics(val_target, val_pred)["rmse"]
        else:
            score = _classification_metrics(val_target.astype(int), val_prob)["roc_auc"]
            if score is None:
                score = _classification_metrics(val_target.astype(int), val_prob)["accuracy"]

        history.append({"epoch": epoch, "train_loss": train_loss / max(len(train_loader), 1), "val_score": float(score)})

        improved = score < best_score if task_type == "regression" else score > best_score
        if improved:
            best_score = score
            patience_left = patience
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])
    test_target, test_pred, test_prob = _predict_cgcnn(
        model=model,
        loader=test_loader,
        device=device,
        task_type=task_type,
        target_mean=target_mean,
        target_std=target_std,
    )
    metrics = (
        _regression_metrics(test_target, test_pred)
        if task_type == "regression"
        else _classification_metrics(test_target.astype(int), test_prob)
    )
    (model_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return ModelOutcome(
        model_name="cgcnn",
        task_type=task_type,
        metrics=metrics,
        predictions=test_pred.tolist(),
        targets=test_target.tolist(),
        probabilities=None if task_type == "regression" else test_prob.tolist(),
        history=history,
        model_path=str(best_path),
    )


def _predict_cgcnn(
    model: _CGCNN,
    loader: DataLoader,
    device: torch.device,
    task_type: str,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets: list[float] = []
    predictions: list[float] = []
    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            target = batch.y.view(-1).detach().cpu().numpy()
            if task_type == "regression":
                pred = (output * target_std + target_mean).detach().cpu().numpy()
                prob = pred
            else:
                prob_tensor = torch.sigmoid(output)
                pred = (prob_tensor >= 0.5).long().detach().cpu().numpy()
                prob = prob_tensor.detach().cpu().numpy()
            targets.extend(target.tolist())
            predictions.extend(np.asarray(pred).reshape(-1).tolist())
            probabilities.extend(np.asarray(prob).reshape(-1).tolist())
    return np.asarray(targets), np.asarray(predictions), np.asarray(probabilities)


def _train_alignn(
    task_type: str,
    splits: dict[str, list[dict[str, Any]]],
    model_dir: Path,
    random_state: int,
    training_config: TrainingRuntimeConfig,
) -> ModelOutcome:
    model_dir.mkdir(parents=True, exist_ok=True)
    del random_state
    train_dataset = _ALIGNNGraphDataset(splits["train"])
    val_dataset = _ALIGNNGraphDataset(splits["val"])
    test_dataset = _ALIGNNGraphDataset(splits["test"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.alignn_batch_size,
        shuffle=True,
        collate_fn=_collate_alignn_graphs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config.alignn_batch_size,
        shuffle=False,
        collate_fn=_collate_alignn_graphs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_config.alignn_batch_size,
        shuffle=False,
        collate_fn=_collate_alignn_graphs,
    )

    device = _dgl_device()
    alignn_config = ALIGNNConfig(
        name="alignn",
        alignn_layers=2,
        gcn_layers=2,
        embedding_features=64,
        hidden_features=128,
        output_features=1,
        classification=task_type == "classification",
        num_classes=2,
    )
    model = ALIGNN(alignn_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.alignn_learning_rate, weight_decay=1e-5)

    train_targets = np.asarray([float(record["target"]) for record in splits["train"]], dtype=float)
    target_mean = float(train_targets.mean()) if task_type == "regression" else 0.0
    target_std = float(train_targets.std(ddof=0)) if task_type == "regression" else 1.0
    if target_std == 0:
        target_std = 1.0

    best_score = np.inf if task_type == "regression" else -np.inf
    best_path = model_dir / "alignn.pt"
    patience = training_config.alignn_patience
    patience_left = patience
    history: list[dict[str, float]] = []

    for epoch in range(1, training_config.alignn_epochs + 1):
        model.train()
        train_loss = 0.0
        for graph, line_graph, lattice, target in train_loader:
            graph = graph.to(device)
            line_graph = line_graph.to(device)
            lattice = lattice.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model([graph, line_graph, lattice])
            if task_type == "classification":
                if output.ndim == 1:
                    output = output.unsqueeze(0)
                loss = F.nll_loss(output, target.view(-1).long())
            else:
                output = output.view(-1)
                normalized = (target.view(-1) - target_mean) / target_std
                loss = F.mse_loss(output, normalized)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        val_target, val_pred, val_prob = _predict_alignn(
            model=model,
            loader=val_loader,
            device=device,
            task_type=task_type,
            target_mean=target_mean,
            target_std=target_std,
        )
        if task_type == "regression":
            score = _regression_metrics(val_target, val_pred)["rmse"]
        else:
            score = _classification_metrics(val_target.astype(int), val_prob)["roc_auc"]
            if score is None:
                score = _classification_metrics(val_target.astype(int), val_prob)["accuracy"]

        history.append({"epoch": epoch, "train_loss": train_loss / max(len(train_loader), 1), "val_score": float(score)})

        improved = score < best_score if task_type == "regression" else score > best_score
        if improved:
            best_score = score
            patience_left = patience
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])
    test_target, test_pred, test_prob = _predict_alignn(
        model=model,
        loader=test_loader,
        device=device,
        task_type=task_type,
        target_mean=target_mean,
        target_std=target_std,
    )
    metrics = (
        _regression_metrics(test_target, test_pred)
        if task_type == "regression"
        else _classification_metrics(test_target.astype(int), test_prob)
    )
    (model_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return ModelOutcome(
        model_name="alignn",
        task_type=task_type,
        metrics=metrics,
        predictions=test_pred.tolist(),
        targets=test_target.tolist(),
        probabilities=None if task_type == "regression" else test_prob.tolist(),
        history=history,
        model_path=str(best_path),
    )


def _predict_alignn(
    model: ALIGNN,
    loader,
    device: torch.device,
    task_type: str,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets: list[float] = []
    predictions: list[float] = []
    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for graph, line_graph, lattice, target in loader:
            output = model([graph.to(device), line_graph.to(device), lattice.to(device)])
            target_np = target.cpu().numpy().reshape(-1)
            if task_type == "classification":
                if output.ndim == 1:
                    output = output.unsqueeze(0)
                prob = torch.exp(output)[:, 1].detach().cpu().numpy()
                pred = np.argmax(output.detach().cpu().numpy(), axis=1)
            else:
                pred = (output.view(-1).detach().cpu().numpy() * target_std) + target_mean
                prob = pred
            targets.extend(target_np.tolist())
            predictions.extend(np.asarray(pred).reshape(-1).tolist())
            probabilities.extend(np.asarray(prob).reshape(-1).tolist())
    return np.asarray(targets), np.asarray(predictions), np.asarray(probabilities)


def _prepare_m3gnet_graph_batch(graph, lattice):
    graphs = dgl.unbatch(graph)
    if lattice.ndim == 2:
        lattices = [lattice]
    else:
        lattices = [lat for lat in lattice]
    for subgraph, lat in zip(graphs, lattices, strict=False):
        subgraph.ndata["pos"] = torch.matmul(subgraph.ndata["frac_coords"], lat)
        subgraph.edata["pbc_offshift"] = torch.matmul(subgraph.edata["pbc_offset"], lat)
    return dgl.batch(graphs)


def _train_m3gnet(
    task_type: str,
    splits: dict[str, list[dict[str, Any]]],
    model_dir: Path,
    training_config: TrainingRuntimeConfig,
) -> ModelOutcome:
    model_dir.mkdir(parents=True, exist_ok=True)
    converter = Structure2Graph(element_types=DEFAULT_ELEMENTS, cutoff=5.0)
    train_structures = records_to_structures(splits["train"])
    val_structures = records_to_structures(splits["val"])
    test_structures = records_to_structures(splits["test"])
    train_targets = np.asarray([float(record["target"]) for record in splits["train"]], dtype=float)
    val_targets = np.asarray([float(record["target"]) for record in splits["val"]], dtype=float)
    test_targets = np.asarray([float(record["target"]) for record in splits["test"]], dtype=float)

    def build_dataset(name: str, structures: list[Structure], targets: np.ndarray) -> MGLDataset:
        cache_dir = model_dir / name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return MGLDataset(
            include_line_graph=True,
            converter=converter,
            threebody_cutoff=4.0,
            structures=structures,
            labels={"target": targets.tolist()},
            directory_name=f"{name}_dataset",
            raw_dir=str(cache_dir),
            save_dir=str(cache_dir),
            clear_processed=True,
            save_cache=False,
        )

    train_dataset = build_dataset("train", train_structures, train_targets)
    val_dataset = build_dataset("val", val_structures, val_targets)
    test_dataset = build_dataset("test", test_structures, test_targets)
    collate = partial(collate_fn_graph, include_line_graph=True)
    train_loader = GraphDataLoader(train_dataset, batch_size=training_config.m3gnet_batch_size, shuffle=True, collate_fn=collate)
    val_loader = GraphDataLoader(val_dataset, batch_size=training_config.m3gnet_batch_size, shuffle=False, collate_fn=collate)
    test_loader = GraphDataLoader(test_dataset, batch_size=training_config.m3gnet_batch_size, shuffle=False, collate_fn=collate)

    device = _dgl_device()
    model = M3GNet(
        element_types=DEFAULT_ELEMENTS,
        nblocks=3,
        units=64,
        cutoff=5.0,
        threebody_cutoff=4.0,
        task_type=task_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.m3gnet_learning_rate, weight_decay=1e-5)
    target_mean = float(train_targets.mean()) if task_type == "regression" else 0.0
    target_std = float(train_targets.std(ddof=0)) if task_type == "regression" else 1.0
    if target_std == 0:
        target_std = 1.0

    best_score = np.inf if task_type == "regression" else -np.inf
    best_path = model_dir / "m3gnet.pt"
    patience = training_config.m3gnet_patience
    patience_left = patience
    history: list[dict[str, float]] = []

    for epoch in range(1, training_config.m3gnet_epochs + 1):
        model.train()
        train_loss = 0.0
        for graph, _lattice, line_graph, state_attr, labels in train_loader:
            optimizer.zero_grad()
            graph = _prepare_m3gnet_graph_batch(graph, _lattice).to(device)
            output = model(graph, state_attr=state_attr.to(device), l_g=line_graph.to(device))
            labels = labels.to(device).view(-1)
            if task_type == "regression":
                normalized = (labels - target_mean) / target_std
                loss = F.mse_loss(output.view(-1), normalized)
            else:
                loss = F.binary_cross_entropy(output.view(-1), labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        val_target, val_pred, val_prob = _predict_m3gnet(
            model=model,
            loader=val_loader,
            device=device,
            task_type=task_type,
            target_mean=target_mean,
            target_std=target_std,
        )
        if task_type == "regression":
            score = _regression_metrics(val_target, val_pred)["rmse"]
        else:
            score = _classification_metrics(val_target.astype(int), val_prob)["roc_auc"]
            if score is None:
                score = _classification_metrics(val_target.astype(int), val_prob)["accuracy"]

        history.append({"epoch": epoch, "train_loss": train_loss / max(len(train_loader), 1), "val_score": float(score)})
        improved = score < best_score if task_type == "regression" else score > best_score
        if improved:
            best_score = score
            patience_left = patience
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])
    test_target, test_pred, test_prob = _predict_m3gnet(
        model=model,
        loader=test_loader,
        device=device,
        task_type=task_type,
        target_mean=target_mean,
        target_std=target_std,
    )
    metrics = (
        _regression_metrics(test_target, test_pred)
        if task_type == "regression"
        else _classification_metrics(test_target.astype(int), test_prob)
    )
    (model_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return ModelOutcome(
        model_name="m3gnet",
        task_type=task_type,
        metrics=metrics,
        predictions=test_pred.tolist(),
        targets=test_target.tolist(),
        probabilities=None if task_type == "regression" else test_prob.tolist(),
        history=history,
        model_path=str(best_path),
    )


def _predict_m3gnet(
    model: M3GNet,
    loader: GraphDataLoader,
    device: torch.device,
    task_type: str,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets: list[float] = []
    predictions: list[float] = []
    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for graph, _lattice, line_graph, state_attr, labels in loader:
            graph = _prepare_m3gnet_graph_batch(graph, _lattice).to(device)
            output = model(graph, state_attr=state_attr.to(device), l_g=line_graph.to(device))
            label_array = labels.view(-1).detach().cpu().numpy()
            if task_type == "regression":
                pred_array = (output.view(-1) * target_std + target_mean).detach().cpu().numpy()
                prob_array = pred_array
            else:
                prob_array = output.view(-1).detach().cpu().numpy()
                pred_array = (prob_array >= 0.5).astype(int)
            targets.extend(label_array.tolist())
            predictions.extend(pred_array.tolist())
            probabilities.extend(prob_array.tolist())
    return np.asarray(targets), np.asarray(predictions), np.asarray(probabilities)


def train_model_suite(
    task_name: str,
    task_type: str,
    splits: dict[str, list[dict[str, Any]]],
    model_root: Path,
    random_state: int,
    training_config: TrainingRuntimeConfig,
) -> dict[str, ModelOutcome]:
    outcomes = {
        "cgcnn": _train_cgcnn(
            task_type=task_type,
            splits=splits,
            model_dir=model_root / "cgcnn",
            training_config=training_config,
        ),
        "alignn": _train_alignn(
            task_type=task_type,
            splits=splits,
            model_dir=model_root / "alignn",
            random_state=random_state,
            training_config=training_config,
        ),
        "m3gnet": _train_m3gnet(
            task_type=task_type,
            splits=splits,
            model_dir=model_root / "m3gnet",
            training_config=training_config,
        ),
    }
    return outcomes
