"""Train the learned backend router.

Pipeline: load the JSONL dataset, embed each prompt with the local
nomic-embed-text model, standardize features, train an L2-regularized softmax
regression with class weights (to counter class imbalance), evaluate on a
held-out split and the golden dogfood cases, and write a JSON weights file the
pure-Python LearnedRouter loads at inference.

numpy is used only here (training time), behind the optional ``[router]``
extra. Inference needs no numpy.

Usage:
    switchboard train-router \
        --dataset router_dataset.jsonl \
        --output config/router_weights.json
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from switchboard.app.services.learned_router import ROUTE_TYPES, RouterWeights
from switchboard.training.router_dataset import (
    RouterExample,
    golden_examples,
)


@dataclass
class TrainingReport:
    total_examples: int
    train_size: int
    holdout_size: int
    holdout_accuracy: float
    golden_accuracy: float
    per_class_accuracy: dict[str, float]
    confusions: list[tuple[str, str, str]]  # (prompt_preview, expected, predicted)


def load_jsonl(path: str | Path) -> list[RouterExample]:
    examples: list[RouterExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        examples.append(
            RouterExample(
                prompt=row["prompt"],
                label=row["label"],
                source=row.get("source", "template"),
            )
        )
    return examples


def _ollama_embed_factory(base_url: str, model: str) -> Callable[[str], list[float]]:
    from switchboard.app.services.semantic_memory import OllamaEmbeddingClient

    return OllamaEmbeddingClient(base_url=base_url, model=model).embed_classification


# Relative trust per example source. Hand-labeled and user-corrected examples
# dominate; bulk-imported public data informs without overpowering them.
DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "golden": 2.0,
    "feedback": 3.0,
    "template": 1.0,
    "external": 0.4,
}


def _source_weight(source: str, weights: dict[str, float]) -> float:
    if source.startswith("external"):
        return weights.get("external", 0.4)
    return weights.get(source, 1.0)


def train(
    examples: Sequence[RouterExample],
    *,
    embed: Callable[[str], list[float]],
    embedding_model: str = "nomic-embed-text",
    classes: tuple[str, ...] = ROUTE_TYPES,
    epochs: int = 300,
    learning_rate: float = 0.5,
    l2: float = 1e-3,
    holdout_fraction: float = 0.15,
    seed: int = 42,
    source_weights: dict[str, float] | None = None,
    golden: Sequence[RouterExample] | None = None,
) -> tuple[RouterWeights, TrainingReport]:
    import numpy as np
    from numpy.typing import NDArray

    rng = np.random.default_rng(seed)
    class_index = {name: i for i, name in enumerate(classes)}

    usable = [ex for ex in examples if ex.label in class_index]
    vectors = np.array([embed(ex.prompt) for ex in usable], dtype=np.float64)
    labels = np.array([class_index[ex.label] for ex in usable], dtype=np.int64)
    prompts = [ex.prompt for ex in usable]

    # Standardize features.
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    std[std == 0] = 1.0
    standardized = (vectors - mean) / std

    # Stratified-ish split: shuffle then slice.
    order = rng.permutation(len(usable))
    holdout_count = max(1, int(len(usable) * holdout_fraction))
    holdout_idx = set(order[:holdout_count].tolist())
    train_mask = np.array([i not in holdout_idx for i in range(len(usable))])

    x_train = standardized[train_mask]
    y_train = labels[train_mask]
    x_hold = standardized[~train_mask]
    y_hold = labels[~train_mask]

    num_classes = len(classes)
    dim = standardized.shape[1]

    # Inverse-frequency class weights to counter imbalance, scaled by
    # per-source trust (golden/feedback > template > external).
    weights_by_source = source_weights or DEFAULT_SOURCE_WEIGHTS
    counts: NDArray[np.float64] = np.bincount(y_train, minlength=num_classes).astype(
        np.float64
    )
    counts[counts == 0] = 1.0
    class_weight = counts.sum() / (num_classes * counts)
    source_factor = np.array(
        [_source_weight(ex.source, weights_by_source) for ex in usable],
        dtype=np.float64,
    )[train_mask]
    sample_weight = class_weight[y_train] * source_factor

    weights = np.zeros((num_classes, dim), dtype=np.float64)
    bias: NDArray[np.float64] = np.zeros(num_classes, dtype=np.float64)
    one_hot = np.eye(num_classes)[y_train]

    for _ in range(epochs):
        logits = x_train @ weights.T + bias
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        error = (probs - one_hot) * sample_weight[:, None]
        grad_w = error.T @ x_train / len(x_train) + l2 * weights
        grad_b = error.mean(axis=0)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    def predict(matrix: np.ndarray) -> np.ndarray:
        logits = matrix @ weights.T + bias
        return logits.argmax(axis=1)

    holdout_pred = predict(x_hold) if len(x_hold) else np.array([], dtype=np.int64)
    holdout_accuracy = (
        float((holdout_pred == y_hold).mean()) if len(x_hold) else 1.0
    )

    per_class: dict[str, float] = {}
    for name, idx in class_index.items():
        mask = y_hold == idx
        if mask.any():
            per_class[name] = float((holdout_pred[mask] == idx).mean())

    confusions: list[tuple[str, str, str]] = []
    hold_positions = [i for i in range(len(usable)) if not train_mask[i]]
    for local_i, global_i in enumerate(hold_positions):
        if local_i < len(holdout_pred) and holdout_pred[local_i] != y_hold[local_i]:
            confusions.append(
                (
                    prompts[global_i][:60],
                    classes[y_hold[local_i]],
                    classes[holdout_pred[local_i]],
                )
            )

    router_weights = RouterWeights(
        classes=classes,
        embedding_model=embedding_model,
        dim=dim,
        mean=mean.tolist(),
        std=std.tolist(),
        weights=weights.tolist(),
        bias=bias.tolist(),
        metadata={
            "trained_examples": len(usable),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
        },
    )

    # Golden accuracy: how many hand-labeled dogfood cases the model gets right.
    golden = list(golden) if golden is not None else golden_examples()
    golden_vectors = (
        np.array([embed(ex.prompt) for ex in golden], dtype=np.float64) - mean
    ) / std
    golden_pred = predict(golden_vectors)
    golden_correct = sum(
        1
        for i, ex in enumerate(golden)
        if ex.label in class_index and golden_pred[i] == class_index[ex.label]
    )
    golden_accuracy = golden_correct / len(golden) if golden else 1.0

    report = TrainingReport(
        total_examples=len(usable),
        train_size=int(train_mask.sum()),
        holdout_size=int((~train_mask).sum()),
        holdout_accuracy=round(holdout_accuracy, 4),
        golden_accuracy=round(golden_accuracy, 4),
        per_class_accuracy={k: round(v, 4) for k, v in per_class.items()},
        confusions=confusions,
    )
    return router_weights, report


def train_from_files(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    base_url: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text",
    embed: Callable[[str], list[float]] | None = None,
) -> TrainingReport:
    examples = load_jsonl(dataset_path)
    embed_fn = embed or _ollama_embed_factory(base_url, embedding_model)
    weights, report = train(examples, embed=embed_fn, embedding_model=embedding_model)
    # Stamp golden accuracy so the feedback-retrain gate has a real baseline
    # to defend from the very first retrain.
    weights.metadata["golden_accuracy"] = report.golden_accuracy
    Path(output_path).write_text(
        json.dumps(weights.to_dict(), indent=2), encoding="utf-8"
    )
    return report


def report_to_text(report: TrainingReport, output_path: str) -> str:
    lines = [
        "Learned router training report",
        "------------------------------",
        f"Examples: {report.total_examples} "
        f"(train {report.train_size}, holdout {report.holdout_size})",
        f"Hold-out accuracy: {report.holdout_accuracy:.1%}",
        f"Golden dogfood accuracy: {report.golden_accuracy:.1%}",
        "Per-class hold-out accuracy:",
    ]
    for name, accuracy in sorted(report.per_class_accuracy.items()):
        lines.append(f"  - {name:<10} {accuracy:.1%}")
    if report.confusions:
        lines.append("Hold-out misclassifications:")
        for preview, expected, predicted in report.confusions[:12]:
            lines.append(f"  - {expected} -> {predicted}: {preview!r}")
    lines.append(f"Wrote weights: {output_path}")
    return "\n".join(lines)
