"""Learned backend router: a tiny embedding classifier over four route types.

Design goals:

- The model replaces *classification*, never *policy*. Privacy enforcement,
  forced-backend selection, tool grounding, availability fallback, and
  follow-up stickiness remain deterministic in SwitchboardCoreService. This
  router only answers "which kind of request is this?".
- Inference is pure Python and dependency-free: a prompt embedding (from the
  local nomic-embed-text model) dotted with a small weight matrix loaded from
  a JSON file. No numpy at inference time, so it runs anywhere the app runs.
- It degrades safely: if the weights file is missing, the embedding model is
  unreachable, or the top-class probability is below a confidence threshold,
  the caller falls back to the deterministic rules.

Route types: ``tool``, ``local``, ``coding``, ``reasoning``. The ``tool``
class means "deterministically answerable" — upstream tool grounding decides
the concrete tool; if none fires, policy treats it as ``local``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROUTE_TYPES = ("tool", "local", "coding", "reasoning")

# Backend a route type maps to when it is the final decision. ``tool`` falls
# back to local because, by the time the router runs, any real tool has
# already grounded the answer upstream.
BACKEND_BY_ROUTE_TYPE = {
    "tool": "ollama",
    "local": "ollama",
    "coding": "codex",
    "reasoning": "claude-code",
}

LOGGER = logging.getLogger(__name__)
_LOGGED_WEIGHT_WARNINGS: set[str] = set()


@dataclass(frozen=True)
class LearnedRouteResult:
    success: bool
    route_type: str = "local"
    backend: str = "ollama"
    confidence: float = 0.0
    probabilities: dict[str, float] | None = None
    latency_ms: int = 0
    model: str = ""
    error: str | None = None


@dataclass(frozen=True)
class RouterWeights:
    """Standardized softmax-regression parameters."""

    classes: tuple[str, ...]
    embedding_model: str
    dim: int
    mean: list[float]
    std: list[float]
    weights: list[list[float]]  # [num_classes][dim]
    bias: list[float]  # [num_classes]
    metadata: dict[str, object]

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        expected_embedding_model: str | None = None,
    ) -> RouterWeights | None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = cls(
            classes=tuple(data["classes"]),
            embedding_model=data.get("embedding_model", "nomic-embed-text"),
            dim=int(data["dim"]),
            mean=[float(x) for x in data["mean"]],
            std=[float(x) for x in data["std"]],
            weights=[[float(x) for x in row] for row in data["weights"]],
            bias=[float(x) for x in data["bias"]],
            metadata=dict(data.get("metadata", {})),
        )
        if expected_embedding_model and not _same_embedding_model(
            weights.embedding_model,
            expected_embedding_model,
        ):
            _log_weight_warning_once(
                str(path),
                (
                    f"weights were trained with {weights.embedding_model!r}, "
                    f"but configured embedder is {expected_embedding_model!r}; "
                    "falling back to deterministic routing"
                ),
            )
            return None
        if not weights.shape_is_consistent():
            _log_weight_warning_once(
                str(path),
                "weights shape does not match recorded embedding dimension; "
                "falling back to deterministic routing",
            )
            return None
        return weights

    def to_dict(self) -> dict[str, object]:
        return {
            "classes": list(self.classes),
            "embedding_model": self.embedding_model,
            "dim": self.dim,
            "mean": self.mean,
            "std": self.std,
            "weights": self.weights,
            "bias": self.bias,
            "metadata": self.metadata,
        }

    def shape_is_consistent(self) -> bool:
        if len(self.mean) != self.dim or len(self.std) != self.dim:
            return False
        if any(len(row) != self.dim for row in self.weights):
            return False
        return len(self.weights) == len(self.classes) and len(self.bias) == len(self.classes)


def _same_embedding_model(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower()


def _log_weight_warning_once(path: str, message: str) -> None:
    key = f"{path}:{message}"
    if key in _LOGGED_WEIGHT_WARNINGS:
        return
    _LOGGED_WEIGHT_WARNINGS.add(key)
    LOGGER.warning("%s: %s", path, message)


def softmax(scores: list[float]) -> list[float]:
    highest = max(scores)
    exps = [math.exp(s - highest) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def predict_probabilities(weights: RouterWeights, vector: list[float]) -> dict[str, float]:
    """Pure-Python standardized softmax-regression forward pass, shared by the
    backend router and the tool dispatcher."""
    standardized = [
        (value - weights.mean[i]) / (weights.std[i] or 1.0)
        for i, value in enumerate(vector)
    ]
    scores = []
    for class_index in range(len(weights.classes)):
        row = weights.weights[class_index]
        score = weights.bias[class_index]
        for i, feature in enumerate(standardized):
            score += row[i] * feature
        scores.append(score)
    probs = softmax(scores)
    return dict(zip(weights.classes, probs, strict=True))


class LearnedRouter:
    def __init__(
        self,
        *,
        weights: RouterWeights,
        embed: Callable[[str], list[float]] | None = None,
        base_url: str = "http://localhost:11434",
        min_confidence: float = 0.55,
        max_prompt_chars: int = 2000,
    ) -> None:
        self.weights = weights
        self.min_confidence = min_confidence
        self.max_prompt_chars = max_prompt_chars
        if embed is not None:
            self._embed = embed
        else:
            # Imported lazily so inference has no hard dependency surface.
            from switchboard.app.services.semantic_memory import (
                OllamaEmbeddingClient,
            )

            self._embed = OllamaEmbeddingClient(
                base_url=base_url,
                model=weights.embedding_model,
            ).embed_classification

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        embed: Callable[[str], list[float]] | None = None,
        base_url: str = "http://localhost:11434",
        min_confidence: float = 0.55,
        expected_embedding_model: str | None = None,
    ) -> LearnedRouter | None:
        weights_path = Path(path)
        if not weights_path.exists():
            return None
        weights = RouterWeights.from_file(
            weights_path,
            expected_embedding_model=expected_embedding_model,
        )
        if weights is None:
            return None
        return cls(
            weights=weights,
            embed=embed,
            base_url=base_url,
            min_confidence=min_confidence,
        )

    def classify(self, prompt: str) -> LearnedRouteResult:
        started = time.perf_counter()
        try:
            vector = self._embed(prompt[: self.max_prompt_chars])
        except Exception as exc:
            return LearnedRouteResult(
                success=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                model=self.weights.embedding_model,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        if len(vector) != self.weights.dim:
            return LearnedRouteResult(
                success=False,
                latency_ms=latency_ms,
                model=self.weights.embedding_model,
                error=(
                    f"embedding dim {len(vector)} does not match trained dim "
                    f"{self.weights.dim}"
                ),
            )
        probabilities = self._predict(vector)
        ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        top_class, top_prob = ranked[0]
        if top_prob < self.min_confidence:
            return LearnedRouteResult(
                success=False,
                route_type=top_class,
                confidence=top_prob,
                probabilities=probabilities,
                latency_ms=latency_ms,
                model=self.weights.embedding_model,
                error=f"low confidence ({top_prob:.2f} < {self.min_confidence:.2f})",
            )
        return LearnedRouteResult(
            success=True,
            route_type=top_class,
            backend=BACKEND_BY_ROUTE_TYPE.get(top_class, "ollama"),
            confidence=top_prob,
            probabilities=probabilities,
            latency_ms=latency_ms,
            model=self.weights.embedding_model,
        )

    def _predict(self, vector: list[float]) -> dict[str, float]:
        return predict_probabilities(self.weights, vector)
