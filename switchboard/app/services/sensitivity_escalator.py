"""Learned sensitivity escalator: catches private content the keywords miss.

The keyword sensitivity classifier is the most consequential deterministic
check in Switchboard: a miss sends private content to a cloud backend. But
keywords only catch phrasings someone thought of in advance ("depression",
"my salary"). This escalator embeds the prompt and classifies
{sensitive, neutral} to catch the long tail ("I've been crying a lot lately",
"I owe more than I make").

Hard rules — this component can only ADD protection:

- It runs only when the keyword classifier said *not* sensitive. Keyword
  positives are final; the model can never de-escalate them.
- Escalation requires confidence >= the threshold. Anything else — low
  confidence, missing weights, unreachable embedder, dimension mismatch —
  means "no opinion" and the keyword verdict stands.
- It biases toward protection by design: the cost of a false escalation is a
  slightly weaker (local) answer; the cost of a miss is a privacy leak.

Same architecture as the router and tool dispatcher: standardized softmax
regression over local embeddings, pure Python at inference.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from switchboard.app.services.learned_router import (
    DEGENERATE_EMBEDDING_ERROR,
    RouterWeights,
    embedding_is_degenerate,
    predict_probabilities,
)

SENSITIVITY_CLASSES = ("sensitive", "neutral")


@dataclass(frozen=True)
class SensitivityEscalation:
    """``escalate`` is the only field policy may act on; it is True only for
    a confident ``sensitive`` prediction."""

    success: bool
    escalate: bool = False
    confidence: float = 0.0
    probabilities: dict[str, float] | None = None
    latency_ms: int = 0
    model: str = ""
    error: str | None = None


class LearnedSensitivityEscalator:
    def __init__(
        self,
        *,
        weights: RouterWeights,
        embed: Callable[[str], list[float]] | None = None,
        base_url: str = "http://localhost:11434",
        min_confidence: float = 0.7,
        max_prompt_chars: int = 2000,
    ) -> None:
        self.weights = weights
        self.min_confidence = min_confidence
        self.max_prompt_chars = max_prompt_chars
        if embed is not None:
            self._embed = embed
        else:
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
        min_confidence: float = 0.7,
        expected_embedding_model: str | None = None,
    ) -> LearnedSensitivityEscalator | None:
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

    def classify(self, prompt: str) -> SensitivityEscalation:
        started = time.perf_counter()
        try:
            vector = self._embed(prompt[: self.max_prompt_chars])
        except Exception as exc:
            return SensitivityEscalation(
                success=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                model=self.weights.embedding_model,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        if len(vector) != self.weights.dim:
            return SensitivityEscalation(
                success=False,
                latency_ms=latency_ms,
                model=self.weights.embedding_model,
                error=(
                    f"embedding dim {len(vector)} does not match trained dim "
                    f"{self.weights.dim}"
                ),
            )
        if embedding_is_degenerate(vector):
            return SensitivityEscalation(
                success=False,
                escalate=False,
                latency_ms=latency_ms,
                model=self.weights.embedding_model,
                error=f"{DEGENERATE_EMBEDDING_ERROR}: near-zero variance",
            )
        probabilities = predict_probabilities(self.weights, vector)
        sensitive_prob = probabilities.get("sensitive", 0.0)
        return SensitivityEscalation(
            success=True,
            escalate=sensitive_prob >= self.min_confidence,
            confidence=sensitive_prob,
            probabilities=probabilities,
            latency_ms=latency_ms,
            model=self.weights.embedding_model,
        )
