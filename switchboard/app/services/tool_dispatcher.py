"""Learned tool dispatcher: a tiny embedding classifier over tool intents.

The regex CapabilityDetector is precise (~99.6% on CLINC150 non-tool
utterances) but narrow (~47% recall on real tool-shaped phrasings). This
dispatcher recovers the missed recall without adding more if-else patterns:

- It runs only when the regexes found *no* tool capability, so the regex fast
  path keeps its behavior and its precision.
- It never grounds an answer by itself. A prediction only counts if the actual
  tool then verifies it: the calculator must parse the expression, a ticker
  must resolve, and so on. Learned recall, verified precision.
- It answers "which tool might handle this?", never routing policy. Privacy,
  forced backends, and availability remain deterministic upstream/downstream.

Same architecture and degradation story as the LearnedRouter: prompt embedding
(nomic-embed-text) -> standardized softmax regression loaded from JSON, pure
Python at inference. Missing weights, an unreachable embedder, low confidence,
or a ``none`` prediction all mean "stay out of the way".
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from switchboard.app.models.capabilities import Capability
from switchboard.app.services.learned_router import (
    RouterWeights,
    predict_probabilities,
)

# Classifier classes. ``none`` is a first-class label so the model can learn
# what NOT to dispatch (small talk, knowledge questions, opinions).
TOOL_CLASSES = (
    "time",
    "date",
    "calculation",
    "unit_conversion",
    "stock_price",
    "news",
    "weather",
    "none",
)

CAPABILITY_BY_TOOL_CLASS: dict[str, Capability] = {
    "time": Capability.CURRENT_TIME,
    "date": Capability.CURRENT_DATE,
    "calculation": Capability.CALCULATION,
    "unit_conversion": Capability.UNIT_CONVERSION,
    "stock_price": Capability.STOCK_PRICE,
    "news": Capability.LATEST_INFO,
    "weather": Capability.WEATHER,
}

# Classes whose tool execution fully verifies the prediction (the tool either
# computes a grounded answer or fails closed). Live classes (news, weather)
# are instead handled by the existing live-data policy, which is already
# honest about what it cannot fetch.
VERIFIED_TOOL_CLASSES = frozenset(
    {"time", "date", "calculation", "unit_conversion", "stock_price"}
)
LIVE_TOOL_CLASSES = frozenset({"news", "weather"})


@dataclass(frozen=True)
class ToolDispatchResult:
    success: bool
    tool_class: str = "none"
    capability: Capability | None = None
    confidence: float = 0.0
    probabilities: dict[str, float] | None = None
    latency_ms: int = 0
    model: str = ""
    error: str | None = None


class LearnedToolDispatcher:
    def __init__(
        self,
        *,
        weights: RouterWeights,
        embed: Callable[[str], list[float]] | None = None,
        base_url: str = "http://localhost:11434",
        min_confidence: float = 0.8,
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
        min_confidence: float = 0.8,
        expected_embedding_model: str | None = None,
    ) -> LearnedToolDispatcher | None:
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

    def classify(self, prompt: str) -> ToolDispatchResult:
        started = time.perf_counter()
        try:
            vector = self._embed(prompt[: self.max_prompt_chars])
        except Exception as exc:
            return ToolDispatchResult(
                success=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                model=self.weights.embedding_model,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        if len(vector) != self.weights.dim:
            return ToolDispatchResult(
                success=False,
                latency_ms=latency_ms,
                model=self.weights.embedding_model,
                error=(
                    f"embedding dim {len(vector)} does not match trained dim "
                    f"{self.weights.dim}"
                ),
            )
        probabilities = predict_probabilities(self.weights, vector)
        ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        top_class, top_prob = ranked[0]
        if top_class == "none" or top_prob < self.min_confidence:
            return ToolDispatchResult(
                success=False,
                tool_class=top_class,
                confidence=top_prob,
                probabilities=probabilities,
                latency_ms=latency_ms,
                model=self.weights.embedding_model,
                error=(
                    None
                    if top_class == "none"
                    else f"low confidence ({top_prob:.2f} < {self.min_confidence:.2f})"
                ),
            )
        return ToolDispatchResult(
            success=True,
            tool_class=top_class,
            capability=CAPABILITY_BY_TOOL_CLASS.get(top_class),
            confidence=top_prob,
            probabilities=probabilities,
            latency_ms=latency_ms,
            model=self.weights.embedding_model,
        )
