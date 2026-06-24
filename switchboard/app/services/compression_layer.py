"""Prompt compression layers for Switchboard Core.

``HeadroomCompressionLayer`` applies Headroom-style heuristic context
compression before routing: long prompts are reduced to a task header,
preserved code/error snippets, and the most recent context window. All
compression statistics are recorded in request metadata so ablation studies
can measure tokens saved against answer quality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from switchboard.app.models.backends import SwitchboardRequest
from switchboard.app.services.context_compression import ContextCompressionService
from switchboard.app.services.cost import CostEstimator


class CompressionLayer(ABC):
    @abstractmethod
    def compress(self, request: SwitchboardRequest) -> SwitchboardRequest:
        raise NotImplementedError

    def compress_context(self, context_text: str) -> tuple[str, dict[str, object]]:
        """Compress the fully assembled context (history + memory + facts +
        request) at the model boundary. Default: no-op."""
        return context_text, {"context_compression_enabled": False}


class NoCompressionLayer(CompressionLayer):
    def compress(self, request: SwitchboardRequest) -> SwitchboardRequest:
        request.metadata.setdefault("compression_enabled", False)
        request.metadata.setdefault("compression_used", False)
        return request


class HeadroomCompressionLayer(CompressionLayer):
    """Measurable, ablatable heuristic prompt compression.

    Wraps :class:`ContextCompressionService` and records original/compressed
    token estimates, the compression ratio, and estimated tokens saved in
    ``request.metadata``. Prompts at or below ``threshold_tokens`` pass
    through unchanged.
    """

    def __init__(
        self,
        *,
        service: ContextCompressionService | None = None,
        threshold_tokens: int = 1000,
    ) -> None:
        self.service = service or ContextCompressionService(
            CostEstimator(),
            threshold_tokens=threshold_tokens,
        )

    def compress(self, request: SwitchboardRequest) -> SwitchboardRequest:
        result = self.service.compress(request.prompt)
        request.metadata.update(
            {
                "compression_enabled": True,
                "compression_used": result.compression_used,
                "compression_original_tokens": result.original_estimated_tokens,
                "compression_compressed_tokens": result.compressed_estimated_tokens,
                "compression_tokens_saved": result.estimated_tokens_saved,
                "compression_ratio": result.compression_ratio,
            }
        )
        if not result.compression_used or not result.compressed_prompt:
            return request
        return request.model_copy(
            update={
                "prompt": result.compressed_prompt,
                "metadata": dict(request.metadata),
            }
        )

    def compress_context(self, context_text: str) -> tuple[str, dict[str, object]]:
        # Structure-aware Headroom compression of the assembled context.
        # Only the <recent_conversation> history is compressed; the
        # instruction preamble, <trusted_facts>, <long_term_memory>, and
        # <current_user_request> blocks survive verbatim so compression can
        # never delete grounded truth (see compress_assembled_context).
        result = self.service.compress_assembled_context(context_text)
        stats: dict[str, object] = {
            "context_compression_enabled": True,
            "context_compression_used": result.compression_used,
            "context_compression_original_tokens": result.original_estimated_tokens,
            "context_compression_compressed_tokens": result.compressed_estimated_tokens,
            "context_compression_tokens_saved": result.estimated_tokens_saved,
            "context_compression_ratio": result.compression_ratio,
            "context_compression_scope": result.scope or "none",
        }
        if result.compression_used and result.compressed_prompt:
            return result.compressed_prompt, stats
        return context_text, stats
