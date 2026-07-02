"""Prompt compression layers for Switchboard Core.

``HeadroomCompressionLayer`` applies Headroom-style heuristic context
compression before routing: long prompts are reduced to a task header,
preserved code/error snippets, and the most recent context window. All
compression statistics are recorded in request metadata so ablation studies
can measure tokens saved against answer quality.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

from switchboard.app.models.backends import SwitchboardRequest
from switchboard.app.services.context_compression import ContextCompressionService
from switchboard.app.services.cost import CostEstimator

LOGGER = logging.getLogger(__name__)
_LOGGED_HEADROOM_WARNINGS: set[str] = set()


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


class HeadroomLibCompressionLayer(CompressionLayer):
    """Optional adapter around the external ``headroom-ai`` library.

    The adapter is intentionally narrow: only ``<recent_conversation>`` content
    is sent to Headroom. Grounded fact blocks and the current request stay
    byte-identical in the assembled context.
    """

    def __init__(
        self,
        *,
        fallback: HeadroomCompressionLayer | None = None,
        compress_fn: Callable[[object], object] | None = None,
        threshold_tokens: int = 1000,
    ) -> None:
        self.fallback = fallback or HeadroomCompressionLayer(threshold_tokens=threshold_tokens)
        self.compress_fn = compress_fn

    def compress(self, request: SwitchboardRequest) -> SwitchboardRequest:
        request = self.fallback.compress(request)
        request.metadata["compression_engine"] = "heuristic"
        return request

    def compress_context(self, context_text: str) -> tuple[str, dict[str, object]]:
        original_tokens = self.fallback.service.cost_estimator.estimate_text_tokens(context_text)
        if original_tokens <= self.fallback.service.threshold_tokens:
            return context_text, {
                "context_compression_enabled": True,
                "context_compression_used": False,
                "context_compression_original_tokens": original_tokens,
                "context_compression_compressed_tokens": original_tokens,
                "context_compression_tokens_saved": 0,
                "context_compression_ratio": 1.0,
                "context_compression_scope": "none",
                "compression_engine": "headroom",
            }

        match = ContextCompressionService._HISTORY_BLOCK_RE.search(context_text)
        if match is None:
            compressed, stats = self.fallback.compress_context(context_text)
            stats["compression_engine"] = "heuristic"
            stats["headroom_fallback_reason"] = "no_recent_conversation_block"
            return compressed, stats
        try:
            compress_fn = self.compress_fn or self._load_headroom_compress()
            compressed_history = self._coerce_result(
                compress_fn([{"role": "user", "content": match.group("history")}])
            )
        except Exception as exc:
            self._log_headroom_warning(f"{type(exc).__name__}: {exc}")
            compressed, stats = self.fallback.compress_context(context_text)
            stats["compression_engine"] = "heuristic"
            stats["headroom_fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return compressed, stats

        if not compressed_history or len(compressed_history) >= len(match.group("history")):
            compressed, stats = self.fallback.compress_context(context_text)
            stats["compression_engine"] = "heuristic"
            stats["headroom_fallback_reason"] = "headroom_no_savings"
            return compressed, stats

        compressed_context = (
            context_text[: match.start("history")]
            + compressed_history
            + context_text[match.end("history") :]
        )
        estimator = self.fallback.service.cost_estimator
        compressed_tokens = estimator.estimate_text_tokens(compressed_context)
        stats = {
            "context_compression_enabled": True,
            "context_compression_used": compressed_tokens < original_tokens,
            "context_compression_original_tokens": original_tokens,
            "context_compression_compressed_tokens": compressed_tokens,
            "context_compression_tokens_saved": max(0, original_tokens - compressed_tokens),
            "context_compression_ratio": round(compressed_tokens / (original_tokens or 1), 4),
            "context_compression_scope": "history_only",
            "compression_engine": "headroom",
        }
        return compressed_context, stats

    def _load_headroom_compress(self) -> Callable[[object], object]:
        for module_name in ("headroom", "headroom_ai"):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            compress = getattr(module, "compress", None)
            if callable(compress):
                return compress
        raise ImportError("headroom-ai is not installed or exposes no compress()")

    def _coerce_result(self, result: object) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            parts: list[str] = []
            for item in result:
                if isinstance(item, dict) and isinstance(item.get("content"), str):
                    parts.append(item["content"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        if isinstance(result, dict):
            content = result.get("content") or result.get("compressed")
            if isinstance(content, str):
                return content
            messages = result.get("messages")
            if isinstance(messages, list):
                return self._coerce_result(messages)
        content = getattr(result, "content", None)
        if isinstance(content, str):
            return content
        raise TypeError("headroom compress() returned unsupported result")

    def _log_headroom_warning(self, message: str) -> None:
        if message in _LOGGED_HEADROOM_WARNINGS:
            return
        _LOGGED_HEADROOM_WARNINGS.add(message)
        LOGGER.warning("Headroom compression unavailable; using heuristic fallback: %s", message)
