"""Shared factory for building a fully configured SwitchboardCoreService.

Used by both the CLI and the local web UI so that preferences in
``config/personal.yaml`` (router mode, compression, semantic memory) apply
consistently across surfaces.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.services.compression_layer import HeadroomCompressionLayer
from switchboard.app.services.container import ServiceContainer
from switchboard.app.services.finance_providers import finance_provider_by_name
from switchboard.app.services.finance_tool import StockPriceTool
from switchboard.app.services.learned_router import LearnedRouter
from switchboard.app.services.llm_router import LlmRouter
from switchboard.app.services.news_tool import NewsTool, news_provider_by_name
from switchboard.app.services.semantic_memory import (
    CachedEmbedder,
    MemoryEmbeddingRepository,
    OllamaEmbeddingClient,
    SemanticMemoryService,
)
from switchboard.app.services.sensitivity_escalator import (
    LearnedSensitivityEscalator,
)
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.services.tool_dispatcher import LearnedToolDispatcher
from switchboard.app.services.tools import ToolRegistry


def build_semantic_memory(
    container: ServiceContainer,
    *,
    embed: Callable[[str], list[float]] | None = None,
) -> SemanticMemoryService:
    preferences = container.personal_config.preferences
    ollama_base_url = (
        container.personal_config.provider_base_url("ollama") or "http://localhost:11434"
    )
    return SemanticMemoryService(
        memory_repository=container.memory_repository,
        embedding_repository=MemoryEmbeddingRepository(container.memory_repository.engine),
        embed=embed,
        embedding_model=preferences.embedding_model,
        base_url=ollama_base_url,
        top_k=preferences.semantic_memory_top_k,
    )


def build_configured_core_service(
    container: ServiceContainer,
    *,
    cwd: Path | None = None,
    router_mode: str | None = None,
    compression: bool | None = None,
    semantic_memory: bool | None = None,
) -> SwitchboardCoreService:
    """Build a core service honoring personal.yaml preferences.

    Explicit keyword arguments override the configured preferences; ``None``
    means "use the preference value".
    """
    preferences = container.personal_config.preferences
    ollama_base_url = (
        container.personal_config.provider_base_url("ollama") or "http://localhost:11434"
    )

    # One cached embedder shared by every learned component, so a prompt is
    # embedded once per request instead of once per component. Components
    # whose weights were trained with a different embedding model fail closed
    # on the dimension check and fall back to their deterministic paths.
    shared_embed = CachedEmbedder(
        OllamaEmbeddingClient(
            base_url=ollama_base_url,
            model=preferences.embedding_model,
        ).embed
    ).embed

    effective_router_mode = router_mode or preferences.router_mode
    llm_router = None
    if effective_router_mode in {"llm", "hybrid"}:
        llm_router = LlmRouter(
            model=preferences.llm_router_model,
            base_url=ollama_base_url,
        )

    learned_router = None
    if effective_router_mode == "learned":
        learned_router = LearnedRouter.from_file(
            preferences.router_weights_path,
            embed=shared_embed,
            min_confidence=preferences.learned_router_min_confidence,
        )
        if learned_router is None:
            # No trained weights yet: fall back to deterministic rules so the
            # app still works before the user runs `switchboard train-router`.
            effective_router_mode = "rules"

    tool_dispatcher = None
    if preferences.tool_dispatcher_enabled:
        # None when weights are missing: the regex detector simply keeps
        # working alone until the user runs `switchboard train-dispatcher`.
        tool_dispatcher = LearnedToolDispatcher.from_file(
            preferences.tool_dispatcher_weights_path,
            embed=shared_embed,
            min_confidence=preferences.tool_dispatcher_min_confidence,
        )

    sensitivity_escalator = None
    if preferences.sensitivity_escalator_enabled:
        # None when weights are missing: keyword sensitivity keeps working
        # alone until the user runs `switchboard train-sensitivity`.
        sensitivity_escalator = LearnedSensitivityEscalator.from_file(
            preferences.sensitivity_weights_path,
            embed=shared_embed,
            min_confidence=preferences.sensitivity_escalator_min_confidence,
        )

    compression_enabled = (
        compression if compression is not None else preferences.compression_enabled
    )
    compression_layer = (
        HeadroomCompressionLayer(threshold_tokens=preferences.compression_threshold_tokens)
        if compression_enabled
        else None
    )

    memory_enabled = (
        semantic_memory if semantic_memory is not None else preferences.semantic_memory_enabled
    )
    memory_service = (
        build_semantic_memory(container, embed=shared_embed) if memory_enabled else None
    )

    # Live-data tools configured in personal.yaml (fall back to env-based
    # defaults inside each tool when the preference is empty).
    tool_registry = None
    if preferences.finance_provider or preferences.news_provider:
        tool_registry = ToolRegistry(
            stock_price_tool=StockPriceTool(
                finance_provider_by_name(preferences.finance_provider)
            )
            if preferences.finance_provider
            else None,
            news_tool=NewsTool(news_provider_by_name(preferences.news_provider))
            if preferences.news_provider
            else None,
        )

    return SwitchboardCoreService(
        registry=BackendRegistry.default(container, cwd=cwd or Path.cwd()),
        metrics=container.backend_metrics_repository,
        container=container,
        router_mode=effective_router_mode,
        llm_router=llm_router,
        learned_router=learned_router,
        compression=compression_layer,
        semantic_memory=memory_service,
        tool_registry=tool_registry,
        tool_dispatcher=tool_dispatcher,
        sensitivity_escalator=sensitivity_escalator,
    )
