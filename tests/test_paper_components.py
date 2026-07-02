"""Tests for the paper-experiment components: LLM router, Headroom-style
compression, and semantic long-term memory."""

from __future__ import annotations

from pathlib import Path

import httpx

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.core.config import Settings
from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    SwitchboardRequest,
    SwitchboardResponse,
)
from switchboard.app.models.personal import PersonalMemoryRead
from switchboard.app.models.telemetry import MemoryItem
from switchboard.app.services.compression_layer import (
    HeadroomCompressionLayer,
    NoCompressionLayer,
)
from switchboard.app.services.container import build_container
from switchboard.app.services.core_factory import build_configured_core_service
from switchboard.app.services.llm_router import (
    ARCH_ROUTER_MODEL,
    LlmRouter,
    OllamaRouterClient,
)
from switchboard.app.services.semantic_memory import (
    MemoryEmbeddingRepository,
    SemanticMemoryService,
    cosine_similarity,
)
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.storage.db import create_db_engine, init_db
from switchboard.app.storage.repositories import MemoryRepository

ROOT = Path(__file__).resolve().parents[1]


class FakeAdapter(AgentAdapter):
    def __init__(self, name: str, *, available: bool = True) -> None:
        self.name = name
        self.available = available
        self.cost_type = BackendCostType.LOCAL
        self.last_prompt: str | None = None

    def is_available(self) -> bool:
        return self.available

    def availability(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=self.available, cost_type=self.cost_type)

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        self.last_prompt = request.prompt
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            content=f"{self.name} answered",
            latency_ms=5,
            success=True,
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


def make_service(
    tmp_path: Path,
    registry: BackendRegistry,
    **kwargs,
) -> SwitchboardCoreService:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'paper.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)
    container.personal_config.preferences.claude_code_web_search = False
    return SwitchboardCoreService(
        registry=registry,
        metrics=container.backend_metrics_repository,
        container=container,
        **kwargs,
    )


def request(prompt: str) -> SwitchboardRequest:
    return SwitchboardRequest(request_id="req_test", prompt=prompt)


# ---------------------------------------------------------------------------
# LLM router
# ---------------------------------------------------------------------------


def test_llm_router_parses_strict_json() -> None:
    router = LlmRouter(complete=lambda _: '{"route_type": "coding", "confidence": 0.92}')
    result = router.classify("fix this bug")
    assert result.success
    assert result.route_type == "coding"
    assert result.backend == "codex"
    assert result.confidence == 0.92


def test_llm_router_parses_json_wrapped_in_prose() -> None:
    router = LlmRouter(
        complete=lambda _: 'Sure! {"route_type": "local", "confidence": 0.7} hope that helps'
    )
    result = router.classify("summarize my note")
    assert result.success
    assert result.backend == "ollama"


def test_llm_router_rejects_invalid_route_type() -> None:
    router = LlmRouter(complete=lambda _: '{"route_type": "spaceship", "confidence": 1.0}')
    result = router.classify("anything")
    assert not result.success
    assert result.error is not None


def test_arch_router_parses_policy_selection() -> None:
    router = LlmRouter(
        model=ARCH_ROUTER_MODEL,
        complete=lambda _: '{"policy": "reasoning", "confidence": 0.77}',
    )
    result = router.classify("compare two system designs")

    assert result.success
    assert result.route_type == "reasoning"
    assert result.backend == "claude-code"
    assert result.confidence == 0.77


def test_arch_router_rejects_unknown_policy() -> None:
    router = LlmRouter(
        model=ARCH_ROUTER_MODEL,
        complete=lambda _: '{"policy": "unknown", "confidence": 0.9}',
    )

    assert not router.classify("anything").success


def test_arch_router_prompt_construction(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"message": {"content": '{"policy": "local", "confidence": 0.8}'}}

    def fake_post(url: str, json: dict[str, object], timeout: float) -> Response:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(httpx, "post", fake_post)

    OllamaRouterClient(model=ARCH_ROUTER_MODEL).complete("summarise this")

    payload = captured["json"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    system = messages[0]["content"]
    assert "- tool:" in system
    assert "- local:" in system
    assert "- coding:" in system
    assert "- reasoning:" in system
    assert "format" not in payload


def test_generic_router_requests_json_schema(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"message": {"content": '{"route_type": "local", "confidence": 0.8}'}}

    def fake_post(url: str, json: dict[str, object], timeout: float) -> Response:
        captured["json"] = json
        return Response()

    monkeypatch.setattr(httpx, "post", fake_post)

    OllamaRouterClient(model="llama3.2:3b").complete("summarise this")

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["format"]["properties"]["route_type"]["enum"] == [
        "tool",
        "coding",
        "reasoning",
        "local",
        "unknown",
    ]


def test_llm_router_handles_unreachable_model() -> None:
    def boom(_: str) -> str:
        raise RuntimeError("connection refused")

    router = LlmRouter(complete=boom)
    result = router.classify("anything")
    assert not result.success
    assert "connection refused" in (result.error or "")


def test_llm_mode_uses_llm_classification(tmp_path: Path) -> None:
    registry = BackendRegistry(
        {"ollama": FakeAdapter("ollama"), "codex": FakeAdapter("codex"),
         "claude-code": FakeAdapter("claude-code")}
    )
    llm_router = LlmRouter(complete=lambda _: '{"route_type": "coding", "confidence": 0.9}')
    service = make_service(tmp_path, registry, router_mode="llm", llm_router=llm_router)

    req = request("an ambiguous prompt with no keywords at all")
    decision = service.route(req)

    assert decision.backend == "codex"
    assert "LLM router" in decision.routing_reason
    assert req.metadata["llm_router_used"] is True
    assert req.metadata["router_mode"] == "llm"


def test_llm_mode_falls_back_to_rules_when_router_down(tmp_path: Path) -> None:
    registry = BackendRegistry(
        {"ollama": FakeAdapter("ollama"), "codex": FakeAdapter("codex"),
         "claude-code": FakeAdapter("claude-code")}
    )

    def boom(_: str) -> str:
        raise RuntimeError("ollama down")

    service = make_service(
        tmp_path, registry, router_mode="llm", llm_router=LlmRouter(complete=boom)
    )

    req = request("Debug this failing pytest run")
    decision = service.route(req)

    assert decision.backend == "codex"  # deterministic rules still match "debug"
    assert "deterministic rules" in decision.routing_reason
    assert req.metadata["llm_router_used"] is False
    assert "llm_router_error" in req.metadata


def test_hybrid_mode_skips_llm_when_rules_confident(tmp_path: Path) -> None:
    registry = BackendRegistry(
        {"ollama": FakeAdapter("ollama"), "codex": FakeAdapter("codex"),
         "claude-code": FakeAdapter("claude-code")}
    )
    calls: list[str] = []

    def tracking_complete(prompt: str) -> str:
        calls.append(prompt)
        return '{"route_type": "local", "confidence": 0.9}'

    service = make_service(
        tmp_path,
        registry,
        router_mode="hybrid",
        llm_router=LlmRouter(complete=tracking_complete),
    )

    req = request("Refactor this repo entrypoint")
    decision = service.route(req)

    assert decision.backend == "codex"
    assert calls == []  # LLM router never consulted
    assert req.metadata["llm_router_skipped_reason"] == "rules_confident"


def test_hybrid_mode_consults_llm_for_unknown_prompts(tmp_path: Path) -> None:
    registry = BackendRegistry(
        {"ollama": FakeAdapter("ollama"), "codex": FakeAdapter("codex"),
         "claude-code": FakeAdapter("claude-code")}
    )
    llm_router = LlmRouter(complete=lambda _: '{"route_type": "local", "confidence": 0.8}')
    service = make_service(tmp_path, registry, router_mode="hybrid", llm_router=llm_router)

    req = request("Lorem ipsum dolor sit amet without routable words")
    decision = service.route(req)

    assert decision.backend == "ollama"
    assert req.metadata["llm_router_used"] is True


def test_rules_mode_never_calls_llm(tmp_path: Path) -> None:
    registry = BackendRegistry({"claude-code": FakeAdapter("claude-code")})
    calls: list[str] = []
    llm_router = LlmRouter(
        complete=lambda p: (calls.append(p), '{"route_type": "coding", "confidence": 1}')[1]
    )
    service = make_service(tmp_path, registry, router_mode="rules", llm_router=llm_router)

    service.route(request("Lorem ipsum dolor"))

    assert calls == []


def test_configured_router_llm_model_is_used(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'arch-router.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)
    container.personal_config.preferences.router_llm_model = ARCH_ROUTER_MODEL

    service = build_configured_core_service(container, router_mode="llm")

    assert service.llm_router is not None
    assert service.llm_router.model == ARCH_ROUTER_MODEL


# ---------------------------------------------------------------------------
# Headroom-style compression
# ---------------------------------------------------------------------------


def test_short_prompts_pass_through_uncompressed() -> None:
    layer = HeadroomCompressionLayer(threshold_tokens=1000)
    req = request("short prompt")
    result = layer.compress(req)
    assert result.prompt == "short prompt"
    assert req.metadata["compression_used"] is False
    assert req.metadata["compression_enabled"] is True


def test_long_prompts_are_compressed_with_stats() -> None:
    layer = HeadroomCompressionLayer(threshold_tokens=100)
    long_prompt = "Summarize this document.\n" + ("filler sentence about nothing. " * 300)
    req = request(long_prompt)
    result = layer.compress(req)
    assert result.prompt != long_prompt
    assert len(result.prompt) < len(long_prompt)
    assert result.metadata["compression_used"] is True
    assert result.metadata["compression_tokens_saved"] > 0
    assert 0 < result.metadata["compression_ratio"] < 1
    assert result.metadata["compression_original_tokens"] > (
        result.metadata["compression_compressed_tokens"]
    )


def test_no_compression_layer_records_disabled_flag() -> None:
    layer = NoCompressionLayer()
    req = request("anything")
    result = layer.compress(req)
    assert result.prompt == "anything"
    assert req.metadata["compression_enabled"] is False


def test_context_boundary_compression_compresses_assembled_context() -> None:
    layer = HeadroomCompressionLayer(threshold_tokens=50)
    long_context = (
        "<recent_conversation>\n"
        + ("User: tell me more about the plan. Assistant: here is detail. " * 200)
        + "\n</recent_conversation>\n<current_user_request>\nfinalize\n</current_user_request>"
    )
    compressed, stats = layer.compress_context(long_context)
    assert stats["context_compression_enabled"] is True
    assert stats["context_compression_used"] is True
    assert stats["context_compression_tokens_saved"] > 0
    assert len(compressed) < len(long_context)


def test_no_compression_layer_context_is_noop() -> None:
    text = "<current_user_request>\nhi\n</current_user_request>"
    compressed, stats = NoCompressionLayer().compress_context(text)
    assert compressed == text
    assert stats["context_compression_enabled"] is False


def test_compression_metadata_reaches_backend_metrics(tmp_path: Path) -> None:
    registry = BackendRegistry({"claude-code": FakeAdapter("claude-code")})
    service = make_service(
        tmp_path,
        registry,
        compression=HeadroomCompressionLayer(threshold_tokens=50),
    )
    long_prompt = "Review this plan.\n" + ("important detail. " * 200)
    response = service.ask(long_prompt, backend="claude-code")
    assert response.success
    metrics = service.metrics_list(limit=1)
    assert metrics[0].metadata["compression_used"] is True


# ---------------------------------------------------------------------------
# Semantic memory
# ---------------------------------------------------------------------------


def fake_embed(text: str) -> list[float]:
    """Deterministic toy embedding: counts of topic words."""
    lowered = text.lower()
    return [
        float(lowered.count("python")),
        float(lowered.count("router")),
        float(lowered.count("recipe")),
        1.0,  # bias term so vectors are never all-zero
    ]


def make_memory_service(tmp_path: Path) -> tuple[SemanticMemoryService, MemoryRepository]:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'memory.db'}")
    init_db(engine)
    memory_repository = MemoryRepository(engine)
    service = SemanticMemoryService(
        memory_repository=memory_repository,
        embedding_repository=MemoryEmbeddingRepository(engine),
        embed=fake_embed,
        min_similarity=0.5,
        top_k=2,
    )
    return service, memory_repository


def add_memory(repo: MemoryRepository, title: str, content: str) -> PersonalMemoryRead:
    return repo.add(MemoryItem(project="personal", title=title, content=content))


def test_cosine_similarity_basics() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0


def test_semantic_search_ranks_by_similarity(tmp_path: Path) -> None:
    service, repo = make_memory_service(tmp_path)
    python_memory = add_memory(repo, "Python style", "Prefer python python python typing.")
    recipe_memory = add_memory(repo, "Curry recipe", "A recipe recipe with spice.")
    assert service.index(python_memory)
    assert service.index(recipe_memory)

    results = service.search(project="personal", query="how do I write python code")

    assert results
    assert results[0][0].id == python_memory.id
    assert results[0][1] > 0.5


def test_semantic_search_falls_back_to_text_search(tmp_path: Path) -> None:
    def broken_embed(_: str) -> list[float]:
        raise RuntimeError("embedding model offline")

    engine = create_db_engine(f"sqlite:///{tmp_path / 'memory2.db'}")
    init_db(engine)
    repo = MemoryRepository(engine)
    service = SemanticMemoryService(
        memory_repository=repo,
        embedding_repository=MemoryEmbeddingRepository(engine),
        embed=broken_embed,
    )
    add_memory(repo, "Router note", "The router prefers local models.")

    results = service.search(project="personal", query="router")

    assert len(results) == 1
    assert results[0][1] == 0.0  # text-search fallback reports zero similarity


def test_index_returns_false_when_embeddings_unavailable(tmp_path: Path) -> None:
    def broken_embed(_: str) -> list[float]:
        raise RuntimeError("offline")

    engine = create_db_engine(f"sqlite:///{tmp_path / 'memory3.db'}")
    init_db(engine)
    repo = MemoryRepository(engine)
    service = SemanticMemoryService(
        memory_repository=repo,
        embedding_repository=MemoryEmbeddingRepository(engine),
        embed=broken_embed,
    )
    memory = add_memory(repo, "Note", "Content")
    assert service.index(memory) is False


def test_memory_facts_injected_into_backend_context(tmp_path: Path) -> None:
    adapter = FakeAdapter("claude-code")
    registry = BackendRegistry({"claude-code": adapter})

    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'paper.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)
    container.personal_config.preferences.claude_code_web_search = False
    memory_repo = container.memory_repository
    memory_service = SemanticMemoryService(
        memory_repository=memory_repo,
        embedding_repository=MemoryEmbeddingRepository(engine),
        embed=fake_embed,
        min_similarity=0.5,
    )
    memory = add_memory(memory_repo, "Router preference", "The router router prefers privacy.")
    memory_service.index(memory)

    service = SwitchboardCoreService(
        registry=registry,
        metrics=container.backend_metrics_repository,
        container=container,
        semantic_memory=memory_service,
    )
    response = service.ask("Tell me about my router preferences", backend="claude-code")

    assert response.success
    assert adapter.last_prompt is not None
    assert "<long_term_memory>" in adapter.last_prompt
    assert "Router preference" in adapter.last_prompt
    metrics = service.metrics_list(limit=1)
    assert metrics[0].metadata["memory_retrieved_count"] >= 1
    assert metrics[0].metadata["semantic_memory_used"] is True


def test_memory_disabled_keeps_context_clean(tmp_path: Path) -> None:
    adapter = FakeAdapter("claude-code")
    registry = BackendRegistry({"claude-code": adapter})
    service = make_service(tmp_path, registry)

    response = service.ask("Tell me about my router preferences", backend="claude-code")

    assert response.success
    assert adapter.last_prompt is not None
    assert "<long_term_memory>" not in adapter.last_prompt
