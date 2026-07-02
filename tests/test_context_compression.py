from __future__ import annotations

from pathlib import Path

from switchboard.app.core.config import Settings
from switchboard.app.models.internal import Sensitivity, TaskType
from switchboard.app.services.compression_layer import (
    HeadroomCompressionLayer,
    HeadroomLibCompressionLayer,
)
from switchboard.app.services.container import build_container
from switchboard.app.services.context_compression import ContextCompressionService
from switchboard.app.services.core_factory import build_configured_core_service
from switchboard.app.services.cost import CostEstimator
from switchboard.app.storage.db import create_db_engine, init_db

ROOT = Path(__file__).resolve().parents[1]

PREAMBLE = (
    "You are replying to the user through Switchboard.\n"
    "Use any trusted facts below to answer the user's request.\n"
    "Return only the final user-facing answer."
)
TRUSTED_FACTS_BLOCK = (
    "<trusted_facts>\n"
    "- Tesla (TSLA) last trade price: $347.82\n"
    "- Do not invent specific facts; say when you cannot access live data.\n"
    "</trusted_facts>"
)
MEMORY_BLOCK = "<long_term_memory>\n- User prefers metric units.\n</long_term_memory>"


def assembled_context(
    *,
    request: str = "what is tesla trading at?",
    history_messages: int = 12,
    include_history: bool = True,
) -> str:
    parts = [PREAMBLE, TRUSTED_FACTS_BLOCK, MEMORY_BLOCK]
    if include_history:
        history_lines = [
            f"{'User' if i % 2 == 0 else 'Assistant'}: message {i} " + ("filler detail " * 60)
            for i in range(history_messages)
        ]
        parts.append(
            "<recent_conversation>\n" + "\n".join(history_lines) + "\n</recent_conversation>"
        )
    parts.append(f"<current_user_request>\n{request}\n</current_user_request>")
    return "\n".join(parts)


def test_context_compression_estimates_token_savings() -> None:
    service = ContextCompressionService(CostEstimator(), threshold_tokens=20)
    prompt = "Repeat this context. " * 200

    result = service.compress(prompt)

    assert result.compression_used
    assert result.compressed_estimated_tokens < result.original_estimated_tokens
    assert result.estimated_tokens_saved > 0
    assert result.warning is not None


def test_context_compression_preserves_code_and_private_warning() -> None:
    service = ContextCompressionService(CostEstimator(), threshold_tokens=20)
    prompt = (
        "Debug this error without using cloud models.\n"
        "```python\nraise ValueError('boom')\n```\n" + ("Private project context. " * 200)
    )

    result = service.compress(
        prompt,
        task_type=TaskType.DEBUGGING,
        sensitivity=Sensitivity.PRIVATE_PERSONAL,
    )

    assert result.compression_used
    assert result.compression_ratio < 1
    assert result.compressed_prompt is not None
    assert "Sensitive/private content warning" in result.compressed_prompt
    assert "raise ValueError" in result.compressed_prompt


# ---------------------------------------------------------------------------
# Structure-aware compression of assembled session contexts (dogfood
# regression 2026-06-12): compression silently dropped <trusted_facts>,
# <long_term_memory>, and honesty directives, leaving the model free to
# fabricate. Only conversation history is compressible; grounded-truth blocks
# must survive byte-identical.
# ---------------------------------------------------------------------------


def test_assembled_context_compression_preserves_fact_blocks_verbatim() -> None:
    service = ContextCompressionService(CostEstimator(), threshold_tokens=200)
    context = assembled_context()

    result = service.compress_assembled_context(context)

    assert result.compression_used
    assert result.scope == "history_only"
    assert result.compressed_prompt is not None
    compressed = result.compressed_prompt
    # Grounded truth survives verbatim, byte-identical including newlines.
    assert compressed.startswith(PREAMBLE)
    assert TRUSTED_FACTS_BLOCK in compressed
    assert MEMORY_BLOCK in compressed
    assert "Tesla (TSLA) last trade price: $347.82" in compressed
    assert "Do not invent specific" in compressed
    request_block = "<current_user_request>\nwhat is tesla trading at?\n</current_user_request>"
    assert request_block in compressed
    # And the savings are real.
    assert result.compressed_estimated_tokens < result.original_estimated_tokens
    assert result.estimated_tokens_saved > 0
    assert len(compressed) < len(context)


def test_assembled_context_preserves_multiline_user_request_verbatim() -> None:
    service = ContextCompressionService(CostEstimator(), threshold_tokens=200)
    request = "line one\n  line two with leading spaces\n\tline three with a tab"
    context = assembled_context(request=request)

    result = service.compress_assembled_context(context)

    assert result.compression_used
    assert result.compressed_prompt is not None
    assert f"<current_user_request>\n{request}\n</current_user_request>" in result.compressed_prompt


def test_assembled_context_below_threshold_is_noop() -> None:
    service = ContextCompressionService(CostEstimator(), threshold_tokens=100_000)
    context = assembled_context()

    result = service.compress_assembled_context(context)

    assert result.compression_used is False
    assert result.compressed_prompt is None
    assert result.compression_ratio == 1.0
    assert result.estimated_tokens_saved == 0


def test_assembled_context_without_history_never_eats_fact_blocks() -> None:
    # Over threshold but with no <recent_conversation> block: every remaining
    # block is grounded truth, so nothing is compressed. Facts win over budget.
    service = ContextCompressionService(CostEstimator(), threshold_tokens=50)
    context = assembled_context(include_history=False)

    result = service.compress_assembled_context(context)

    assert result.compression_used is False
    assert result.compressed_prompt is None
    assert result.scope == "history_only"


def test_arbitrary_text_falls_back_to_whole_text_compression() -> None:
    service = ContextCompressionService(CostEstimator(), threshold_tokens=20)
    raw_prompt = "Summarize this. " + ("filler sentence about nothing. " * 200)

    result = service.compress_assembled_context(raw_prompt)

    assert result.compression_used
    assert result.scope == "whole_text"
    assert result.compressed_prompt is not None
    assert result.compressed_estimated_tokens < result.original_estimated_tokens


def test_pathological_inputs_stay_safe_via_fallback_and_structured_paths() -> None:
    service = ContextCompressionService(CostEstimator(), threshold_tokens=20)
    pathological_cases = [
        "x" * 50_000,  # 50k single line, no newlines
        "data\x00with\x00nul\x00bytes " * 500,
        r"regex metachars (?P<bad>[a-z]+\\ ${} .*+?[]() |^$ " * 300,
    ]
    for nasty in pathological_cases:
        # Raw path: must not raise.
        raw_result = service.compress_assembled_context(nasty)
        assert raw_result.scope == "whole_text"
        assert raw_result.compressed_prompt is not None
        # Structured path: nasty content embedded in history must not raise
        # and must not damage the request block.
        context = (
            f"{PREAMBLE}\n{TRUSTED_FACTS_BLOCK}\n"
            f"<recent_conversation>\nUser: {nasty}\n</recent_conversation>\n"
            "<current_user_request>\nis this safe?\n</current_user_request>"
        )
        structured = service.compress_assembled_context(context)
        assert structured.scope == "history_only"
        if structured.compressed_prompt is not None:
            assert TRUSTED_FACTS_BLOCK in structured.compressed_prompt
            assert (
                "<current_user_request>\nis this safe?\n</current_user_request>"
                in structured.compressed_prompt
            )


def test_raw_compress_path_unchanged_for_assembled_looking_text() -> None:
    # The legacy compress() entry point (raw prompt path) keeps whole-text
    # behavior and never sets a scope.
    service = ContextCompressionService(CostEstimator(), threshold_tokens=20)
    result = service.compress(assembled_context())
    assert result.compression_used
    assert result.scope is None


def test_headroom_layer_compress_context_reports_scope_and_keeps_facts() -> None:
    layer = HeadroomCompressionLayer(threshold_tokens=200)
    context = assembled_context()

    compressed, stats = layer.compress_context(context)

    assert stats["context_compression_enabled"] is True
    assert stats["context_compression_used"] is True
    assert stats["context_compression_scope"] == "history_only"
    assert stats["context_compression_tokens_saved"] > 0
    assert TRUSTED_FACTS_BLOCK in compressed
    assert MEMORY_BLOCK in compressed
    request_block = "<current_user_request>\nwhat is tesla trading at?\n</current_user_request>"
    assert request_block in compressed
    assert len(compressed) < len(context)


def test_headroom_lib_layer_only_passes_recent_conversation_to_headroom() -> None:
    seen: list[object] = []

    def fake_compress(messages: object) -> object:
        seen.append(messages)
        return [{"role": "user", "content": "compressed history"}]

    layer = HeadroomLibCompressionLayer(
        compress_fn=fake_compress,
        threshold_tokens=200,
    )
    context = assembled_context()

    compressed, stats = layer.compress_context(context)

    assert stats["compression_engine"] == "headroom"
    assert TRUSTED_FACTS_BLOCK in compressed
    assert MEMORY_BLOCK in compressed
    assert (
        "<current_user_request>\nwhat is tesla trading at?\n</current_user_request>"
        in compressed
    )
    assert "compressed history" in compressed
    assert seen == [
        [
            {
                "role": "user",
                "content": context.split("<recent_conversation>\n", 1)[1].split(
                    "\n</recent_conversation>",
                    1,
                )[0],
            }
        ]
    ]


def test_headroom_lib_layer_respects_threshold_before_external_call() -> None:
    called = False

    def fake_compress(messages: object) -> object:
        nonlocal called
        called = True
        return [{"role": "user", "content": "compressed history"}]

    layer = HeadroomLibCompressionLayer(
        compress_fn=fake_compress,
        threshold_tokens=100_000,
    )
    context = assembled_context()

    compressed, stats = layer.compress_context(context)

    assert compressed == context
    assert called is False
    assert stats["compression_engine"] == "headroom"
    assert stats["context_compression_used"] is False
    assert stats["context_compression_scope"] == "none"


def test_headroom_lib_layer_falls_back_to_heuristic_on_runtime_error() -> None:
    def broken_compress(messages: object) -> object:
        raise RuntimeError("model download failed")

    layer = HeadroomLibCompressionLayer(
        compress_fn=broken_compress,
        threshold_tokens=200,
    )
    context = assembled_context()

    compressed, stats = layer.compress_context(context)

    assert stats["compression_engine"] == "heuristic"
    assert "model download failed" in str(stats["headroom_fallback_reason"])
    assert TRUSTED_FACTS_BLOCK in compressed
    assert MEMORY_BLOCK in compressed


def test_headroom_compression_engine_configures_lib_layer(tmp_path) -> None:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'headroom.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)
    container.personal_config.preferences.compression_enabled = True
    container.personal_config.preferences.compression_engine = "headroom"

    service = build_configured_core_service(container)

    assert isinstance(service.compression, HeadroomLibCompressionLayer)
