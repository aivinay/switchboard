from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from switchboard.app.core.config import Settings, get_settings
from switchboard.app.main import create_app
from switchboard.app.models.catalogue import ModelCatalogue, ModelKind, QualityTier
from switchboard.app.providers.base import ProviderResponse
from switchboard.app.providers.ollama import OllamaProviderAdapter
from switchboard.app.services.answer_quality import AnswerQualityHeuristic
from switchboard.app.services.local_runtime import (
    OllamaRuntimeService,
    RuntimeCommandResult,
)
from switchboard.app.services.personal_switchboard import PersonalSwitchboardService
from switchboard.cli import (
    doctor_command,
    feedback_command,
    loaded_models_command,
    personal_ask_command,
)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_LIST_LOADED_MODELS = OllamaRuntimeService.list_loaded_models


@pytest.fixture(autouse=True)
def no_real_ollama_runtime(monkeypatch) -> None:
    monkeypatch.setattr(OllamaRuntimeService, "list_loaded_models", lambda self: set())


def personal_config(
    tmp_path: Path,
    *,
    ollama: bool = True,
    openai: bool = False,
    allow_cloud: bool = False,
    performance_mode: str = "balanced",
    max_loaded_models: int = 2,
    reuse_hot_model_if_good_enough: bool = True,
    prefer_hot_model_for_simple_tasks: bool = True,
    unload_after_benchmark: bool = True,
) -> Path:
    path = tmp_path / "personal.yaml"
    path.write_text(
        f"""
profile:
  user_id: "local-user"
  default_project: "personal"
preferences:
  default_mode: "auto"
  local_first: true
  prefer_free_models: true
  allow_cloud: {str(allow_cloud).lower()}
  require_confirmation_for_scarce_models: true
  private_mode: true
  cache_routing: true
  cache_answers: false
savings:
  default_baseline_model: "manual/claude-web"
  premium_unit_value_usd: null
  assume_premium_for_unknown: false
local_runtime:
  performance_mode: "{performance_mode}"
  max_loaded_models: {max_loaded_models}
  keep_alive: "10m"
  reuse_hot_model_if_good_enough: {str(reuse_hot_model_if_good_enough).lower()}
  model_switch_penalty_ms: 3000
  prefer_hot_model_for_simple_tasks: {str(prefer_hot_model_for_simple_tasks).lower()}
  unload_after_benchmark: {str(unload_after_benchmark).lower()}
providers:
  mock:
    type: "mock"
    enabled: true
  ollama:
    type: "local"
    base_url: "http://localhost:11434"
    enabled: {str(ollama).lower()}
  openai:
    type: "cloud_api"
    env_api_key: "OPENAI_API_KEY"
    enabled: {str(openai).lower()}
    scarce: true
  claude_web:
    type: "manual_subscription"
    enabled: true
    scarce: true
  chatgpt_web:
    type: "manual_subscription"
    enabled: true
    scarce: true
  codex:
    type: "manual_subscription"
    enabled: true
    scarce: true
""",
        encoding="utf-8",
    )
    return path


def personal_client(tmp_path: Path, config_path: Path) -> TestClient:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'daily_use.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(config_path),
    )
    return TestClient(create_app(settings))


def cli_args(prompt: str, **overrides: object) -> argparse.Namespace:
    defaults = {
        "prompt": prompt,
        "project": None,
        "no_cache": True,
        "show_prompt": False,
        "show_metadata": False,
        "strict": False,
        "force_model": None,
        "allow_cloud_once": False,
        "override_reason": None,
        "baseline": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_model_catalogue_includes_recommended_ollama_pack() -> None:
    catalogue = ModelCatalogue.from_yaml(ROOT / "config" / "models.yaml")
    model_ids = {model.model_id for model in catalogue.models}

    assert "ollama/llama3.2:3b" in model_ids
    assert "ollama/gemma4:e4b" in model_ids
    assert "ollama/gemma4:12b" in model_ids
    assert "ollama/qwen3.5:9b" in model_ids
    assert "ollama/gpt-oss:20b" in model_ids
    assert "ollama/embeddinggemma" in model_ids
    assert "ollama/nomic-embed-text" in model_ids
    llama = catalogue.get("ollama/llama3.2:3b")
    nomic = catalogue.get("ollama/nomic-embed-text")
    assert llama is not None
    assert llama.provider_model_name == "llama3.2:3b"
    assert nomic is not None
    assert nomic.kind == ModelKind.LOCAL_EMBEDDING
    assert nomic.quality_tier == QualityTier.EMBEDDING
    assert nomic.must_never_be_selected_for_chat
    assert not catalogue.get("ollama/gemma4:31b").enabled  # type: ignore[union-attr]
    assert not catalogue.get("ollama/qwen3-coder:30b").enabled  # type: ignore[union-attr]


def test_default_personal_config_enables_ollama() -> None:
    config_path = ROOT / "config" / "personal.yaml"

    from switchboard.app.models.personal import PersonalConfig

    config = PersonalConfig.from_yaml(config_path)

    assert config.provider_enabled("ollama")
    assert config.local_runtime.performance_mode == "balanced"
    assert config.local_runtime.max_loaded_models == 2
    assert config.preferences.claude_code_web_search is True


def test_requested_bullet_count_detection_variants() -> None:
    quality = AnswerQualityHeuristic()

    assert quality.requested_bullet_count("Summarise in three bullets.") == 3
    assert quality.requested_bullet_count("Summarise in 3 bullets.") == 3
    assert quality.requested_bullet_count("Summarise in five bullets.") == 5
    assert quality.requested_bullet_count("Summarise in 5 bullet points.") == 5
    assert quality.requested_bullet_count("Give me 4 points.") == 4


def test_actual_bullet_count_detection_variants() -> None:
    quality = AnswerQualityHeuristic()

    answer = "\u2022 One\n- Two\n* Three\n1. Four\n2) Five"

    assert quality.bullet_count(answer) == 5
    assert quality.empty_bullet_count("\u2022\n- Two\n3)") == 2


def test_source_limitation_note_mismatch_warns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def mismatched_note(self, request, model_profile):  # noqa: ANN001
        return ProviderResponse(
            content=(
                "Only 1 distinct fact was present in the source.\n\n"
                "- The sender cannot attend tomorrow's meeting.\n"
                "- The sender can send notes instead."
            ),
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=18,
        )

    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", mismatched_note)
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/ask",
        json={
            "prompt": (
                "Summarise this email in three bullets: I cannot attend tomorrow's "
                "meeting but can send notes."
            ),
            "use_cache": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["quality_warning"]
    assert any("source-limitation note says 1" in note for note in body["quality_notes"])


def test_nomic_embed_text_is_never_selected_for_chat(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post("/personal/route", json={"prompt": "Summarise this email."})

    assert response.status_code == 200
    assert response.json()["recommended_model"] != "ollama/nomic-embed-text"


def test_loaded_embedding_model_is_never_reused_for_chat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        OllamaRuntimeService,
        "list_loaded_models",
        lambda self: {"ollama/nomic-embed-text"},
    )
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post("/personal/route", json={"prompt": "Summarise this email."})

    assert response.status_code == 200
    assert response.json()["recommended_model"] == "ollama/gemma4:e4b"


def test_local_simple_tasks_prefer_gemma4_e4b_when_ollama_enabled(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post("/personal/route", json={"prompt": "Summarise this email."})

    assert response.status_code == 200
    assert response.json()["recommended_model"] == "ollama/gemma4:e4b"


def test_hot_general_model_is_reused_for_simple_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        OllamaRuntimeService,
        "list_loaded_models",
        lambda self: {"ollama/gemma4:12b"},
    )
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post("/personal/route", json={"prompt": "Summarise this email."})

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_model"] == "ollama/gemma4:12b"
    assert body["selected_model_loaded"]
    assert body["model_switch_avoided"]
    assert not body["cold_start_expected"]
    assert "HOT_MODEL_REUSED" in body["reason_codes"]
    assert "MODEL_SWITCH_AVOIDED" in body["reason_codes"]


def test_coding_prefers_qwen_coder_when_ollama_enabled(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/route",
        json={"prompt": "Debug this Python error: TypeError: NoneType is not subscriptable."},
    )

    assert response.status_code == 200
    assert response.json()["recommended_model"] == "ollama/qwen3.5:9b"


def test_general_reasoning_prefers_gemma4_when_ollama_enabled(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/route",
        json={"prompt": "What is the difference between a router and a gateway?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_model"] == "ollama/gemma4:12b"
    assert "OLLAMA_GENERAL_MODEL_SELECTED" in body["reason_codes"]


def test_coding_does_not_reuse_non_coding_hot_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        OllamaRuntimeService,
        "list_loaded_models",
        lambda self: {"ollama/gemma4:12b"},
    )
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/route",
        json={"prompt": "Debug this Python code: for customer in customers: print(customer_id)"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_model"] == "ollama/qwen3.5:9b"
    assert not body["selected_model_loaded"]
    assert body["cold_start_expected"]
    assert "SPECIALIST_MODEL_SWITCH_JUSTIFIED" in body["reason_codes"]


def test_memory_saver_reuses_loaded_good_enough_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        OllamaRuntimeService,
        "list_loaded_models",
        lambda self: {"ollama/gemma4:12b"},
    )
    client = personal_client(
        tmp_path,
        personal_config(
            tmp_path,
            ollama=True,
            performance_mode="memory_saver",
            max_loaded_models=1,
        ),
    )

    response = client.post("/personal/route", json={"prompt": "Summarise this meeting note."})

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_model"] == "ollama/gemma4:12b"
    assert body["performance_mode"] == "memory_saver"
    assert "MEMORY_SAVER_MODE_ACTIVE" in body["reason_codes"]
    assert "MODEL_SWITCH_AVOIDED" in body["reason_codes"]


def test_balanced_mode_uses_loaded_coding_specialist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        OllamaRuntimeService,
        "list_loaded_models",
        lambda self: {"ollama/gemma4:12b", "ollama/qwen3.5:9b"},
    )
    client = personal_client(
        tmp_path,
        personal_config(tmp_path, ollama=True, performance_mode="balanced"),
    )

    response = client.post(
        "/personal/route",
        json={"prompt": "Debug this Python function and explain the bug."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_model"] == "ollama/qwen3.5:9b"
    assert body["selected_model_loaded"]
    assert body["performance_mode"] == "balanced"
    assert "BALANCED_RUNTIME_MODE_ACTIVE" in body["reason_codes"]
    assert "OLLAMA_MODEL_ALREADY_LOADED" in body["reason_codes"]


def test_private_complex_reasoning_prefers_local_reasoning_model(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/route",
        json={
            "prompt": (
                "This is private medical information. Compare three treatment planning "
                "tradeoffs and identify risks."
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_model"] == "ollama/gpt-oss:20b"
    assert body["route_kind"] == "local"
    assert "PERSONAL_PRIVATE_MODE_CLOUD_BLOCKED" in body["reason_codes"]


def test_sensitive_simple_task_uses_medium_local_not_frontier(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/route",
        json={
            "prompt": (
                "Summarise my private medical letter and list follow-up questions "
                "for my doctor."
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_model"] == "ollama/gemma4:12b"
    assert body["route_kind"] == "local"
    assert "SENSITIVE_BUT_SIMPLE_TASK" in body["reason_codes"]
    assert "SENSITIVITY_DOES_NOT_IMPLY_FRONTIER" in body["reason_codes"]


def test_switchboard_ask_uses_ollama_when_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def fake_complete_chat(self, request, model_profile):  # noqa: ANN001
        return ProviderResponse(
            content="Three concise bullets from the real local model.",
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=8,
        )

    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", fake_complete_chat)
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/ask",
        json={
            "prompt": (
                "Summarise this email in three bullet points: I cannot attend tomorrow's "
                "meeting but can send notes."
            ),
            "use_cache": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "called"
    assert body["answer"] == "Three concise bullets from the real local model."
    assert body["recommendation"]["recommended_model"] == "ollama/gemma4:e4b"
    assert "Mock response from" not in body["answer"]


def test_summary_prompt_wrapper_includes_grounding_instruction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen_messages = []

    async def fake_complete_chat(self, request, model_profile):  # noqa: ANN001
        seen_messages.extend(request.messages)
        return ProviderResponse(
            content=(
                "- The sender cannot attend tomorrow's meeting.\n"
                "- The sender can send notes instead.\n"
                "Only 2 distinct facts were present in the source."
            ),
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=16,
        )

    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", fake_complete_chat)
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/ask",
        json={
            "prompt": (
                "Summarise this email in five bullets: I cannot attend tomorrow's "
                "meeting but can send notes."
            ),
            "use_cache": False,
        },
    )

    assert response.status_code == 200
    assert seen_messages[0].role == "system"
    assert "Do not invent facts" in seen_messages[0].content
    assert "SOURCE_GROUNDED_SUMMARY_PROMPT" in response.json()["recommendation"][
        "reason_codes"
    ]
    assert not response.json()["quality_warning"]


def test_summary_strict_flag_makes_wrapper_more_explicit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen_messages = []

    async def fake_complete_chat(self, request, model_profile):  # noqa: ANN001
        seen_messages.extend(request.messages)
        return ProviderResponse(
            content="- The sender cannot attend.",
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=6,
        )

    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", fake_complete_chat)
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/ask",
        json={
            "prompt": "Summarise this email: I cannot attend.",
            "strict": True,
            "use_cache": False,
        },
    )

    assert response.status_code == 200
    assert "Strict mode is enabled." in seen_messages[0].content


def test_cli_ask_output_distinguishes_ollama_provider(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    async def fake_complete_chat(self, request, model_profile):  # noqa: ANN001
        return ProviderResponse(
            content="- Cannot attend.\n- Can send notes.\n- Follow up later.",
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=10,
        )

    config_path = personal_config(tmp_path, ollama=True)
    monkeypatch.setenv("ICP_PERSONAL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ICP_DATABASE_URL", f"sqlite:///{tmp_path / 'cli_ollama.db'}")
    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", fake_complete_chat)
    get_settings.cache_clear()

    personal_ask_command(
        cli_args(
            "Summarise this email in three bullets: I cannot attend tomorrow's meeting."
        )
    )

    output = capsys.readouterr().out
    assert "Model: ollama/gemma4:e4b" in output
    assert "Provider: Ollama" in output
    assert "Provider status:" not in output
    assert "Route: local model" in output
    assert "Premium saved: 1.0 unit(s)" in output
    assert "Called model:" not in output
    assert "Request ID: req_" in output
    assert "Next step: Local Ollama model was called" not in output
    assert "local/mock provider was called" not in output

    request_id_match = re.search(r"Request ID: (req_[a-f0-9]+)", output)
    assert request_id_match is not None
    feedback_command(
        argparse.Namespace(
            request_id=request_id_match.group(1),
            rating="too-weak",
            note="Needed stricter grounding.",
            preferred_model="ollama/gemma4:12b",
        )
    )
    feedback_output = capsys.readouterr().out
    assert request_id_match.group(1) in feedback_output
    assert '"rating": "too-weak"' in feedback_output
    get_settings.cache_clear()


def test_cli_ask_show_metadata_includes_operational_details(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    async def fake_complete_chat(self, request, model_profile):  # noqa: ANN001
        return ProviderResponse(
            content="- Cannot attend.\n- Can send notes.\n- Follow up later.",
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=10,
        )

    config_path = personal_config(tmp_path, ollama=True)
    monkeypatch.setenv("ICP_PERSONAL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ICP_DATABASE_URL", f"sqlite:///{tmp_path / 'cli_ollama_debug.db'}")
    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", fake_complete_chat)
    get_settings.cache_clear()

    personal_ask_command(
        cli_args(
            "Summarise this email in three bullets: I cannot attend tomorrow's meeting.",
            show_metadata=True,
        )
    )

    output = capsys.readouterr().out
    assert "Called model: True" in output
    assert "Request ID: req_" in output
    get_settings.cache_clear()


def test_cli_ask_output_distinguishes_mock_provider(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = personal_config(tmp_path, ollama=False)
    monkeypatch.setenv("ICP_PERSONAL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ICP_DATABASE_URL", f"sqlite:///{tmp_path / 'cli_mock.db'}")
    get_settings.cache_clear()

    personal_ask_command(cli_args("Summarise this email in three bullets."))

    output = capsys.readouterr().out
    assert "Demo mock response only from mock/small" in output
    assert "Provider: Demo mock" in output
    assert "Provider status:" not in output
    assert "Route: demo mock" in output
    assert "Called model:" not in output
    assert "Request ID: req_" in output
    assert "Next step: Demo mock provider was called" not in output
    assert "local/mock provider was called" not in output
    get_settings.cache_clear()


def test_cli_ask_manual_recommendation_includes_request_id(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = personal_config(tmp_path, ollama=False)
    monkeypatch.setenv("ICP_PERSONAL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ICP_DATABASE_URL", f"sqlite:///{tmp_path / 'cli_manual.db'}")
    get_settings.cache_clear()

    personal_ask_command(
        cli_args(
            "Create a multi-step strategy for launching a local-first developer tool.",
            force_model="manual/claude-web",
            override_reason="Demo premium recommendation.",
        )
    )

    output = capsys.readouterr().out
    assert "Provider: Manual recommendation" in output
    assert "Route: manual recommendation" in output
    assert "Request ID: req_" in output
    assert "Called model:" not in output
    assert "Switchboard did not call the provider" in output
    get_settings.cache_clear()


def test_switchboard_ask_falls_back_to_mock_when_ollama_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def unavailable(self, request, model_profile):  # noqa: ANN001
        raise RuntimeError("Ollama is unavailable")

    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", unavailable)
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/ask",
        json={"prompt": "Summarise this email in three bullets.", "use_cache": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "called"
    assert body["answer"].startswith("Fell back to mock because Ollama was unavailable.")
    assert body["recommendation"]["recommended_model"] == "mock/small"
    assert "MOCK_FALLBACK_USED" in body["recommendation"]["reason_codes"]


def test_short_source_with_fewer_accurate_bullets_does_not_warn(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def two_bullets(self, request, model_profile):  # noqa: ANN001
        return ProviderResponse(
            content=(
                "- The sender cannot attend tomorrow's meeting.\n"
                "- The sender can send notes instead.\n"
                "Only 2 distinct facts were present in the source."
            ),
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=8,
        )

    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", two_bullets)
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/ask",
        json={
            "prompt": (
                "Summarise this email in three bullets: I cannot attend tomorrow's "
                "meeting but can send notes."
            ),
            "use_cache": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert not body["quality_warning"]
    assert "Follow up later" not in body["answer"]


def test_five_bullet_short_source_with_speculative_padding_warns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def padded_summary(self, request, model_profile):  # noqa: ANN001
        return ProviderResponse(
            content=(
                "- The sender cannot attend tomorrow's meeting.\n"
                "- The sender can send notes instead.\n"
                "- No replacement or proxy will attend.\n"
                "- The absence may impact the discussion.\n"
                "- Notes can be obtained by contacting the sender."
            ),
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=32,
        )

    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", padded_summary)
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/ask",
        json={
            "prompt": (
                "Summarise this email in five bullets: I cannot attend tomorrow's "
                "meeting but can send notes."
            ),
            "use_cache": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["quality_warning"]
    assert (
        "You asked for 5 bullets, but the source appears to contain only 2 distinct "
        "facts. The response may be padded."
    ) in body["quality_notes"]
    assert "Some bullets appear speculative rather than directly supported by the source." in body[
        "quality_notes"
    ]
    assert (
        "For source-grounded summaries, prefer fewer accurate bullets over invented bullets."
        in body["quality_notes"]
    )
    assert body["recommendation"]["route_kind"] == "local"
    assert body["recommendation"]["called_model"]
    assert body["recommendation"]["recommended_provider"] == "ollama"
    assert "switchboard ask '<same prompt>' --force-model ollama/gemma4:12b" in body[
        "suggested_next_step"
    ]
    premium_suggestion = "switchboard ask '<same prompt>' --backend claude-code"
    assert premium_suggestion in body["suggested_next_step"]


def test_requested_json_with_invalid_json_warns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def invalid_json(self, request, model_profile):  # noqa: ANN001
        return ProviderResponse(
            content="name: Vinay",
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=4,
        )

    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", invalid_json)
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/ask",
        json={"prompt": "Return JSON for this contact: Vinay.", "use_cache": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["quality_warning"]
    assert "You asked for JSON, but the response does not appear to be valid JSON." in body[
        "quality_notes"
    ]


def test_requested_one_sentence_with_multi_sentence_answer_warns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def two_sentences(self, request, model_profile):  # noqa: ANN001
        return ProviderResponse(
            content="Routing picks a model. It also records metadata.",
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=8,
        )

    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", two_sentences)
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/ask",
        json={"prompt": "Explain model routing in one sentence.", "use_cache": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["quality_warning"]
    assert any(
        note.startswith("You asked for one sentence, but the response appears to contain ")
        for note in body["quality_notes"]
    )


def test_manual_force_model_is_recommendation_only(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=False))

    response = client.post(
        "/personal/ask",
        json={
            "prompt": "Compare three pricing strategies for a solo SaaS launch.",
            "force_model": "manual/claude-web",
            "override_reason": "I want Claude for this.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "requires_confirmation"
    assert body["recommendation"]["route_kind"] == "manual_subscription"
    assert not body["recommendation"]["called_model"]
    assert body["recommendation"]["recommended_only"]


def test_manual_codex_recommendation_does_not_require_loaded_local_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(OllamaRuntimeService, "list_loaded_models", lambda self: set())
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/route",
        json={
            "prompt": "Review this public Python architecture and suggest tests.",
            "force_model": "manual/codex",
            "override_reason": "I want Codex for this code review.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommended_model"] == "manual/codex"
    assert body["route_kind"] == "manual_subscription"
    assert not body["selected_model_loaded"]
    assert body["requires_confirmation"]


def test_disabled_force_model_returns_clear_error(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/route",
        json={
            "prompt": "Debug this Python error.",
            "force_model": "ollama/qwen3-coder:30b",
        },
    )

    assert response.status_code == 400
    assert "disabled or unavailable" in response.json()["detail"]["message"]


def test_force_embedding_model_for_chat_is_rejected(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=True))

    response = client.post(
        "/personal/route",
        json={
            "prompt": "Summarise this public note.",
            "force_model": "ollama/nomic-embed-text",
        },
    )

    assert response.status_code == 400
    assert "not valid for chat responses" in response.json()["detail"]["message"]


def test_cloud_force_model_blocked_when_cloud_disabled(tmp_path: Path) -> None:
    client = personal_client(
        tmp_path,
        personal_config(tmp_path, ollama=False, openai=True, allow_cloud=False),
    )

    response = client.post(
        "/personal/route",
        json={"prompt": "Summarise this public note.", "force_model": "openai/gpt-4.1-mini"},
    )

    assert response.status_code == 400
    assert "allow_cloud=false" in response.json()["detail"]["message"]
    record = client.get("/personal/history").json()[0]
    assert record["user_forced_model"] == "openai/gpt-4.1-mini"
    assert record["override_safety_blocked"]


def test_override_telemetry_fields_are_populated(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=False))

    route = client.post(
        "/personal/route",
        json={
            "prompt": "Summarise this meeting note.",
            "force_model": "mock/medium",
            "override_reason": "Need a slightly stronger local draft.",
        },
    )
    history = client.get("/personal/history")

    assert route.status_code == 200
    record = history.json()[0]
    assert record["router_selected_model"] == "mock/small"
    assert record["user_forced_model"] == "mock/medium"
    assert record["final_selected_model"] == "mock/medium"
    assert record["override_used"]
    assert record["override_reason"] == "Need a slightly stronger local draft."


def test_feedback_preferred_model_is_stored_and_summarized(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=False))
    route = client.post("/personal/route", json={"prompt": "Summarise this note."})

    feedback = client.post(
        "/personal/feedback",
        json={
            "request_id": route.json()["request_id"],
            "rating": "wrong-route",
            "preferred_model": "manual/claude-web",
        },
    )
    usage = client.get("/personal/usage")

    assert feedback.status_code == 200
    assert feedback.json()["preferred_model"] == "manual/claude-web"
    assert usage.json()["feedback"]["wrong_route"] == 1
    assert usage.json()["feedback"]["preferred_models"]["manual/claude-web"] == 1


def test_previous_feedback_can_prefer_stronger_local_model(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=False))
    first = client.post(
        "/personal/route",
        json={"prompt": "Summarise this note.", "project": "demo"},
    )

    feedback = client.post(
        "/personal/feedback",
        json={
            "request_id": first.json()["request_id"],
            "rating": "too-weak",
            "preferred_model": "mock/medium",
        },
    )
    second = client.post(
        "/personal/route",
        json={"prompt": "Summarise this other note.", "project": "demo"},
    )

    assert feedback.status_code == 200
    assert second.status_code == 200
    body = second.json()
    assert first.json()["recommended_model"] == "mock/small"
    assert body["router_selected_model"] == "mock/small"
    assert body["recommended_model"] == "mock/medium"
    assert "PREVIOUS_FEEDBACK_CONSIDERED" in body["reason_codes"]


def test_previous_feedback_does_not_auto_apply_manual_premium(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=False))
    first = client.post(
        "/personal/route",
        json={"prompt": "Summarise this note.", "project": "demo"},
    )

    client.post(
        "/personal/feedback",
        json={
            "request_id": first.json()["request_id"],
            "rating": "too-weak",
            "preferred_model": "manual/claude-web",
        },
    )
    second = client.post(
        "/personal/route",
        json={"prompt": "Summarise this other note.", "project": "demo"},
    )

    assert second.status_code == 200
    body = second.json()
    assert body["recommended_model"] == "mock/small"
    assert "PREVIOUS_FEEDBACK_CONSIDERED" not in body["reason_codes"]


def test_savings_counts_saved_and_spent_premium_units(tmp_path: Path) -> None:
    client = personal_client(tmp_path, personal_config(tmp_path, ollama=False))

    original = client.post("/personal/route", json={"prompt": "Summarise this note."})
    client.post(
        "/personal/route",
        json={
            "prompt": "Compare three pricing strategies for a solo SaaS launch.",
            "force_model": "manual/claude-web",
            "original_request_id": original.json()["request_id"],
            "escalation_used": True,
        },
    )
    savings = client.get("/personal/savings?days=7")

    assert savings.status_code == 200
    body = savings.json()
    assert body["premium_units_saved"] >= 1
    assert body["premium_units_spent"] >= 1


def test_doctor_handles_missing_ollama_gracefully(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = personal_config(tmp_path, ollama=True)
    monkeypatch.setenv("ICP_PERSONAL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ICP_DATABASE_URL", f"sqlite:///{tmp_path / 'doctor.db'}")
    get_settings.cache_clear()

    def missing_ollama(*args, **kwargs):  # noqa: ANN002, ANN003
        raise FileNotFoundError("ollama")

    def failed_health(*args, **kwargs):  # noqa: ANN002, ANN003
        raise httpx.ConnectError("ollama unavailable")

    monkeypatch.setattr("httpx.get", failed_health)
    monkeypatch.setattr(subprocess, "run", missing_ollama)

    doctor_command(argparse.Namespace())

    output = capsys.readouterr().out
    assert "Ollama provider enabled: True" in output
    assert "Ollama installed models: unavailable" in output
    get_settings.cache_clear()


def test_loaded_models_command_handles_missing_ollama_gracefully(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = personal_config(tmp_path, ollama=True)
    monkeypatch.setenv("ICP_PERSONAL_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ICP_DATABASE_URL", f"sqlite:///{tmp_path / 'loaded.db'}")
    get_settings.cache_clear()

    def missing_ollama(*args, **kwargs):  # noqa: ANN002, ANN003
        raise FileNotFoundError("ollama")

    monkeypatch.setattr(OllamaRuntimeService, "list_loaded_models", ORIGINAL_LIST_LOADED_MODELS)
    monkeypatch.setattr(subprocess, "run", missing_ollama)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("offline")),
    )

    loaded_models_command(argparse.Namespace(format="text"))

    output = capsys.readouterr().out
    assert "Ollama provider enabled: True" in output
    assert "Loaded Ollama models: none detected" in output
    get_settings.cache_clear()


def test_bench_models_uses_mock_provider_without_real_ollama(
    client: TestClient,
) -> None:
    service = PersonalSwitchboardService(client.app.state.container)
    results = asyncio.run(service.bench_models())

    assert results
    assert any(result["model"] == "mock/small" for result in results)


def test_bench_models_unloads_ollama_models_after_each_benchmark(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = personal_client(
        tmp_path,
        personal_config(tmp_path, ollama=True, unload_after_benchmark=True),
    )
    unloaded: list[str] = []

    async def fake_complete_chat(self, request, model_profile):  # noqa: ANN001
        return ProviderResponse(
            content="ok",
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=1,
        )

    def fake_unload(self, model_id: str) -> RuntimeCommandResult:  # noqa: ANN001
        unloaded.append(model_id)
        return RuntimeCommandResult(True, f"Unloaded {model_id}.", model_id)

    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", fake_complete_chat)
    monkeypatch.setattr(OllamaRuntimeService, "unload_model", fake_unload)

    service = PersonalSwitchboardService(client.app.state.container)
    results = asyncio.run(service.bench_models())

    ollama_results = [result for result in results if result["provider"] == "ollama"]
    assert ollama_results
    assert unloaded == [result["model"] for result in ollama_results]
    assert all(result["unloaded_after_benchmark"] for result in ollama_results)
