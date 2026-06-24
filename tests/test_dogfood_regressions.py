"""Regression tests distilled from a real dogfooding session (2026-06-11).

Each test encodes a failure observed while using the product:

1. Live-data questions (news/weather/stocks) burned subscription quota just to
   produce a disclaimer, and a local model fabricated news headlines.
2. Claude Code asked the user to grant a tool permission that can never be
   granted in non-interactive mode.
3. Claude answered personal questions with "I'm built for software
   engineering" persona leakage.
4. "How to get out of depression?" was routed to a subscription backend even
   with private mode enabled.
"""

from __future__ import annotations

from pathlib import Path

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.backends.cli_agents import ClaudeCodeCliAdapter
from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.core.config import Settings
from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    SwitchboardRequest,
    SwitchboardResponse,
)
from switchboard.app.services.container import build_container
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.storage.db import create_db_engine, init_db

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "switchboard" / "app" / "static"


class FakeAdapter(AgentAdapter):
    def __init__(self, name: str, *, cost_type: BackendCostType = BackendCostType.LOCAL) -> None:
        self.name = name
        self.cost_type = cost_type
        self.prompts: list[str] = []

    def is_available(self) -> bool:
        return True

    def availability(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=True, cost_type=self.cost_type)

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        self.prompts.append(request.prompt)
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            content=f"{self.name} answered",
            latency_ms=5,
            success=True,
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


class DownAdapter(FakeAdapter):
    """A configured backend that is currently unavailable (e.g. Ollama not
    running). Production registries always contain adapters, so outage edge
    cases go through is_available(), never through a missing adapter."""

    def is_available(self) -> bool:
        return False

    def availability(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=False, cost_type=self.cost_type)


def full_registry() -> tuple[BackendRegistry, dict[str, FakeAdapter]]:
    adapters = {
        "ollama": FakeAdapter("ollama"),
        "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": FakeAdapter("claude-code", cost_type=BackendCostType.SUBSCRIPTION),
    }
    return BackendRegistry(adapters), adapters


def make_service(tmp_path: Path, registry: BackendRegistry, **kwargs) -> SwitchboardCoreService:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'dogfood.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)
    # Tests must not depend on the developer's live personal.yaml toggles.
    container.personal_config.preferences.claude_code_web_search = False
    return SwitchboardCoreService(
        registry=registry,
        metrics=container.backend_metrics_repository,
        container=container,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Live-data questions: never spend premium, never fabricate
# ---------------------------------------------------------------------------


def test_claude_web_search_route_instructs_claude_to_actually_search(tmp_path: Path) -> None:
    # Dogfood: "How's the weather in california" routed to Claude (web search
    # enabled) but Claude answered "I don't have access to live weather data"
    # from memory — premium quota spent on a non-answer. When we route to
    # Claude FOR web search, the context must direct it to search.
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)
    service.container.personal_config.preferences.claude_code_web_search = True

    response = service.ask("how's the weather in california", new_session=True)

    assert response.backend == "claude-code"
    prompt = adapters["claude-code"].prompts[-1]
    assert "use WebSearch now" in prompt
    # The generic "no provider configured" honesty fact must NOT be sent on
    # this route — it is what nudged Claude into the no-answer disclaimer.
    assert "no live-data provider configured" not in prompt


def test_corporate_event_questions_are_grounded_or_guarded(tmp_path: Path) -> None:
    # Dogfood (2026-06-12, SpaceX IPO day of all days): "did spacex go
    # public" was answered with a fabricated 2021 IPO at $102/share. The
    # question must be treated as live info: news-grounded when a provider
    # is configured, honesty-guarded otherwise — never answered bare.
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask("did spacex go public", new_session=True)

    assert response.backend == "ollama"
    prompt = adapters["ollama"].prompts[-1]
    # Either live headlines were fetched (news tool) or the anti-fabrication
    # instruction is present; both prevent an invented IPO.
    assert ("Live headlines fetched" in prompt) or ("Do not invent specific" in prompt)


def test_identity_question_gets_grounded_identity_facts(tmp_path: Path) -> None:
    # Dogfood: "who made switchboard?" was answered with a fabricated vendor
    # ("developed by Meta AI"). The context contract must carry trusted
    # identity facts so the model never invents one.
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask("who made switchboard?", new_session=True)

    assert response.success
    prompt = adapters[response.backend].prompts[-1]
    assert "local-first personal AI router" in prompt
    assert "never invent a company" in prompt


def test_unconfigured_stock_question_routes_local(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask("what is the stock price of Amazon?")

    assert response.backend == "ollama"
    assert response.cost_type == BackendCostType.LOCAL
    assert adapters["claude-code"].prompts == []
    assert adapters["codex"].prompts == []


def test_unconfigured_news_question_routes_local_with_honesty_fact(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask("give me latest news of india?")

    assert response.backend == "ollama"
    prompt = adapters["ollama"].prompts[-1]
    assert "Do not invent specific" in prompt
    assert "cannot access live data" in prompt


def test_unconfigured_weather_question_routes_local(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask("what is the weather in Delhi?")

    assert response.backend == "ollama"
    assert "Do not invent specific" in adapters["ollama"].prompts[-1]


def test_live_data_routes_to_claude_when_web_search_enabled(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)
    service.container.personal_config.preferences.claude_code_web_search = True

    response = service.ask("give me latest news of india?")

    assert response.backend == "claude-code"
    assert "web search is enabled" in response.routing_reason


def test_officeholder_questions_are_detected_as_latest_info(tmp_path: Path) -> None:
    """Dogfood regression: "what is the name of us president" was answered
    from stale training data instead of being treated as current info."""
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    for prompt in (
        "what is the name of us president",
        "who is the president of the united states?",
        "who's the prime minister of india",
        "who is the ceo of microsoft?",
    ):
        response = service.ask(prompt, new_session=True)
        record = service.metrics_list(limit=1)[0]
        assert "latest_info" in record.metadata["detected_capabilities"], prompt
        # Without a provider: local model + anti-fabrication instruction.
        assert response.backend == "ollama", prompt
        assert "Do not invent specific" in adapters["ollama"].prompts[-1]


def test_officeholder_questions_use_claude_when_web_search_enabled(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)
    service.container.personal_config.preferences.claude_code_web_search = True

    response = service.ask("who is the us president right now?")

    assert response.backend == "claude-code"
    assert "web search is enabled" in response.routing_reason


def test_natural_date_phrasings_are_tool_grounded(tmp_path: Path) -> None:
    """Dogfood regression: "what's the date today" got a fabricated 2023 date
    because the capability detector only matched a few exact phrasings."""
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    for prompt in (
        "what's the date today",
        "whats the date",
        "what is the date today?",
        "what day is it?",
        "date today",
        "what's the time",
    ):
        response = service.ask(prompt, new_session=True)
        assert response.success, prompt
        record = service.metrics_list(limit=1)[0]
        assert record.metadata.get("grounded_by_tool") or record.metadata.get(
            "answered_by_tool"
        ), f"{prompt!r} was not grounded by the time tool"


def test_time_questions_still_short_circuit_via_tool(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask("what time is it in Tokyo right now?")

    assert response.success
    # Deterministic tool grounding: the answer must not be fabricated by a
    # subscription backend; the model that answers receives trusted facts.
    all_premium_prompts = adapters["claude-code"].prompts + adapters["codex"].prompts
    for prompt in all_premium_prompts:
        assert "<trusted_facts>" in prompt


# ---------------------------------------------------------------------------
# Privacy: sensitive prompts stay local instead of being blocked or leaked
# ---------------------------------------------------------------------------


def test_depression_question_routes_local_under_private_mode(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask("how to get out of depression?")

    assert response.backend == "ollama"
    assert response.success
    assert "Private mode" in response.routing_reason
    assert adapters["claude-code"].prompts == []


def test_relationship_question_routes_local_under_private_mode(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask("I need advice about my relationship problems")

    assert response.backend == "ollama"
    assert adapters["claude-code"].prompts == []


def test_sensitive_prompt_blocked_when_no_local_backend(tmp_path: Path) -> None:
    adapters = {
        "claude-code": FakeAdapter("claude-code", cost_type=BackendCostType.SUBSCRIPTION),
    }
    service = make_service(tmp_path, BackendRegistry(adapters))

    response = service.ask("how to get out of depression?")

    # Without a local model, the prompt must be blocked, never silently leaked.
    assert not response.success
    assert adapters["claude-code"].prompts == []
    assert "private mode" in (response.error_message or "").lower()


def test_forced_backend_with_sensitive_prompt_still_blocked(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask("how to get out of depression?", backend="claude-code")

    assert not response.success
    assert adapters["claude-code"].prompts == []


# ---------------------------------------------------------------------------
# Persona and permission-prompt leakage
# ---------------------------------------------------------------------------


def test_context_contract_forbids_coding_persona_and_permission_requests(
    tmp_path: Path,
) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    service.ask("Review this architecture for a model router")

    prompt = adapters["claude-code"].prompts[-1]
    assert "general-purpose personal assistant" in prompt
    assert "coding-only" in prompt
    assert "Never ask the user to grant tool permissions" in prompt


def test_routing_still_sends_coding_to_codex(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask("how to reverse a linked list in java")

    assert response.backend == "codex"


def test_greetings_stay_local(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    assert service.ask("hi").backend == "ollama"
    assert service.ask("how are you doing?").backend == "ollama"


# ---------------------------------------------------------------------------
# Session 3 dogfood: web-dev prompts to Codex; follow-ups stay sticky
# ---------------------------------------------------------------------------


def test_web_project_prompt_routes_to_codex(tmp_path: Path) -> None:
    """Dogfood regression: "create me a project that has a login page" went to
    a 3B local model, which produced SQL-injectable code with md5 passwords."""
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    for prompt in (
        "create me a project that has a login page with personal images stored",
        "build me an app with a signup page",
        "make a website with html and css",
    ):
        response = service.ask(prompt, new_session=True)
        assert response.backend == "codex", prompt


def test_short_followup_sticks_with_previous_backend(tmp_path: Path) -> None:
    """Dogfood regression: "Can you do it yourself" was re-routed blindly
    instead of continuing with the model from the previous turn."""
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    first = service.ask("create me a project that has a login page", new_session=True)
    assert first.backend == "codex"

    second = service.ask("Can you do it yourself", session_id=first.session_id)

    assert second.backend == "codex"
    assert "continuing with the same model" in second.routing_reason.lower()


def test_long_new_topic_is_not_sticky(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    first = service.ask("create me a project with a login page", new_session=True)
    assert first.backend == "codex"

    # A full new question with its own signal must be routed on its merits.
    second = service.ask(
        "summarize this short private note about my weekend plans",
        session_id=first.session_id,
    )
    assert second.backend == "ollama"


# ---------------------------------------------------------------------------
# Session 4 dogfood (2026-06-12): unknown tasks fail CLOSED to the local
# model, and conversational weather phrasing gets the live-data policy
# ---------------------------------------------------------------------------


def test_unknown_smalltalk_defaults_to_free_local_model(tmp_path: Path) -> None:
    """Tester finding: "im running a 10k saturday..." burned subscription
    quota because unknown tasks failed open to Claude. Local-first is the
    product thesis: premium is a deliberate exception, never the default."""
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask(
        "im running a 10k saturday, how many miles is that again lol",
        new_session=True,
    )

    assert response.backend == "ollama"
    assert response.cost_type == BackendCostType.LOCAL
    assert "local-first default" in response.routing_reason
    assert adapters["claude-code"].prompts == []
    assert adapters["codex"].prompts == []


def test_keyword_free_sensitive_prompt_stays_local_by_default(tmp_path: Path) -> None:
    """Tester finding: a private disclosure with no privacy keywords reached a
    subscription backend whenever the learned sensitivity escalator was
    inactive. The deterministic floor must be safe on its own: prompts with
    no routable signal never leave the box."""
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask(
        "i havent been sleeping. i lost my job and still havent told my wife",
        new_session=True,
    )

    assert response.backend == "ollama"
    assert adapters["claude-code"].prompts == []
    assert adapters["codex"].prompts == []


def test_unknown_task_without_local_backend_falls_back_to_claude(tmp_path: Path) -> None:
    adapters = {
        "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": FakeAdapter("claude-code", cost_type=BackendCostType.SUBSCRIPTION),
    }
    service = make_service(tmp_path, BackendRegistry(dict(adapters)))

    response = service.ask("what should I do next?", new_session=True)

    assert response.backend == "claude-code"
    assert "fell back" in response.routing_reason


def test_conversational_weather_routes_local_with_honesty_instruction(
    tmp_path: Path,
) -> None:
    """Tester finding: "is it gonna rain in seattle tomorrow?" escaped the
    weather detector, routed to a subscription backend with no
    anti-fabrication instruction."""
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    for prompt in (
        "is it gonna rain in seattle tomorrow?",
        "will there be snow this weekend?",
        "chance of rain tomorrow in portland",
        "is it going to rain tonight",
    ):
        response = service.ask(prompt, new_session=True)
        record = service.metrics_list(limit=1)[0]
        assert "weather" in record.metadata["detected_capabilities"], prompt
        assert response.backend == "ollama", prompt
        assert "Do not invent specific" in adapters["ollama"].prompts[-1], prompt


def test_conversational_weather_uses_claude_web_search_when_enabled(
    tmp_path: Path,
) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)
    service.container.personal_config.preferences.claude_code_web_search = True

    response = service.ask("is it gonna rain in seattle tomorrow?", new_session=True)

    assert response.backend == "claude-code"
    prompt = adapters["claude-code"].prompts[-1]
    assert "use WebSearch now" in prompt
    assert "no live-data provider configured" not in prompt


# ---------------------------------------------------------------------------
# Session 5 dogfood (2026-06-12): filler greetings must not preempt coding,
# sticky follow-ups inherit the live-data honesty fact, and date arithmetic
# is calculator-grade tool work
# ---------------------------------------------------------------------------


def test_filler_greeting_does_not_preempt_coding_detection(tmp_path: Path) -> None:
    """Tester finding (N7): "hey can you debug this python traceback" routed
    to Ollama because filler greetings matched the local-keyword check before
    coding keywords were ever consulted. Voice-style prompts starting with
    hey/ok/quick must still reach the coding model."""
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    for prompt in (
        "hey can you debug this python traceback for me: KeyError in my loop",
        "ok quick question - whats wrong with my javascript promise chain",
        "hey, fix this failing pytest run",
        "thanks! now refactor this function to be iterative",
    ):
        response = service.ask(prompt, new_session=True)
        assert response.backend == "codex", prompt


def test_filler_greeting_does_not_preempt_reasoning_detection(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask(
        "ok, compare postgres vs mongodb tradeoffs for me",
        new_session=True,
    )

    assert response.backend == "claude-code"


def test_simple_requests_with_fillers_still_stay_local(tmp_path: Path) -> None:
    """Counter-case for N7: genuine simple/local tasks phrased with the same
    fillers must STILL go to the free local model."""
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    for prompt in (
        "hey, rewrite this paragraph to sound friendlier",
        "quick summary of this note please",
        "hey",
        "ok thanks!",
        "good morning! what's up",
    ):
        response = service.ask(prompt, new_session=True)
        assert response.backend == "ollama", prompt
        assert response.cost_type == BackendCostType.LOCAL, prompt


def test_sticky_followup_to_live_data_turn_keeps_honesty_fact(tmp_path: Path) -> None:
    """Tester finding (N5b): "what is tesla trading at" -> "and microsoft?"
    stuck with Ollama (correct) but dropped the live-data honesty fact,
    leaving the local model free to invent an MSFT price."""
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    first = service.ask("what is tesla stock trading at right now", new_session=True)
    assert first.backend == "ollama"
    assert "Do not invent specific" in adapters["ollama"].prompts[-1]

    second = service.ask("and microsoft?", session_id=first.session_id)

    assert second.backend == "ollama"
    assert "continuing with the same model" in second.routing_reason.lower()
    assert "Do not invent specific" in adapters["ollama"].prompts[-1]
    assert "cannot access live data" in adapters["ollama"].prompts[-1]

    # Chained elliptical follow-ups stay covered too.
    third = service.ask("and apple?", session_id=first.session_id)
    assert third.backend == "ollama"
    assert "Do not invent specific" in adapters["ollama"].prompts[-1]


def test_compression_preserves_trusted_facts_and_request_in_long_sessions(tmp_path: Path) -> None:
    """Tester finding (round 6, must-fix 1): with compression enabled, a long
    session pushed the assembled context over the threshold and the whole-text
    heuristic deleted <trusted_facts> and the honesty directives — the exact
    blocks that stop a local model from fabricating prices and headlines.
    Compression must only touch <recent_conversation>; grounded truth and the
    user's request survive verbatim."""
    from switchboard.app.services.compression_layer import HeadroomCompressionLayer

    registry, adapters = full_registry()
    service = make_service(
        tmp_path,
        registry,
        compression=HeadroomCompressionLayer(threshold_tokens=400),
    )

    first = service.ask(
        "lets chat about my week. " + ("interesting detail " * 60),
        new_session=True,
    )
    for i in range(6):
        service.ask(
            f"more thoughts {i}. " + ("interesting detail " * 60),
            session_id=first.session_id,
        )

    response = service.ask("give me latest news of india?", session_id=first.session_id)

    assert response.backend == "ollama"
    prompt = adapters["ollama"].prompts[-1]
    # Compression really ran on the conversation history...
    assert "Heuristic compression" in prompt
    assert "<recent_conversation>" in prompt
    # ...but every grounded-truth block survived verbatim.
    assert "Do not invent specific" in prompt
    assert "cannot access live data" in prompt
    assert "never invent a company" in prompt  # instruction preamble
    assert (
        "<current_user_request>\ngive me latest news of india?\n</current_user_request>" in prompt
    )


def test_sticky_followup_to_non_live_turn_gets_no_honesty_fact(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    first = service.ask("create me a project that has a login page", new_session=True)
    assert first.backend == "codex"

    second = service.ask("Can you do it yourself", session_id=first.session_id)

    assert second.backend == "codex"
    assert "Do not invent specific" not in adapters["codex"].prompts[-1]


def test_date_arithmetic_is_tool_grounded(tmp_path: Path) -> None:
    """Tester finding (N6): "what date is 45 days from today" was neither
    tool-grounded nor dated, so the model had to guess today's date. Date
    arithmetic is calculator-grade work for the deterministic TimeTool."""
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    for prompt in (
        "what date is 45 days from today",
        "what will the date be in 30 days",
        "15 days from today, what will the date be",
    ):
        response = service.ask(prompt, new_session=True)
        assert response.success, prompt
        assert response.backend == "ollama", prompt
        record = service.metrics_list(limit=1)[0]
        assert "current_date" in record.metadata["detected_capabilities"], prompt
        assert record.metadata.get("grounded_by_tool"), prompt
        assert "days from today is" in adapters["ollama"].prompts[-1], prompt


# ---------------------------------------------------------------------------
# Round 4 tester findings (2026-06-12): backend-outage edge cases
# ---------------------------------------------------------------------------


def test_health_disclosure_never_leaks_to_subscription_when_ollama_down(
    tmp_path: Path,
) -> None:
    """Tester finding (privacy leak): "i think i have an STD what do i do"
    with Ollama down was sent verbatim to Codex. The learned sensitivity
    escalator uses Ollama embeddings, so it is dead exactly when Ollama is
    down — the keyword floor alone must catch physical-health disclosures,
    and the fallback path must refuse rather than leak."""
    for prompt in (
        "i think i have an STD what do i do",
        "i was diagnosed with herpes last week and i feel ashamed",
        "i just found out im pregnant and dont know what to do",
        "my test results came back and im scared to open them",
        "i think i have a kidney infection, should i go to the er",
    ):
        adapters = {
            "ollama": DownAdapter("ollama"),
            "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
            "claude-code": FakeAdapter("claude-code", cost_type=BackendCostType.SUBSCRIPTION),
        }
        service = make_service(tmp_path, BackendRegistry(dict(adapters)))

        response = service.ask(prompt, new_session=True)

        assert not response.success, prompt
        assert "private mode" in (response.error_message or "").lower(), prompt
        assert "local model unavailable" in (response.error_message or "").lower(), prompt
        assert adapters["codex"].prompts == [], prompt
        assert adapters["claude-code"].prompts == [], prompt


def test_health_disclosure_routes_local_when_ollama_up(tmp_path: Path) -> None:
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask("i think i have an STD what do i do", new_session=True)

    assert response.success
    assert response.backend == "ollama"
    assert "Private mode" in response.routing_reason
    assert adapters["codex"].prompts == []
    assert adapters["claude-code"].prompts == []


def test_cpp_std_vocabulary_still_routes_to_codex(tmp_path: Path) -> None:
    """Guard for the new STD/STI health keywords: C++/stdlib vocabulary must
    stay a coding signal, never a health disclosure."""
    registry, adapters = full_registry()
    service = make_service(tmp_path, registry)

    response = service.ask(
        "fix this c++ code: using namespace std; int main() { std::cout << 1; }",
        new_session=True,
    )

    assert response.backend == "codex"
    assert adapters["codex"].prompts


def test_tool_grounded_answer_returned_when_all_backends_down(tmp_path: Path) -> None:
    """Tester finding (data loss): with every backend down, a tool-grounded
    request returned "Ollama is unavailable." even though the time tool had
    already computed the answer. The trusted grounding must be returned
    directly: free, sanitized, honest reason, stored in the session."""
    adapters = {
        "ollama": DownAdapter("ollama"),
        "codex": DownAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": DownAdapter("claude-code", cost_type=BackendCostType.SUBSCRIPTION),
    }
    service = make_service(tmp_path, BackendRegistry(dict(adapters)))

    response = service.ask("what time is it in tokyo", new_session=True)

    assert response.success
    assert "Tokyo" in (response.content or "")
    assert response.cost_type == BackendCostType.LOCAL
    assert response.estimated_cost_usd == 0.0
    assert "trusted grounding directly" in response.routing_reason
    assert response.message_id  # the answer is stored as an assistant turn
    record = service.metrics_list(limit=1)[0]
    assert record.success
    assert record.metadata.get("grounded_by_tool") is True


def test_forced_backend_outage_still_returns_honest_error(tmp_path: Path) -> None:
    """The direct tool answer must not override an explicit backend choice:
    forcing a down backend keeps the honest unavailability error."""
    adapters = {
        "ollama": DownAdapter("ollama"),
        "codex": DownAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": DownAdapter("claude-code", cost_type=BackendCostType.SUBSCRIPTION),
    }
    service = make_service(tmp_path, BackendRegistry(dict(adapters)))

    response = service.ask("what time is it in tokyo", new_session=True, backend="codex")

    assert not response.success
    assert "unavailable" in (response.error_message or "").lower()


def test_tool_grounded_fallback_reason_reflects_grounding(tmp_path: Path) -> None:
    """Tester finding (minor): when a tool-grounded request falls back off
    Ollama to a subscription model, the reason claimed "local-first default
    keeps it on the free local model" — misleading. It must say the premium
    model is only formatting trusted facts."""
    adapters = {
        "ollama": DownAdapter("ollama"),
        "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": FakeAdapter("claude-code", cost_type=BackendCostType.SUBSCRIPTION),
    }
    service = make_service(tmp_path, BackendRegistry(dict(adapters)))

    response = service.ask("what time is it in tokyo", new_session=True)

    assert response.success
    assert "Tool-grounded request" in response.routing_reason
    assert "trusted facts" in response.routing_reason
    assert "local-first default" not in response.routing_reason
    # The premium model still receives the grounding it is formatting.
    assert "<trusted_facts>" in adapters["claude-code"].prompts[-1]


# ---------------------------------------------------------------------------
# Claude Code adapter: WebSearch flag
# ---------------------------------------------------------------------------


def test_claude_adapter_omits_web_search_by_default() -> None:
    command = ClaudeCodeCliAdapter().build_command(
        SwitchboardRequest(request_id="req_t", prompt="hello")
    )
    assert "--allowedTools=WebSearch" not in command
    assert "--disallowedTools=Edit,Write,Bash" in command


def test_claude_adapter_adds_web_search_when_enabled() -> None:
    command = ClaudeCodeCliAdapter(allow_web_search=True).build_command(
        SwitchboardRequest(request_id="req_t", prompt="latest news")
    )
    assert "--allowedTools=WebSearch" in command
    assert "--disallowedTools=Edit,Write,Bash" in command


# ---------------------------------------------------------------------------
# Frontend safety: the markdown renderer must escape before innerHTML
# ---------------------------------------------------------------------------


def test_frontend_markdown_renderer_escapes_html() -> None:
    javascript = (STATIC / "app.js").read_text(encoding="utf-8")
    # Every innerHTML sink must be fed by renderBlocks, which escapes first.
    assert javascript.count("innerHTML") == 1
    assert "renderBlocks(plain)" in javascript
    assert "escapeHtml" in javascript
    # Links restricted to http(s); no javascript: URLs can be injected.
    assert "https?:\\/\\/" in javascript
    # Code content uses textContent (never innerHTML).
    assert "codeEl.textContent = code" in javascript
