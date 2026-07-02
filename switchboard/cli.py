from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx

from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.core.config import (
    DEFAULT_CONFIG_FILES,
    Settings,
    get_settings,
    packaged_config_path,
    user_config_dir,
)
from switchboard.app.models.backends import BackendRouteDecision, SwitchboardResponse
from switchboard.app.models.catalogue import ModelCatalogue
from switchboard.app.models.personal import (
    FeedbackCreate,
    PersonalConfig,
    PersonalMemoryCreate,
    PersonalPromptRequest,
    PersonalRouteResponse,
)
from switchboard.app.services.container import build_container
from switchboard.app.services.core_factory import (
    build_configured_core_service,
    build_semantic_memory,
)
from switchboard.app.services.local_runtime import OllamaRuntimeService
from switchboard.app.services.model_recommendations import (
    apply_local_model_pack,
    detect_total_ram_bytes,
    recommend_local_model_pack,
)
from switchboard.app.services.personal_switchboard import (
    PersonalRoutingError,
    PersonalSwitchboardService,
)
from switchboard.app.services.provider_status import (
    finance_provider_status,
    news_provider_status,
    web_provider_status,
)
from switchboard.app.services.semantic_memory import EmbeddingUnavailableError
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.storage.db import create_db_engine, init_db
from switchboard.evals.quality_bench import (
    DEFAULT_CONDITIONS,
    OllamaJudge,
    QualityBenchRunner,
)
from switchboard.evals.quality_bench import (
    report_to_text as quality_report_to_text,
)
from switchboard.evals.real_providers import RealProviderRunner
from switchboard.evals.real_smoke import RealSmokeRunner
from switchboard.evals.reports import report_to_json, report_to_text, write_report
from switchboard.evals.runner import EvalRunner
from switchboard.evals.types import EvalCase


def build_service() -> PersonalSwitchboardService:
    settings: Settings = get_settings()
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    return PersonalSwitchboardService(build_container(settings, engine))


def build_core_service(
    *,
    router_mode: str | None = None,
    compression: bool | None = None,
    semantic_memory: bool | None = None,
) -> SwitchboardCoreService:
    settings: Settings = get_settings()
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)
    return build_configured_core_service(
        container,
        cwd=Path.cwd(),
        router_mode=router_mode,
        compression=compression,
        semantic_memory=semantic_memory,
    )


def build_eval_runner(*, mock: bool = True) -> EvalRunner:
    return EvalRunner(settings=get_settings(), cwd=Path.cwd(), mock=mock)


def build_real_smoke_runner(
    *,
    timeout_s: int = 90,
    case_timeouts: dict[str, int] | None = None,
) -> RealSmokeRunner:
    return RealSmokeRunner(
        settings=get_settings(),
        cwd=Path.cwd(),
        timeout_s=timeout_s,
        case_timeouts=case_timeouts,
    )


def print_json(payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    print(json.dumps(payload, indent=2))


USER_REASON_LABELS = {
    "CACHE_HIT": "Cached routing decision reused",
    "CLASSIFIER_CODE_HINT": "Coding/debugging task",
    "CLASSIFIER_HIGH_COMPLEXITY": "High complexity",
    "CLASSIFIER_LOW_COMPLEXITY": "Low complexity",
    "CLASSIFIER_MEDIUM_COMPLEXITY": "Medium complexity",
    "CLASSIFIER_REGULATED_HINT": "Sensitive content detected",
    "CLASSIFIER_SUMMARY_HINT": "Simple summary",
    "CLOUD_DISABLED": "Cloud disabled",
    "CODING_TASK": "Coding/debugging task",
    "COMPLEX_REASONING_REQUEST": "Complex reasoning",
    "COLD_START_POSSIBLE": "Ollama cold start may happen",
    "EMBEDDING_MODEL_SKIPPED_FOR_CHAT": "Embedding-only model skipped for chat",
    "FINANCIAL_PLANNING_DETECTED": "Regulated financial content detected",
    "BALANCED_RUNTIME_MODE_ACTIVE": "Balanced runtime mode",
    "HOT_MODEL_GOOD_ENOUGH": "Already-loaded model is good enough",
    "HOT_MODEL_REUSED": "Already-loaded local model reused",
    "LEGAL_SENSITIVE_CONTENT": "Sensitive legal content detected",
    "LOCAL_MODEL_PREFERRED": "Local model preferred",
    "LOCAL_OLLAMA_MODEL_SELECTED": "Local Ollama model selected",
    "LOW_CONFIDENCE_SAFE_LOCAL_ROUTE": "Ambiguous request kept local",
    "LOW_LATENCY_MODE_ACTIVE": "Low-latency runtime mode",
    "MANUAL_PREMIUM_RECOMMENDED": "Manual premium recommendation only",
    "MEDICAL_SENSITIVE_CONTENT": "Sensitive medical content detected",
    "MEMORY_SAVER_MODE_ACTIVE": "Memory-saver runtime mode",
    "MODEL_SWITCH_AVOIDED": "Cold model switch avoided",
    "MOCK_FALLBACK_USED": "Mock fallback used",
    "OLLAMA_CODING_MODEL_SELECTED": "Ollama coding model selected",
    "OLLAMA_FAST_MODEL_SELECTED": "Ollama fast model selected",
    "OLLAMA_GENERAL_MODEL_SELECTED": "Ollama general model selected",
    "OLLAMA_MODEL_ALREADY_LOADED": "Selected Ollama model already loaded",
    "OLLAMA_MODEL_NOT_LOADED": "Selected Ollama model is cold",
    "OLLAMA_REASONING_MODEL_SELECTED": "Ollama reasoning model selected",
    "PERSONAL_AMBIGUOUS_REQUEST_KEPT_LOCAL": "Ambiguous request kept local",
    "PERSONAL_CLOUD_DISABLED_PREMIUM_RECOMMENDATION_ONLY": (
        "Cloud disabled, so premium tool is recommendation-only"
    ),
    "PERSONAL_CODING_LOCAL_MODEL_PREFERRED": "Local coding model preferred",
    "PERSONAL_COMPLEX_REASONING_LOCAL_MODEL_PREFERRED": "Local reasoning model preferred",
    "PERSONAL_LOCAL_FIRST_ENABLED": "Local-first enabled",
    "PERSONAL_PRIVATE_MODE_CLOUD_BLOCKED": "Cloud blocked for private content",
    "PERSONAL_SCARCE_MODEL_NOT_CALLED_AUTOMATICALLY": "No web automation performed",
    "PERSONAL_SENSITIVE_SIMPLE_TASK_KEPT_LOCAL": "Sensitive simple task kept local",
    "PERSONAL_SIMPLE_TASK_ROUTED_TO_FREE_LOCAL_MODEL": "Premium model avoided",
    "PRIVATE_MODE_ENABLED": "Private mode enabled",
    "PRIVATE_PERSONAL_CONTENT": "Private personal content detected",
    "PREVIOUS_FEEDBACK_CONSIDERED": "Previous feedback considered",
    "PROMPT_INJECTION_ATTEMPT": "Request tried to override privacy settings; ignored",
    "SCARCE_MODEL_REQUIRES_CONFIRMATION": (
        "Confirmation required before using scarce premium tool"
    ),
    "SECURITY_ROUTING_OVERRIDE_ATTEMPT": "Routing override ignored",
    "SENSITIVE_BUT_SIMPLE_TASK": "Sensitive but simple task",
    "SENSITIVITY_DOES_NOT_IMPLY_FRONTIER": "Sensitivity controls data handling, not model size",
    "SIMPLE_SUMMARISATION": "Simple summary",
    "SPECIALIST_MODEL_SWITCH_JUSTIFIED": "Specialist model is worth loading",
    "USER_ALLOW_CLOUD_ONCE": "Cloud allowed once by user",
    "USER_FORCE_MODEL_APPLIED": "User-selected model applied",
    "USER_FORCE_MODEL_REQUESTED": "User requested a specific model",
}


def user_facing_reasons(response: PersonalRouteResponse) -> list[str]:
    reasons: list[str] = []
    seen: set[str] = set()
    for code in response.reason_codes:
        label = USER_REASON_LABELS.get(code)
        if label and label not in seen:
            reasons.append(label)
            seen.add(label)
    if not reasons:
        reasons.append("Local-first routing applied")
    return reasons


def next_step_for_route(response: PersonalRouteResponse) -> str:
    reason_codes = set(response.reason_codes)
    if "PROMPT_INJECTION_ATTEMPT" in reason_codes:
        return (
            "Request tried to override privacy settings; ignored. Private mode remains "
            "active; no cloud or manual provider was called."
        )
    if response.route_kind == "manual_subscription":
        return (
            "Manual premium recommendation only. Switchboard did not call the provider; "
            "copy the ready-to-paste prompt only if the task is worth scarce usage."
        )
    if response.requires_confirmation:
        return "Confirmation is required before this scarce route can be called."
    if response.route_kind == "cloud_api":
        return "Cloud API route is callable only because allow_cloud is enabled for this profile."
    if "MODEL_SWITCH_AVOIDED" in reason_codes:
        return (
            f"Using {response.recommended_model} because it is already loaded and good "
            "enough for this request."
        )
    if "SPECIALIST_MODEL_SWITCH_JUSTIFIED" in reason_codes:
        return (
            f"Loading {response.recommended_model} is justified because the task needs "
            "a specialist model."
        )
    if "PERSONAL_PRIVATE_MODE_CLOUD_BLOCKED" in reason_codes:
        if "PERSONAL_SENSITIVE_SIMPLE_TASK_KEPT_LOCAL" in reason_codes:
            return (
                "Private medical or sensitive content detected, so cloud is blocked. "
                "Sensitive content affects where data may go; complexity affects model strength."
            )
        return (
            "Private mode blocked cloud routing; use the local result or redact before "
            "any manual step."
        )
    if "LOW_CONFIDENCE_SAFE_LOCAL_ROUTE" in reason_codes:
        return "Ambiguous prompt kept local; add task details if the first answer is too weak."
    if response.called_model:
        return f"{provider_call_status(response)}; review quality before trying premium."
    return "Local/mock route is ready; no premium subscription or cloud API is used."


def provider_display_name(response: PersonalRouteResponse) -> str:
    if response.recommended_provider == "ollama":
        return "Ollama"
    if response.recommended_provider == "mock":
        return "Demo mock"
    if response.route_kind == "cloud_api":
        return response.recommended_provider
    if response.route_kind == "manual_subscription":
        return "Manual recommendation"
    return response.recommended_provider


def provider_call_status(response: PersonalRouteResponse) -> str:
    if response.route_kind == "manual_subscription":
        return "Manual recommendation only; no provider was called"
    if not response.called_model:
        return "No provider was called"
    if response.recommended_provider == "ollama":
        return "Local Ollama model was called"
    if response.recommended_provider == "mock":
        return "Demo mock provider was called"
    if response.route_kind == "cloud_api":
        return "Cloud API provider was called"
    return f"{provider_display_name(response)} provider was called"


def ask_route_display_name(response: PersonalRouteResponse) -> str:
    if response.route_kind == "local":
        return "local model"
    if response.route_kind == "mock":
        return "demo mock"
    if response.route_kind == "manual_subscription":
        return "manual recommendation"
    if response.route_kind == "cloud_api":
        return "cloud API"
    return response.route_kind.replace("_", " ")


def print_ask_metadata(response: PersonalRouteResponse, show_debug_metadata: bool = False) -> None:
    print("\n---")
    print(f"Model: {response.recommended_model}")
    print(f"Provider: {provider_display_name(response)}")
    print(f"Route: {ask_route_display_name(response)}")
    print(f"Premium saved: {response.estimated_premium_units_saved} unit(s)")
    if "PREVIOUS_FEEDBACK_CONSIDERED" in response.reason_codes:
        print("Feedback: previous feedback considered")
    print(f"Request ID: {response.request_id}")
    if show_debug_metadata:
        print(f"Called model: {response.called_model}")


def print_backend_response(
    response: SwitchboardResponse,
    *,
    show_metadata: bool = False,
) -> None:
    if response.success and response.content:
        print(response.content)
    elif response.error_message:
        print(f"Error: {response.error_message}")
        hint = backend_error_hint(response)
        if hint:
            print(f"Hint: {hint}")
    else:
        print("No response content returned.")
    if response.stderr and show_metadata:
        print("\nStderr:")
        print(response.stderr.rstrip())
    print("\n---")
    print(f"Backend: {response.backend}")
    if response.selected_model:
        print(f"Model: {response.selected_model}")
    print(f"Success: {response.success}")
    print(f"Latency: {response.latency_ms}ms")
    if response.exit_code is not None:
        print(f"Exit code: {response.exit_code}")
    print(f"Cost type: {response.cost_type.value}")
    if response.routing_reason:
        print(f"Routing: {response.routing_reason}")
    if show_metadata and response.session_id:
        print(f"Session ID: {response.session_id}")
    print(f"Request ID: {response.request_id}")


def backend_error_hint(response: SwitchboardResponse) -> str | None:
    error = response.error_message or ""
    if response.backend == "codex" and "not supported when using Codex" in error:
        return (
            "Codex rejected the forced model for this account. Retry without "
            "--force-model to use the configured Codex default, or choose a model "
            "your Codex CLI supports."
        )
    if response.backend == "codex" and "unexpected argument" in error:
        return "Update Switchboard or retry without custom Codex flags."
    if response.backend == "claude-code" and "Input must be provided" in error:
        return "Update Switchboard; this usually means a Claude CLI option swallowed the prompt."
    return None


CORE_BACKEND_BY_FORCE_MODEL = {
    "codex": "codex",
    "claude": "claude-code",
    "claude-code": "claude-code",
    "ollama": "ollama",
}


def core_backend_and_model(
    backend: str | None,
    force_model: str | None,
) -> tuple[str | None, str | None]:
    resolved_backend = None if backend in {None, "auto"} else backend
    model = force_model
    if not force_model:
        return resolved_backend, model
    if force_model.startswith("manual/"):
        raise SystemExit(
            f"{force_model} is a manual subscription catalogue entry, not a "
            "callable Switchboard Core backend. Choose "
            "--backend codex, --backend claude-code, or --backend ollama."
        )
    forced_backend = CORE_BACKEND_BY_FORCE_MODEL.get(force_model)
    if forced_backend is None and force_model.startswith("ollama/"):
        forced_backend = "ollama"
    if forced_backend is None:
        return resolved_backend, model
    if resolved_backend is not None and resolved_backend != forced_backend:
        raise SystemExit(
            f"--force-model {force_model} implies backend {forced_backend}, "
            f"but --backend {resolved_backend} was selected."
        )
    if force_model in {"codex", "claude", "claude-code", "ollama"}:
        model = None
    return forced_backend, model


def core_route_next_step(decision: BackendRouteDecision, model: str | None) -> str:
    if decision.forced_backend and model:
        return (
            "switchboard ask "
            f"--backend {decision.backend} --force-model {model} '<same prompt>'"
        )
    if decision.forced_backend:
        return f"switchboard ask --backend {decision.backend} '<same prompt>'"
    if model:
        return f"switchboard ask --force-model {model} '<same prompt>'"
    return "switchboard ask '<same prompt>'"


def print_core_route(
    decision: BackendRouteDecision,
    *,
    prompt: str,
    model: str | None = None,
    show_prompt: bool = False,
    debug: bool = False,
) -> None:
    print(f"Recommendation: {decision.display_model}")
    print(f"Backend: {decision.backend}")
    print(f"Route type: {decision.route_type}")
    print(f"Fallback used: {decision.fallback_used}")
    if decision.fallback_from:
        print(f"Fallback from: {decision.fallback_from}")
    print(f"Forced backend: {decision.forced_backend}")
    if model:
        print(f"Model override: {model}")
    print(f"Routing: {decision.routing_reason}")
    print(f"Next step: {core_route_next_step(decision, model)}")
    if show_prompt:
        print("\nPrompt:")
        print(prompt)
    if debug:
        print("\nRaw route decision:")
        print_json(decision)


def provider_status_text(name: str, configured: bool) -> str:
    status = "configured" if configured else "not configured"
    if name in {"", "none", "unconfigured"}:
        return status
    return f"{name} {status}"


def claude_code_web_search_available(config: PersonalConfig) -> bool:
    if not config.preferences.claude_code_web_search:
        return False
    executable = os.getenv("SWITCHBOARD_CLAUDE_CODE_EXECUTABLE", "claude")
    return shutil.which(executable) is not None


def live_latest_status_text(
    *,
    news_name: str,
    news_configured: bool,
    web_configured: bool,
    claude_web_search_available: bool,
) -> str:
    if news_configured:
        return f"configured ({news_name})"
    if web_configured:
        return "available via direct web search"
    if claude_web_search_available:
        return "available via Claude Code WebSearch"
    return "not configured"


def print_route(
    response: PersonalRouteResponse,
    show_prompt: bool = False,
    debug: bool = False,
) -> None:
    print(f"Recommendation: {response.recommended_model}")
    print(f"Provider: {response.recommended_provider}")
    print(f"Route kind: {response.route_kind}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Cost estimate: ${response.estimated_cost_usd:.6f}")
    print(f"Premium impact: {response.estimated_premium_units} unit(s)")
    print(f"Premium saved: {response.estimated_premium_units_saved} unit(s)")
    print(f"Requires confirmation: {response.requires_confirmation}")
    print(f"Privacy: {response.privacy_note}")
    print(f"Next step: {next_step_for_route(response)}")
    if response.performance_mode:
        print(f"Performance mode: {response.performance_mode}")
    if response.loaded_local_models:
        print(f"Loaded local models: {', '.join(response.loaded_local_models)}")
    else:
        print("Loaded local models: none detected")
    if response.selected_model_loaded is not None:
        print(f"Selected model loaded: {response.selected_model_loaded}")
    print(f"Model switch avoided: {response.model_switch_avoided}")
    print(f"Cold start expected: {response.cold_start_expected}")
    if response.next_best_alternative:
        print(f"Next best alternative: {response.next_best_alternative}")
    print("Why:")
    for reason in user_facing_reasons(response):
        print(f"  - {reason}")
    if debug:
        print("Raw reason codes:")
        for code in response.reason_codes:
            print(f"  - {code}")
    if debug and response.uncertainty_reasons:
        print("Uncertainty:")
        for reason in response.uncertainty_reasons:
            print(f"  - {reason}")
    if show_prompt and response.premium_prompt:
        print("\nReady-to-paste prompt:")
        print(response.premium_prompt.ready_to_paste_prompt)
    print(f"\nrequest_id: {response.request_id}")


def route_command(args: argparse.Namespace) -> None:
    backend, model = core_backend_and_model(None, args.force_model)
    service = build_core_service()
    response = service.preview_route(
        args.prompt,
        backend=backend,
        project=args.project,
        model=model,
        metadata={"surface": "cli_route"},
    )
    print_core_route(
        response,
        prompt=args.prompt,
        model=model,
        show_prompt=args.show_prompt,
        debug=args.debug or args.show_reasons,
    )


def ask_command(args: argparse.Namespace) -> None:
    requested_backend = getattr(args, "backend", None) or "auto"
    backend, model = core_backend_and_model(
        requested_backend,
        getattr(args, "force_model", None),
    )
    print(
        f"Calling backend {requested_backend} with timeout {getattr(args, 'timeout', 120)}s...",
        flush=True,
    )
    backend_response = build_core_service(
        router_mode=getattr(args, "router", None),
        compression=(False if getattr(args, "no_compression", False) else None),
        semantic_memory=(True if getattr(args, "memory", False) else None),
    ).ask(
        args.prompt,
        backend=backend,
        project=args.project,
        model=model,
        timeout_s=getattr(args, "timeout", 120),
        session_id=(
            None if getattr(args, "new_session", False) else getattr(args, "session", None)
        ),
        new_session=getattr(args, "new_session", False),
    )
    print_backend_response(backend_response, show_metadata=args.show_metadata)
    if not backend_response.success:
        raise SystemExit(1)


def personal_ask_command(args: argparse.Namespace) -> None:
    service = build_service()
    try:
        personal_response = asyncio.run(
            service.ask(
                PersonalPromptRequest(
                    prompt=args.prompt,
                    project=args.project,
                    use_cache=not args.no_cache,
                    strict=args.strict,
                    force_model=args.force_model,
                    allow_cloud_once=args.allow_cloud_once,
                    override_reason=args.override_reason,
                    baseline_model=args.baseline,
                )
            )
        )
    except PersonalRoutingError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    if personal_response.answer:
        print(personal_response.answer)
    else:
        print(personal_response.recommendation.explanation)
        print(f"\nNext step: {next_step_for_route(personal_response.recommendation)}")
        if personal_response.suggested_compressed_prompt:
            print("\nSuggested compressed prompt:")
            print(personal_response.suggested_compressed_prompt)
        if args.show_prompt and personal_response.recommendation.premium_prompt:
            print("\nReady-to-paste prompt:")
            print(personal_response.recommendation.premium_prompt.ready_to_paste_prompt)
    if personal_response.quality_warning:
        print("\nQuality warning:")
        for note in personal_response.quality_notes:
            print(f"  - {note}")
        if personal_response.suggested_next_step:
            print(personal_response.suggested_next_step)
    print_ask_metadata(personal_response.recommendation, show_debug_metadata=args.show_metadata)


def models_command(args: argparse.Namespace) -> None:
    if args.recommend:
        settings = get_settings()
        catalogue = ModelCatalogue.from_yaml(settings.models_config_path)
        total_ram_bytes = detect_total_ram_bytes()
        recommendation = recommend_local_model_pack(
            catalogue,
            total_ram_bytes=total_ram_bytes,
        )
        if total_ram_bytes is None:
            print(f"Detected RAM: unknown (using {recommendation.tier} tier)")
        else:
            print(
                "Detected RAM: "
                f"{total_ram_bytes / (1024**3):.1f} GiB ({recommendation.tier} tier)"
            )
        print("Recommended Ollama pack:")
        for role in recommendation.roles:
            print(f"  {role.role:10} {role.model_id}  (pull: ollama pull {role.ollama_tag})")
            if role.notes and "Ollama >= 0.14.3" in role.notes:
                print(f"             note: {role.notes}")
        for note in recommendation.notes:
            print(f"Note: {note}")
        print("Pull commands:")
        for command in recommendation.pull_commands:
            print(f"  {command}")
        print("No models were pulled automatically.")
        if args.apply:
            if not args.yes:
                answer = input(
                    "Apply these local role mappings to personal.yaml/models.yaml? "
                    "Type 'apply' to continue: "
                )
                if answer.strip().lower() != "apply":
                    print("No changes made.")
                    return
            apply_local_model_pack(
                personal_config_path=settings.personal_config_path,
                models_config_path=settings.models_config_path,
                recommendation=recommendation,
            )
            print("Updated local role mappings and enabled selected model profiles.")
        return

    for model in build_service().models():
        enabled = "enabled" if model.provider_enabled and model.enabled else "disabled"
        scarce = "scarce" if model.scarce else "not scarce"
        print(
            f"{model.model_id:38} {model.kind:22} {enabled:9} {scarce:11} "
            f"good_for={','.join(model.good_for)}"
        )


def backends_command(args: argparse.Namespace) -> None:
    service = build_core_service()
    settings = get_settings()
    config = PersonalConfig.from_yaml(settings.personal_config_path)
    preferences = config.preferences
    if args.format == "json":
        print_json([backend.model_dump(mode="json") for backend in service.backends()])
        return
    for backend in service.backends():
        status = "available" if backend.available else "unavailable"
        path = f" path={backend.path}" if backend.path else ""
        details = f" details={backend.details}" if backend.details else ""
        print(f"{backend.name:12} {status:11} cost_type={backend.cost_type.value}{path}{details}")
        if backend.warning:
            print(f"  warning: {backend.warning}")
    web_name, web_configured = web_provider_status()
    finance_name, finance_configured = finance_provider_status(preferences.finance_provider)
    news_name, news_configured = news_provider_status(preferences.news_provider)
    web_status = "configured" if web_configured else "not configured"
    if not web_configured and claude_code_web_search_available(config):
        web_status = f"{web_status} (optional; Claude Code WebSearch available)"
    print(
        f"web-search  {web_status:15} "
        f"provider={web_name}"
    )
    print(
        f"news        {'configured' if news_configured else 'not configured':15} "
        f"provider={news_name}"
    )
    print(
        f"finance     {'configured' if finance_configured else 'not configured':15} "
        f"provider={finance_name}"
    )


def metrics_command(args: argparse.Namespace) -> None:
    service = build_core_service()
    if args.view == "summary":
        print_json(service.metrics_summary())
        return
    records = service.metrics_list(limit=args.last)
    if args.format == "json":
        print_json([record.model_dump(mode="json") for record in records])
        return
    for record in records:
        status = "ok" if record.success else "error"
        model = record.selected_model or "-"
        print(
            f"{record.created_at.isoformat()} {record.backend:12} {status:5} "
            f"model={model} latency={record.latency_ms}ms request_id={record.request_id}"
        )
        if record.routing_reason:
            print(f"  routing: {record.routing_reason}")
        if record.error_message:
            print(f"  error: {record.error_message}")


def quota_command(args: argparse.Namespace) -> None:
    status = build_core_service().quota_status()
    if args.format == "json":
        print_json(status)
        return
    enabled = "enabled" if status.get("enabled") else "disabled (budgets unset)"
    print("Premium quota ledger: estimate-only, from local backend metrics")
    print(f"Quota-aware routing: {enabled}")
    windows = status.get("windows", {})
    if not isinstance(windows, dict):
        return
    for backend in ("codex", "claude-code"):
        window = windows.get(backend)
        if not isinstance(window, dict):
            continue
        budget = window.get("budget")
        budget_text = "unset" if budget is None else str(budget)
        remaining = window.get("remaining")
        remaining_text = "-" if remaining is None else str(remaining)
        state = "constrained" if window.get("constrained") else "ok"
        print(
            f"{window.get('label', backend):12} {state:12} "
            f"used={window.get('used', 0)}/{budget_text} "
            f"window={window.get('window')} remaining={remaining_text}"
        )


def usage_command(args: argparse.Namespace) -> None:
    usage = build_service().usage()
    print(f"Total requests: {usage.get('total_requests', 0)}")
    print(f"Local/mock requests: {usage.get('local_requests', 0)}")
    print(f"Cloud requests: {usage.get('cloud_requests', 0)}")
    print(f"Manual recommendations: {usage.get('manual_recommendations', 0)}")
    print(f"Estimated API spend: ${usage.get('estimated_api_spend_usd', 0)}")
    print(f"Estimated premium units saved: {usage.get('estimated_premium_units_saved', 0)}")
    print(f"Cache hits: {usage.get('cache_hits', 0)}")
    print(f"Cache misses: {usage.get('cache_misses', 0)}")
    print(f"Feedback: {json.dumps(usage.get('feedback', {}), sort_keys=True)}")


def savings_command(args: argparse.Namespace) -> None:
    since = None
    days = args.days
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=UTC)
        days = None
    savings = build_service().savings(days=days, since=since)
    if args.format == "json":
        print_json(savings)
        return
    print(f"Total requests: {savings.get('total_requests', 0)}")
    print(f"Local/model calls: {savings.get('local_model_calls', 0)}")
    print(f"Local Ollama calls: {savings.get('local_ollama_calls', 0)}")
    print(f"Mock calls: {savings.get('mock_calls', 0)}")
    print(f"Cloud calls: {savings.get('cloud_calls', 0)}")
    print(f"Manual premium recommendations: {savings.get('manual_premium_recommendations', 0)}")
    print(f"Premium units saved: {savings.get('premium_units_saved', 0)}")
    print(f"Premium units spent: {savings.get('premium_units_spent', 0)}")
    print(f"Estimated API spend: ${savings.get('estimated_api_spend_usd', 0)}")
    print(f"Estimated API cost saved: ${savings.get('estimated_api_cost_saved_usd', 0)}")
    print(f"Top task types saved: {json.dumps(savings.get('top_task_types_saved', {}))}")
    print(f"Top models used: {json.dumps(savings.get('top_models_used', {}))}")
    print(f"Overrides: {savings.get('overrides_count', 0)}")
    print(f"Escalations: {savings.get('escalations_count', 0)}")
    print(f"Cache hits/misses: {savings.get('cache_hits', 0)}/{savings.get('cache_misses', 0)}")
    print(f"Feedback: {json.dumps(savings.get('feedback', {}), sort_keys=True)}")
    print(
        "Baseline assumptions: "
        f"{json.dumps(savings.get('baseline_assumptions', {}), sort_keys=True)}"
    )


def demo_command(args: argparse.Namespace) -> None:
    print(
        """
Switchboard Demo Checklist
--------------------------
1. Check local setup:
   switchboard doctor
   switchboard backends

2. Start the local UI:
   switchboard ui
   Open http://127.0.0.1:8080/ui

3. Try these prompts in Auto mode:
   Fix this failing Python test
   Expected: Auto chooses Codex for coding/debugging.

   Review this architecture for a model router
   Expected: Auto chooses Claude for reasoning/design review.

   Answer locally: summarize this sentence
   Expected: Auto chooses Ollama for local/simple work.

   Time in India
   Expected: TimeTool grounds the selected model; the model gives the final answer.

   Weather in India
   Expected: If Claude Code is available, Auto uses Claude WebSearch; otherwise
   Switchboard stays honest instead of inventing live weather.

4. Show shared session context:
   Remember: Switchboard routes between Codex, Claude, and Ollama.
   Then switch model and ask:
   What did I ask you to remember?

5. Show evidence:
   switchboard eval --mock
   switchboard eval-real-smoke --fast
   switchboard metrics summary

The UI intentionally shows friendly model labels and answers only. Routing internals,
metrics, stdout/stderr, and costs stay out of the chat view but are recorded locally.
""".strip()
    )


def _eval_case_preview(case: EvalCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "category": case.category,
        "name": case.name,
        "prompt_char_count": len(case.prompt),
        "expected_backend": case.expected_backend,
        "expected_route_type": case.expected_route_type,
        "expected_tool": case.expected_tool,
        "expected_capability": case.expected_capability,
        "should_call_model": case.should_call_model,
        "step_count": len(case.steps),
    }


def eval_command(args: argparse.Namespace) -> None:
    runner = build_eval_runner(mock=args.mock)
    suite = args.suite
    if args.dry_run:
        cases = runner.list_cases(suite, limit=args.limit)
        previews = [_eval_case_preview(case) for case in cases]
        if args.json:
            print_json({"suite": suite, "cases": previews})
            return
        print("Switchboard Eval Cases")
        print("----------------------")
        print(f"Suite: {suite}")
        for preview in previews:
            expectation = (
                preview["expected_tool"]
                or preview["expected_backend"]
                or preview["expected_route_type"]
                or "structural"
            )
            print(f"- {preview['case_id']} [{preview['category']}]: {expectation}")
        return

    report = runner.run(suite, backend=args.backend, limit=args.limit)
    if args.output:
        write_report(Path(args.output), report)
    if args.json:
        print(report_to_json(report))
    else:
        print(report_to_text(report))
        if args.output:
            print(f"\nWrote JSON report: {args.output}")
    if report.failed:
        raise SystemExit(1)


def eval_real_smoke_command(args: argparse.Namespace) -> None:
    tags = set(args.tag or [])
    if args.fast:
        tags.add("fast")
    runner = build_real_smoke_runner(
        timeout_s=args.timeout,
        case_timeouts=_parse_case_timeouts(args.case_timeout),
    )
    report = runner.run(limit=args.limit, tags=tags or None)
    if args.output:
        write_report(Path(args.output), report)
    if args.json:
        print(report_to_json(report))
    else:
        print(report_to_text(report))
        if args.output:
            print(f"\nWrote JSON report: {args.output}")
    if report.failed:
        raise SystemExit(1)


def eval_real_providers_command(args: argparse.Namespace) -> None:
    runner = RealProviderRunner(timeout_s=args.timeout)
    report = runner.run()
    if args.output:
        write_report(Path(args.output), report)
    if args.json:
        print(report_to_json(report))
    else:
        print(report_to_text(report))
        if args.output:
            print(f"\nWrote JSON report: {args.output}")
    if report.failed:
        raise SystemExit(1)


def _parse_case_timeouts(values: list[str] | None) -> dict[str, int]:
    timeouts: dict[str, int] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(
                "Invalid --case-timeout value. Expected format: case_id=seconds"
            )
        case_id, seconds_text = value.split("=", 1)
        case_id = case_id.strip()
        try:
            seconds = int(seconds_text)
        except ValueError as exc:
            raise SystemExit(
                "Invalid --case-timeout seconds. Expected an integer value."
            ) from exc
        if not case_id or seconds <= 0:
            raise SystemExit(
                "Invalid --case-timeout value. Case ID must be set and seconds > 0."
            )
        timeouts[case_id] = seconds
    return timeouts


def _require_prompt_for_replay(
    args: argparse.Namespace,
    service: PersonalSwitchboardService,
) -> str:
    previous = service.container.personal_telemetry_repository.get(args.request_id)
    if previous is None:
        raise SystemExit(f"Unknown request_id: {args.request_id}")
    if args.prompt:
        return args.prompt
    raise SystemExit(
        "Prompt bodies are not stored by default. Re-run with --prompt or enable prompt logging."
    )


def rerun_command(args: argparse.Namespace) -> None:
    service = build_service()
    prompt = _require_prompt_for_replay(args, service)
    try:
        response = asyncio.run(
            service.ask(
                PersonalPromptRequest(
                    prompt=prompt,
                    project=args.project,
                    use_cache=False,
                    force_model=args.model,
                    override_reason=args.override_reason,
                    baseline_model=args.baseline,
                    original_request_id=args.request_id,
                )
            )
        )
    except PersonalRoutingError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    if response.answer:
        print(response.answer)
    else:
        print(response.recommendation.explanation)
    print(f"\nrequest_id: {response.request_id}")


def escalate_command(args: argparse.Namespace) -> None:
    service = build_service()
    prompt = _require_prompt_for_replay(args, service)
    try:
        response = service.route(
            PersonalPromptRequest(
                prompt=prompt,
                project=args.project,
                use_cache=False,
                force_model=args.to,
                override_reason=args.override_reason,
                baseline_model=args.baseline,
                original_request_id=args.request_id,
                escalation_used=True,
            )
        )
    except PersonalRoutingError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print_route(response, show_prompt=args.show_prompt, debug=args.debug or args.show_reasons)


def feedback_examples_command(args: argparse.Namespace) -> None:
    from switchboard.training.feedback_loop import FeedbackExampleStore

    settings: Settings = get_settings()
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    store = FeedbackExampleStore(engine)
    if args.purge:
        removed = store.purge()
        print(f"Purged {removed} stored feedback example(s) and context snapshot(s).")
        return
    counts = store.counts()
    print(f"Stored feedback examples: {counts['total']}")
    print(f"Unprocessed wrong-model corrections: {counts['unprocessed_wrong_model']}")


def _embedding_unavailable_exit(model: str) -> SystemExit:
    return SystemExit(
        f"Embedding model unreachable — is Ollama running with '{model}' pulled? "
        f"(ollama pull {model})"
    )


def configured_embedding_model(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    settings = get_settings()
    config = PersonalConfig.from_yaml(settings.personal_config_path)
    return config.preferences.embedding_model


def train_router_command(args: argparse.Namespace) -> None:
    from switchboard.training.augment import augment_examples
    from switchboard.training.router_dataset import (
        build_dataset,
        class_counts,
        write_jsonl,
    )
    from switchboard.training.train_router import (
        report_to_text,
        train_from_files,
    )

    dataset_path = args.dataset or "router_dataset.jsonl"
    if not args.dataset or not Path(dataset_path).exists():
        examples = build_dataset()
        if args.external:
            from switchboard.training.external_datasets import (
                load_or_fetch_external,
            )

            print("Fetching public datasets (CLINC150, dolly-15k, CodeAlpaca)...", flush=True)
            external = load_or_fetch_external("data/external_router_examples.jsonl")
            print(f"External examples: {len(external)}")
            examples = examples + external
        if args.augment:
            print("Augmenting dataset with Claude paraphrases (one-time)...", flush=True)
            examples = augment_examples(examples, limit=args.augment_limit)
        from switchboard.training.router_dataset import relabel_toolable

        examples, relabeled = relabel_toolable(examples)
        if relabeled:
            print(f"Tools-first relabeling: {relabeled} example(s) moved to 'tool'")
        write_jsonl(examples, dataset_path)
        print(f"Built dataset: {len(examples)} examples -> {dataset_path}")
        print(f"Class counts: {class_counts(examples)}")

    embedding_model = configured_embedding_model(args.embedding_model)
    print("Embedding and training (requires Ollama embedding model)...", flush=True)
    try:
        report = train_from_files(
            dataset_path=dataset_path,
            output_path=args.output,
            embedding_model=embedding_model,
        )
    except (EmbeddingUnavailableError, httpx.HTTPError) as exc:
        raise _embedding_unavailable_exit(embedding_model) from exc
    print(report_to_text(report, args.output))


def train_dispatcher_command(args: argparse.Namespace) -> None:
    from switchboard.app.services.tool_dispatcher import TOOL_CLASSES
    from switchboard.training.router_dataset import class_counts
    from switchboard.training.tool_dispatcher_dataset import (
        dispatcher_golden_examples,
        load_or_build_dispatcher_dataset,
    )
    from switchboard.training.train_router import report_to_text, train

    print("Building dispatcher dataset (CLINC150 + templates)...", flush=True)
    examples = load_or_build_dispatcher_dataset(args.dataset)
    print(f"Dataset: {len(examples)} examples")
    print(f"Class counts: {class_counts(examples)}")

    embedding_model = configured_embedding_model(args.embedding_model)
    print("Embedding and training (requires Ollama embedding model)...", flush=True)
    from switchboard.app.services.semantic_memory import OllamaEmbeddingClient

    embed = OllamaEmbeddingClient(model=embedding_model).embed_classification
    try:
        weights, report = train(
            examples,
            embed=embed,
            embedding_model=embedding_model,
            classes=TOOL_CLASSES,
            golden=dispatcher_golden_examples(),
        )
    except (EmbeddingUnavailableError, httpx.HTTPError) as exc:
        raise _embedding_unavailable_exit(embedding_model) from exc
    weights.metadata["golden_accuracy"] = report.golden_accuracy
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(weights.to_dict(), indent=2), encoding="utf-8")
    print(report_to_text(report, args.output))


def train_sensitivity_command(args: argparse.Namespace) -> None:
    from switchboard.app.services.sensitivity_escalator import (
        SENSITIVITY_CLASSES,
    )
    from switchboard.training.router_dataset import class_counts
    from switchboard.training.sensitivity_dataset import (
        sensitivity_examples,
        sensitivity_golden_examples,
    )
    from switchboard.training.train_router import report_to_text, train

    examples = sensitivity_examples()
    print(f"Dataset: {len(examples)} examples")
    print(f"Class counts: {class_counts(examples)}")

    embedding_model = configured_embedding_model(args.embedding_model)
    print("Embedding and training (requires Ollama embedding model)...", flush=True)
    from switchboard.app.services.semantic_memory import OllamaEmbeddingClient

    embed = OllamaEmbeddingClient(model=embedding_model).embed_classification
    try:
        weights, report = train(
            examples,
            embed=embed,
            embedding_model=embedding_model,
            classes=SENSITIVITY_CLASSES,
            golden=sensitivity_golden_examples(),
        )
    except (EmbeddingUnavailableError, httpx.HTTPError) as exc:
        raise _embedding_unavailable_exit(embedding_model) from exc
    weights.metadata["golden_accuracy"] = report.golden_accuracy
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(weights.to_dict(), indent=2), encoding="utf-8")
    print(report_to_text(report, args.output))


def bench_quality_command(args: argparse.Namespace) -> None:
    conditions = DEFAULT_CONDITIONS
    if args.condition:
        wanted = set(args.condition)
        conditions = tuple(c for c in DEFAULT_CONDITIONS if c.name in wanted)
        unknown = wanted - {c.name for c in DEFAULT_CONDITIONS}
        if unknown:
            known = ", ".join(c.name for c in DEFAULT_CONDITIONS)
            raise SystemExit(f"Unknown condition(s): {', '.join(sorted(unknown))}. Known: {known}")
    judge = None
    if not args.mock and args.judge_model:
        judge = OllamaJudge(model=args.judge_model)
    runner = QualityBenchRunner(
        settings=get_settings(),
        mock=args.mock,
        judge=judge,
        conditions=conditions,
        timeout_s=args.timeout,
        cwd=Path.cwd(),
    )
    report = runner.run(
        limit=args.limit,
        categories=tuple(args.category) if args.category else None,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    if args.json:
        print_json(report)
    else:
        print(quality_report_to_text(report))


def bench_models_command(args: argparse.Namespace) -> None:
    results = asyncio.run(build_service().bench_models())
    if args.format == "json":
        print_json(results)
        return
    if any(result.get("unload_message") for result in results):
        print("Benchmark mode unloads Ollama models after each smoke test.\n")
    for result in results:
        status = "ok" if result["reachable"] and result["non_empty_response"] else "error"
        print(
            f"{result['model']:34} {status:5} latency={result['latency_ms']}ms "
            f"install={result.get('install_command') or '-'}"
        )
        if result.get("error"):
            print(f"  error: {result['error']}")
        if result.get("unload_message"):
            print(f"  unload: {result['unload_message']}")


def loaded_models_command(args: argparse.Namespace) -> None:
    status = build_service().loaded_models()
    if args.format == "json":
        print_json(status)
        return
    print(f"Ollama provider enabled: {status['ollama_enabled']}")
    print(f"Performance mode: {status['performance_mode']}")
    loaded = cast("list[str]", status.get("loaded_models", []))
    if loaded:
        print("Loaded Ollama models:")
        for model_id in loaded:
            print(f"  - {model_id}")
    else:
        print(
            "Loaded Ollama models: none detected. If Ollama is running, try "
            "`ollama ps`; otherwise this is safe to ignore."
        )


def warm_command(args: argparse.Namespace) -> None:
    result = build_service().warm_model(args.model, allow_embedding=args.allow_embedding)
    print(result.message)
    if not result.ok:
        raise SystemExit(1)


def unload_command(args: argparse.Namespace) -> None:
    result = build_service().unload_model(args.model)
    print(result.message)
    if not result.ok:
        raise SystemExit(1)


def memory_add_command(args: argparse.Namespace) -> None:
    service = build_service()
    memory = service.add_memory(
        PersonalMemoryCreate(
            title=args.title,
            content=args.content,
            project=args.project,
            tags=args.tag,
        )
    )
    print_json(memory)
    if service.container.personal_config.preferences.semantic_memory_enabled:
        indexed = build_semantic_memory(service.container).index(memory)
        if indexed:
            print("Semantic embedding indexed.")
        else:
            print(
                "Semantic embedding not indexed (embedding model unavailable); "
                "text search fallback will be used."
            )


def memory_search_command(args: argparse.Namespace) -> None:
    service = build_service()
    print_json(
        [item.model_dump(mode="json") for item in service.search_memory(args.query, args.project)]
    )


def feedback_command(args: argparse.Namespace) -> None:
    service = build_service()
    result = service.add_feedback(
        FeedbackCreate(
            request_id=args.request_id,
            rating=args.rating,
            note=args.note,
            preferred_model=args.preferred_model,
        )
    )
    print_json(result)


def _ollama_model_name(model_id: str) -> str:
    return model_id.split("/", 1)[-1]


def _parse_ollama_list(output: str) -> set[str]:
    installed: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("name"):
            continue
        installed.add(stripped.split()[0])
    return installed


def doctor_command(args: argparse.Namespace) -> None:
    settings = get_settings()
    config_path = Path(settings.personal_config_path)
    print(f"Personal config exists: {config_path.exists()} ({config_path})")
    config = PersonalConfig.from_yaml(config_path)
    catalogue = ModelCatalogue.from_yaml(settings.models_config_path)
    print("Models config loads: yes")
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    print("Database reachable: yes")
    print(f"Private mode: {config.preferences.private_mode}")
    print(f"Allow cloud: {config.preferences.allow_cloud}")
    print(f"Prompt logging disabled: {not settings.log_prompts}")
    print(f"Response logging disabled: {not settings.log_responses}")
    print("Runtime context: available")
    print("Time tool: available")
    web_name, web_configured = web_provider_status()
    finance_name, finance_configured = finance_provider_status(config.preferences.finance_provider)
    news_name, news_configured = news_provider_status(config.preferences.news_provider)
    claude_web_search_available = claude_code_web_search_available(config)
    weather_status = (
        "available via direct web search"
        if web_configured
        else (
            "available via Claude Code WebSearch"
            if claude_web_search_available
            else "not configured"
        )
    )
    latest_status = live_latest_status_text(
        news_name=news_name,
        news_configured=news_configured,
        web_configured=web_configured,
        claude_web_search_available=claude_web_search_available,
    )
    print(f"Weather tool: {weather_status}")
    print(f"Live/latest info tool: {latest_status}")
    web_status = provider_status_text(web_name, web_configured)
    if not web_configured and claude_web_search_available:
        web_status = f"{web_status} (optional; Claude Code WebSearch available)"
    print(f"Web search provider: {web_status}")
    print(f"News provider: {provider_status_text(news_name, news_configured)}")
    print(f"Finance provider: {provider_status_text(finance_name, finance_configured)}")
    print("Session store: available")
    print("Context builder: available")
    print("Default recent messages: 12")
    print(f"Performance mode: {config.local_runtime.performance_mode}")
    print(f"Max loaded models: {config.local_runtime.max_loaded_models}")
    print(f"Ollama keep_alive: {config.local_runtime.keep_alive}")
    print("Local model recommendation: run `switchboard models --recommend`")
    print("GLM 4.7 Flash requires Ollama >= 0.14.3.")

    ollama = config.providers.get("ollama")
    print(f"Ollama provider enabled: {bool(ollama and ollama.enabled)}")
    runtime = OllamaRuntimeService(config)
    if ollama and ollama.enabled:
        base_url = ollama.base_url or "http://localhost:11434"
        try:
            response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
            response.raise_for_status()
            print("Ollama server reachable: yes")
        except httpx.HTTPError as exc:
            print(f"Ollama server reachable: no ({exc})")
            print("Route still works; ask may fall back to mock until Ollama is running.")
        configured = [
            model.provider_model_name or _ollama_model_name(model.model_id)
            for model in catalogue.models
            if model.provider == "ollama" and model.enabled and model.is_chat_selectable
        ]
        try:
            result = subprocess.run(
                ["ollama", "list"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            print(f"Ollama installed models: unavailable ({exc})")
            installed: set[str] = set()
        else:
            installed = _parse_ollama_list(result.stdout)
            print(f"Ollama installed models: {', '.join(sorted(installed)) or 'none found'}")
        missing = [name for name in configured if name not in installed]
        if missing:
            print("Configured Ollama models missing:")
            for name in missing:
                print(f"  - {name} (install: ollama pull {name})")
        loaded = sorted(runtime.list_loaded_models())
        print(f"Currently loaded Ollama models: {', '.join(loaded) or 'none detected'}")
    else:
        print("Local model answering is disabled. Route still works with mock/manual providers.")

    lmstudio = config.providers.get("lmstudio")
    if lmstudio and lmstudio.enabled and lmstudio.base_url:
        try:
            httpx.get(lmstudio.base_url, timeout=2)
            print("lmstudio reachable: yes")
        except httpx.HTTPError:
            print("lmstudio reachable: no")

    for provider_name, provider in config.providers.items():
        if provider.enabled and provider.env_api_key:
            print(f"{provider_name} env var present: {bool(os.getenv(provider.env_api_key))}")

    print(f"Metrics store: {settings.database_url}")
    print("Switchboard Core backends:")
    container = build_container(settings, engine)
    core = SwitchboardCoreService(
        registry=BackendRegistry.default(container, cwd=Path.cwd()),
        metrics=container.backend_metrics_repository,
        container=container,
    )
    for backend in core.backends():
        status = "available" if backend.available else "unavailable"
        detail = f" ({backend.path})" if backend.path else ""
        print(f"  - {backend.name}: {status}{detail}")
        if backend.warning:
            print(f"    warning: {backend.warning}")


def init_command(args: argparse.Namespace) -> None:
    settings = get_settings()
    personal_path = Path(settings.personal_config_path).expanduser()
    if personal_path.resolve().parent == packaged_config_path("personal.yaml").parent.resolve():
        personal_path = user_config_dir() / "personal.yaml"
    config_dir = personal_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)

    wrote: list[Path] = []
    skipped: list[Path] = []
    for name in DEFAULT_CONFIG_FILES:
        target = personal_path if name == "personal.yaml" else config_dir / name
        if target.exists() and not args.force:
            skipped.append(target)
            continue
        shutil.copyfile(packaged_config_path(name), target)
        wrote.append(target)

    for path in wrote:
        print(f"Wrote config: {path}")
    for path in skipped:
        print(f"Config already exists: {path}")


def ui_command(args: argparse.Namespace) -> None:
    import uvicorn

    url = f"http://{args.host}:{args.port}/ui"
    print(f"Switchboard UI running at {url}", flush=True)
    uvicorn.run(
        "switchboard.app.main:app",
        host=args.host,
        port=args.port,
    )


def add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output")
    parser.add_argument(
        "--backend",
        choices=["auto", "ollama", "codex", "claude-code"],
        default="auto",
    )
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--limit", type=int)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="switchboard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    route = subparsers.add_parser(
        "route",
        help="Preview the core backend route without calling a model",
    )
    route.add_argument("prompt")
    route.add_argument("--project")
    route.add_argument("--show-prompt", action="store_true")
    route.add_argument("--debug", action="store_true")
    route.add_argument("--show-reasons", action="store_true")
    route.add_argument("--force-model")
    route.set_defaults(func=route_command)

    ask = subparsers.add_parser("ask", help="Ask through the core router")
    ask.add_argument("prompt")
    ask.add_argument("--project")
    ask.add_argument("--backend", choices=["auto", "ollama", "codex", "claude-code"])
    ask.add_argument(
        "--router",
        choices=["rules", "llm", "hybrid", "learned"],
        help="Router mode for backend asks (default: personal.yaml router_mode)",
    )
    ask.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable Headroom-style prompt compression for this request",
    )
    ask.add_argument(
        "--memory",
        action="store_true",
        help="Enable semantic long-term memory retrieval for this request",
    )
    ask.add_argument("--session")
    ask.add_argument("--new-session", action="store_true")
    ask.add_argument("--timeout", type=int, default=120)
    ask.add_argument("--show-metadata", action="store_true")
    ask.add_argument("--force-model")
    ask.set_defaults(func=ask_command)

    models = subparsers.add_parser("models", help="List configured models")
    models.add_argument(
        "--recommend",
        action="store_true",
        help="Recommend an Ollama model pack for this machine",
    )
    models.add_argument(
        "--apply",
        action="store_true",
        help="Apply the recommended local role mappings after confirmation",
    )
    models.add_argument(
        "--yes",
        action="store_true",
        help="Confirm --apply noninteractively",
    )
    models.set_defaults(func=models_command)

    backends = subparsers.add_parser("backends", help="List Switchboard Core backends")
    backends.add_argument("--format", choices=["text", "json"], default="text")
    backends.set_defaults(func=backends_command)

    metrics = subparsers.add_parser("metrics", help="Show Switchboard Core backend metrics")
    metrics.add_argument("view", nargs="?", choices=["summary"], default="list")
    metrics.add_argument("--last", type=int, default=20)
    metrics.add_argument("--format", choices=["text", "json"], default="text")
    metrics.set_defaults(func=metrics_command)

    quota = subparsers.add_parser("quota", help="Show user-declared premium quota usage")
    quota.add_argument("--format", choices=["text", "json"], default="text")
    quota.set_defaults(func=quota_command)

    usage = subparsers.add_parser("usage", help="Show personal usage summary")
    usage.set_defaults(func=usage_command)

    savings = subparsers.add_parser("savings", help="Show personal savings ledger")
    savings.add_argument("--days", type=int, default=7)
    savings.add_argument("--since")
    savings.add_argument("--format", choices=["text", "json"], default="text")
    savings.set_defaults(func=savings_command)

    demo = subparsers.add_parser("demo", help="Run four demo prompts")
    demo.set_defaults(func=demo_command)

    eval_all = subparsers.add_parser("eval", help="Run deterministic Switchboard evals")
    add_eval_arguments(eval_all)
    eval_all.set_defaults(func=eval_command, suite="all")

    eval_routing = subparsers.add_parser("eval-routing", help="Run routing evals")
    add_eval_arguments(eval_routing)
    eval_routing.set_defaults(func=eval_command, suite="routing")

    eval_tools = subparsers.add_parser("eval-tools", help="Run deterministic tool evals")
    add_eval_arguments(eval_tools)
    eval_tools.set_defaults(func=eval_command, suite="tools")

    eval_session = subparsers.add_parser("eval-session", help="Run shared-session evals")
    add_eval_arguments(eval_session)
    eval_session.set_defaults(func=eval_command, suite="session")

    eval_real_smoke = subparsers.add_parser(
        "eval-real-smoke",
        help="Run optional real-backend smoke evals",
    )
    eval_real_smoke.add_argument("--json", action="store_true")
    eval_real_smoke.add_argument("--output")
    eval_real_smoke.add_argument("--limit", type=int)
    eval_real_smoke.add_argument("--timeout", type=int, default=90)
    eval_real_smoke.add_argument("--case-timeout", action="append", default=[])
    eval_real_smoke.add_argument("--fast", action="store_true")
    eval_real_smoke.add_argument("--tag", action="append", choices=["fast", "slow"])
    eval_real_smoke.set_defaults(func=eval_real_smoke_command)

    eval_real_providers = subparsers.add_parser(
        "eval-real-providers",
        help="Run optional real web/finance provider smoke evals",
    )
    eval_real_providers.add_argument("--json", action="store_true")
    eval_real_providers.add_argument("--output")
    eval_real_providers.add_argument("--timeout", type=int, default=120)
    eval_real_providers.set_defaults(func=eval_real_providers_command)

    rerun = subparsers.add_parser("rerun", help="Rerun a previous request with a chosen model")
    rerun.add_argument("request_id")
    rerun.add_argument("--model", required=True)
    rerun.add_argument("--prompt")
    rerun.add_argument("--project")
    rerun.add_argument("--override-reason")
    rerun.add_argument("--baseline")
    rerun.set_defaults(func=rerun_command)

    escalate = subparsers.add_parser("escalate", help="Escalate a previous request to a model")
    escalate.add_argument("request_id")
    escalate.add_argument("--to", required=True)
    escalate.add_argument("--prompt")
    escalate.add_argument("--project")
    escalate.add_argument("--show-prompt", action="store_true")
    escalate.add_argument("--debug", action="store_true")
    escalate.add_argument("--show-reasons", action="store_true")
    escalate.add_argument("--override-reason")
    escalate.add_argument("--baseline")
    escalate.set_defaults(func=escalate_command)

    bench = subparsers.add_parser("bench-models", help="Smoke-test enabled local/mock models")
    bench.add_argument("--format", choices=["text", "json"], default="text")
    bench.set_defaults(func=bench_models_command)

    bench_quality = subparsers.add_parser(
        "bench-quality",
        help="Run the 100-case quality benchmark across routing ablation conditions",
    )
    bench_quality.add_argument("--mock", action="store_true", help="Use mock backends and judge")
    bench_quality.add_argument("--limit", type=int, help="Limit number of cases per condition")
    bench_quality.add_argument(
        "--category",
        action="append",
        choices=["coding", "reasoning", "summarization", "private", "grounding"],
        help="Restrict to one or more case categories",
    )
    bench_quality.add_argument(
        "--condition",
        action="append",
        help="Run only the named conditions (default: all)",
    )
    bench_quality.add_argument("--judge-model", help="Ollama judge model (default gemma4:12b)")
    bench_quality.add_argument("--timeout", type=int, default=120)
    bench_quality.add_argument("--json", action="store_true")
    bench_quality.add_argument("--output", help="Write the full JSON report to this file")
    bench_quality.set_defaults(func=bench_quality_command)

    train_router = subparsers.add_parser(
        "train-router",
        help="Train the learned embedding router from synthetic data",
    )
    train_router.add_argument("--dataset", help="JSONL dataset path (built if missing)")
    train_router.add_argument(
        "--output", default="config/router_weights.json", help="Weights output path"
    )
    train_router.add_argument(
        "--embedding-model",
        help="Embedding model for training (default: preferences.embedding_model)",
    )
    train_router.add_argument(
        "--external",
        action="store_true",
        help="Include public datasets (CLINC150, dolly-15k, CodeAlpaca; cached locally)",
    )
    train_router.add_argument(
        "--augment", action="store_true", help="Add Claude-CLI paraphrases (uses quota)"
    )
    train_router.add_argument("--augment-limit", type=int, help="Cap examples to augment")
    train_router.set_defaults(func=train_router_command)

    train_dispatcher = subparsers.add_parser(
        "train-dispatcher",
        help="Train the learned tool dispatcher (CLINC150 + templates; cached locally)",
    )
    train_dispatcher.add_argument(
        "--dataset",
        default="data/tool_dispatcher_examples.jsonl",
        help="JSONL dataset cache path (built if missing)",
    )
    train_dispatcher.add_argument(
        "--output",
        default="config/tool_dispatcher_weights.json",
        help="Weights output path",
    )
    train_dispatcher.add_argument(
        "--embedding-model",
        help="Embedding model for training (default: preferences.embedding_model)",
    )
    train_dispatcher.set_defaults(func=train_dispatcher_command)

    train_sensitivity = subparsers.add_parser(
        "train-sensitivity",
        help="Train the learned sensitivity escalator (privacy long-tail)",
    )
    train_sensitivity.add_argument(
        "--output",
        default="config/sensitivity_weights.json",
        help="Weights output path",
    )
    train_sensitivity.add_argument(
        "--embedding-model",
        help="Embedding model for training (default: preferences.embedding_model)",
    )
    train_sensitivity.set_defaults(func=train_sensitivity_command)

    feedback_examples = subparsers.add_parser(
        "feedback-examples",
        help="Inspect or purge stored thumbs-down training snapshots",
    )
    feedback_examples.add_argument("--purge", action="store_true")
    feedback_examples.set_defaults(func=feedback_examples_command)

    loaded = subparsers.add_parser("loaded-models", help="Show currently loaded Ollama models")
    loaded.add_argument("--format", choices=["text", "json"], default="text")
    loaded.set_defaults(func=loaded_models_command)

    warm = subparsers.add_parser("warm", help="Warm an Ollama model with keep_alive")
    warm.add_argument("model")
    warm.add_argument("--allow-embedding", action="store_true")
    warm.set_defaults(func=warm_command)

    unload = subparsers.add_parser("unload", help="Unload an Ollama model")
    unload.add_argument("model")
    unload.set_defaults(func=unload_command)

    memory = subparsers.add_parser("memory", help="Local memory commands")
    memory_subparsers = memory.add_subparsers(dest="memory_command", required=True)

    memory_add = memory_subparsers.add_parser("add", help="Add a memory item")
    memory_add.add_argument("--title", required=True)
    memory_add.add_argument("--content", required=True)
    memory_add.add_argument("--project")
    memory_add.add_argument("--tag", action="append", default=[])
    memory_add.set_defaults(func=memory_add_command)

    memory_search = memory_subparsers.add_parser("search", help="Search local memory")
    memory_search.add_argument("query")
    memory_search.add_argument("--project")
    memory_search.set_defaults(func=memory_search_command)

    feedback = subparsers.add_parser("feedback", help="Rate a routing decision")
    feedback.add_argument("request_id")
    feedback.add_argument(
        "--rating",
        required=True,
        choices=["good", "bad", "too-expensive", "too-weak", "wrong-route"],
    )
    feedback.add_argument("--note")
    feedback.add_argument("--preferred-model")
    feedback.set_defaults(func=feedback_command)

    doctor = subparsers.add_parser("doctor", help="Check local configuration")
    doctor.set_defaults(func=doctor_command)

    init = subparsers.add_parser("init", help="Create a starter personal config")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=init_command)

    ui = subparsers.add_parser("ui", help="Start the local Switchboard chat UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8080)
    ui.set_defaults(func=ui_command)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
