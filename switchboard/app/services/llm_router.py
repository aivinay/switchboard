"""LLM-based route classification for Switchboard Core.

The LLM router asks a small local Ollama model to classify a prompt into one of
the existing Switchboard route types. It never calls cloud or subscription
backends, so routing itself stays local-first and private.

Router modes (configured in personal.yaml or via CLI):

- ``rules``: Phase A deterministic keyword rules only (default, unchanged).
- ``llm``: the local LLM router classifies every prompt; deterministic rules
  remain as a fallback when the router model is unavailable or returns an
  unparseable result.
- ``hybrid``: deterministic rules first; the LLM router is consulted only when
  the rules cannot classify the prompt (route type ``unknown``).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

ARCH_ROUTER_MODEL = "hf.co/katanemo/Arch-Router-1.5B.gguf"
ROUTE_TYPES = ("tool", "coding", "reasoning", "local", "unknown")

BACKEND_BY_ROUTE_TYPE = {
    "tool": "ollama",
    "coding": "codex",
    "reasoning": "claude-code",
    "local": "ollama",
    # Local-first: an "unknown" classification is not evidence the prompt
    # needs a premium model, so it stays on the free local one.
    "unknown": "ollama",
}

ROUTER_SYSTEM_PROMPT = (
    "You are a routing classifier for a local AI switchboard. "
    "Classify the user request into exactly one route type:\n"
    "- coding: writing, debugging, refactoring, or testing code; repository work\n"
    "- reasoning: multi-step architecture, system design, tradeoff analysis, "
    "long-form planning, or professional review work that genuinely needs a "
    "frontier model\n"
    "- local: everything a small local model handles well, including greetings, "
    "chit-chat, simple general-knowledge questions with stable answers (science "
    "facts, definitions, how-things-work), short rewrites and summaries, and "
    "private or personal topics\n"
    "- unknown: none of the above clearly applies\n"
    "Prefer local unless the request clearly needs deep multi-step reasoning or "
    "coding tools.\n"
    "Respond with ONLY a JSON object, no prose:\n"
    '{"route_type": "<coding|reasoning|local|unknown>", "confidence": <0.0-1.0>}'
)

ROUTER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "route_type": {
            "type": "string",
            "enum": ["tool", "coding", "reasoning", "local", "unknown"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["route_type", "confidence"],
}

ARCH_ROUTER_POLICY_PROMPT = (
    "You are a policy router. Select exactly one named policy for the user request.\n"
    "Policies:\n"
    "- tool: deterministically answerable by tools such as time, math, units, "
    "stock, weather, news, or current facts.\n"
    "- local: small, simple, or private tasks for the local model.\n"
    "- coding: code, repositories, web/app development, algorithms, debugging, "
    "or tests.\n"
    "- reasoning: architecture, design, tradeoffs, planning, or professional review.\n"
    'Return JSON only: {"policy": "<tool|local|coding|reasoning>", '
    '"confidence": <0.0-1.0>}'
)


class LlmRouterUnavailableError(RuntimeError):
    """Raised when the router model cannot be reached."""


@dataclass(frozen=True)
class LlmRouteResult:
    success: bool
    route_type: str = "unknown"
    backend: str = "ollama"
    confidence: float = 0.0
    latency_ms: int = 0
    model: str = ""
    error: str | None = None


class OllamaRouterClient:
    """Minimal synchronous Ollama chat client used only for route classification."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout_s: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def complete(self, prompt: str) -> str:
        arch_router = is_arch_router_model(self.model)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": ARCH_ROUTER_POLICY_PROMPT
                    if arch_router
                    else ROUTER_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 64},
        }
        if not arch_router:
            payload["format"] = ROUTER_JSON_SCHEMA
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LlmRouterUnavailableError(f"Ollama router model unreachable: {exc}") from exc
        body = response.json()
        return str(body.get("message", {}).get("content", ""))


class LlmRouter:
    """Classifies prompts with a small local model and maps them to backends."""

    def __init__(
        self,
        *,
        complete: Callable[[str], str] | None = None,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 15.0,
        max_prompt_chars: int = 2000,
    ) -> None:
        self.model = model
        self.arch_router = is_arch_router_model(model)
        self.max_prompt_chars = max_prompt_chars
        self._complete = complete or OllamaRouterClient(
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
        ).complete

    def classify(self, prompt: str) -> LlmRouteResult:
        started = time.perf_counter()
        truncated = prompt[: self.max_prompt_chars]
        try:
            raw = self._complete(truncated)
        except Exception as exc:
            return LlmRouteResult(
                success=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                model=self.model,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        parsed = self._parse(raw)
        if parsed is None:
            return LlmRouteResult(
                success=False,
                latency_ms=latency_ms,
                model=self.model,
                error="Router model returned an unparseable classification.",
            )
        route_type, confidence = parsed
        return LlmRouteResult(
            success=True,
            route_type=route_type,
            backend=BACKEND_BY_ROUTE_TYPE[route_type],
            confidence=confidence,
            latency_ms=latency_ms,
            model=self.model,
        )

    def _parse(self, raw: str) -> tuple[str, float] | None:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        route_key = "policy" if self.arch_router else "route_type"
        route_type = str(data.get(route_key, "")).strip().lower()
        allowed_routes = (
            {"tool", "local", "coding", "reasoning"}
            if self.arch_router
            else ROUTE_TYPES
        )
        if route_type not in allowed_routes:
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return route_type, max(0.0, min(1.0, confidence))


def is_arch_router_model(model: str) -> bool:
    return "arch-router" in model.lower()
