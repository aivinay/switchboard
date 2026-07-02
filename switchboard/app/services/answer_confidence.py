from __future__ import annotations

import time
from dataclasses import dataclass

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.models.backends import SwitchboardRequest


@dataclass(frozen=True)
class AnswerConfidenceResult:
    passed: bool
    score: float
    latency_ms: int = 0
    verdict: str = ""
    error: str | None = None

    @property
    def unavailable(self) -> bool:
        return self.error is not None


class AnswerConfidenceService:
    """Local answer-confidence check used before optional premium escalation."""

    def check(
        self,
        *,
        adapter: AgentAdapter,
        request: SwitchboardRequest,
        answer: str,
        threshold: float,
        selected_model: str | None = None,
    ) -> AnswerConfidenceResult:
        started = time.perf_counter()
        try:
            check_response = adapter.ask(
                SwitchboardRequest(
                    request_id=f"{request.request_id}_confidence",
                    prompt=self._check_prompt(request.prompt, answer),
                    project=request.project,
                    model=selected_model or request.model,
                    timeout_s=min(request.timeout_s, 30),
                    private_mode=request.private_mode,
                    metadata={
                        **request.metadata,
                        "answer_confidence_check": True,
                    },
                )
            )
        except Exception as exc:
            return AnswerConfidenceResult(
                passed=True,
                score=1.0,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        if not check_response.success:
            return AnswerConfidenceResult(
                passed=True,
                score=1.0,
                latency_ms=latency_ms,
                error=check_response.error_message or "confidence check failed",
            )
        verdict = (check_response.content or check_response.stdout or "").strip()
        score = self._score(verdict=verdict, answer=answer, prompt=request.prompt)
        return AnswerConfidenceResult(
            passed=score >= threshold,
            score=score,
            latency_ms=latency_ms,
            verdict=verdict,
        )

    def _check_prompt(self, prompt: str, answer: str) -> str:
        return (
            "Answer only YES or NO.\n"
            "Is the answer correct, complete, and responsive to the user request?\n\n"
            f"User request:\n{prompt}\n\n"
            f"Answer:\n{answer}\n"
        )

    def _score(self, *, verdict: str, answer: str, prompt: str) -> float:
        normalized_verdict = verdict.strip().lower()
        heuristic = self._heuristic_score(prompt=prompt, answer=answer)
        if normalized_verdict.startswith("yes"):
            return max(heuristic, 0.8)
        if normalized_verdict.startswith("no"):
            return min(heuristic, 0.2)
        return heuristic

    def _heuristic_score(self, *, prompt: str, answer: str) -> float:
        normalized_answer = answer.strip().lower()
        normalized_prompt = prompt.lower()
        if not normalized_answer:
            return 0.0
        score = 0.65
        if len(normalized_answer) < 80 and any(
            marker in normalized_prompt
            for marker in ("analyze", "compare", "architecture", "debug", "plan")
        ):
            score -= 0.25
        if any(
            marker in normalized_answer
            for marker in ("i cannot", "i can't", "error", "unavailable")
        ):
            score -= 0.3
        if "json" in normalized_prompt and not normalized_answer.startswith(("{", "[")):
            score -= 0.2
        return max(0.0, min(1.0, score))
