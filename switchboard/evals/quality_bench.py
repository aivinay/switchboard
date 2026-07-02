"""Task-quality benchmark with ablation conditions for the Switchboard paper.

The benchmark runs the 100-case quality dataset under a matrix of routing
conditions and reports, per condition:

- mean answer quality (1-5, judged by a local LLM judge or a deterministic
  mock judge)
- premium (subscription) units consumed
- privacy violations (private cases that left the local backend)
- latency and success rates

Baselines:

- ``always_premium``: every prompt forced to the strongest subscription
  backend (claude-code) — the "just use Claude for everything" baseline.
- ``always_local``: every prompt forced to local Ollama.

Ablations vary router mode (rules / hybrid / llm), Headroom-style
compression, and semantic memory.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.core.config import Settings, get_settings
from switchboard.app.models.backends import BackendCostType
from switchboard.app.services.compression_layer import HeadroomCompressionLayer
from switchboard.app.services.container import build_container
from switchboard.app.services.llm_router import LlmRouter
from switchboard.app.services.semantic_memory import (
    MemoryEmbeddingRepository,
    SemanticMemoryService,
)
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.storage.db import create_db_engine, init_db
from switchboard.evals.mock_adapters import mock_registry
from switchboard.evals.quality_dataset import QualityCase, quality_cases


@dataclass(frozen=True)
class BenchCondition:
    name: str
    router_mode: str = "rules"  # rules | llm | hybrid | learned
    compression: bool = False
    memory: bool = False
    forced_backend: str | None = None  # baselines force one backend for all prompts
    # Learned tool dispatcher + sensitivity escalator (product defaults).
    # One ablation condition disables them so their end-to-end contribution
    # is attributable rather than constant across the matrix.
    learned_assists: bool = True


DEFAULT_CONDITIONS: tuple[BenchCondition, ...] = (
    BenchCondition(name="always_premium", forced_backend="claude-code"),
    BenchCondition(name="always_local", forced_backend="ollama"),
    BenchCondition(name="rules"),
    BenchCondition(name="hybrid", router_mode="hybrid"),
    BenchCondition(name="hybrid_no_assists", router_mode="hybrid", learned_assists=False),
    BenchCondition(name="llm", router_mode="llm"),
    BenchCondition(name="hybrid_compression", router_mode="hybrid", compression=True),
    BenchCondition(name="hybrid_memory", router_mode="hybrid", memory=True),
    BenchCondition(
        name="hybrid_full",
        router_mode="hybrid",
        compression=True,
        memory=True,
    ),
    BenchCondition(name="learned", router_mode="learned"),
)


@dataclass
class JudgeScore:
    score: float  # 1.0 - 5.0
    rationale: str = ""
    judged: bool = True


JUDGE_PROMPT_TEMPLATE = (
    "You are a strict grader of AI assistant answers. Most answers have flaws; "
    "use the FULL 1-5 scale and justify every point.\n\n"
    "Score anchors:\n"
    "5 = flawless: every rubric requirement met, fully correct, concise, nothing invented.\n"
    "4 = good: rubric met but with minor flaws (small omission, mild verbosity, weak edge "
    "case handling).\n"
    "3 = adequate: core of the rubric met but a clear requirement is missing, partially "
    "wrong, or padded with filler.\n"
    "2 = poor: addresses the topic but misses most rubric requirements or contains a "
    "significant error or invented fact.\n"
    "1 = failing: wrong, empty, off-topic, or fabricated.\n\n"
    "Mandatory deductions: -1 if any rubric requirement is unmet; -1 for any invented "
    "fact or fabricated precision; -1 for significant filler or repetition. "
    "A score of 5 must be rare and fully earned.\n\n"
    "Task category: {category}\n"
    "User request:\n{prompt}\n\n"
    "Grading rubric: {rubric}\n\n"
    "Assistant answer:\n{answer}\n\n"
    "First identify which rubric requirements are met or unmet, then score.\n"
    'Respond with ONLY a JSON object: {{"score": <integer 1-5>, '
    '"rationale": "<one sentence naming the unmet requirements, if any>"}}'
)


class MockJudge:
    """Deterministic keyword-overlap judge for CI and structural tests."""

    def score(self, case: QualityCase, answer: str) -> JudgeScore:
        if not answer.strip():
            return JudgeScore(score=1.0, rationale="Empty answer.")
        lowered = answer.lower()
        if not case.keywords:
            return JudgeScore(score=3.0, rationale="No keywords to check.")
        hits = sum(1 for keyword in case.keywords if keyword in lowered)
        fraction = hits / len(case.keywords)
        return JudgeScore(
            score=round(1.0 + 4.0 * fraction, 2),
            rationale=f"Matched {hits}/{len(case.keywords)} rubric keywords.",
        )


class OllamaJudge:
    """LLM judge backed by a local Ollama model."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "gemma4:12b",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def score(self, case: QualityCase, answer: str) -> JudgeScore:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            category=case.category,
            prompt=case.prompt[:4000],
            rubric=case.rubric,
            answer=answer[:4000],
        )
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            raw = str(response.json().get("message", {}).get("content", ""))
        except httpx.HTTPError as exc:
            return JudgeScore(score=0.0, rationale=f"Judge unavailable: {exc}", judged=False)
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return JudgeScore(score=0.0, rationale="Unparseable judge output.", judged=False)
        try:
            data = json.loads(match.group(0))
            score = float(data.get("score", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            return JudgeScore(score=0.0, rationale="Unparseable judge output.", judged=False)
        return JudgeScore(
            score=max(1.0, min(5.0, score)),
            rationale=str(data.get("rationale", ""))[:300],
        )


@dataclass
class CaseRunRecord:
    case_id: str
    category: str
    condition: str
    backend: str = ""
    route_type: str = ""
    success: bool = False
    latency_ms: int = 0
    cost_type: str = "unknown"
    premium_unit_used: bool = False
    privacy_violation: bool = False
    score: float = 0.0
    judged: bool = False
    judge_rationale: str = ""
    answer_preview: str = ""
    error: str | None = None


@dataclass
class ConditionSummary:
    condition: str
    cases: int = 0
    successes: int = 0
    judged_cases: int = 0
    mean_score: float = 0.0
    mean_score_by_category: dict[str, float] = field(default_factory=dict)
    premium_units: int = 0
    premium_rate: float = 0.0
    privacy_violations: int = 0
    privacy_violation_rate: float = 0.0
    mean_latency_ms: float = 0.0
    success_rate: float = 0.0


class QualityBenchRunner:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        mock: bool = True,
        judge: object | None = None,
        conditions: tuple[BenchCondition, ...] = DEFAULT_CONDITIONS,
        timeout_s: int = 120,
        cwd: Path | None = None,
        service_factory: Callable[[BenchCondition], SwitchboardCoreService] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.mock = mock
        self.judge = judge or (MockJudge() if mock else OllamaJudge())
        self.conditions = conditions
        self.timeout_s = timeout_s
        self.cwd = cwd or Path.cwd()
        self._service_factory = service_factory

    def run(
        self,
        *,
        limit: int | None = None,
        categories: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        cases = quality_cases()
        if categories:
            cases = [case for case in cases if case.category in categories]
        if limit is not None:
            cases = self._stratified_sample(cases, limit)
        records: list[CaseRunRecord] = []
        summaries: list[ConditionSummary] = []
        for condition in self.conditions:
            service = self._build_service(condition)
            condition_records = [
                self._run_case(service, condition, case) for case in cases
            ]
            records.extend(condition_records)
            summaries.append(self._summarize(condition, condition_records))
        return {
            "suite": "quality_bench",
            "mode": "mock" if self.mock else "real",
            "case_count": len(cases),
            "conditions": [asdict(summary) for summary in summaries],
            "records": [asdict(record) for record in records],
        }

    @staticmethod
    def _stratified_sample(cases: list[QualityCase], limit: int) -> list[QualityCase]:
        """Round-robin across categories so small pilots cover every category."""
        by_category: dict[str, list[QualityCase]] = {}
        for case in cases:
            by_category.setdefault(case.category, []).append(case)
        queues = list(by_category.values())
        sampled: list[QualityCase] = []
        index = 0
        while len(sampled) < limit and any(queues):
            queue = queues[index % len(queues)]
            if queue:
                sampled.append(queue.pop(0))
            index += 1
            if index > limit * len(queues) + len(cases):
                break
        return sampled

    def _run_case(
        self,
        service: SwitchboardCoreService,
        condition: BenchCondition,
        case: QualityCase,
    ) -> CaseRunRecord:
        record = CaseRunRecord(
            case_id=case.case_id,
            category=case.category,
            condition=condition.name,
        )
        started = time.perf_counter()
        try:
            response = service.ask(
                case.prompt,
                backend=condition.forced_backend,
                project="quality_bench",
                timeout_s=self.timeout_s,
                metadata={
                    "source": "quality_bench",
                    "bench_condition": condition.name,
                    "bench_case_id": case.case_id,
                },
                new_session=True,
            )
        except Exception as exc:  # defensive: a bench crash should not stop the run
            record.error = f"{type(exc).__name__}: {exc}"
            record.latency_ms = int((time.perf_counter() - started) * 1000)
            return record
        record.backend = response.backend
        record.success = response.success
        record.latency_ms = response.latency_ms or int((time.perf_counter() - started) * 1000)
        record.cost_type = response.cost_type.value
        record.premium_unit_used = response.cost_type in {
            BackendCostType.SUBSCRIPTION,
            BackendCostType.API,
        }
        # A privacy violation means the prompt was actually ANSWERED by a
        # non-local backend. Requests blocked by private mode never reach the
        # backend, so they count as failures, not leaks.
        record.privacy_violation = bool(
            case.expected_local_only
            and response.success
            and bool(response.content)
            and response.backend not in {"ollama", "switchboard", "time"}
        )
        record.error = response.error_message
        answer = response.content or ""
        record.answer_preview = answer[:160]
        if response.success and answer:
            judge_score = self.judge.score(case, answer)  # type: ignore[attr-defined]
            record.score = judge_score.score
            record.judged = judge_score.judged
            record.judge_rationale = judge_score.rationale
        return record

    def _summarize(
        self,
        condition: BenchCondition,
        records: list[CaseRunRecord],
    ) -> ConditionSummary:
        summary = ConditionSummary(condition=condition.name, cases=len(records))
        if not records:
            return summary
        judged = [record for record in records if record.judged]
        summary.successes = sum(1 for record in records if record.success)
        summary.judged_cases = len(judged)
        summary.success_rate = round(summary.successes / len(records), 4)
        summary.premium_units = sum(1 for record in records if record.premium_unit_used)
        summary.premium_rate = round(summary.premium_units / len(records), 4)
        summary.privacy_violations = sum(1 for record in records if record.privacy_violation)
        private_cases = [record for record in records if record.category == "private"]
        summary.privacy_violation_rate = round(
            summary.privacy_violations / len(private_cases), 4
        ) if private_cases else 0.0
        summary.mean_latency_ms = round(
            statistics.mean(record.latency_ms for record in records), 1
        )
        if judged:
            summary.mean_score = round(
                statistics.mean(record.score for record in judged), 3
            )
            by_category: dict[str, list[float]] = {}
            for record in judged:
                by_category.setdefault(record.category, []).append(record.score)
            summary.mean_score_by_category = {
                category: round(statistics.mean(scores), 3)
                for category, scores in sorted(by_category.items())
            }
        return summary

    def _build_service(self, condition: BenchCondition) -> SwitchboardCoreService:
        if self._service_factory is not None:
            return self._service_factory(condition)
        engine = create_db_engine(self.settings.database_url)
        init_db(engine)
        container = build_container(self.settings, engine)
        preferences = container.personal_config.preferences
        # Benchmark determinism: results must not depend on the developer's
        # live personal.yaml toggles or on live network providers. Grounding
        # cases are scored on honesty about missing providers, so providers
        # are pinned off; web search would add non-deterministic premium
        # routing; feedback storage would pollute the local DB and could
        # trigger retraining mid-run.
        preferences.claude_code_web_search = False
        preferences.finance_provider = ""
        preferences.news_provider = ""
        preferences.store_feedback_examples = False
        ollama_base_url = (
            container.personal_config.provider_base_url("ollama") or "http://localhost:11434"
        )
        if self.mock:
            registry, _ = mock_registry(None)
        else:
            registry = BackendRegistry.default(container, cwd=self.cwd)
        llm_router = None
        if condition.router_mode in {"llm", "hybrid"}:
            if self.mock:
                llm_router = LlmRouter(
                    complete=lambda _: '{"route_type": "unknown", "confidence": 0.0}'
                )
            else:
                llm_router = LlmRouter(
                    model=preferences.llm_router_model,
                    base_url=ollama_base_url,
                )
        memory_service = None
        if condition.memory and not self.mock:
            memory_service = SemanticMemoryService(
                memory_repository=container.memory_repository,
                embedding_repository=MemoryEmbeddingRepository(engine),
                embedding_model=preferences.embedding_model,
                base_url=ollama_base_url,
                top_k=preferences.semantic_memory_top_k,
            )
        learned_router = None
        if condition.router_mode == "learned":
            from switchboard.app.services.learned_router import LearnedRouter

            learned_router = LearnedRouter.from_file(
                preferences.router_weights_path,
                base_url=ollama_base_url,
                min_confidence=preferences.learned_router_min_confidence,
                expected_embedding_model=preferences.embedding_model,
            )
        # The learned tool dispatcher and sensitivity escalator are product
        # defaults; benchmark conditions measure the system WITH them (they
        # fail closed to deterministic behavior in mock mode or when their
        # weights are absent, so older setups are unaffected).
        tool_dispatcher = None
        sensitivity_escalator = None
        if not self.mock and condition.learned_assists:
            from switchboard.app.services.semantic_memory import (
                CachedEmbedder,
                OllamaEmbeddingClient,
            )
            from switchboard.app.services.sensitivity_escalator import (
                LearnedSensitivityEscalator,
            )
            from switchboard.app.services.tool_dispatcher import (
                LearnedToolDispatcher,
            )

            shared_embed = CachedEmbedder(
                OllamaEmbeddingClient(
                    base_url=ollama_base_url,
                    model=preferences.embedding_model,
                ).embed_classification
            ).embed
            if preferences.tool_dispatcher_enabled:
                tool_dispatcher = LearnedToolDispatcher.from_file(
                    preferences.tool_dispatcher_weights_path,
                    embed=shared_embed,
                    min_confidence=preferences.tool_dispatcher_min_confidence,
                    expected_embedding_model=preferences.embedding_model,
                )
            if preferences.sensitivity_escalator_enabled:
                sensitivity_escalator = LearnedSensitivityEscalator.from_file(
                    preferences.sensitivity_weights_path,
                    embed=shared_embed,
                    min_confidence=preferences.sensitivity_escalator_min_confidence,
                    expected_embedding_model=preferences.embedding_model,
                )
        return SwitchboardCoreService(
            registry=registry,
            metrics=container.backend_metrics_repository,
            container=container,
            router_mode=condition.router_mode,
            llm_router=llm_router,
            learned_router=learned_router,
            tool_dispatcher=tool_dispatcher,
            sensitivity_escalator=sensitivity_escalator,
            compression=(
                HeadroomCompressionLayer(
                    threshold_tokens=preferences.compression_threshold_tokens
                )
                if condition.compression
                else None
            ),
            semantic_memory=memory_service,
        )


def report_to_text(report: dict[str, object]) -> str:
    conditions: list[dict[str, Any]] = report["conditions"]  # type: ignore[assignment]
    lines = [
        f"Quality benchmark ({report['mode']}): "
        f"{report['case_count']} cases x {len(conditions)} conditions",
        "",
        f"{'condition':<22}{'score':>7}{'premium%':>10}{'privacy!':>10}"
        f"{'success%':>10}{'latency':>9}",
    ]
    for summary in conditions:
        lines.append(
            f"{summary['condition']:<22}"
            f"{summary['mean_score']:>7.2f}"
            f"{summary['premium_rate'] * 100:>9.1f}%"
            f"{summary['privacy_violations']:>10}"
            f"{summary['success_rate'] * 100:>9.1f}%"
            f"{summary['mean_latency_ms']:>7.0f}ms"
        )
    return "\n".join(lines)
