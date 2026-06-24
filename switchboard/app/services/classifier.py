from __future__ import annotations

import re

from switchboard.app.models.internal import (
    ClassificationResult,
    Complexity,
    LatencyClass,
    NormalizedRequest,
    Sensitivity,
    TaskType,
)
from switchboard.app.utils.secret_patterns import contains_secret_format

CODE_HINTS = {
    "```",
    "401",
    "api error",
    "code",
    "python",
    "regex",
    "regular expression",
    "sql query",
    "terraform",
    "typescript",
    "javascript",
    "stack trace",
    "traceback",
    "debug",
    "refactor",
    "function",
    "unit test",
    "variable",
    "snippet",
}
DEBUG_HINTS = {
    "401",
    "auth error",
    "authentication bug",
    "bug report",
    "debug",
    "duplicate rows",
    "failing test",
    "fix",
    "compile",
    "exception",
    "race condition",
    "traceback",
    "stack trace",
    "segmentation fault",
    "token leak",
}
FILENAME_PATTERN = re.compile(
    r"\b[\w\-]+\.(py|ts|tsx|js|jsx|go|rs|java|rb|php|sql|yaml|yml|json)\b"
)
PII_PATTERN = re.compile(
    r"\b(\d{3}-\d{2}-\d{4}|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b",
    re.IGNORECASE,
)
REGULATED_HINTS = {
    "bank",
    "bank statement",
    "bank letter",
    "compensation",
    "custody",
    "financial",
    "finance",
    "debt",
    "divorce",
    "equity",
    "health insurance",
    "immigration",
    "insurance denial",
    "investment",
    "lawyer",
    "legal",
    "landlord",
    "medication",
    "contract",
    "mortgage",
    "payroll",
    "medical",
    "patient",
    "diagnosis",
    "salary",
    "settlement",
    "tax",
    "therapy",
    "therapist",
    "visa",
    "gdpr",
    "hipaa",
    "regulated",
}
LEGAL_HINTS = {
    "contract",
    "custody",
    "divorce",
    "dpa",
    "immigration",
    "landlord",
    "lawyer",
    "legal",
    "liability",
    "clause",
    "settlement",
    "terms",
    "nda",
    "visa",
}
MEDICAL_HINTS = {
    "clinical",
    "clinic",
    "diagnosis",
    "doctor",
    "health",
    "health insurance",
    "medical",
    "medication",
    "patient",
    "therapy",
    "therapist",
    "treatment",
}
FINANCIAL_HINTS = {
    "account number",
    "bank",
    "bank statement",
    "budget",
    "capital",
    "compensation",
    "creditor",
    "debt",
    "equity",
    "financial",
    "finance",
    "investment",
    "mortgage",
    "payroll",
    "salary",
    "tax",
    "valuation",
    "forecast",
}
CONFIDENTIAL_HINTS = {
    "access token",
    "api key",
    "acquisition",
    "board deck",
    "confidential",
    "credentials",
    "customer complaints",
    "secret",
    "proprietary",
    "internal only",
    "customer data",
    "customer message",
    "customer messages",
    "customer feedback",
    "support ticket",
    "employee",
    "grievance",
    "incident log",
    "production incident",
    "token",
}
# Physical-health condition names that mark a prompt as private wherever
# they appear (round-4 tester finding: "i think i have an STD" leaked to a
# subscription backend while Ollama was down — the keyword floor is the last
# line of defense and must work without any learned component). Word-bounded
# so code identifiers never match ("std::vector", "archive" contains "hiv"),
# and guarded against C++/stdlib vocabulary ("using namespace std",
# "std.h", "the std library").
PERSONAL_HEALTH_TOPIC_PATTERN = re.compile(
    r"(?<!namespace )(?<!namespace\t)"
    r"\b(?:stds?|stis?|hiv|herpes|chlamydia|gonorrh(?:o)?ea|syphilis)\b"
    r"(?!\s*::|\s*\.\s*h\b|\s+(?:lib|libs|library|crate|header|module|namespace)\b)"
)
# First-person illness disclosures without a named condition. Both halves
# are required so "i think i have a bug in my code" stays non-sensitive
# while "i think i have an infection" stays on the local model.
PERSONAL_HEALTH_DISCLOSURE_PATTERN = re.compile(
    r"\b(?:i think i have|i might have|i'?m worried i have|do i have|"
    r"i'?ve got|i caught|i contracted)\b"
    r"[^.!?\n]{0,60}?"
    r"\b(?:infection|disease|illness|virus|cancer|tumou?r|condition|disorder)\b"
)
PRIVATE_PERSONAL_HINTS = {
    "date of birth",
    "private",
    # Bare "personal" is too broad ("personal website", "personal images" in
    # build requests); require a possessive or data-bearing phrase.
    "my personal",
    "personal details",
    "personal information",
    "personal data",
    "personal life",
    "passport",
    "do not share",
    "my diary",
    "my notes",
    "my medical",
    "my finances",
    # Mental health and emotional wellbeing stay on the local model.
    "depression",
    "depressed",
    "anxiety",
    "anxious",
    "mental health",
    "panic attack",
    "suicidal",
    "self-harm",
    "self harm",
    "grief",
    "grieving",
    "lonely",
    "loneliness",
    "burnout",
    "my relationship",
    "breakup",
    "break up",
    "my salary",
    "my therapist",
    # Physical-health and diagnosis phrasings (round-4 tester finding). Bare
    # condition acronyms (STD/STI/HIV/...) live in
    # PERSONAL_HEALTH_TOPIC_PATTERN so they stay word-bounded.
    "diagnosed with",
    "my symptoms",
    "my test results",
    "test results came back",
    "tested positive",
    "pregnant",
    "pregnancy",
    "abortion",
    "miscarriage",
    "addiction",
    "addicted",
    "rehab",
    "overdose",
}
PROMPT_INJECTION_HINTS = {
    "allow_cloud=false",
    "bypass",
    "bypass governance",
    "bypass policy",
    "bypass safety",
    "call frontier",
    "disable policy",
    "disable private mode",
    "disregard safety",
    "do not classify this as regulated",
    "forget previous rules",
    "ignore local-first",
    "ignore policy",
    "ignore private mode",
    "ignore privacy settings",
    "ignore routing policy",
    "ignore the router",
    "ignore the word private",
    "mark as public",
    "mark this as public",
    "not sensitive",
    "override tenant policy",
    "pretend cloud is allowed",
    "route this",
    "safety override",
    "send to cloud",
    "system override",
    "treat this as low risk",
    "use cloud despite",
    "use cloud anyway",
}
SIMPLE_TASK_HINTS = {
    "summarise",
    "summarize",
    "summary",
    "classify",
    "extract",
    "rewrite",
    "proofread",
}
REASONING_HINTS = {
    "compare",
    "tradeoff",
    "trade-off",
    "architecture",
    "design",
    "plan",
    "strategy",
    "analyse",
    "analyze",
    "evaluate",
    "risk",
}


class RequestClassifier:
    """A transparent heuristic classifier that can be replaced by ML later."""

    def classify(self, request: NormalizedRequest) -> ClassificationResult:
        text = "\n".join(message.content for message in request.messages)
        lower = text.lower()
        metadata = request.metadata
        reason_codes: list[str] = []
        uncertainty_reasons: list[str] = []
        matched_task_groups: list[str] = []

        if any(hint in lower for hint in PROMPT_INJECTION_HINTS):
            reason_codes.extend(["PROMPT_INJECTION_ATTEMPT", "SECURITY_ROUTING_OVERRIDE_ATTEMPT"])
            uncertainty_reasons.append("PROMPT_INJECTION_ATTEMPT")

        task_type = self._task_type(lower, text, metadata, reason_codes, matched_task_groups)
        complexity = self._complexity(lower, text, task_type, reason_codes)
        sensitivity = self._sensitivity(lower, text, metadata, reason_codes)
        latency_class = self._latency_class(lower, text, metadata, reason_codes)
        confidence = self._confidence(
            task_type,
            complexity,
            sensitivity,
            text,
            reason_codes,
            matched_task_groups,
            uncertainty_reasons,
        )

        return ClassificationResult(
            task_type=task_type,
            complexity=complexity,
            sensitivity=sensitivity,
            latency_class=latency_class,
            confidence=confidence,
            reason_codes=reason_codes,
            uncertainty_reasons=uncertainty_reasons,
        )

    def _task_type(
        self,
        lower: str,
        text: str,
        metadata: dict[str, object],
        reason_codes: list[str],
        matched_task_groups: list[str],
    ) -> TaskType:
        explicit = metadata.get("task_type")
        if isinstance(explicit, str) and explicit in set(TaskType):
            reason_codes.extend(
                ["CLASSIFIER_METADATA_TASK_TYPE", "GOVERNANCE_METADATA_TASK_TYPE_USED"]
            )
            return TaskType(explicit)

        has_code_block = "```" in lower
        has_stack_trace = any(hint in lower for hint in {"stack trace", "traceback", "exception"})
        has_filename = FILENAME_PATTERN.search(text) is not None
        has_debug = any(hint in lower for hint in DEBUG_HINTS) or has_filename
        has_summary = any(
            hint in lower
            for hint in {
                "bullet summary",
                "bullets",
                "list the deadlines",
                "summarise",
                "summarize",
                "summary",
                "tldr",
                "tl;dr",
            }
        )
        has_rewrite = any(
            hint in lower
            for hint in {
                "draft a",
                "draft a short reply",
                "make this better",
                "make this clearer",
                "polish",
                "proofread",
                "rewrite",
            }
        )
        has_extraction = any(
            hint in lower
            for hint in {
                "entities",
                "extract",
                "fields",
                "last four",
                "parse",
                "turn this into json",
            }
        )
        has_classification = (
            any(hint in lower for hint in {"classify", "categorise", "categorize", "label this"})
            and "do not classify" not in lower
        )
        has_factual_qa = any(
            hint in lower for hint in {"what is", "who is", "when did", "where is", "answer this"}
        )
        has_architecture = any(
            hint in lower
            for hint in {
                "architecture",
                "system design",
                "technical design",
                "database schema",
                "benchmark plan",
                "queue",
                "idempotency",
                "terraform",
            }
        ) or ("design" in lower and any(hint in lower for hint in {"schema", "scaling"}))
        has_planning = any(hint in lower for hint in {"plan", "strategy", "roadmap", "launch"})
        has_reasoning = any(
            hint in lower
            for hint in {
                "compare",
                "tradeoff",
                "trade-off",
                "analyse",
                "analyze",
                "evaluate",
                "explain where",
                "risk",
                "should i",
                "reason through",
                "financial model",
                "5-year",
            }
        )
        has_schema_creation = "json schema" in lower and any(
            hint in lower for hint in {"create", "generate", "write"}
        )

        if has_code_block or has_stack_trace or has_debug:
            matched_task_groups.append("coding")
        if has_summary:
            matched_task_groups.append("summarisation")
        if has_rewrite:
            matched_task_groups.append("rewrite")
        if has_extraction:
            matched_task_groups.append("extraction")
        if has_classification:
            matched_task_groups.append("classification")
        if has_factual_qa:
            matched_task_groups.append("factual_qa")
        if has_architecture:
            matched_task_groups.append("architecture")
        if has_planning:
            matched_task_groups.append("planning")
        if has_reasoning:
            matched_task_groups.append("reasoning")

        if has_stack_trace:
            matched_task_groups.append("debugging")
            reason_codes.extend(
                [
                    "CLASSIFIER_CODE_HINT",
                    "CTO_CODING_TASK_DETECTED",
                    "STACK_TRACE_DETECTED",
                ]
            )
            return TaskType.DEBUGGING
        if has_code_block:
            matched_task_groups.append("coding")
            reason_codes.extend(
                ["CLASSIFIER_CODE_HINT", "CTO_CODING_TASK_DETECTED", "CODE_BLOCK_DETECTED"]
            )
            return TaskType.CODING
        if has_debug and not has_summary and not has_classification:
            matched_task_groups.append("debugging")
            reason_codes.extend(["CLASSIFIER_CODE_HINT", "CTO_CODING_TASK_DETECTED"])
            return TaskType.DEBUGGING
        if has_schema_creation:
            matched_task_groups.append("coding")
            reason_codes.extend(["CLASSIFIER_CODE_HINT", "CTO_CODING_TASK_DETECTED"])
            return TaskType.CODING
        if "refactor" in lower:
            matched_task_groups.append("coding")
            reason_codes.extend(["CLASSIFIER_CODE_HINT", "CTO_CODING_TASK_DETECTED"])
            return TaskType.CODING
        if has_summary:
            matched_task_groups.append("summarisation")
            reason_codes.extend(
                [
                    "CLASSIFIER_SUMMARY_HINT",
                    "CTO_SUMMARISATION_TASK_DETECTED",
                    "SIMPLE_SUMMARISATION",
                ]
            )
            return TaskType.SUMMARISATION
        if has_rewrite:
            matched_task_groups.append("rewrite")
            reason_codes.extend(["CLASSIFIER_REWRITE_HINT", "SIMPLE_REWRITE"])
            return TaskType.REWRITE
        if has_extraction:
            matched_task_groups.append("extraction")
            reason_codes.extend(["CLASSIFIER_EXTRACTION_HINT", "CTO_EXTRACTION_TASK_DETECTED"])
            return TaskType.EXTRACTION
        if has_classification:
            matched_task_groups.append("classification")
            reason_codes.extend(
                ["CLASSIFIER_CLASSIFICATION_HINT", "CTO_CLASSIFICATION_TASK_DETECTED"]
            )
            return TaskType.CLASSIFICATION
        if any(
            hint in lower for hint in {"write a story", "poem", "creative", "brainstorm", "tagline"}
        ):
            matched_task_groups.append("creative")
            reason_codes.extend(["CLASSIFIER_CREATIVE_HINT", "CTO_CREATIVE_TASK_DETECTED"])
            return TaskType.CREATIVE
        if has_factual_qa:
            matched_task_groups.append("factual_qa")
            reason_codes.extend(["CLASSIFIER_FACTUAL_QA_HINT", "CTO_FACTUAL_QA_TASK_DETECTED"])
            return TaskType.FACTUAL_QA
        if any(hint in lower for hint in CODE_HINTS):
            matched_task_groups.append("coding")
            reason_codes.extend(["CLASSIFIER_CODE_HINT", "CTO_CODING_TASK_DETECTED"])
            return TaskType.CODING
        if has_architecture:
            matched_task_groups.append("architecture")
            reason_codes.extend(["CLASSIFIER_REASONING_HINT", "ARCHITECTURE_DESIGN_REQUEST"])
            return TaskType.ARCHITECTURE_DESIGN
        if (
            has_reasoning
            and not has_planning
            and any(
                hint in lower
                for hint in {"compare", "tradeoff", "trade-off", "analyse", "analyze", "evaluate"}
            )
        ):
            matched_task_groups.append("reasoning")
            reason_codes.extend(["CLASSIFIER_REASONING_HINT", "CTO_REASONING_TASK_DETECTED"])
            return TaskType.REASONING
        if has_planning:
            matched_task_groups.append("planning")
            reason_codes.extend(["CLASSIFIER_REASONING_HINT", "PLANNING_REQUEST"])
            return TaskType.PLANNING
        if any(hint in lower for hint in {"tool", "agent", "browser", "api call", "function call"}):
            matched_task_groups.append("agentic")
            reason_codes.extend(["CLASSIFIER_AGENTIC_HINT", "CTO_AGENTIC_TOOL_USE_DETECTED"])
            return TaskType.AGENTIC_TOOL_USE
        if has_reasoning:
            matched_task_groups.append("reasoning")
            reason_codes.extend(["CLASSIFIER_REASONING_HINT", "CTO_REASONING_TASK_DETECTED"])
            return TaskType.REASONING

        reason_codes.extend(["CLASSIFIER_UNKNOWN_TASK", "CTO_TASK_TYPE_UNCLEAR"])
        return TaskType.UNKNOWN

    def _complexity(
        self,
        lower: str,
        text: str,
        task_type: TaskType,
        reason_codes: list[str],
    ) -> Complexity:
        word_count = len(re.findall(r"\w+", text))
        if (
            word_count > 900
            or "financial model" in lower
            or "5-year" in lower
            or "race condition" in lower
            or "rollback-safe" in lower
            or "schema migration" in lower
            or "multi-step" in lower
            or (task_type == TaskType.REASONING and "risk" in lower)
            or task_type == TaskType.AGENTIC_TOOL_USE
            or (
                task_type in {TaskType.CODING, TaskType.DEBUGGING}
                and any(
                    hint in lower
                    for hint in {
                        "race condition",
                        "segmentation fault",
                        "stack trace",
                        "traceback",
                    }
                )
            )
            or (
                task_type
                in {
                    TaskType.ARCHITECTURE_DESIGN,
                    TaskType.PLANNING,
                    TaskType.REASONING,
                    TaskType.AGENTIC_TOOL_USE,
                }
                and any(hint in lower for hint in REASONING_HINTS)
            )
        ):
            reason_codes.extend(
                [
                    "CLASSIFIER_HIGH_COMPLEXITY",
                    "CTO_HIGH_COMPLEXITY_REQUEST",
                    "COMPLEX_REASONING_REQUEST",
                ]
            )
            return Complexity.HIGH

        if (
            word_count > 250
            or task_type
            in {
                TaskType.FACTUAL_QA,
                TaskType.REASONING,
                TaskType.CODING,
                TaskType.DEBUGGING,
                TaskType.ARCHITECTURE_DESIGN,
                TaskType.PLANNING,
            }
            or (
                task_type == TaskType.CREATIVE
                and any(hint in lower for hint in {"evaluate", "risk", "risks"})
            )
            or (
                task_type != TaskType.UNKNOWN
                and any(hint in lower for hint in {"compare", "analyse", "analyze"})
            )
        ):
            reason_codes.extend(["CLASSIFIER_MEDIUM_COMPLEXITY", "CTO_MEDIUM_COMPLEXITY_REQUEST"])
            return Complexity.MEDIUM

        reason_codes.extend(["CLASSIFIER_LOW_COMPLEXITY", "CFO_LOW_COMPLEXITY_COST_OPTIMISABLE"])
        return Complexity.LOW

    def _sensitivity(
        self, lower: str, text: str, metadata: dict[str, object], reason_codes: list[str]
    ) -> Sensitivity:
        explicit = metadata.get("sensitivity")
        if isinstance(explicit, str) and explicit in set(Sensitivity):
            reason_codes.extend(
                ["CLASSIFIER_METADATA_SENSITIVITY", "GOVERNANCE_METADATA_SENSITIVITY_USED"]
            )
            return Sensitivity(explicit)

        if any(hint in lower for hint in REGULATED_HINTS):
            if any(hint in lower for hint in FINANCIAL_HINTS):
                reason_codes.append("FINANCIAL_PLANNING_DETECTED")
            if any(hint in lower for hint in LEGAL_HINTS):
                reason_codes.append("LEGAL_SENSITIVE_CONTENT")
            if any(hint in lower for hint in MEDICAL_HINTS):
                reason_codes.append("MEDICAL_SENSITIVE_CONTENT")
            reason_codes.extend(["CLASSIFIER_REGULATED_HINT", "SECURITY_REGULATED_DATA_DETECTED"])
            return Sensitivity.REGULATED
        # Keyword-floor backstop (round-7 finding F3): bare secret material —
        # AKIA keys, JWTs, PEM blocks, env-style assignments, credentialed
        # URLs — carries no privacy keyword, yet the verbatim current request
        # would reach a subscription backend. A recognized secret FORMAT marks
        # the request CONFIDENTIAL so existing policy keeps it local. Checked
        # against the original text: AKIA/PEM formats are case-sensitive.
        if contains_secret_format(text):
            reason_codes.extend(
                ["CLASSIFIER_SECRET_FORMAT_HINT", "SECURITY_SECRET_MATERIAL_DETECTED"]
            )
            return Sensitivity.CONFIDENTIAL
        if any(hint in lower for hint in CONFIDENTIAL_HINTS):
            reason_codes.extend(
                ["CLASSIFIER_CONFIDENTIAL_HINT", "SECURITY_CONFIDENTIAL_DATA_DETECTED"]
            )
            return Sensitivity.CONFIDENTIAL
        if (
            any(hint in lower for hint in PRIVATE_PERSONAL_HINTS)
            or PII_PATTERN.search(lower)
            or PERSONAL_HEALTH_TOPIC_PATTERN.search(lower)
            or PERSONAL_HEALTH_DISCLOSURE_PATTERN.search(lower)
        ):
            reason_codes.extend(["CLASSIFIER_PRIVATE_PERSONAL_HINT", "PRIVATE_PERSONAL_CONTENT"])
            return Sensitivity.PRIVATE_PERSONAL
        if "internal" in lower:
            reason_codes.extend(["CLASSIFIER_INTERNAL_HINT", "SECURITY_INTERNAL_DATA_DETECTED"])
            return Sensitivity.INTERNAL

        reason_codes.extend(["CLASSIFIER_PUBLIC_DEFAULT", "SECURITY_PUBLIC_DATA_ASSUMED"])
        return Sensitivity.PUBLIC

    def _latency_class(
        self,
        lower: str,
        text: str,
        metadata: dict[str, object],
        reason_codes: list[str],
    ) -> LatencyClass:
        explicit = metadata.get("latency_class")
        if isinstance(explicit, str) and explicit in set(LatencyClass):
            reason_codes.extend(["CLASSIFIER_METADATA_LATENCY", "GOVERNANCE_METADATA_LATENCY_USED"])
            return LatencyClass(explicit)

        if len(text) > 6000 or any(hint in lower for hint in {"batch", "overnight", "full report"}):
            reason_codes.extend(["CLASSIFIER_BATCH_LATENCY", "CTO_BATCH_LATENCY_ACCEPTABLE"])
            return LatencyClass.BATCH

        reason_codes.extend(["CLASSIFIER_INTERACTIVE_DEFAULT", "CTO_INTERACTIVE_LATENCY_EXPECTED"])
        return LatencyClass.INTERACTIVE

    def _confidence(
        self,
        task_type: TaskType,
        complexity: Complexity,
        sensitivity: Sensitivity,
        text: str,
        reason_codes: list[str],
        matched_task_groups: list[str],
        uncertainty_reasons: list[str],
    ) -> float:
        confidence = 0.72
        if task_type == TaskType.UNKNOWN:
            confidence -= 0.25
            uncertainty_reasons.append("TASK_TYPE_UNCLEAR")
        if sensitivity == Sensitivity.UNKNOWN:
            confidence -= 0.1
        if complexity == Complexity.HIGH:
            confidence -= 0.02
        if any(code.startswith("CLASSIFIER_METADATA") for code in reason_codes):
            confidence += 0.12
        if any(
            code in reason_codes
            for code in {
                "CODE_BLOCK_DETECTED",
                "STACK_TRACE_DETECTED",
                "FINANCIAL_PLANNING_DETECTED",
                "LEGAL_SENSITIVE_CONTENT",
                "MEDICAL_SENSITIVE_CONTENT",
                "SIMPLE_SUMMARISATION",
                "SIMPLE_REWRITE",
            }
        ):
            confidence += 0.08
        if len(set(matched_task_groups)) > 1:
            confidence -= 0.15
            uncertainty_reasons.append("MULTIPLE_TASK_TYPE_HINTS")
        if len(re.findall(r"\w+", text)) <= 3 and task_type == TaskType.UNKNOWN:
            confidence -= 0.15
            uncertainty_reasons.append("SHORT_AMBIGUOUS_PROMPT")
        return max(0.05, min(0.98, round(confidence, 2)))
