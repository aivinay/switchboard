from __future__ import annotations

import re

from switchboard.app.models.internal import Sensitivity, TaskType
from switchboard.app.models.personal import CompressionResult
from switchboard.app.services.cost import CostEstimator


class ContextCompressionService:
    # Matches the <recent_conversation> block emitted by ContextBuilder.build,
    # where the open/close tags each sit on their own line. Non-greedy so an
    # injected close tag inside message content (always mid-line after the
    # "User:"/"Assistant:" prefix) cannot swallow later blocks.
    _HISTORY_BLOCK_RE = re.compile(
        r"^<recent_conversation>\n(?P<history>.*?)\n</recent_conversation>$",
        flags=re.DOTALL | re.MULTILINE,
    )

    def __init__(self, cost_estimator: CostEstimator, threshold_tokens: int = 1000) -> None:
        self.cost_estimator = cost_estimator
        self.threshold_tokens = threshold_tokens

    def compress(
        self,
        prompt: str,
        task_type: TaskType | None = None,
        sensitivity: Sensitivity | None = None,
    ) -> CompressionResult:
        original_tokens = self.cost_estimator.estimate_text_tokens(prompt)
        if original_tokens <= self.threshold_tokens:
            return self._noop_result(original_tokens)
        compressed = self._heuristic_compress(prompt, task_type, sensitivity)
        compressed_tokens = self.cost_estimator.estimate_text_tokens(compressed)
        return CompressionResult(
            original_estimated_tokens=original_tokens,
            compressed_estimated_tokens=compressed_tokens,
            compression_used=True,
            estimated_tokens_saved=max(0, original_tokens - compressed_tokens),
            compression_ratio=round(compressed_tokens / original_tokens, 4),
            compressed_prompt=compressed,
            warning="Heuristic compression used; review before sending to a scarce model.",
        )

    def compress_assembled_context(self, context: str) -> CompressionResult:
        """Structure-aware compression for assembled session contexts.

        Only the ``<recent_conversation>`` block is compressible. The
        instruction preamble, ``<trusted_facts>``, ``<long_term_memory>``,
        and ``<current_user_request>`` blocks are grounded truth and must
        survive verbatim (byte-identical, including newlines) — dropping a
        trusted fact turns compression into a fabrication risk. If, after
        compressing history, the context is still over the token threshold,
        we do NOT start eating fact blocks: facts win over budget.

        Text that does not match the assembled-context structure (raw
        prompts, pathological inputs) falls back to the whole-text heuristic.
        """
        original_tokens = self.cost_estimator.estimate_text_tokens(context)
        if original_tokens <= self.threshold_tokens:
            return self._noop_result(original_tokens, scope="none")
        if not self._looks_like_assembled_context(context):
            result = self.compress(context)
            return result.model_copy(update={"scope": "whole_text"})
        match = self._HISTORY_BLOCK_RE.search(context)
        if match is None:
            # Assembled context with no conversation history: every remaining
            # block is grounded truth, so there is nothing safe to compress.
            return self._noop_result(original_tokens, scope="history_only")
        history = match.group("history")
        compressed_history = self._heuristic_compress(history, None, None)
        if len(compressed_history) >= len(history):
            return self._noop_result(original_tokens, scope="history_only")
        compressed = (
            context[: match.start("history")] + compressed_history + context[match.end("history") :]
        )
        compressed_tokens = self.cost_estimator.estimate_text_tokens(compressed)
        if compressed_tokens >= original_tokens:
            return self._noop_result(original_tokens, scope="history_only")
        return CompressionResult(
            original_estimated_tokens=original_tokens,
            compressed_estimated_tokens=compressed_tokens,
            compression_used=True,
            estimated_tokens_saved=original_tokens - compressed_tokens,
            compression_ratio=round(compressed_tokens / original_tokens, 4),
            compressed_prompt=compressed,
            warning="Heuristic compression used; review before sending to a scarce model.",
            scope="history_only",
        )

    def _looks_like_assembled_context(self, context: str) -> bool:
        if not context.rstrip().endswith("</current_user_request>"):
            return False
        if context.startswith("<current_user_request>\n"):
            return True
        return "\n<current_user_request>\n" in context

    def _noop_result(self, original_tokens: int, *, scope: str | None = None) -> CompressionResult:
        return CompressionResult(
            original_estimated_tokens=original_tokens,
            compressed_estimated_tokens=original_tokens,
            compression_used=False,
            estimated_tokens_saved=0,
            compression_ratio=1.0,
            scope=scope,
        )

    def _heuristic_compress(
        self,
        prompt: str,
        task_type: TaskType | None,
        sensitivity: Sensitivity | None,
    ) -> str:
        normalized = re.sub(r"\s+", " ", prompt).strip()
        first_line = prompt.strip().splitlines()[0][:800] if prompt.strip() else ""
        code_snippets = re.findall(r"```.*?```", prompt, flags=re.DOTALL)
        error_lines = [
            line
            for line in prompt.splitlines()
            if any(marker in line.lower() for marker in {"traceback", "error", "exception"})
        ][:20]
        task_note = f"Task: {first_line}"
        if task_type in {TaskType.CODING, TaskType.DEBUGGING} and code_snippets:
            task_note += "\n\nCode/error context to preserve:\n" + "\n\n".join(code_snippets[:2])
        elif error_lines:
            task_note += "\n\nError context to preserve:\n" + "\n".join(error_lines)
        if sensitivity in {
            Sensitivity.CONFIDENTIAL,
            Sensitivity.REGULATED,
            Sensitivity.PRIVATE_PERSONAL,
        }:
            task_note = (
                "Sensitive/private content warning: keep this local unless explicitly approved.\n"
                + task_note
            )

        compressed = (
            "Heuristic compression. Preserve the user's task and critical details.\n\n"
            f"{task_note}\n\n"
            f"Opening context:\n{normalized[:500]}\n\n"
            f"Most recent context:\n{normalized[-1000:]}"
        )
        original_tokens = self.cost_estimator.estimate_text_tokens(prompt)
        if self.cost_estimator.estimate_text_tokens(compressed) >= original_tokens:
            compressed = (
                "Heuristic compression. Preserve the user's task and critical details.\n\n"
                f"Most recent context:\n{normalized[-2000:]}"
            )
        return compressed
