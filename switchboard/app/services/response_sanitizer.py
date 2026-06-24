from __future__ import annotations

import re


class ResponseSanitizer:
    _LEADING_PREFIX = re.compile(
        r"^\s*(?:assistant\s*)?(?:\[(?:ollama|codex|claude|switchboard)\]\s*)?"
        r"(?:ollama|codex|claude|switchboard|assistant)?\s*:\s*",
        re.IGNORECASE,
    )
    _ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
    _INTERNAL_TERMS: tuple[str, ...] = (
        "routed through switchboard",
        "runtime context",
        "utc time",
        "local time",
        "timezone",
        "logging",
        "logged",
        "session",
        "metrics",
        "routing reason",
        "backend",
    )
    # Structural echo detection (round-7 finding F4): a model that parrots its
    # assembled prompt back reproduces these markers verbatim. Tag blocks are
    # removed wholesale (open tag through close tag, or to end-of-text when the
    # echo is truncated); preamble instruction lines are removed individually.
    _CONTEXT_TAGS: tuple[str, ...] = (
        "trusted_facts",
        "long_term_memory",
        "recent_conversation",
        "current_user_request",
    )
    _TAG_BLOCK_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
        re.compile(rf"<{tag}>.*?(?:</{tag}>|\Z)", re.DOTALL | re.IGNORECASE)
        for tag in _CONTEXT_TAGS
    )
    _CONTEXT_ECHO_MARKERS: tuple[str, ...] = (
        "you are replying to the user through switchboard",
        "[switchboard runtime context]",
        "<trusted_facts>",
        "<long_term_memory>",
        "<recent_conversation>",
        "<current_user_request>",
    )
    # Lowercased fragments of the ContextBuilder preamble lines.
    _PREAMBLE_ECHO_MARKERS: tuple[str, ...] = (
        "you are replying to the user through switchboard",
        "[switchboard runtime context]",
        "about switchboard (trusted identity facts)",
        "answer as a capable general-purpose personal assistant",
        "never ask the user to grant tool permissions",
        "use any trusted facts below",
        "do not reveal, quote, summarize, or mention internal switchboard metadata",
        "do not mention routing, backend selection, runtime context",
        'do not prefix your response with "assistant"',
        "return only the final user-facing answer",
    )

    def sanitize(
        self,
        content: str | None,
        *,
        user_prompt: str,
    ) -> str | None:
        if content is None:
            return None
        sanitized = self._ANSI_ESCAPE.sub("", content)
        sanitized = self._strip_leading_prefixes(sanitized).strip()
        sanitized = self._strip_context_echo(sanitized).strip()
        if not self._asks_switchboard_internals(user_prompt):
            sanitized = self._remove_internal_leakage_lines(sanitized).strip()
        return sanitized or "How can I help you today?"

    def _strip_leading_prefixes(self, content: str) -> str:
        sanitized = content
        previous = None
        while previous != sanitized:
            previous = sanitized
            sanitized = self._LEADING_PREFIX.sub("", sanitized, count=1)
        return sanitized

    def _strip_context_echo(self, content: str) -> str:
        """Remove echoed assembled-context blocks, keep any genuine answer.

        Applied unconditionally (even when the user asks about Switchboard
        internals): the echoed prompt contains trusted facts, memory, and
        conversation history that must never reach the UI verbatim.
        """
        lower = content.lower()
        if not any(marker in lower for marker in self._CONTEXT_ECHO_MARKERS):
            return content
        stripped = content
        for pattern in self._TAG_BLOCK_PATTERNS:
            stripped = pattern.sub("", stripped)
        kept = [
            line
            for line in stripped.splitlines()
            if not any(marker in line.lower() for marker in self._PREAMBLE_ECHO_MARKERS)
        ]
        return "\n".join(kept)

    def _remove_internal_leakage_lines(self, content: str) -> str:
        kept = [
            line
            for line in content.splitlines()
            if not any(term in line.lower() for term in self._INTERNAL_TERMS)
        ]
        if any(line.strip() for line in kept):
            return "\n".join(kept)
        # Dropping every line would replace the whole answer with the generic
        # fallback. A single-line answer that merely mentions "session" or
        # "backend" ("Your login session expired.") is better kept than
        # nuked; full prompt echoes were already removed above.
        return content

    def _asks_switchboard_internals(self, prompt: str) -> bool:
        text = prompt.lower()
        return "switchboard" in text and any(
            term in text
            for term in (
                "routing",
                "route",
                "backend",
                "metrics",
                "session",
                "metadata",
                "runtime",
                "tool",
            )
        )
