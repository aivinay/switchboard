from __future__ import annotations

import json
import re

from switchboard.app.models.catalogue import QualityTier
from switchboard.app.models.internal import ClassificationResult, Complexity, TaskType
from switchboard.app.models.personal import PersonalRouteResponse

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

SPECULATIVE_SUMMARY_MARKERS = (
    "may",
    "might",
    "likely",
    "no specific details",
    "not mentioned",
    "can be obtained by contacting",
    "may impact",
    "no replacement",
    "proxy",
)


class AnswerQualityHeuristic:
    def assess(
        self,
        answer: str | None,
        prompt: str,
        classification: ClassificationResult,
        route: PersonalRouteResponse,
        selected_tier: QualityTier | None,
    ) -> tuple[bool, list[str], str | None]:
        notes: list[str] = []
        normalized = (answer or "").strip().lower()

        if not normalized:
            notes.append("No answer was produced.")
        if normalized and len(normalized) < 80 and classification.complexity == Complexity.HIGH:
            notes.append("Answer appears too short for the requested analysis.")
        if (
            normalized.startswith("demo mock response only")
            and classification.complexity == Complexity.HIGH
        ):
            notes.append("Mock/demo answer is not a real model answer for this complex task.")
        if any(marker in normalized for marker in {"i cannot", "i can't", "error", "unavailable"}):
            notes.append("Answer may contain a refusal or provider failure.")
        if classification.task_type == TaskType.EXTRACTION and not any(
            marker in normalized for marker in {"{", "[", ":", "-"}
        ):
            notes.append("Structured extraction requested, but output does not appear structured.")
        if (
            classification.complexity == Complexity.HIGH
            and selected_tier is not None
            and selected_tier != QualityTier.FRONTIER
        ):
            notes.append("Local answer may be insufficient for this complex reasoning task.")
        if route.confidence < 0.55:
            notes.append("Classifier confidence was low for this task.")
        notes.extend(self._format_notes(prompt, answer or "", classification))

        suggested = None
        if notes:
            suggested = self._suggested_next_step(notes)
        return bool(notes), notes, suggested

    def _format_notes(
        self,
        prompt: str,
        answer: str,
        classification: ClassificationResult,
    ) -> list[str]:
        prompt_lower = prompt.lower()
        notes: list[str] = []
        source = (
            self.source_text_from_prompt(prompt)
            if classification.task_type == TaskType.SUMMARISATION
            else None
        )
        source_fact_count = self.distinct_fact_count(source) if source else None
        expected_items = self.requested_bullet_count(prompt)
        if expected_items is not None:
            actual_items = self.bullet_count(answer)
            empty_items = self.empty_bullet_count(answer)
            if actual_items == 0:
                notes.append(
                    "You asked for bullets, but the response does not appear to use "
                    "bullet formatting."
                )
            if empty_items:
                notes.append("The response contains empty bullet placeholders.")
            elif source_fact_count is not None and source_fact_count < expected_items:
                if actual_items > source_fact_count:
                    notes.append(
                        "You asked for "
                        f"{expected_items} bullets, but the source appears to contain "
                        f"only {source_fact_count} distinct facts. The response may be "
                        "padded."
                    )
            elif actual_items != expected_items:
                notes.append(
                    "You asked for "
                    f"{expected_items} bullets, but the response appears to contain "
                    f"{actual_items}."
                )
        if source and self._has_speculative_summary_marker(answer):
            notes.append(
                "Some bullets appear speculative rather than directly supported by the source."
            )
        if source and self._is_short_source(source) and self.bullet_count(answer) >= 4:
            notes.append("Source is short; summary may be padded.")
        limitation_mismatch = self._source_limitation_note_mismatch(answer)
        if source and limitation_mismatch:
            notes.append(limitation_mismatch)
        if self._has_source_summary_warning(notes):
            notes.append(
                "For source-grounded summaries, prefer fewer accurate bullets over "
                "invented bullets."
            )
        if self._asks_for_json(prompt_lower) and not self._looks_like_json(answer):
            notes.append("You asked for JSON, but the response does not appear to be valid JSON.")
        if "table" in prompt_lower and not self._looks_like_table(answer):
            notes.append("You asked for a table, but the response does not appear to be a table.")
        if "one sentence" in prompt_lower or "1 sentence" in prompt_lower:
            sentence_count = self._sentence_count(answer)
            if sentence_count > 1:
                notes.append(
                    "You asked for one sentence, but the response appears to contain "
                    f"{sentence_count} sentences."
                )
        return notes

    def requested_bullet_count(self, prompt: str) -> int | None:
        prompt_lower = prompt.lower()
        match = re.search(r"\b(\d+)\s+(?:bullet\s+points?|bullets?|points?)\b", prompt_lower)
        if match:
            return int(match.group(1))
        for word, count in NUMBER_WORDS.items():
            if re.search(rf"\b{word}\s+(?:bullet\s+points?|bullets?|points?)\b", prompt_lower):
                return count
        return None

    def bullet_count(self, answer: str) -> int:
        count = 0
        for line in answer.splitlines():
            stripped = line.strip()
            if re.match(r"^([-*+•]|\d+[.)])\s+", stripped):
                count += 1
        return count

    def empty_bullet_count(self, answer: str) -> int:
        count = 0
        for line in answer.splitlines():
            stripped = line.strip()
            if re.match(r"^([-*+•]|\d+[.)])$", stripped):
                count += 1
        return count

    def source_text_from_prompt(self, prompt: str) -> str | None:
        marker_match = re.search(
            r"\b(?:source|text|email|ticket|transcript|note|notes)\s*:\s*(.+)$",
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if marker_match:
            return marker_match.group(1).strip().strip("\"'")
        if ":" in prompt:
            before, after = prompt.split(":", 1)
            if len(before.split()) >= 2 and len(after.split()) >= 2:
                return after.strip().strip("\"'")
        return None

    def distinct_fact_count(self, source: str | None) -> int:
        if not source:
            return 0
        clauses = re.split(
            r"[.;!?]\s+|\s+(?:but|and|also)\s+",
            source.strip(),
            flags=re.IGNORECASE,
        )
        facts = [
            clause
            for clause in clauses
            if len(re.findall(r"\b[\w']+\b", clause.strip())) >= 2
        ]
        return max(1, len(facts)) if source.strip() else 0

    def _has_speculative_summary_marker(self, answer: str) -> bool:
        normalized = answer.lower()
        for marker in SPECULATIVE_SUMMARY_MARKERS:
            if " " in marker:
                if marker in normalized:
                    return True
            elif re.search(rf"\b{re.escape(marker)}\b", normalized):
                return True
        return False

    def _is_short_source(self, source: str) -> bool:
        return len(re.findall(r"\b[\w']+\b", source)) <= 25

    def _has_source_summary_warning(self, notes: list[str]) -> bool:
        return any(
            note
            in {
                "Some bullets appear speculative rather than directly supported by the source.",
                "Source is short; summary may be padded.",
                "The response contains empty bullet placeholders.",
            }
            or "source appears to contain only" in note
            or "source-limitation note" in note
            for note in notes
        )

    def _source_limitation_note_mismatch(self, answer: str) -> str | None:
        match = re.search(
            r"only\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"distinct facts?",
            answer,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        raw_count = match.group(1).lower()
        stated_count = int(raw_count) if raw_count.isdigit() else NUMBER_WORDS.get(raw_count)
        actual_count = self.bullet_count(answer)
        if stated_count is not None and actual_count and stated_count != actual_count:
            return (
                "The source-limitation note says "
                f"{stated_count} distinct fact(s), but the response has {actual_count} "
                "factual bullet(s)."
            )
        return None

    def _asks_for_json(self, prompt_lower: str) -> bool:
        return "json" in prompt_lower

    def _looks_like_json(self, answer: str) -> bool:
        stripped = answer.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            return False
        return True

    def _looks_like_table(self, answer: str) -> bool:
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        pipe_rows = [line for line in lines if line.count("|") >= 2]
        if len(pipe_rows) >= 2:
            return True
        return any("\t" in line for line in lines)

    def _sentence_count(self, answer: str) -> int:
        normalized = " ".join(answer.strip().split())
        if not normalized:
            return 0
        sentences = re.findall(r"[^.!?]+[.!?](?=\s|$)", normalized)
        if sentences:
            return len(sentences)
        return 1

    def _suggested_next_step(self, notes: list[str]) -> str:
        del notes
        return (
            "Next steps:\n"
            "- Retry locally with stricter grounding.\n"
            "- Force a stronger local model:\n"
            "  switchboard ask '<same prompt>' --force-model ollama/gemma4:12b\n"
            "- If this is important enough for premium:\n"
            "  switchboard ask '<same prompt>' --backend claude-code"
        )
