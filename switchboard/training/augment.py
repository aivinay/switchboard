"""Optional Claude-CLI augmentation for the router dataset.

For each template example, ask the locally authenticated Claude Code CLI to
produce a few natural paraphrases that keep the same routing intent. This
diversifies phrasing beyond the hand-written templates. It is opt-in (costs
subscription quota) and fully deterministic in how it parses output; if the
CLI is unavailable, augmentation is skipped and the base dataset is used.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable

from switchboard.training.router_dataset import RouterExample

_PROMPT_TEMPLATE = (
    "You are helping build a router training set. Paraphrase the user request "
    "below into {n} different natural phrasings (casual and formal) that a real "
    "person might type. Keep the SAME intent and topic. Do not answer it. "
    "Respond with ONLY a JSON array of strings.\n\nRequest: {prompt}"
)


def _claude_available() -> bool:
    return shutil.which("claude") is not None


def _default_paraphraser(prompt: str, n: int, timeout_s: int) -> list[str]:
    command = [
        "claude",
        "--print",
        "--output-format=json",
        "--no-session-persistence",
        "--disallowedTools=Edit,Write,Bash",
        _PROMPT_TEMPLATE.format(n=n, prompt=prompt),
    ]
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell.
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout)
        text = payload.get("result", result.stdout)
    except json.JSONDecodeError:
        text = result.stdout
    return _parse_paraphrases(text)


def _parse_paraphrases(text: str) -> list[str]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [str(item).strip() for item in items if isinstance(item, str) and item.strip()]


def augment_examples(
    examples: list[RouterExample],
    *,
    per_example: int = 2,
    timeout_s: int = 60,
    paraphrase: Callable[[str, int, int], list[str]] | None = None,
    limit: int | None = None,
) -> list[RouterExample]:
    """Return base examples plus Claude paraphrases (label preserved).

    Only template examples are augmented; golden cases are left verbatim.
    """
    paraphraser = paraphrase or _default_paraphraser
    if paraphrase is None and not _claude_available():
        return list(examples)

    augmented = list(examples)
    seen = {ex.prompt.lower() for ex in examples}
    candidates = [ex for ex in examples if ex.source == "template"]
    if limit is not None:
        candidates = candidates[:limit]

    for example in candidates:
        try:
            paraphrases = paraphraser(example.prompt, per_example, timeout_s)
        except Exception:
            continue
        for phrase in paraphrases:
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            augmented.append(
                RouterExample(prompt=phrase, label=example.label, source="claude_augment")
            )
    return augmented
