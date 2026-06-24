from __future__ import annotations

PROMPT_BODY_MARKERS = (
    "Runtime context:",
    "Recent conversation:",
    "Current request:",
    "\nuser\n",
    "\nassistant\n",
)


def sanitize_provider_error(
    error_message: str | None,
    *,
    prompt: str | None = None,
    backend: str | None = None,
    max_chars: int = 1000,
) -> str | None:
    if not error_message:
        return None
    stripped = error_message.strip()
    if not stripped:
        return None

    prompt_text = (prompt or "").strip()
    may_include_prompt = any(marker in stripped for marker in PROMPT_BODY_MARKERS)
    if prompt_text and prompt_text in stripped:
        may_include_prompt = True
    if may_include_prompt:
        label = f"{backend} backend" if backend else "Provider"
        return (
            f"{label} error output redacted because it may contain prompt, response, "
            "or shared context content."
        )
    if len(stripped) <= max_chars:
        return stripped
    return f"{stripped[:max_chars].rstrip()}..."
