"""Shared secret-format detection and redaction (round-7 tester findings F2/F3).

Two consumers share these compiled patterns so they can never drift apart:

* ``ContextBuilder._clean_content`` (app/services/session_context.py) redacts
  every piece of assembled context — conversation history, session summaries,
  trusted facts, and semantic-memory facts — before it is shared with any
  model backend.
* ``RequestClassifier._sensitivity`` (app/services/classifier.py) uses
  ``contains_secret_format`` as a keyword-floor backstop: a recognized secret
  FORMAT in the raw request marks it CONFIDENTIAL so existing policy routes
  it to the local model instead of a subscription backend.

Patterns are format-anchored on purpose: prose like "the key is under the
mat", "password hygiene tips", or "JWT is a token format" must never match,
and code such as ``sorted(items, key=len)`` or ``using namespace std`` must
never be corrupted. Bare ``key`` therefore only matches with an ``_``/``-``
prefixed name (``ACCESS_KEY=...``), never as a standalone word.
"""

from __future__ import annotations

import re

# Env-style credential names: SECRET/TOKEN/PASSWORD/... may stand alone or be
# prefixed (DATABASE_PASSWORD, AWS_SECRET_ACCESS_KEY); bare KEY requires a
# prefixed segment so "monkey=funny" and "key=len" stay untouched.
_ENV_SECRET_NAME = (
    r"(?:[A-Za-z0-9]+[_-])*"
    r"(?:api[_-]?key|secret|token|password|passwd|pwd|credentials?|authorization)"
    r"|(?:[A-Za-z0-9]+[_-])+key"
)

PEM_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
# A BEGIN marker without its END (truncated paste) must still take the whole
# tail with it — the base64 body lines carry the actual key material.
PEM_PRIVATE_KEY_DANGLING = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*\Z",
    re.DOTALL,
)
SK_API_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
AWS_ACCESS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
JWT_TOKEN = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}")
BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+\S+")
# postgres://user:pass@host — redact only the password, keep user and host.
URL_CREDENTIALS = re.compile(r"://([^/\s:@]+):([^/\s@]+)@")
ENV_ASSIGNMENT = re.compile(rf"(?i)\b({_ENV_SECRET_NAME})\s*[=:]\s*\S+")
# Prose disclosures: "my password is hunter2". Requires is/was plus a value so
# "password hygiene tips" never matches.
PROSE_PASSWORD = re.compile(r"(?i)\b(password|passphrase|passcode)\s+(is|was)\s+\S+")

# Order matters: multiline PEM blocks first (their base64 body would otherwise
# leak line by line), URL credentials before the env-name pattern so the
# username/host survive, value-bearing patterns last.
SECRET_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (PEM_PRIVATE_KEY_BLOCK, "[REDACTED_PRIVATE_KEY]"),
    (PEM_PRIVATE_KEY_DANGLING, "[REDACTED_PRIVATE_KEY]"),
    (SK_API_KEY, "[REDACTED_SECRET]"),
    (AWS_ACCESS_KEY, "[REDACTED_AWS_KEY]"),
    (JWT_TOKEN, "[REDACTED_JWT]"),
    (BEARER_TOKEN, "Bearer [REDACTED]"),
    (URL_CREDENTIALS, r"://\1:[REDACTED]@"),
    (ENV_ASSIGNMENT, r"\1=[REDACTED]"),
    (PROSE_PASSWORD, r"\1 \2 [REDACTED]"),
)

SECRET_FORMAT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    pattern for pattern, _ in SECRET_REPLACEMENTS
)


def redact_secrets(text: str) -> str:
    """Redact recognized secret formats, leaving all other content intact.

    Must be called on the FULL text (not per line): PEM private-key blocks
    span multiple lines and their body lines carry no per-line marker.
    """
    redacted = text
    for pattern, replacement in SECRET_REPLACEMENTS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def contains_secret_format(text: str) -> bool:
    """True when the text contains a recognized secret FORMAT.

    Case-sensitive patterns (AKIA keys, PEM markers) need the original text,
    not a lowercased copy. Text that merely discusses secrets ("regex for AWS
    keys is AKIA[0-9A-Z]{16}") only matches when it embeds a real-shaped
    value — and erring local in that case is the privacy-safe direction.
    """
    return any(pattern.search(text) for pattern in SECRET_FORMAT_PATTERNS)
