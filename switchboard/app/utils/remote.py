from __future__ import annotations

import ipaddress
import os

REMOTE_MUTATION_ENV = "SWITCHBOARD_ALLOW_REMOTE_MUTATIONS"
TRUTHY_REMOTE_VALUES = {"1", "true", "yes", "on"}


def host_is_loopback(host: str | None) -> bool:
    if not host:
        return True
    normalized = host.strip().lower().strip("[]")
    if normalized in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def remote_mutations_allowed() -> bool:
    return os.getenv(REMOTE_MUTATION_ENV, "").strip().lower() in TRUTHY_REMOTE_VALUES
