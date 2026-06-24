"""Harness for the tester/developer agent loop.

Runs prompts through the REAL SwitchboardCoreService — real capability
detector, deterministic tools, policy layer, context builder — with recording
fake model backends (no Ollama/Codex/Claude needed). For each prompt it
reports the routing decision, cost, grounding metadata, and the exact context
sent to the chosen backend, so an automated tester can judge behavior.

Usage:
    python scripts/agent_harness.py prompts.json [results.json]

prompts.json: [{"prompt": "...", "session": "optional-name",
                "web_search": false}, ...]
Prompts sharing a "session" run in the same conversation, in order.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from switchboard.app.backends.base import AgentAdapter  # noqa: E402
from switchboard.app.backends.registry import BackendRegistry  # noqa: E402
from switchboard.app.core.config import Settings  # noqa: E402
from switchboard.app.models.backends import (  # noqa: E402
    BackendCostType,
    BackendInfo,
    SwitchboardRequest,
    SwitchboardResponse,
)
from switchboard.app.services.container import build_container  # noqa: E402
from switchboard.app.services.switchboard_core import (  # noqa: E402
    SwitchboardCoreService,
)
from switchboard.app.storage.db import create_db_engine, init_db  # noqa: E402


class RecordingAdapter(AgentAdapter):
    def __init__(self, name: str, *, cost_type: BackendCostType) -> None:
        self.name = name
        self.cost_type = cost_type
        self.prompts: list[str] = []

    def is_available(self) -> bool:
        return True

    def availability(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=True, cost_type=self.cost_type)

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        self.prompts.append(request.prompt)
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            content=f"[{self.name} mock answer]",
            latency_ms=5,
            success=True,
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


def main() -> None:
    prompts_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("harness_results.json")
    cases = json.loads(prompts_path.read_text(encoding="utf-8"))

    tmp = Path(tempfile.mkdtemp())
    adapters = {
        "ollama": RecordingAdapter("ollama", cost_type=BackendCostType.LOCAL),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code", cost_type=BackendCostType.SUBSCRIPTION
        ),
    }
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp / 'harness.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)
    container.personal_config.preferences.claude_code_web_search = False
    service = SwitchboardCoreService(
        registry=BackendRegistry(dict(adapters)),
        metrics=container.backend_metrics_repository,
        container=container,
    )

    sessions: dict[str, str | None] = {}
    results = []
    for case in cases:
        prompt = case["prompt"]
        session_name = case.get("session") or f"solo-{len(results)}"
        container.personal_config.preferences.claude_code_web_search = bool(
            case.get("web_search", False)
        )
        before = {name: len(a.prompts) for name, a in adapters.items()}
        response = service.ask(
            prompt,
            session_id=sessions.get(session_name),
            new_session=session_name not in sessions,
        )
        sessions[session_name] = response.session_id
        record = service.metrics_list(limit=1)[0]
        backend_prompt = None
        for name, adapter in adapters.items():
            if len(adapter.prompts) > before[name]:
                backend_prompt = {"backend": name, "context": adapter.prompts[-1]}
        results.append(
            {
                "prompt": prompt,
                "session": session_name,
                "backend": response.backend,
                "cost_type": str(response.cost_type),
                "success": response.success,
                "answer": response.content,
                "routing_reason": response.routing_reason,
                "metadata": record.metadata,
                "context_sent_to_backend": backend_prompt,
            }
        )

    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Ran {len(results)} prompts -> {out_path}")


if __name__ == "__main__":
    main()
