"""Closed feedback loop: store thumbs-down snapshots, auto-retrain with a
golden-accuracy gate.

Flow:

1. On every ask (when ``store_feedback_examples`` is enabled) the assembled
   context is kept briefly in ``RecentContextRecord`` (capped) so a later
   thumbs-down can snapshot exactly what the model saw. Sensitivity-flagged
   requests (private-mode reroute or learned escalation) are never
   snapshotted: their assembled context must not persist on disk.
2. A thumbs-down stores a ``FeedbackExampleRecord``: prompt, full context,
   response, route taken, and — when the user picked "wrong model" — the
   corrected backend. Bodies are stored only for explicit thumbs-downs, one
   example per request_id (latest verdict wins). Sensitive requests store
   ``context_text=""``; the prompt is kept because a wrong-model correction
   trains the router from the (prompt, label) pair alone, and submitting it
   is the user's deliberate choice.
3. When unprocessed wrong-model examples reach a threshold, retraining is
   triggered in a background thread. The router trains on (prompt, corrected
   label) pairs only — never on full context — to avoid train/serve skew.
4. Golden gate: new weights are deployed only if golden dogfood accuracy does
   not regress; otherwise they are saved beside the live weights as
   ``*.rejected.json`` for inspection.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import desc, func
from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select

from switchboard.app.models.telemetry import (
    BackendMetricRecord,
    FeedbackExampleRecord,
    RecentContextRecord,
)

# User-facing backend choice -> router training label.
ROUTE_TYPE_BY_BACKEND = {
    "ollama": "local",
    "codex": "coding",
    "claude-code": "reasoning",
}

_RECENT_CONTEXT_CAP = 200


class FeedbackExampleStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # -- recent context snapshots ------------------------------------------

    def save_recent_context(self, request_id: str, context_text: str) -> None:
        with Session(self.engine) as session:
            existing = session.exec(
                select(RecentContextRecord).where(
                    RecentContextRecord.request_id == request_id
                )
            ).first()
            if existing is None:
                session.add(
                    RecentContextRecord(request_id=request_id, context_text=context_text)
                )
            else:
                existing.context_text = context_text
                session.add(existing)
            session.commit()
            # Cap retention: drop oldest beyond the cap.
            ids = session.exec(
                select(RecentContextRecord.id).order_by(
                    desc(col(RecentContextRecord.created_at))
                )
            ).all()
            for stale_id in ids[_RECENT_CONTEXT_CAP:]:
                stale = session.get(RecentContextRecord, stale_id)
                if stale is not None:
                    session.delete(stale)
            session.commit()

    def get_recent_context(self, request_id: str) -> str:
        with Session(self.engine) as session:
            record = session.exec(
                select(RecentContextRecord).where(
                    RecentContextRecord.request_id == request_id
                )
            ).first()
            return record.context_text if record else ""

    # -- feedback examples ---------------------------------------------------

    def add_example(self, record: FeedbackExampleRecord) -> None:
        """Insert, or replace the existing example for the same request_id.

        Latest verdict wins: a user changing their mind must not double-count
        toward the retrain threshold, so the unprocessed count reflects
        distinct requests. Replacement resets ``processed``/``gate_failed`` so
        the new verdict is eligible for the next training run.
        """
        with Session(self.engine) as session:
            existing = session.exec(
                select(FeedbackExampleRecord).where(
                    FeedbackExampleRecord.request_id == record.request_id
                )
            ).first()
            if existing is None:
                session.add(record)
            else:
                existing.rating = record.rating
                existing.detail = record.detail
                existing.corrected_backend = record.corrected_backend
                existing.prompt = record.prompt
                existing.context_text = record.context_text
                existing.response_text = record.response_text
                existing.route_type = record.route_type
                existing.backend = record.backend
                existing.confidence = record.confidence
                existing.processed = False
                existing.gate_failed = False
                existing.created_at = record.created_at
                session.add(existing)
            session.commit()

    def unprocessed_wrong_model_count(self) -> int:
        with Session(self.engine) as session:
            statement = (
                select(func.count())
                .select_from(FeedbackExampleRecord)
                .where(FeedbackExampleRecord.processed == False)  # noqa: E712
                .where(FeedbackExampleRecord.detail == "wrong_model")
            )
            return int(session.exec(statement).one())

    def unprocessed_wrong_model_examples(self) -> list[FeedbackExampleRecord]:
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(FeedbackExampleRecord)
                    .where(FeedbackExampleRecord.processed == False)  # noqa: E712
                    .where(FeedbackExampleRecord.detail == "wrong_model")
                ).all()
            )

    def unprocessed_bad_answer_examples(self) -> list[FeedbackExampleRecord]:
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(FeedbackExampleRecord)
                    .where(FeedbackExampleRecord.processed == False)  # noqa: E712
                    .where(FeedbackExampleRecord.detail == "bad_answer")
                ).all()
            )

    def request_metadata(self, request_id: str) -> dict[str, object]:
        """Metadata of the metrics record for a request (empty if unknown)."""
        with Session(self.engine) as session:
            record = session.exec(
                select(BackendMetricRecord).where(
                    BackendMetricRecord.request_id == request_id
                )
            ).first()
            if record is None:
                return {}
            try:
                return dict(json.loads(record.metadata_json))
            except (ValueError, TypeError):
                return {}

    def mark_processed(self, ids: list[int], *, gate_failed: bool = False) -> None:
        with Session(self.engine) as session:
            for record_id in ids:
                record = session.get(FeedbackExampleRecord, record_id)
                if record is not None:
                    record.processed = True
                    record.gate_failed = gate_failed
                    session.add(record)
            session.commit()

    def counts(self) -> dict[str, int]:
        with Session(self.engine) as session:
            total = int(
                session.exec(select(func.count()).select_from(FeedbackExampleRecord)).one()
            )
        return {
            "total": total,
            "unprocessed_wrong_model": self.unprocessed_wrong_model_count(),
        }

    def purge(self) -> int:
        with Session(self.engine) as session:
            examples = session.exec(select(FeedbackExampleRecord)).all()
            contexts = session.exec(select(RecentContextRecord)).all()
            count = len(examples) + len(contexts)
            for record in [*examples, *contexts]:
                session.delete(record)
            session.commit()
            return count


def retrain_with_feedback(
    *,
    engine: Engine,
    dataset_path: str | Path,
    weights_path: str | Path,
    embed: Callable[[str], list[float]] | None = None,
    embedding_model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
    min_golden_accuracy_drop: float = 0.0,
) -> dict[str, object]:
    """Retrain including wrong-model feedback; deploy only if the golden gate
    passes. Returns a status dict (never raises for expected failures)."""
    from switchboard.training.router_dataset import (
        RouterExample,
        build_dataset,
    )

    try:
        from switchboard.training.train_router import train
    except ImportError as exc:  # numpy missing
        return {"status": "skipped", "reason": f"training unavailable: {exc}"}

    store = FeedbackExampleStore(engine)
    feedback_records = store.unprocessed_wrong_model_examples()
    feedback_examples = [
        RouterExample(
            prompt=record.prompt,
            label=ROUTE_TYPE_BY_BACKEND[record.corrected_backend],
            source="feedback",
        )
        for record in feedback_records
        if record.corrected_backend in ROUTE_TYPE_BY_BACKEND and record.prompt.strip()
    ]

    examples = build_dataset() + feedback_examples
    external_cache = Path("data/external_router_examples.jsonl")
    if external_cache.exists():
        from switchboard.training.external_datasets import (
            load_or_fetch_external,
        )

        examples = examples + load_or_fetch_external(external_cache)
    # Feedback examples are real usage: weight them by duplication.
    examples = examples + feedback_examples * 2
    from switchboard.training.router_dataset import relabel_toolable

    examples, _ = relabel_toolable(examples)

    if embed is None:
        from switchboard.app.services.semantic_memory import (
            OllamaEmbeddingClient,
        )

        embed = OllamaEmbeddingClient(base_url=base_url, model=embedding_model).embed

    try:
        weights, report = train(examples, embed=embed, embedding_model=embedding_model)
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    # Golden gate: compare against the live weights' recorded golden accuracy.
    previous_golden = 0.0
    weights_file = Path(weights_path)
    if weights_file.exists():
        try:
            previous_meta = json.loads(weights_file.read_text(encoding="utf-8")).get(
                "metadata", {}
            )
            previous_golden = float(previous_meta.get("golden_accuracy", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            previous_golden = 0.0

    weights.metadata["golden_accuracy"] = report.golden_accuracy
    weights.metadata["feedback_examples"] = len(feedback_examples)
    record_ids = [r.id for r in feedback_records if r.id is not None]

    if report.golden_accuracy + min_golden_accuracy_drop < previous_golden:
        rejected_path = weights_file.with_suffix(".rejected.json")
        rejected_path.write_text(json.dumps(weights.to_dict(), indent=2), encoding="utf-8")
        store.mark_processed(record_ids, gate_failed=True)
        return {
            "status": "rejected",
            "reason": (
                f"golden accuracy regressed: {report.golden_accuracy:.2%} < "
                f"{previous_golden:.2%}"
            ),
            "rejected_weights": str(rejected_path),
        }

    # Atomic deploy: write to a temp file, then rename.
    tmp_path = weights_file.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(weights.to_dict(), indent=2), encoding="utf-8")
    tmp_path.replace(weights_file)
    store.mark_processed(record_ids, gate_failed=False)
    return {
        "status": "deployed",
        "golden_accuracy": report.golden_accuracy,
        "holdout_accuracy": report.holdout_accuracy,
        "feedback_examples": len(feedback_examples),
        "weights": str(weights_file),
    }


def retrain_dispatcher_with_feedback(
    *,
    engine: Engine,
    dataset_path: str | Path = "data/tool_dispatcher_examples.jsonl",
    weights_path: str | Path = "config/tool_dispatcher_weights.json",
    embed: Callable[[str], list[float]] | None = None,
    embedding_model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> dict[str, object]:
    """Teach the tool dispatcher from thumbs-downs.

    A "bad answer" on a response the dispatcher grounded means the dispatcher
    should not have fired: the prompt becomes a ``none`` training example.
    Same golden gate and atomic deploy as the router. Bad-answer feedback on
    non-dispatcher responses is left untouched (and unprocessed) — it carries
    no routing signal.
    """
    from switchboard.app.services.tool_dispatcher import TOOL_CLASSES
    from switchboard.training.router_dataset import RouterExample
    from switchboard.training.tool_dispatcher_dataset import (
        dispatcher_golden_examples,
    )

    try:
        from switchboard.training.train_router import train
    except ImportError as exc:  # numpy missing
        return {"status": "skipped", "reason": f"training unavailable: {exc}"}

    store = FeedbackExampleStore(engine)
    dispatcher_records = [
        record
        for record in store.unprocessed_bad_answer_examples()
        if record.prompt.strip()
        and store.request_metadata(record.request_id).get("tool_dispatcher_used")
    ]
    if not dispatcher_records:
        return {"status": "skipped", "reason": "no dispatcher-grounded bad answers"}

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        return {
            "status": "skipped",
            "reason": "dispatcher dataset missing: run train-dispatcher first",
        }
    from switchboard.training.tool_dispatcher_dataset import (
        load_or_build_dispatcher_dataset,
    )

    feedback_examples = [
        RouterExample(prompt=record.prompt, label="none", source="feedback")
        for record in dispatcher_records
    ]
    examples = (
        load_or_build_dispatcher_dataset(dataset_file) + feedback_examples * 3
    )

    if embed is None:
        from switchboard.app.services.semantic_memory import (
            OllamaEmbeddingClient,
        )

        embed = OllamaEmbeddingClient(base_url=base_url, model=embedding_model).embed

    try:
        weights, report = train(
            examples,
            embed=embed,
            embedding_model=embedding_model,
            classes=TOOL_CLASSES,
            golden=dispatcher_golden_examples(),
        )
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    weights_file = Path(weights_path)
    previous_golden = 0.0
    if weights_file.exists():
        try:
            previous_meta = json.loads(weights_file.read_text(encoding="utf-8")).get(
                "metadata", {}
            )
            previous_golden = float(previous_meta.get("golden_accuracy", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            previous_golden = 0.0

    weights.metadata["golden_accuracy"] = report.golden_accuracy
    weights.metadata["feedback_examples"] = len(feedback_examples)
    record_ids = [r.id for r in dispatcher_records if r.id is not None]

    if report.golden_accuracy < previous_golden:
        rejected_path = weights_file.with_suffix(".rejected.json")
        rejected_path.write_text(json.dumps(weights.to_dict(), indent=2), encoding="utf-8")
        store.mark_processed(record_ids, gate_failed=True)
        return {
            "status": "rejected",
            "reason": (
                f"dispatcher golden accuracy regressed: {report.golden_accuracy:.2%} < "
                f"{previous_golden:.2%}"
            ),
            "rejected_weights": str(rejected_path),
        }

    tmp_path = weights_file.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(weights.to_dict(), indent=2), encoding="utf-8")
    tmp_path.replace(weights_file)
    store.mark_processed(record_ids, gate_failed=False)
    return {
        "status": "deployed",
        "golden_accuracy": report.golden_accuracy,
        "feedback_examples": len(feedback_examples),
        "weights": str(weights_file),
    }


def maybe_trigger_retraining(
    *,
    engine: Engine,
    threshold: int,
    weights_path: str | Path,
    dataset_path: str | Path = "router_dataset.jsonl",
    run_async: bool = True,
    retrain: Callable[..., dict[str, object]] | None = None,
) -> bool:
    """Kick off background retraining when enough wrong-model feedback has
    accumulated. Returns True if retraining was triggered."""
    store = FeedbackExampleStore(engine)
    if store.unprocessed_wrong_model_count() < threshold:
        return False

    retrain_fn = retrain or retrain_with_feedback

    def _run() -> None:
        # The background thread must never crash the app.
        with contextlib.suppress(Exception):
            retrain_fn(
                engine=engine,
                dataset_path=dataset_path,
                weights_path=weights_path,
            )
        # The dispatcher learns from its own feedback signal (bad answers on
        # dispatcher-grounded responses); cheap no-op when there are none.
        with contextlib.suppress(Exception):
            retrain_dispatcher_with_feedback(engine=engine)

    if run_async:
        thread = threading.Thread(target=_run, name="router-retrain", daemon=True)
        thread.start()
    else:
        _run()
    return True
