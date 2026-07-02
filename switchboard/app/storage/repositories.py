from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.engine import Engine
from sqlmodel import Session, col, desc, select

from switchboard.app.models.personal import FeedbackRead, PersonalMemoryRead
from switchboard.app.models.sessions import (
    ChatMessageRead,
    ChatMessageRecord,
    ChatSessionRead,
    ChatSessionRecord,
)
from switchboard.app.models.telemetry import (
    BackendMetricRead,
    BackendMetricRecord,
    FeedbackExampleRecord,
    FeedbackRecord,
    MemoryItem,
    PersonalTelemetryRead,
    PersonalTelemetryRecord,
    RoutingCacheRecord,
    TelemetryRead,
    TelemetryRecord,
    utc_now,
)
from switchboard.app.utils.ids import new_request_id
from switchboard.app.utils.redaction import sanitize_provider_error


def telemetry_to_read(record: TelemetryRecord) -> TelemetryRead:
    return TelemetryRead(
        request_id=record.request_id,
        tenant_id=record.tenant_id,
        application_id=record.application_id,
        workflow_id=record.workflow_id,
        routing_mode=record.routing_mode,
        task_type=record.task_type,
        complexity=record.complexity,
        sensitivity=record.sensitivity,
        classifier_confidence=record.classifier_confidence,
        requested_model=record.requested_model,
        selected_model=record.selected_model,
        shadow_recommended_model=record.shadow_recommended_model,
        policy_version=record.policy_version,
        reason_codes=json.loads(record.reason_codes_json or "[]"),
        estimated_cost_usd=record.estimated_cost_usd,
        estimated_baseline_cost_usd=record.estimated_baseline_cost_usd,
        estimated_latency_ms=record.estimated_latency_ms,
        actual_latency_ms=record.actual_latency_ms,
        provider=record.provider,
        fallback_used=record.fallback_used,
        status=record.status,
        error_code=record.error_code,
        created_at=record.created_at,
    )


def personal_telemetry_to_read(record: PersonalTelemetryRecord) -> PersonalTelemetryRead:
    return PersonalTelemetryRead(
        request_id=record.request_id,
        user_id=record.user_id,
        project=record.project,
        mode=record.mode,
        task_type=record.task_type,
        complexity=record.complexity,
        sensitivity=record.sensitivity,
        selected_model=record.selected_model,
        selected_provider=record.selected_provider,
        route_kind=record.route_kind,
        scarce_model=record.scarce_model,
        required_confirmation=record.required_confirmation,
        called_model=record.called_model,
        recommended_only=record.recommended_only,
        estimated_input_tokens=record.estimated_input_tokens,
        estimated_output_tokens=record.estimated_output_tokens,
        estimated_cost_usd=record.estimated_cost_usd,
        estimated_premium_units=record.estimated_premium_units,
        estimated_premium_units_saved=record.estimated_premium_units_saved,
        router_selected_model=record.router_selected_model,
        user_forced_model=record.user_forced_model,
        final_selected_model=record.final_selected_model,
        override_used=bool(record.override_used),
        override_reason=record.override_reason,
        override_safety_blocked=bool(record.override_safety_blocked),
        escalation_used=bool(record.escalation_used),
        original_request_id=record.original_request_id,
        original_model=record.original_model,
        escalated_to_model=record.escalated_to_model,
        escalation_reason=record.escalation_reason,
        manual_recommendation=bool(record.manual_recommendation),
        premium_unit_spent=record.premium_unit_spent or 0.0,
        premium_unit_saved=record.premium_unit_saved or 0.0,
        estimated_api_cost_saved=record.estimated_api_cost_saved or 0.0,
        baseline_model=record.baseline_model,
        baseline_route_kind=record.baseline_route_kind,
        baseline_source=record.baseline_source or "config_default",
        feedback_rating=record.feedback_rating,
        selected_model_loaded=record.selected_model_loaded,
        model_switch_avoided=bool(record.model_switch_avoided),
        cold_start_expected=bool(record.cold_start_expected),
        performance_mode=record.performance_mode,
        loaded_local_models=json.loads(record.loaded_local_models_json or "[]"),
        reason_codes=json.loads(record.reason_codes_json or "[]"),
        status=record.status,
        cache_hit=record.cache_hit,
        created_at=record.created_at,
    )


def backend_metric_to_read(record: BackendMetricRecord) -> BackendMetricRead:
    return BackendMetricRead(
        request_id=record.request_id,
        backend=record.backend,
        selected_model=record.selected_model,
        project=record.project,
        prompt_char_count=record.prompt_char_count,
        latency_ms=record.latency_ms,
        success=record.success,
        error_message=sanitize_provider_error(record.error_message, backend=record.backend),
        exit_code=record.exit_code,
        routing_reason=record.routing_reason,
        cost_type=record.cost_type,
        estimated_cost_usd=record.estimated_cost_usd,
        private_mode=record.private_mode,
        metadata=json.loads(record.metadata_json or "{}"),
        created_at=record.created_at,
    )


def chat_session_to_read(record: ChatSessionRecord) -> ChatSessionRead:
    return ChatSessionRead(
        session_id=record.session_id,
        title=record.title,
        summary=record.summary,
        private=record.private,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def chat_message_to_read(record: ChatMessageRecord) -> ChatMessageRead:
    return ChatMessageRead(
        message_id=record.message_id,
        session_id=record.session_id,
        role=record.role,
        content=record.content,
        display_model=record.display_model,
        backend=record.backend,
        tool_name=record.tool_name,
        metadata=json.loads(record.metadata_json or "{}"),
        created_at=record.created_at,
    )


def memory_to_read(record: MemoryItem) -> PersonalMemoryRead:
    return PersonalMemoryRead(
        id=record.id or 0,
        project=record.project,
        title=record.title,
        content=record.content,
        tags=json.loads(record.tags_json or "[]"),
        created_at=record.created_at.isoformat(),
    )


@dataclass
class TelemetryRepository:
    engine: Engine

    def add(self, record: TelemetryRecord) -> TelemetryRecord:
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def list(self, limit: int = 100) -> list[TelemetryRead]:
        with Session(self.engine) as session:
            statement = (
                select(TelemetryRecord).order_by(desc(TelemetryRecord.created_at)).limit(limit)
            )
            return [telemetry_to_read(record) for record in session.exec(statement).all()]

    def get(self, request_id: str) -> TelemetryRead | None:
        with Session(self.engine) as session:
            statement = select(TelemetryRecord).where(TelemetryRecord.request_id == request_id)
            record = session.exec(statement).first()
            return telemetry_to_read(record) if record else None

    def summary(self) -> dict[str, object]:
        with Session(self.engine) as session:
            records = session.exec(select(TelemetryRecord)).all()

        total_cost = sum(record.estimated_cost_usd for record in records)
        baseline_cost = sum(record.estimated_baseline_cost_usd for record in records)
        latencies = [
            record.actual_latency_ms for record in records if record.actual_latency_ms is not None
        ]
        average_latency = sum(latencies) / len(latencies) if latencies else 0

        return {
            "total_requests": len(records),
            "requests_by_task_type": dict(Counter(record.task_type for record in records)),
            "requests_by_selected_model": dict(
                Counter(record.selected_model or "none" for record in records)
            ),
            "estimated_total_cost_usd": round(total_cost, 8),
            "estimated_baseline_cost_usd": round(baseline_cost, 8),
            "estimated_savings_usd": round(baseline_cost - total_cost, 8),
            "average_latency_ms": round(average_latency, 2),
            "escalation_fallback_count": sum(1 for record in records if record.fallback_used),
        }


@dataclass
class PersonalTelemetryRepository:
    engine: Engine

    def add(self, record: PersonalTelemetryRecord) -> PersonalTelemetryRecord:
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def list(self, limit: int = 100) -> list[PersonalTelemetryRead]:
        with Session(self.engine) as session:
            statement = (
                select(PersonalTelemetryRecord)
                .order_by(desc(PersonalTelemetryRecord.created_at))
                .limit(limit)
            )
            return [personal_telemetry_to_read(record) for record in session.exec(statement).all()]

    def get(self, request_id: str) -> PersonalTelemetryRead | None:
        with Session(self.engine) as session:
            statement = select(PersonalTelemetryRecord).where(
                PersonalTelemetryRecord.request_id == request_id
            )
            record = session.exec(statement).first()
            return personal_telemetry_to_read(record) if record else None

    def summary(self) -> dict[str, object]:
        with Session(self.engine) as session:
            records = session.exec(select(PersonalTelemetryRecord)).all()

        route_kind_counts = Counter(record.route_kind for record in records)
        return {
            "total_requests": len(records),
            "local_requests": route_kind_counts.get("local", 0)
            + route_kind_counts.get("mock", 0)
            + route_kind_counts.get("openai_compatible_local", 0),
            "cloud_requests": route_kind_counts.get("cloud_api", 0),
            "manual_recommendations": sum(
                1 for record in records if record.route_kind == "manual_subscription"
            ),
            "estimated_api_spend_usd": round(
                sum(record.estimated_cost_usd for record in records), 8
            ),
            "estimated_premium_units_saved": round(
                sum(record.premium_unit_saved for record in records), 2
            ),
            "estimated_premium_units_spent": round(
                sum(record.premium_unit_spent for record in records), 2
            ),
            "model_distribution": dict(
                Counter(record.selected_model or "none" for record in records)
            ),
            "cache_hits": self.cache_hits(),
            "cache_misses": sum(1 for record in records if not record.cache_hit),
            "estimated_premium_units_saved_from_cache": round(float(self.cache_hits()), 2),
            "feedback": self.feedback_summary(),
        }

    def savings(
        self,
        days: int | None = 7,
        since: datetime | None = None,
    ) -> dict[str, object]:
        cutoff = since
        if cutoff is None and days is not None:
            cutoff = datetime.now(UTC) - timedelta(days=days)
        with Session(self.engine) as session:
            statement = select(PersonalTelemetryRecord)
            if cutoff is not None:
                statement = statement.where(PersonalTelemetryRecord.created_at >= cutoff)
            records = session.exec(statement).all()

        route_kind_counts = Counter(record.route_kind for record in records)
        provider_counts = Counter(record.selected_provider or "none" for record in records)
        task_savings: dict[str, float] = {}
        for record in records:
            if record.premium_unit_saved or record.estimated_api_cost_saved:
                task_savings[record.task_type] = task_savings.get(record.task_type, 0.0) + (
                    record.premium_unit_saved + record.estimated_api_cost_saved
                )
        baseline_counts = Counter(record.baseline_model or "none" for record in records)
        return {
            "days": days,
            "since": cutoff.date().isoformat() if cutoff else None,
            "total_requests": len(records),
            "local_model_calls": route_kind_counts.get("local", 0)
            + route_kind_counts.get("mock", 0)
            + route_kind_counts.get("openai_compatible_local", 0),
            "local_ollama_calls": provider_counts.get("ollama", 0),
            "mock_calls": provider_counts.get("mock", 0),
            "cloud_calls": route_kind_counts.get("cloud_api", 0),
            "manual_premium_recommendations": sum(
                1 for record in records if record.manual_recommendation
            ),
            "premium_units_saved": round(sum(record.premium_unit_saved for record in records), 2),
            "premium_units_spent": round(sum(record.premium_unit_spent for record in records), 2),
            "estimated_api_spend_usd": round(
                sum(record.estimated_cost_usd for record in records), 8
            ),
            "estimated_api_cost_saved_usd": round(
                sum(record.estimated_api_cost_saved for record in records), 8
            ),
            "top_task_types_saved": dict(
                sorted(task_savings.items(), key=lambda item: item[1], reverse=True)[:5]
            ),
            "top_models_used": dict(
                Counter(
                    record.final_selected_model or record.selected_model or "none"
                    for record in records
                )
            ),
            "overrides_count": sum(1 for record in records if record.override_used),
            "escalations_count": sum(1 for record in records if record.escalation_used),
            "cache_savings": sum(1 for record in records if record.cache_hit),
            "cache_hits": sum(1 for record in records if record.cache_hit),
            "cache_misses": sum(1 for record in records if not record.cache_hit),
            "feedback": self.feedback_summary(),
            "baseline_assumptions": dict(baseline_counts),
        }

    def cache_hits(self) -> int:
        with Session(self.engine) as session:
            records = session.exec(select(RoutingCacheRecord)).all()
        return sum(record.hit_count for record in records)

    def feedback_summary(self) -> dict[str, object]:
        with Session(self.engine) as session:
            records = session.exec(select(FeedbackRecord)).all()
        counts = Counter(record.rating for record in records)
        preferred = Counter(record.preferred_model for record in records if record.preferred_model)
        negative = sum(
            counts.get(rating, 0)
            for rating in {"bad", "too-expensive", "too-weak", "wrong-route"}
        )
        return {
            "total": len(records),
            "positive": counts.get("good", 0),
            "negative": negative,
            "bad": counts.get("bad", 0),
            "too_expensive": counts.get("too-expensive", 0),
            "too_weak": counts.get("too-weak", 0),
            "wrong_route": counts.get("wrong-route", 0),
            "preferred_models": dict(preferred.most_common(5)),
        }

    def preferred_model_from_feedback(
        self,
        project: str,
        task_type: str,
        current_model: str,
    ) -> str | None:
        with Session(self.engine) as session:
            feedback_records = session.exec(
                select(FeedbackRecord)
                .where(col(FeedbackRecord.preferred_model).is_not(None))
                .order_by(desc(FeedbackRecord.created_at))
                .limit(50)
            ).all()
            candidates: list[str] = []
            for feedback in feedback_records:
                if feedback.rating not in {"too-weak", "wrong-route"}:
                    continue
                route = session.exec(
                    select(PersonalTelemetryRecord).where(
                        PersonalTelemetryRecord.request_id == feedback.request_id
                    )
                ).first()
                if route is None:
                    continue
                if route.project != project or route.task_type != task_type:
                    continue
                previous_models = {route.final_selected_model, route.selected_model}
                if current_model not in previous_models:
                    continue
                if feedback.preferred_model:
                    candidates.append(feedback.preferred_model)
        if not candidates:
            return None
        return Counter(candidates).most_common(1)[0][0]

    def get_cache(self, cache_key: str) -> RoutingCacheRecord | None:
        with Session(self.engine) as session:
            statement = select(RoutingCacheRecord).where(RoutingCacheRecord.cache_key == cache_key)
            record = session.exec(statement).first()
            if record is None:
                return None
            record.hit_count += 1
            record.updated_at = datetime.now(UTC)
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def set_cache(
        self,
        cache_key: str,
        project: str,
        mode: str,
        route_json: str,
    ) -> None:
        with Session(self.engine) as session:
            existing = session.exec(
                select(RoutingCacheRecord).where(RoutingCacheRecord.cache_key == cache_key)
            ).first()
            if existing:
                existing.route_json = route_json
                existing.updated_at = datetime.now(UTC)
                session.add(existing)
            else:
                session.add(
                    RoutingCacheRecord(
                        cache_key=cache_key,
                        project=project,
                        mode=mode,
                        route_json=route_json,
                    )
                )
            session.commit()

    def add_feedback(self, record: FeedbackRecord) -> FeedbackRead:
        with Session(self.engine) as session:
            existing_records = session.exec(
                select(FeedbackRecord)
                .where(FeedbackRecord.request_id == record.request_id)
                .order_by(col(FeedbackRecord.created_at), col(FeedbackRecord.id))
            ).all()
            existing = existing_records[0] if existing_records else None
            if existing is None:
                stored = record
                session.add(stored)
            else:
                stored = existing
                stored.rating = record.rating
                stored.note = record.note
                stored.preferred_model = record.preferred_model
                stored.created_at = record.created_at
                session.add(stored)
                for duplicate in existing_records[1:]:
                    session.delete(duplicate)
            route = session.exec(
                select(PersonalTelemetryRecord).where(
                    PersonalTelemetryRecord.request_id == record.request_id
                )
            ).first()
            if route is not None:
                route.feedback_rating = record.rating
                session.add(route)
            session.commit()
            session.refresh(stored)
            return FeedbackRead(
                request_id=stored.request_id,
                rating=stored.rating,
                note=stored.note,
                preferred_model=stored.preferred_model,
                created_at=stored.created_at.isoformat(),
            )

    def delete_feedback(self, request_id: str) -> bool:
        deleted = False
        with Session(self.engine) as session:
            records = session.exec(
                select(FeedbackRecord).where(FeedbackRecord.request_id == request_id)
            ).all()
            for record in records:
                session.delete(record)
                deleted = True
            examples = session.exec(
                select(FeedbackExampleRecord).where(
                    FeedbackExampleRecord.request_id == request_id
                )
            ).all()
            for example in examples:
                session.delete(example)
                deleted = True
            route = session.exec(
                select(PersonalTelemetryRecord).where(
                    PersonalTelemetryRecord.request_id == request_id
                )
            ).first()
            if route is not None:
                route.feedback_rating = None
                session.add(route)
            session.commit()
        return deleted

    def feedback_by_request_ids(self, request_ids: Sequence[str]) -> dict[str, FeedbackRead]:
        if not request_ids:
            return {}
        with Session(self.engine) as session:
            records = session.exec(
                select(FeedbackRecord)
                .where(col(FeedbackRecord.request_id).in_(request_ids))
                .order_by(desc(FeedbackRecord.created_at), desc(FeedbackRecord.id))
            ).all()
        by_request: dict[str, FeedbackRead] = {}
        for record in records:
            if record.request_id in by_request:
                continue
            by_request[record.request_id] = FeedbackRead(
                request_id=record.request_id,
                rating=record.rating,
                note=record.note,
                preferred_model=record.preferred_model,
                created_at=record.created_at.isoformat(),
            )
        return by_request


@dataclass
class BackendMetricsRepository:
    engine: Engine

    def add(self, record: BackendMetricRecord) -> BackendMetricRecord:
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def get(self, request_id: str) -> BackendMetricRead | None:
        with Session(self.engine) as session:
            record = session.exec(
                select(BackendMetricRecord).where(
                    BackendMetricRecord.request_id == request_id
                )
            ).first()
            return backend_metric_to_read(record) if record else None

    def list_since(
        self,
        *,
        since: datetime,
        limit: int = 5000,
    ) -> list[BackendMetricRead]:
        with Session(self.engine) as session:
            statement = (
                select(BackendMetricRecord)
                .where(BackendMetricRecord.created_at >= since)
                .order_by(desc(BackendMetricRecord.created_at))
                .limit(limit)
            )
            return [backend_metric_to_read(record) for record in session.exec(statement).all()]

    def list(self, limit: int = 20) -> list[BackendMetricRead]:
        with Session(self.engine) as session:
            statement = (
                select(BackendMetricRecord)
                .order_by(desc(BackendMetricRecord.created_at))
                .limit(limit)
            )
            return [backend_metric_to_read(record) for record in session.exec(statement).all()]

    def successful_call_count(
        self,
        *,
        backend: str,
        since: datetime,
        until: datetime | None = None,
    ) -> int:
        with Session(self.engine) as session:
            statement = select(BackendMetricRecord).where(
                BackendMetricRecord.backend == backend,
                col(BackendMetricRecord.success).is_(True),
                BackendMetricRecord.created_at >= since,
            )
            if until is not None:
                statement = statement.where(BackendMetricRecord.created_at <= until)
            records = session.exec(statement).all()
            return len(records)

    def summary(self) -> dict[str, object]:
        with Session(self.engine) as session:
            records = session.exec(select(BackendMetricRecord)).all()

        by_backend = Counter(record.backend for record in records)
        recent_error_records = sorted(
            (record for record in records if not record.success),
            key=lambda record: record.created_at,
            reverse=True,
        )[:5]
        success_by_backend: dict[str, float] = {}
        average_latency_by_backend: dict[str, float] = {}
        for backend in by_backend:
            backend_records = [record for record in records if record.backend == backend]
            successes = sum(1 for record in backend_records if record.success)
            success_by_backend[backend] = round(successes / len(backend_records), 4)
            average_latency_by_backend[backend] = round(
                sum(record.latency_ms for record in backend_records) / len(backend_records),
                2,
            )
        session_ids: set[str] = set()
        for record in records:
            metadata = json.loads(record.metadata_json or "{}")
            session_id = metadata.get("session_id")
            if isinstance(session_id, str) and session_id:
                session_ids.add(session_id)

        return {
            "total_requests": len(records),
            "requests_by_backend": dict(by_backend),
            "session_count": len(session_ids),
            "success_rate_by_backend": success_by_backend,
            "average_latency_ms_by_backend": average_latency_by_backend,
            "recent_errors": [
                {
                    "request_id": record.request_id,
                    "backend": record.backend,
                    "error_message": sanitize_provider_error(
                        record.error_message,
                        backend=record.backend,
                    ),
                    "exit_code": record.exit_code,
                    "created_at": record.created_at.isoformat(),
                }
                for record in recent_error_records
            ],
        }


@dataclass
class ContextStore:
    engine: Engine

    def create_session(
        self,
        *,
        session_id: str | None = None,
        title: str | None = None,
        private: bool = False,
    ) -> ChatSessionRead:
        with Session(self.engine) as session:
            existing = None
            if session_id:
                existing = session.get(ChatSessionRecord, session_id)
            if existing:
                return chat_session_to_read(existing)
            record = ChatSessionRecord(
                session_id=session_id or new_request_id("session"),
                title=title,
                private=private,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return chat_session_to_read(record)

    def get_session(self, session_id: str) -> ChatSessionRead | None:
        with Session(self.engine) as session:
            record = session.get(ChatSessionRecord, session_id)
            return chat_session_to_read(record) if record else None

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        private: bool | None = None,
    ) -> ChatSessionRead | None:
        with Session(self.engine) as session:
            record = session.get(ChatSessionRecord, session_id)
            if record is None:
                return None
            if title is not None:
                record.title = title
            if private is not None:
                record.private = private
            record.updated_at = utc_now()
            session.add(record)
            session.commit()
            session.refresh(record)
            return chat_session_to_read(record)

    def append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        display_model: str | None = None,
        backend: str | None = None,
        tool_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ChatMessageRead:
        with Session(self.engine) as session:
            session_record = session.get(ChatSessionRecord, session_id)
            if session_record is None:
                raise ValueError(f"Unknown session_id: {session_id}")
            now = utc_now()
            record = ChatMessageRecord(
                message_id=new_request_id("msg"),
                session_id=session_id,
                role=role,
                content=content,
                display_model=display_model,
                backend=backend,
                tool_name=tool_name,
                metadata_json=json.dumps(metadata or {}),
                created_at=now,
            )
            session_record.updated_at = now
            session.add(record)
            session.add(session_record)
            session.commit()
            session.refresh(record)
            return chat_message_to_read(record)

    def list_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[ChatMessageRead]:
        with Session(self.engine) as session:
            statement = (
                select(ChatMessageRecord)
                .where(ChatMessageRecord.session_id == session_id)
                .order_by(col(ChatMessageRecord.created_at), col(ChatMessageRecord.id))
            )
            if limit is not None:
                statement = statement.limit(limit)
            return [chat_message_to_read(record) for record in session.exec(statement).all()]

    def get_recent_messages(self, session_id: str, *, limit: int = 12) -> list[ChatMessageRead]:
        with Session(self.engine) as session:
            statement = (
                select(ChatMessageRecord)
                .where(ChatMessageRecord.session_id == session_id)
                .order_by(desc(ChatMessageRecord.created_at), desc(ChatMessageRecord.id))
                .limit(limit)
            )
            records = list(session.exec(statement).all())
            records.reverse()
            return [chat_message_to_read(record) for record in records]

    def update_session_summary(
        self,
        session_id: str,
        summary: str | None,
    ) -> ChatSessionRead | None:
        with Session(self.engine) as session:
            record = session.get(ChatSessionRecord, session_id)
            if record is None:
                return None
            record.summary = summary
            record.updated_at = utc_now()
            session.add(record)
            session.commit()
            session.refresh(record)
            return chat_session_to_read(record)


@dataclass
class MemoryRepository:
    engine: Engine

    def add(self, item: MemoryItem) -> PersonalMemoryRead:
        with Session(self.engine) as session:
            session.add(item)
            session.commit()
            session.refresh(item)
            return memory_to_read(item)

    def search(self, project: str, query: str, limit: int = 20) -> list[PersonalMemoryRead]:
        escaped_query = (
            query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        pattern = f"%{escaped_query}%"
        with Session(self.engine) as session:
            statement = (
                select(MemoryItem)
                .where(MemoryItem.project == project)
                .where(
                    (col(MemoryItem.title).like(pattern, escape="\\"))
                    | (col(MemoryItem.content).like(pattern, escape="\\"))
                )
                .order_by(desc(MemoryItem.created_at))
                .limit(limit)
            )
            return [memory_to_read(record) for record in session.exec(statement).all()]
