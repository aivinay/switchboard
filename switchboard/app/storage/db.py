from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, create_engine


def create_db_engine(database_url: str) -> Engine:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)
    if engine.dialect.name == "sqlite":
        _migrate_sqlite(engine)


def _migrate_sqlite(engine: Engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "personaltelemetryrecord" in tables:
        _add_missing_columns(
            engine,
            "personaltelemetryrecord",
            {
                "router_selected_model": "VARCHAR",
                "user_forced_model": "VARCHAR",
                "final_selected_model": "VARCHAR",
                "override_used": "BOOLEAN DEFAULT 0",
                "override_reason": "VARCHAR",
                "override_safety_blocked": "BOOLEAN DEFAULT 0",
                "escalation_used": "BOOLEAN DEFAULT 0",
                "original_request_id": "VARCHAR",
                "original_model": "VARCHAR",
                "escalated_to_model": "VARCHAR",
                "escalation_reason": "VARCHAR",
                "manual_recommendation": "BOOLEAN DEFAULT 0",
                "premium_unit_spent": "FLOAT DEFAULT 0.0",
                "premium_unit_saved": "FLOAT DEFAULT 0.0",
                "estimated_api_cost_saved": "FLOAT DEFAULT 0.0",
                "baseline_model": "VARCHAR",
                "baseline_route_kind": "VARCHAR",
                "baseline_source": "VARCHAR DEFAULT 'config_default'",
                "feedback_rating": "VARCHAR",
                "selected_model_loaded": "BOOLEAN",
                "model_switch_avoided": "BOOLEAN DEFAULT 0",
                "cold_start_expected": "BOOLEAN DEFAULT 0",
                "performance_mode": "VARCHAR",
                "loaded_local_models_json": "VARCHAR DEFAULT '[]'",
            },
        )
    if "feedbackrecord" in tables:
        _add_missing_columns(
            engine,
            "feedbackrecord",
            {
                "preferred_model": "VARCHAR",
            },
        )
    if "chatsessionrecord" in tables:
        _add_missing_columns(
            engine,
            "chatsessionrecord",
            {
                "private": "BOOLEAN DEFAULT 0",
            },
        )


def _add_missing_columns(engine: Engine, table_name: str, columns: dict[str, str]) -> None:
    inspector = inspect(engine)
    existing = {column["name"] for column in inspector.get_columns(table_name)}
    with engine.begin() as connection:
        for column_name, column_type in columns.items():
            if column_name not in existing:
                connection.execute(
                    text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                )
