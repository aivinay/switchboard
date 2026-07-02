from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel

from switchboard.app.models.telemetry import utc_now


class ChatSessionRecord(SQLModel, table=True):
    session_id: str = SQLField(primary_key=True)
    title: str | None = None
    summary: str | None = None
    private: bool = SQLField(default=False, index=True)
    created_at: datetime = SQLField(default_factory=utc_now, index=True)
    updated_at: datetime = SQLField(default_factory=utc_now, index=True)


class ChatMessageRecord(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    message_id: str = SQLField(index=True, unique=True)
    session_id: str = SQLField(index=True)
    role: str = SQLField(index=True)
    content: str
    display_model: str | None = None
    backend: str | None = SQLField(default=None, index=True)
    tool_name: str | None = SQLField(default=None, index=True)
    metadata_json: str = "{}"
    created_at: datetime = SQLField(default_factory=utc_now, index=True)


class ChatSessionRead(BaseModel):
    session_id: str
    title: str | None = None
    summary: str | None = None
    private: bool = False
    created_at: datetime
    updated_at: datetime


class ChatMessageRead(BaseModel):
    message_id: str
    session_id: str
    role: str
    content: str
    display_model: str | None = None
    backend: str | None = None
    tool_name: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
