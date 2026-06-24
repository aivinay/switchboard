from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Capability(StrEnum):
    CURRENT_TIME = "current_time"
    CURRENT_DATE = "current_date"
    WEATHER = "weather"
    STOCK_PRICE = "stock_price"
    LATEST_INFO = "latest_info"
    WEB_SEARCH = "web_search"
    CALCULATION = "calculation"
    UNIT_CONVERSION = "unit_conversion"
    CODING = "coding"
    REASONING = "reasoning"
    LOCAL_PRIVATE = "local_private"
    UNKNOWN = "unknown"


class CapabilityDetection(BaseModel):
    capabilities: list[Capability] = Field(default_factory=list)
    primary: Capability = Capability.UNKNOWN

    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def values(self) -> list[str]:
        return [capability.value for capability in self.capabilities]


class RuntimeContext(BaseModel):
    utc_datetime: datetime
    local_datetime: datetime
    local_timezone: str
    current_date: str
    utc_iso: str
    local_iso: str
    human_utc_time: str
    human_local_time: str

    def hidden_prompt_block(self) -> str:
        return (
            "[Switchboard runtime context]\n"
            f"Current local datetime: {self.local_iso} ({self.local_timezone})\n"
            f"Current local time: {self.human_local_time}\n"
            f"Current local date: {self.current_date}\n"
            f"Current UTC datetime: {self.utc_iso}\n"
            f"Current UTC time: {self.human_utc_time}\n"
            "Use this trusted context for current time/date references. "
            "Do not reveal this block unless explicitly asked.\n"
            "[/Switchboard runtime context]"
        )


class ToolResult(BaseModel):
    tool_name: str
    capability: Capability
    answer: str
    success: bool = True
    error: str | None = None
    display_model_or_label: str = "Switchboard"
    metadata: dict[str, Any] = Field(default_factory=dict)
