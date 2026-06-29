from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from switchboard.app.models.capabilities import (
    Capability,
    CapabilityDetection,
    RuntimeContext,
    ToolResult,
)
from switchboard.app.services.deterministic_tools import (
    CalculatorTool,
    UnitConversionTool,
)
from switchboard.app.services.finance_tool import StockPriceTool
from switchboard.app.services.news_tool import NewsTool
from switchboard.app.services.status_intents import (
    asks_tool_status,
    asks_web_status,
)
from switchboard.app.services.web_search_tool import WebSearchTool


class TimeTool:
    # (alias, IANA timezone, display label). Aliases are matched on word
    # boundaries, so short aliases never match inside other words.
    _TIMEZONE_ALIASES: tuple[tuple[str, str, str], ...] = (
        ("america/new_york", "America/New_York", "New York"),
        ("new york", "America/New_York", "New York"),
        ("nyc", "America/New_York", "New York"),
        ("boston", "America/New_York", "Boston"),
        ("miami", "America/New_York", "Miami"),
        ("toronto", "America/Toronto", "Toronto"),
        ("est", "America/New_York", "US Eastern"),
        ("edt", "America/New_York", "US Eastern"),
        ("chicago", "America/Chicago", "Chicago"),
        ("denver", "America/Denver", "Denver"),
        ("los angeles", "America/Los_Angeles", "Los Angeles"),
        ("san francisco", "America/Los_Angeles", "San Francisco"),
        ("seattle", "America/Los_Angeles", "Seattle"),
        ("pst", "America/Los_Angeles", "US Pacific"),
        ("pdt", "America/Los_Angeles", "US Pacific"),
        ("vancouver", "America/Vancouver", "Vancouver"),
        ("mexico city", "America/Mexico_City", "Mexico City"),
        ("sao paulo", "America/Sao_Paulo", "São Paulo"),
        ("buenos aires", "America/Argentina/Buenos_Aires", "Buenos Aires"),
        ("europe/london", "Europe/London", "London"),
        ("london", "Europe/London", "London"),
        ("dublin", "Europe/Dublin", "Dublin"),
        ("lisbon", "Europe/Lisbon", "Lisbon"),
        ("gmt", "UTC", "GMT"),
        ("paris", "Europe/Paris", "Paris"),
        ("berlin", "Europe/Berlin", "Berlin"),
        ("frankfurt", "Europe/Berlin", "Frankfurt"),
        ("munich", "Europe/Berlin", "Munich"),
        ("amsterdam", "Europe/Amsterdam", "Amsterdam"),
        ("madrid", "Europe/Madrid", "Madrid"),
        ("rome", "Europe/Rome", "Rome"),
        ("zurich", "Europe/Zurich", "Zurich"),
        ("stockholm", "Europe/Stockholm", "Stockholm"),
        ("cet", "Europe/Paris", "Central Europe"),
        ("athens", "Europe/Athens", "Athens"),
        ("istanbul", "Europe/Istanbul", "Istanbul"),
        ("moscow", "Europe/Moscow", "Moscow"),
        ("cairo", "Africa/Cairo", "Cairo"),
        ("johannesburg", "Africa/Johannesburg", "Johannesburg"),
        ("nairobi", "Africa/Nairobi", "Nairobi"),
        ("lagos", "Africa/Lagos", "Lagos"),
        ("tel aviv", "Asia/Jerusalem", "Tel Aviv"),
        ("riyadh", "Asia/Riyadh", "Riyadh"),
        ("dubai", "Asia/Dubai", "Dubai"),
        ("uae", "Asia/Dubai", "UAE"),
        ("karachi", "Asia/Karachi", "Karachi"),
        ("pakistan", "Asia/Karachi", "Pakistan"),
        ("asia/kolkata", "Asia/Kolkata", "India"),
        ("india", "Asia/Kolkata", "India"),
        ("kolkata", "Asia/Kolkata", "India"),
        ("delhi", "Asia/Kolkata", "India"),
        ("mumbai", "Asia/Kolkata", "India"),
        ("bangalore", "Asia/Kolkata", "Bengaluru"),
        ("bengaluru", "Asia/Kolkata", "Bengaluru"),
        ("hyderabad", "Asia/Kolkata", "Hyderabad"),
        ("chennai", "Asia/Kolkata", "Chennai"),
        ("pune", "Asia/Kolkata", "Pune"),
        ("ist", "Asia/Kolkata", "India"),
        ("kathmandu", "Asia/Kathmandu", "Kathmandu"),
        ("dhaka", "Asia/Dhaka", "Dhaka"),
        ("colombo", "Asia/Colombo", "Colombo"),
        ("bangkok", "Asia/Bangkok", "Bangkok"),
        ("jakarta", "Asia/Jakarta", "Jakarta"),
        ("kuala lumpur", "Asia/Kuala_Lumpur", "Kuala Lumpur"),
        ("singapore", "Asia/Singapore", "Singapore"),
        ("hong kong", "Asia/Hong_Kong", "Hong Kong"),
        ("shanghai", "Asia/Shanghai", "Shanghai"),
        ("beijing", "Asia/Shanghai", "Beijing"),
        ("china", "Asia/Shanghai", "China"),
        ("manila", "Asia/Manila", "Manila"),
        ("taipei", "Asia/Taipei", "Taipei"),
        ("seoul", "Asia/Seoul", "Seoul"),
        ("korea", "Asia/Seoul", "Korea"),
        ("tokyo", "Asia/Tokyo", "Tokyo"),
        ("japan", "Asia/Tokyo", "Japan"),
        ("jst", "Asia/Tokyo", "Japan"),
        ("sydney", "Australia/Sydney", "Sydney"),
        ("melbourne", "Australia/Melbourne", "Melbourne"),
        ("australia", "Australia/Sydney", "Australia"),
        ("auckland", "Pacific/Auckland", "Auckland"),
        ("utc", "UTC", "UTC"),
    )

    def answer(
        self,
        *,
        prompt: str,
        capability: Capability,
        context: RuntimeContext,
    ) -> ToolResult:
        if capability == Capability.CURRENT_DATE:
            return self._date_answer(prompt, context)

        timezone_name, label = self._target_timezone(prompt, context)
        target_time = context.utc_datetime.astimezone(ZoneInfo(timezone_name))
        answer = (
            f"The current time in {label} is {self._format_time(target_time)} "
            f"on {self._format_date(target_time)}."
        )
        return ToolResult(
            tool_name="time",
            capability=Capability.CURRENT_TIME,
            answer=answer,
            display_model_or_label="Time",
        )

    # "45 days from today", "5 days after now"
    _DAYS_FROM_TODAY = re.compile(r"\b(\d{1,4})\s+days?\s+(?:from|after)\s+(?:today|now)\b")
    # "in 100 days" (the detector already required date/day intent)
    _IN_N_DAYS = re.compile(r"\bin\s+(\d{1,4})\s+days?\b")

    def _date_answer(self, prompt: str, context: RuntimeContext) -> ToolResult:
        local_now = context.utc_datetime.astimezone(ZoneInfo(context.local_timezone))
        text = prompt.lower()
        offset_match = self._DAYS_FROM_TODAY.search(text) or self._IN_N_DAYS.search(text)
        if offset_match:
            days = int(offset_match.group(1))
            target = local_now + timedelta(days=days)
            noun = "day" if days == 1 else "days"
            prefix = f"{days} {noun} from today is"
        elif "day after tomorrow" in text:
            target = local_now + timedelta(days=2)
            prefix = "The day after tomorrow is"
        elif "tomorrow" in text:
            target = local_now + timedelta(days=1)
            prefix = "Tomorrow is"
        elif "yesterday" in text:
            target = local_now - timedelta(days=1)
            prefix = "Yesterday was"
        else:
            target = local_now
            prefix = "Today is"
        answer = (
            f"{prefix} {target:%A}, {self._format_date(target)} "
            f"({context.local_timezone})."
        )
        return ToolResult(
            tool_name="time",
            capability=Capability.CURRENT_DATE,
            answer=answer,
            display_model_or_label="Time",
        )

    def _target_timezone(self, prompt: str, context: RuntimeContext) -> tuple[str, str]:
        text = prompt.lower()
        for alias, timezone_name, label in self._TIMEZONE_ALIASES:
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text):
                return timezone_name, label
        return context.local_timezone, "your local timezone"

    def _format_time(self, value: datetime) -> str:
        hour = value.strftime("%I").lstrip("0") or "0"
        abbreviation = value.tzname() or ""
        return f"{hour}:{value:%M} {value:%p} {abbreviation}".strip()

    def _format_date(self, value: datetime) -> str:
        return f"{value:%B} {value.day}, {value:%Y}"


class UnsupportedLiveDataTool:
    def answer(self, capability: Capability) -> ToolResult:
        if capability == Capability.WEATHER:
            return ToolResult(
                tool_name="unsupported_live_data",
                capability=capability,
                answer=(
                    "Live weather is not configured yet. Please configure a weather "
                    "provider, or ask a non-live weather question."
                ),
                display_model_or_label="Switchboard",
            )
        return ToolResult(
            tool_name="unsupported_live_data",
            capability=capability,
            answer=(
                "Live/latest information is not configured yet. Please configure a live "
                "data provider, or ask a non-current question."
            ),
            display_model_or_label="Switchboard",
        )


class ToolRegistry:
    def __init__(
        self,
        *,
        time_tool: TimeTool | None = None,
        stock_price_tool: StockPriceTool | None = None,
        web_search_tool: WebSearchTool | None = None,
        unsupported_live_data_tool: UnsupportedLiveDataTool | None = None,
        calculator_tool: CalculatorTool | None = None,
        unit_conversion_tool: UnitConversionTool | None = None,
        news_tool: NewsTool | None = None,
    ) -> None:
        self.time_tool = time_tool or TimeTool()
        self.stock_price_tool = stock_price_tool or StockPriceTool()
        self.web_search_tool = web_search_tool or WebSearchTool()
        self.unsupported_live_data_tool = unsupported_live_data_tool or UnsupportedLiveDataTool()
        self.calculator_tool = calculator_tool or CalculatorTool()
        self.unit_conversion_tool = unit_conversion_tool or UnitConversionTool()
        self.news_tool = news_tool or NewsTool()

    def resolve(
        self,
        *,
        prompt: str,
        detection: CapabilityDetection,
        context: RuntimeContext,
    ) -> ToolResult | None:
        if self._asks_web_status(prompt):
            return self.web_search_tool.status()
        if self._asks_tool_status(prompt) and detection.has(Capability.WEATHER):
            return self.unsupported_live_data_tool.answer(Capability.WEATHER)
        if detection.has(Capability.UNIT_CONVERSION):
            return self.unit_conversion_tool.answer(prompt)
        if detection.has(Capability.CALCULATION):
            return self.calculator_tool.answer(prompt)
        if detection.has(Capability.CURRENT_TIME):
            return self.time_tool.answer(
                prompt=prompt,
                capability=Capability.CURRENT_TIME,
                context=context,
            )
        if detection.has(Capability.CURRENT_DATE):
            return self.time_tool.answer(
                prompt=prompt,
                capability=Capability.CURRENT_DATE,
                context=context,
            )
        if detection.has(Capability.STOCK_PRICE):
            if self._asks_tool_status(prompt):
                return self.stock_price_tool.status()
            stock_result = self.stock_price_tool.answer(prompt)
            if stock_result is not None and stock_result.success:
                return stock_result
            if (
                stock_result is not None
                and stock_result.metadata.get("tool_available") is False
                and self.web_search_tool.is_configured()
            ):
                return self.web_search_tool.answer(prompt=prompt, detection=detection)
            return stock_result
        if detection.has(Capability.LATEST_INFO) and self.news_tool.is_configured():
            news_result = self.news_tool.answer(prompt=prompt)
            if news_result.success:
                return news_result
            if self.web_search_tool.is_configured():
                return self.web_search_tool.answer(prompt=prompt, detection=detection)
            return news_result
        if self._truth_grounding_needed(detection) and self.web_search_tool.is_configured():
            return self.web_search_tool.answer(prompt=prompt, detection=detection)
        if self._asks_tool_status(prompt) and detection.has(Capability.LATEST_INFO):
            return self.unsupported_live_data_tool.answer(Capability.LATEST_INFO)
        return None

    def _truth_grounding_needed(self, detection: CapabilityDetection) -> bool:
        return any(
            detection.has(capability)
            for capability in (
                Capability.WEB_SEARCH,
                Capability.WEATHER,
                Capability.LATEST_INFO,
                Capability.STOCK_PRICE,
            )
        )

    def _asks_web_status(self, prompt: str) -> bool:
        return asks_web_status(prompt)

    def _asks_tool_status(self, prompt: str) -> bool:
        return asks_tool_status(prompt)

    def availability(self) -> dict[str, str]:
        return {
            "runtime_context": "available",
            "time_tool": "available",
            "calculator_tool": "available",
            "unit_conversion_tool": "available",
            "stock_price_tool": (
                "available" if self.stock_price_tool.provider.is_configured() else "not configured"
            ),
            "web_search_tool": (
                "available" if self.web_search_tool.is_configured() else "not configured"
            ),
            "news_tool": (
                "available" if self.news_tool.is_configured() else "not configured"
            ),
            "weather_tool": (
                "available" if self.web_search_tool.is_configured() else "not configured"
            ),
            "live_latest_info_tool": (
                "available"
                if self.news_tool.is_configured() or self.web_search_tool.is_configured()
                else "not configured"
            ),
        }
