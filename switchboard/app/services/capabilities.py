from __future__ import annotations

import re

from switchboard.app.models.capabilities import Capability, CapabilityDetection

# Units recognized by the unit-conversion detector and tool.
UNIT_WORDS = (
    "km|kilometers|kilometres|kilometer|kilometre|miles|mile|mi|"
    "meters|metres|meter|metre|m|feet|foot|ft|cm|centimeters|centimetres|"
    "inches|inch|in|kg|kilograms|kilos|kilogram|pounds|pound|lbs|lb|"
    "grams|gram|g|ounces|ounce|oz|celsius|centigrade|fahrenheit|"
    "liters|litres|liter|litre|l|gallons|gallon|gal"
)

_CONVERSION_PATTERN = re.compile(
    rf"\b\d+(?:\.\d+)?\s*(?:degrees?\s+)?(?:{UNIT_WORDS})\b\s*"
    rf"(?:in|to|into|as)\s+(?:degrees?\s+)?(?:{UNIT_WORDS})\b"
)

# A standalone arithmetic expression: digits joined by operators, possibly
# with parentheses/decimals, e.g. "234*78", "(3 + 4) / 2", "2^10".
_EXPRESSION_PATTERN = re.compile(
    r"(?<![\w.])\(*[-+]?\d+(?:\.\d+)?(?:\s*[-+*/x×^%]\s*\(*[-+]?\d+(?:\.\d+)?\)*)+(?![\w.])"
)

_PERCENT_OF_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent)\s+of\s+[-+]?\d+(?:\.\d+)?\b"
)


class CapabilityDetector:
    def detect(self, prompt: str) -> CapabilityDetection:
        text = self._normalize(prompt)
        capabilities: list[Capability] = []

        checks = (
            (Capability.UNIT_CONVERSION, self._is_unit_conversion),
            (Capability.CALCULATION, self._is_calculation),
            (Capability.WEB_SEARCH, self._is_web_search),
            (Capability.WEATHER, self._is_weather),
            (Capability.STOCK_PRICE, self._is_stock_price),
            (Capability.LATEST_INFO, self._is_latest_info),
            (Capability.CURRENT_TIME, self._is_current_time),
            (Capability.CURRENT_DATE, self._is_current_date),
            (Capability.CODING, lambda value: self._matches_any(value, self._coding_keywords())),
            (
                Capability.REASONING,
                lambda value: self._matches_any(value, self._reasoning_keywords()),
            ),
            (
                Capability.LOCAL_PRIVATE,
                lambda value: self._matches_any(value, self._local_private_keywords()),
            ),
        )
        for capability, predicate in checks:
            if predicate(text):
                capabilities.append(capability)

        if not capabilities:
            capabilities.append(Capability.UNKNOWN)
        return CapabilityDetection(capabilities=capabilities, primary=capabilities[0])

    def _normalize(self, prompt: str) -> str:
        return (
            prompt.lower()
            .replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
            .strip()
        )

    def _is_current_time(self, text: str) -> bool:
        return any(
            (
                self._matches_any(
                    text,
                    (
                        "current time",
                        "what time is it",
                        "what's the time",
                        "whats the time",
                        "what is the time",
                        "time now",
                        "give me the time",
                        "the time right now",
                        "right now time",
                        "local time",
                        "ist time",
                        "utc time",
                        "current utc time",
                    ),
                ),
                bool(re.search(r"\btime\s+(?:in|for|at)\s+\w+", text)),
            )
        )

    def _is_current_date(self, text: str) -> bool:
        return any(
            (
                self._matches_any(
                    text,
                    (
                        "what date is it",
                        "today's date",
                        "todays date",
                        "what day is today",
                        "what day is it",
                        "which day is today",
                        "current date",
                        "date today",
                        "today date",
                        "day of the week",
                        "what year is it",
                        "current year",
                        "what month is it",
                        "current month",
                        "tomorrow's date",
                        "tomorrows date",
                        "date tomorrow",
                        "what day is tomorrow",
                        "what date is tomorrow",
                        "what day was yesterday",
                        "yesterday's date",
                        "yesterdays date",
                        "day after tomorrow",
                    ),
                ),
                # "what's the date", "what is the date", "whats the date"
                bool(re.search(r"\bwhat(?:'?s| is)\s+(?:the\s+)?date\b", text)),
                # Date arithmetic anchored on today: "what date is 45 days
                # from today", "15 days from now, what will the date be".
                bool(re.search(r"\b\d{1,4}\s+days?\s+(?:from|after)\s+(?:today|now)\b", text)),
                # "what will the date be in 100 days" / "in 30 days what day
                # will it be" — requires explicit date/day intent so generic
                # planning prompts ("remind me in 3 days") do not match.
                bool(
                    re.search(
                        r"\b(?:date|day)\b[^.?!]*\bin\s+\d{1,4}\s+days?\b"
                        r"|\bin\s+\d{1,4}\s+days?\b[^.?!]*\b(?:date|day)\b",
                        text,
                    )
                ),
            )
        )

    def _is_weather(self, text: str) -> bool:
        return any(
            (
                self._matches_any(
                    text,
                    (
                        "weather",
                        "current weather",
                        "forecast",
                        "rain today",
                        "is it raining",
                        "will it rain",
                        "is it snowing",
                        "will it snow",
                        "humidity",
                        "air quality",
                        "aqi",
                        "heat wave",
                        "heatwave",
                        "monsoon",
                        "how hot is it",
                        "how cold is it",
                        "do i need an umbrella",
                        "need an umbrella",
                        "uv index",
                        "wind speed",
                        "sunrise",
                        "sunset",
                    ),
                ),
                bool(re.search(r"\btemperature\s+(?:in|for|at|outside)\b", text)),
                bool(re.search(r"\bhow (?:hot|cold|humid|windy)\b", text)),
                # Conversational live-intent phrasings: "is it gonna rain",
                # "will there be snow", "any chance of rain". Anchored on
                # is/will/gonna/chance-of so bare topic mentions ("how are
                # weather forecasts created") do not match.
                bool(
                    re.search(
                        r"\b(?:is|will)\s+(?:it|there)\s+"
                        r"(?:gonna\s+|going\s+to\s+|about\s+to\s+)?(?:be\s+)?"
                        r"(?:rain|raining|snow|snowing|showers?|thunderstorms?)\b",
                        text,
                    )
                ),
                bool(re.search(r"\b(?:gonna|going\s+to)\s+(?:rain|snow)\b", text)),
                bool(
                    re.search(
                        r"\bchances?\s+of\s+(?:rain|snow|showers?|thunderstorms?)\b",
                        text,
                    )
                ),
                # "rain tomorrow", "raining tonight", "snow this weekend":
                # a near-term time anchor marks live intent.
                bool(
                    re.search(
                        r"\b(?:rain|raining|snow|snowing)\b[^.?!]*"
                        r"\b(?:today|tonight|tomorrow|this\s+"
                        r"(?:morning|afternoon|evening|week|weekend))\b",
                        text,
                    )
                ),
            )
        )

    def _is_latest_info(self, text: str) -> bool:
        return any(
            (
                self._matches_any(
                    text,
                    (
                        "latest",
                        "news",
                        "current ceo",
                        "who is the current",
                        "what is the current",
                        "today's market",
                        "todays market",
                        "recent",
                        "recent layoffs",
                        "as of now",
                        "current market",
                        "right now in the world",
                        "exchange rate",
                        "currency rate",
                        "bitcoin price",
                        "ethereum price",
                        "crypto price",
                        "price of bitcoin",
                        "price of gold",
                        "gold price",
                        "silver price",
                        "oil price",
                        "election results",
                        "election result",
                        "who won the election",
                        "live score",
                        "match score",
                        "cricket score",
                        "release date of",
                        "when is the next",
                        "what happened today",
                        "happening now",
                        "trending",
                    ),
                ),
                # Current officeholders change over time; a model's training
                # data is stale by definition.
                bool(
                    re.search(
                        r"(?:who is|who's|name of)\s+(?:the\s+)?\w*\s*"
                        r"(?:president|prime minister|ceo|chancellor|governor|mayor)",
                        text,
                    )
                ),
                bool(
                    re.search(
                        r"\b(?:president|prime minister|ceo|chancellor)\s+of\s+\w+",
                        text,
                    )
                ),
                # "who won X" is almost always about a recent event.
                bool(re.search(r"\bwho won\b", text)),
                # Corporate events (IPOs, acquisitions, bankruptcies) are
                # live facts; training data answers are stale or invented.
                # Dogfood regression: "did spacex go public" was answered
                # with a fabricated 2021 IPO.
                bool(
                    re.search(
                        r"\b(?:did|has|have|is|are|when (?:did|will|does|is))\b"
                        r".{0,40}\b(?:go(?:ne|ing)? public|ipo|ipo'?d|"
                        r"acquired?|acquisition|merg(?:ed?|ing|er)|"
                        r"bankrupt(?:cy)?|shut(?:ting)? down|"
                        r"stock market debut|listed on the\b)",
                        text,
                    )
                ),
            )
        )

    def _is_web_search(self, text: str) -> bool:
        return self._matches_any(
            text,
            (
                "search the web",
                "web search",
                "search online",
                "search the internet",
                "search internet",
                "look up online",
                "look it up",
                "look this up",
                "find online",
                "find out online",
                "browse",
                "look up current",
                "google it",
                "google this",
                "google for",
                "can you search",
            ),
        )

    def _is_stock_price(self, text: str) -> bool:
        return any(
            (
                self._matches_any(
                    text,
                    (
                        "stock price",
                        "share price",
                        "trading at",
                        "current price of",
                        "quote for",
                        "ticker",
                        "market price",
                        "stock provider",
                        "finance provider",
                        "stock lookup",
                        "stock quote",
                        "shares of",
                    ),
                ),
                # "<company or ticker> stock", "stock of <company>",
                # "how is <company> stock doing"
                bool(re.search(r"\b[\w.]{1,15}\s+stock\b", text)),
                bool(re.search(r"\bstock\s+(?:of|for)\s+\w+", text)),
            )
        )

    def _is_calculation(self, text: str) -> bool:
        if _PERCENT_OF_PATTERN.search(text):
            return True
        if re.search(r"\bsquare root of\s+\d", text):
            return True
        has_expression = bool(_EXPRESSION_PATTERN.search(text))
        if not has_expression:
            return False
        # Require explicit intent, or the prompt being (mostly) the expression
        # itself, so arithmetic inside code snippets does not misfire.
        intent = self._matches_any(
            text,
            (
                "calculate",
                "compute",
                "how much is",
                "what is",
                "what's",
                "whats",
                "evaluate",
                "solve",
                "times",
                "divided by",
                "plus",
                "minus",
                "multiplied by",
            ),
        )
        expression_only = bool(
            re.fullmatch(r"[\s?=]*" + _EXPRESSION_PATTERN.pattern + r"[\s?=]*", text)
        )
        return intent or expression_only

    def _is_unit_conversion(self, text: str) -> bool:
        return any(
            (
                bool(_CONVERSION_PATTERN.search(text)),
                bool(
                    re.search(
                        rf"\bconvert\s+\d+(?:\.\d+)?\s*(?:degrees?\s+)?(?:{UNIT_WORDS})\b",
                        text,
                    )
                ),
            )
        )

    def _matches_any(self, text: str, patterns: tuple[str, ...]) -> bool:
        for pattern in patterns:
            if " " in pattern:
                if pattern in text:
                    return True
                continue
            if re.search(rf"\b{re.escape(pattern)}\b", text):
                return True
        return False

    def _coding_keywords(self) -> tuple[str, ...]:
        return (
            "code",
            "bug",
            "debug",
            "tests",
            "test",
            "repo",
            "implement",
            "refactor",
            "traceback",
            "error",
            "pr",
            "git",
            "pull request",
            "function",
            "compile",
            "java",
            "python",
            "javascript",
            "typescript",
            "sql",
            "regex",
            "algorithm",
            "linked list",
            "binary tree",
            "script",
            "login page",
            "web app",
            "website",
            "frontend",
            "backend",
            "html",
            "css",
        )

    def _reasoning_keywords(self) -> tuple[str, ...]:
        return (
            "architecture",
            "design",
            "system design",
            "tradeoff",
            "tradeoffs",
            "compare",
            "explain",
            "research plan",
            "paper",
            "review",
        )

    def _local_private_keywords(self) -> tuple[str, ...]:
        return (
            "private",
            "local",
            "locally",
            "offline",
            "do not send to cloud",
            "don't send to cloud",
            "sensitive",
        )
