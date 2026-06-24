"""Synthetic dataset generation for the Switchboard prompt router.

Generates a labeled dataset for training a tiny embedding classifier that routes
prompts into four classes: ``tool``, ``local``, ``coding`` and ``reasoning``.

The dataset is built by expanding hand-written templates with slot fillers
(combinatorially, then capped per class for balance) plus a small set of
hand-labeled "golden" dogfood cases the trained model must get right.

Deterministic: all randomness is seeded with 42. No network, stdlib only.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from dataclasses import dataclass

SEED = 42
PER_CLASS_CAP = 260


@dataclass(frozen=True)
class RouterExample:
    prompt: str
    label: str  # one of: tool, local, coding, reasoning
    source: str  # "template" or "golden"


def _expand(templates: list[str], slots: dict[str, list[str]]) -> list[str]:
    """Expand templates combinatorially over the slot vocabularies.

    Only the slots actually referenced by a template are filled, so templates
    with no slots are emitted verbatim. Output order is deterministic.
    """
    out: list[str] = []
    for template in templates:
        used = [name for name in slots if "{" + name + "}" in template]
        if not used:
            out.append(template)
            continue
        vocab_lists = [slots[name] for name in used]
        for combo in itertools.product(*vocab_lists):
            mapping = dict(zip(used, combo, strict=True))
            out.append(template.format(**mapping))
    return out


def _cap(prompts: list[str], cap: int, rng: random.Random) -> list[str]:
    """Deduplicate, then sample down to ``cap`` deterministically."""
    seen: dict[str, None] = {}
    for prompt in prompts:
        text = prompt.strip()
        if text:
            seen.setdefault(text, None)
    unique = sorted(seen)
    if len(unique) <= cap:
        return unique
    chosen = rng.sample(unique, cap)
    return sorted(chosen)


# --------------------------------------------------------------------------- #
# tool
# --------------------------------------------------------------------------- #


def _tool_prompts(rng: random.Random) -> list[str]:
    cities = ["tokyo", "delhi", "new york", "london", "dubai", "paris", "sydney"]
    companies = ["apple", "tesla", "infosys", "NVDA", "amazon", "microsoft", "google"]
    countries = ["india", "the us", "japan", "france", "brazil", "germany"]
    math_exprs = ["234 * 78", "15% of 240", "1024 / 16", "sqrt of 144", "37 + 89 * 2"]
    units = ["10 km to miles", "100 F to C", "5 kg to pounds", "3 liters to gallons"]
    news_topics = ["india", "technology", "the stock market", "the elections", "sports"]
    roles = ["president", "PM", "CEO"]
    sports = ["the lakers game", "the world cup final", "yesterday's cricket match"]

    templates = [
        "what time is it in {city}",
        "what's the current time in {city} right now",
        "tell me the time in {city}",
        "what is the weather in {city}",
        "how's the weather looking in {city} today",
        "give me today's forecast for {city}",
        "what's the date today",
        "what day of the week is it",
        "what is {math}",
        "calculate {math} for me",
        "can you compute {math}",
        "convert {unit}",
        "how much is {unit}",
        "what's the stock price of {company}",
        "current share price for {company}",
        "how is {company} stock doing today",
        "give me the latest news of {topic}",
        "what's the latest news on {topic}",
        "who is the {role} of {country}",
        "what is the exchange rate for usd to inr",
        "convert 100 dollars to euros",
        "what's the score of {sport}",
        "who won {sport}",
    ]
    slots = {
        "city": cities,
        "company": companies,
        "country": countries,
        "math": math_exprs,
        "unit": units,
        "topic": news_topics,
        "role": roles,
        "sport": sports,
    }
    return _cap(_expand(templates, slots), PER_CLASS_CAP, rng)


# --------------------------------------------------------------------------- #
# local
# --------------------------------------------------------------------------- #


def _local_prompts(rng: random.Random) -> list[str]:
    greetings = [
        "hi",
        "hello there",
        "hey, how's it going",
        "good morning",
        "what's up",
        "how are you doing today",
        "nice to meet you",
    ]
    phenomena = [
        "the sky blue",
        "the ocean salty",
        "the grass green",
        "the sun bright",
        "ice slippery",
        "the moon visible at night",
        "snow white",
        "fire hot",
    ]
    concepts = [
        "a noun",
        "photosynthesis",
        "gravity",
        "a metaphor",
        "democracy",
        "inflation",
        "an ecosystem",
        "a black hole",
    ]
    feelings = ["anxious", "depressed", "lonely", "overwhelmed", "stressed", "burnt out"]
    relations = ["partner", "boyfriend", "girlfriend", "spouse", "best friend", "coworker"]
    personal_things = ["photos", "details", "diary", "messages", "notes", "calendar"]
    rewrite_lines = [
        "the meeting was good",
        "thanks for your help yesterday",
        "we shipped the feature on time",
        "please find the attached report",
    ]

    templates = [
        "why is {phenomenon}",
        "can you explain why {phenomenon}",
        "what is {concept}",
        "give me a simple definition of {concept}",
        "rewrite this sentence to be clearer: {line}",
        "summarize this in one line: {line}",
        "make this shorter: {line}",
        "tell me a fun fact",
        "what's a good word for happy",
        "I've been feeling {feeling} lately, any advice",
        "how do I cope when I feel {feeling}",
        "I'm having problems with my {relation}, what should I do",
        "I had a fight with my {relation} and feel terrible",
        "can you help me organize my personal {thing}",
        "please keep my personal {thing} private",
        "is my current salary fair for my role",
        "should I tell my boss about my personal situation",
        "I have a small health concern I want to talk about",
        "how do I deal with grief after a loss",
        "I feel like no one understands me right now",
    ]
    slots = {
        "phenomenon": phenomena,
        "concept": concepts,
        "line": rewrite_lines,
        "feeling": feelings,
        "relation": relations,
        "thing": personal_things,
    }
    expanded = _expand(templates, slots) + greetings
    return _cap(expanded, PER_CLASS_CAP, rng)


# --------------------------------------------------------------------------- #
# coding
# --------------------------------------------------------------------------- #


def _coding_prompts(rng: random.Random) -> list[str]:
    langs = ["python", "java", "javascript", "go", "rust", "c++", "typescript"]
    tasks = [
        "reverses a string",
        "checks if a number is prime",
        "sorts a list of integers",
        "reads a csv file",
        "calls a REST endpoint",
        "validates an email address",
    ]
    structs = ["a linked list", "a binary tree", "a hash map", "a stack", "a queue"]
    algos = ["binary search", "quicksort", "dijkstra's algorithm", "BFS", "merge sort"]
    features = ["login", "signup", "dashboard", "search", "profile"]
    apps = ["a website", "a web app", "a REST API", "a mobile app", "a CLI tool"]

    templates = [
        "write a {lang} function that {task}",
        "show me {lang} code that {task}",
        "debug this {lang} code, it keeps crashing",
        "refactor this {lang} function to be cleaner",
        "implement {struct} in {lang}",
        "how do I reverse {struct}",
        "implement {algo} from scratch",
        "explain how {algo} works with code",
        "create a project with a {feature} page",
        "build {app} with a {feature} page",
        "make a website with html and css",
        "write a sql query to join two tables",
        "fix this failing test",
        "why is my {lang} unit test failing",
    ]
    slots = {
        "lang": langs,
        "task": tasks,
        "struct": structs,
        "algo": algos,
        "feature": features,
        "app": apps,
    }
    return _cap(_expand(templates, slots), PER_CLASS_CAP, rng)


# --------------------------------------------------------------------------- #
# reasoning
# --------------------------------------------------------------------------- #


def _reasoning_prompts(rng: random.Random) -> list[str]:
    systems = [
        "a chat application",
        "a payment system",
        "a recommendation engine",
        "a model router",
        "a notification service",
    ]
    scales = [
        "a million users",
        "ten thousand requests per second",
        "global multi-region traffic",
        "a small startup",
    ]
    options_a = ["postgres", "rest", "monolith", "sql", "kafka"]
    options_b = ["mongodb", "graphql", "microservices", "nosql", "rabbitmq"]
    use_cases = [
        "a write-heavy workload",
        "a read-heavy workload",
        "real-time analytics",
        "an event-driven system",
    ]
    choices = [
        "event sourcing",
        "sharding the database",
        "adopting kubernetes",
        "going serverless",
    ]
    plans = [
        "a migration from monolith to microservices",
        "a zero-downtime database upgrade",
        "a product launch",
        "scaling the team",
    ]

    templates = [
        "design {system} for {scale}",
        "how would you architect {system} at {scale}",
        "compare {a} vs {b} for {use_case}",
        "what's better for {use_case}, {a} or {b}",
        "review this architecture and flag scaling risks",
        "plan {plan} step by step",
        "what are the tradeoffs of {choice}",
        "walk me through the pros and cons of {choice}",
        "evaluate the long-term maintainability of {choice}",
    ]
    slots = {
        "system": systems,
        "scale": scales,
        "a": options_a,
        "b": options_b,
        "use_case": use_cases,
        "choice": choices,
        "plan": plans,
    }
    return _cap(_expand(templates, slots), PER_CLASS_CAP, rng)


def template_examples() -> list[RouterExample]:
    rng = random.Random(SEED)
    examples: list[RouterExample] = []
    for label, builder in (
        ("tool", _tool_prompts),
        ("local", _local_prompts),
        ("coding", _coding_prompts),
        ("reasoning", _reasoning_prompts),
    ):
        for prompt in builder(rng):
            examples.append(RouterExample(prompt=prompt, label=label, source="template"))
    return _dedupe(examples)


def golden_examples() -> list[RouterExample]:
    pairs = [
        ("what's the date today", "tool"),
        ("current time", "tool"),
        ("time in tokyo", "tool"),
        ("what is 234 * 78", "tool"),
        ("convert 10 km to miles", "tool"),
        ("whats the stock price of LAC", "tool"),
        ("stock price of Amazon", "tool"),
        ("give me latest news of india", "tool"),
        ("what is the name of us president", "tool"),
        ("what is the weather in Delhi", "tool"),
        ("hi", "local"),
        ("how are you doing?", "local"),
        ("how am i doing?", "local"),
        ("why is sky blue", "local"),
        ("what is the color of my tshirt", "local"),
        ("how to get out of depression?", "local"),
        ("how to make a girl fall in love?", "local"),
        ("I need advice about my relationship problems", "local"),
        ("summarize this short private note about my weekend plans", "local"),
        ("rewrite this note to be clearer", "local"),
        ("how to reverse a linked list in java", "coding"),
        ("how to revert a binary tree", "coding"),
        ("create me a project that has a login page with personal images stored", "coding"),
        ("build me an app with a signup page", "coding"),
        ("make a website with html and css", "coding"),
        ("fix this failing pytest run", "coding"),
        ("write a python function that prints today's date", "coding"),
        ("review this architecture for a model router", "reasoning"),
        ("design a database schema and evaluate scaling risk", "reasoning"),
        ("compare postgres vs mongodb for a write-heavy workload", "reasoning"),
        ("plan a migration from monolith to microservices", "reasoning"),
        ("what are the tradeoffs of event sourcing", "reasoning"),
    ]
    return [RouterExample(prompt=p, label=lbl, source="golden") for p, lbl in pairs]


def _dedupe(examples: list[RouterExample]) -> list[RouterExample]:
    seen: set[str] = set()
    out: list[RouterExample] = []
    for ex in examples:
        if ex.prompt in seen:
            continue
        seen.add(ex.prompt)
        out.append(ex)
    return out


def build_dataset() -> list[RouterExample]:
    return _dedupe(template_examples() + golden_examples())


def class_counts(examples: list[RouterExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex.label] = counts.get(ex.label, 0) + 1
    return counts


def write_jsonl(examples: list[RouterExample], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            record = {"prompt": ex.prompt, "label": ex.label, "source": ex.source}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "router_dataset.jsonl"
    dataset = build_dataset()
    print(f"total: {len(dataset)}")
    print(f"class_counts: {class_counts(dataset)}")
    write_jsonl(dataset, out_path)
    print(f"wrote {len(dataset)} examples to {out_path}")


def relabel_toolable(examples: list[RouterExample]) -> tuple[list[RouterExample], int]:
    """Tools-first dataset cleaning: examples a shipped tool can VERIFIABLY
    answer are relabeled ``tool``.

    Verification beats keyword matching here: "is it raining in delhi" is a
    tool question, but "how are weather forecasts created?" merely mentions
    the topic. So deterministic tools (calculator, unit conversion) are
    actually executed and must succeed; time/date and stock require their
    specific detector patterns plus, for stocks, an offline ticker
    resolution; live-info capabilities require live-intent phrasing
    (latest/current/news), which their detector patterns already enforce.
    Bare-topic weather mentions are NOT relabeled.

    Golden cases and user feedback are never relabeled: hand truth outranks
    heuristics.
    """
    import re as _re

    from switchboard.app.models.capabilities import Capability
    from switchboard.app.services.capabilities import CapabilityDetector
    from switchboard.app.services.deterministic_tools import (
        CalculatorTool,
        UnitConversionTool,
    )
    from switchboard.app.services.finance_providers import (
        UnconfiguredFinanceProvider,
    )
    from switchboard.app.services.finance_tool import StockPriceTool

    detector = CapabilityDetector()
    calculator = CalculatorTool()
    converter = UnitConversionTool()
    stock = StockPriceTool(UnconfiguredFinanceProvider())
    # Keep in sync with CapabilityDetector._is_weather: live-intent phrasings
    # only, so bare topic mentions ("how are weather forecasts created?")
    # are never relabeled tool.
    live_weather = _re.compile(
        r"is it (?:raining|snowing)|will it (?:rain|snow)"
        r"|(?:is|will) (?:it|there) (?:gonna |going to |about to )?(?:be )?"
        r"(?:rain|raining|snow|snowing|showers?|thunderstorms?)\b"
        r"|(?:gonna|going to) (?:rain|snow)"
        r"|chances? of (?:rain|snow|showers?|thunderstorms?)"
        r"|(?:rain|raining|snow|snowing)\b[^.?!]*"
        r"\b(?:today|tonight|tomorrow|this (?:morning|afternoon|evening|week|weekend))"
        r"|forecast (?:for|in|this|today|tomorrow|tonight)|temperature"
        r"|umbrella|how (?:hot|cold|humid|windy)|air quality|\baqi\b|uv index"
    )

    def verifiably_toolable(prompt: str) -> bool:
        detection = detector.detect(prompt)
        # Mirror the inference-side guard: prompts that also need coding or
        # reasoning work are not tool questions, even if they mention dates,
        # numbers, or live topics ("write code to print the current date").
        if detection.has(Capability.CODING) or detection.has(Capability.REASONING):
            return False
        if detection.has(Capability.CALCULATION):
            return calculator.answer(prompt).success
        if detection.has(Capability.UNIT_CONVERSION):
            return converter.answer(prompt).success
        if detection.has(Capability.CURRENT_TIME) or detection.has(Capability.CURRENT_DATE):
            return True  # TimeTool always succeeds for detected patterns.
        if detection.has(Capability.STOCK_PRICE):
            symbol, _ = stock.resolve_symbol(prompt)
            return symbol is not None
        if detection.has(Capability.WEATHER):
            return bool(live_weather.search(prompt.lower()))
        # LATEST_INFO / WEB_SEARCH detector patterns already require live
        # intent (latest/current/news/exchange rate/who won...).
        return detection.has(Capability.LATEST_INFO) or detection.has(Capability.WEB_SEARCH)

    relabeled: list[RouterExample] = []
    changed = 0
    for example in examples:
        if example.source in {"golden", "feedback"} or example.label == "tool":
            relabeled.append(example)
            continue
        if verifiably_toolable(example.prompt):
            relabeled.append(
                RouterExample(prompt=example.prompt, label="tool", source=example.source)
            )
            changed += 1
        else:
            relabeled.append(example)
    return relabeled, changed
