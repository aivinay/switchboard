"""Training data for the learned sensitivity escalator.

Classes: ``sensitive`` (must stay on the local model) vs ``neutral``.

The escalator exists to catch sensitive phrasings the keyword hints miss, so
the sensitive templates deliberately avoid trigger words where possible
("I've been crying a lot lately" rather than "I have depression"). Neutral
examples reuse the router templates (tool/coding/reasoning/factual prompts)
plus build-request distractors that historically caused false privacy
reroutes ("login page with my personal images").

The golden set encodes both promises: known keyword misses MUST escalate,
and ordinary build/work requests MUST NOT.
"""

from __future__ import annotations

import random

from switchboard.training.router_dataset import RouterExample, build_dataset

# Sensitive phrasings grouped by theme; most avoid the keyword hints on
# purpose. Expanded with light template variation below.
_SENSITIVE_SEEDS: dict[str, list[str]] = {
    "emotional": [
        "i've been crying a lot lately and i don't know why",
        "i can't get out of bed most mornings anymore",
        "everything feels pointless these days",
        "i feel like nobody would notice if i disappeared",
        "i haven't slept properly in weeks because my mind races",
        "i snapped at my kids again and i hate myself for it",
        "i feel empty even when good things happen",
        "lately i just want to be alone all the time",
        "i keep replaying an embarrassing moment over and over",
        "i'm scared i'm becoming someone i don't like",
    ],
    "relationships": [
        "i think my marriage is falling apart",
        "my wife and i haven't really talked in months",
        "i found messages on my husband's phone that worry me",
        "i don't know how to tell my partner i'm unhappy",
        "my best friend stopped talking to me and it hurts",
        "i'm still in love with my ex and it's been a year",
        "my parents keep pressuring me about having children",
        "i feel invisible to my family",
    ],
    "health": [
        "my chest hurts when i climb the stairs, should i be worried",
        "i found a lump and i'm too scared to get it checked",
        "i've been drinking more than i want to admit",
        "i keep forgetting things and it's scaring me",
        "the test results came back and i don't understand them",
        "i haven't told anyone about my eating habits",
        "i missed my pills three days in a row",
        "my hands shake when i'm stressed, is that normal",
    ],
    "money_work": [
        "i owe more money than i make and it's getting worse",
        "i'm behind on rent and too ashamed to ask for help",
        "how do i tell my boss i'm struggling to keep up",
        "i think i'm about to be fired and i can't tell my family",
        "my side of the family keeps asking me for money",
        "i lied on my expense report and i regret it",
        "i make less than everyone on my team and it eats at me",
    ],
    "identity_docs": [
        "here is my home address and phone number, draft the letter",
        "my aadhaar number is in this document, summarize it",
        "this file has my bank account and routing numbers",
        "the attachment includes my children's school details",
        "use my real name and birthday to fill this form",
    ],
}

# Neutral distractors: emotionally-flavored or "personal"-flavored requests
# that are actually ordinary work. These historically caused false reroutes.
_NEUTRAL_DISTRACTORS = [
    "build a login page for my project with my personal images",
    "create a personal website with a photo gallery",
    "write a sad story about a robot who loses its memory",
    "my code is depressing, refactor this mess",
    "the server keeps dying, can you debug it",
    "this api is killing me, why does it return 401",
    "write a breakup letter from the perspective of a printer",
    "design a database schema for a therapy booking startup",
    "summarize this article about rising healthcare costs",
    "what is the divorce rate in european countries",
    "explain how antidepressants work chemically",
    "compare term and whole life insurance products in general",
    "write documentation for the payments module",
    "how do banks calculate mortgage interest, in general",
    "draft a blog post about burnout culture in tech",
    "what programming language should i learn first",
]

_VARIATION_PREFIXES = ["", "hey, ", "honestly ", "i need to say this somewhere: ", "lately "]


def sensitivity_examples(*, seed: int = 42) -> list[RouterExample]:
    """Sensitive seeds (with light variation) + neutral router templates and
    distractors, deterministically shuffled."""
    rng = random.Random(seed)
    examples: list[RouterExample] = []
    for prompts in _SENSITIVE_SEEDS.values():
        for prompt in prompts:
            examples.append(
                RouterExample(prompt=prompt, label="sensitive", source="template")
            )
            prefix = rng.choice(_VARIATION_PREFIXES)
            if prefix:
                examples.append(
                    RouterExample(
                        prompt=f"{prefix}{prompt}", label="sensitive", source="template"
                    )
                )
    for prompt in _NEUTRAL_DISTRACTORS:
        examples.append(RouterExample(prompt=prompt, label="neutral", source="template"))
    # Router templates are ordinary requests: a broad neutral sample.
    neutral_pool = [
        RouterExample(prompt=example.prompt, label="neutral", source=example.source)
        for example in build_dataset()
        if example.label in {"tool", "coding", "reasoning", "local"}
    ]
    rng.shuffle(neutral_pool)
    golden_prompts = {ex.prompt.lower() for ex in sensitivity_golden_examples()}
    merged = examples + neutral_pool[:400]
    seen: set[str] = set()
    unique: list[RouterExample] = []
    for example in merged:
        key = example.prompt.lower()
        if key in seen or key in golden_prompts:
            continue
        seen.add(key)
        unique.append(example)
    rng.shuffle(unique)
    return unique


def sensitivity_golden_examples() -> list[RouterExample]:
    """Deploy gate: keyword misses that MUST escalate, and ordinary requests
    that MUST NOT (including the historic 'personal images' false positive)."""
    must_escalate = [
        "i've been crying every night this week",
        "i think my marriage is over and i'm lost",
        "i found a lump and i'm scared to see anyone about it",
        "i owe so much money i can't sleep",
        "i feel like a burden to everyone around me",
        "my drinking is getting out of hand",
    ]
    must_not_escalate = [
        "build a login page with my personal images for the gallery",
        "refactor this depressing legacy code",
        "what is the divorce rate in sweden",
        "explain how ssris work pharmacologically",
        "design a schema for a mental health appointment app",
        "summarize this news article about hospital funding",
        "what time is it in tokyo",
        "write a python function to merge two sorted lists",
    ]
    return [
        RouterExample(prompt=prompt, label="sensitive", source="golden")
        for prompt in must_escalate
    ] + [
        RouterExample(prompt=prompt, label="neutral", source="golden")
        for prompt in must_not_escalate
    ]
