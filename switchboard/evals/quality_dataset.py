"""Hand-curated quality benchmark dataset for the Switchboard router.

This module defines 100 evaluation cases used to measure end-to-end answer
quality across the backends Switchboard routes to: ``codex`` (coding CLI
agent), ``claude-code`` (reasoning CLI agent), and ``ollama`` (local models).

Categories:

* ``coding`` (25): small, self-contained programming tasks.
* ``reasoning`` (25): architecture, design, and planning questions.
* ``summarization`` (25): faithful compression of a supplied passage.
* ``private`` (15): benign tasks over sensitive content that must stay local.
* ``grounding`` (10): time, date, and live-data questions that test honest
  handling of information the model cannot know without grounding.

Each case carries a short rubric and a keyword set used by automated scorers
as a cheap proxy for rubric coverage. All passages, companies, and people in
the prompts are invented; private cases use first names only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    category: str  # one of: coding, reasoning, summarization, private, grounding
    prompt: str
    rubric: str  # 1-2 sentences: what a good answer must contain/do
    keywords: tuple[str, ...]  # 3-6 lowercase terms a good answer would likely mention
    expected_local_only: bool = False  # True only for private cases
    expected_route_type: str | None = None  # coding | reasoning | local | None


def _coding_cases() -> list[QualityCase]:
    return [
        QualityCase(
            case_id="q_code_001",
            category="coding",
            prompt=(
                "Write a Python function merge_intervals(intervals: list[tuple[int, int]]) -> "
                "list[tuple[int, int]] that merges overlapping closed intervals and returns the "
                "result sorted by start. Include a brief docstring."
            ),
            rubric=(
                "Provides a correct implementation that sorts by start and merges overlapping "
                "intervals, handling the empty-list case."
            ),
            keywords=("sort", "merge", "overlap", "intervals"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_002",
            category="coding",
            prompt=(
                "Find the bug in this Python function and show the fixed version:\n\n"
                "def add_tag(tag, tags=[]):\n"
                "    tags.append(tag)\n"
                "    return tags"
            ),
            rubric=(
                "Identifies the mutable default argument shared across calls and fixes it with "
                "a None default and a fresh list inside the function."
            ),
            keywords=("mutable", "default", "argument", "none", "shared"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_003",
            category="coding",
            prompt=(
                "My Python script fails with: TypeError: 'NoneType' object is not subscriptable "
                "on the line `name = result[0]`. Explain the most likely causes and how to track "
                "down which one applies."
            ),
            rubric=(
                "Explains that result is None, lists common causes such as a function returning "
                "None or a failed lookup, and suggests checking the value before indexing."
            ),
            keywords=("none", "return", "subscript", "check", "debug"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_004",
            category="coding",
            prompt=(
                "Write pytest tests for a function slugify(title: str) -> str that lowercases, "
                "replaces spaces with hyphens, and strips punctuation. Cover normal input, "
                "repeated spaces, an empty string, and unicode accents."
            ),
            rubric=(
                "Provides runnable pytest tests, ideally parametrized, covering the listed edge "
                "cases with clear expected values."
            ),
            keywords=("pytest", "parametrize", "assert", "slugify", "empty"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_005",
            category="coding",
            prompt=(
                "Write a regular expression that matches ISO-8601 calendar dates like 2026-06-11 "
                "(four-digit year, two-digit month 01-12, two-digit day 01-31) and rejects "
                "obviously invalid months like 13. Explain each part briefly."
            ),
            rubric=(
                "Gives a regex with anchors that constrains month to 01-12 and day to 01-31, and "
                "explains the groups; may note it does not validate days per month."
            ),
            keywords=("regex", "anchor", "month", "day", "group"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_006",
            category="coding",
            prompt=(
                "Explain what `git rebase -i HEAD~3` does, when you would use it instead of "
                "merge, and one risk to be aware of when the commits are already pushed."
            ),
            rubric=(
                "Explains interactive rebase over the last three commits (reorder, squash, "
                "reword), contrasts with merge, and warns about rewriting shared history."
            ),
            keywords=("rebase", "interactive", "squash", "history", "force"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_007",
            category="coding",
            prompt=(
                "Refactor this Python function to reduce nesting and improve readability, "
                "keeping behavior identical:\n\n"
                "def process(order):\n"
                "    if order is not None:\n"
                "        if order.items:\n"
                "            if order.status == 'open':\n"
                "                total = sum(i.price for i in order.items)\n"
                "                return total\n"
                "            else:\n"
                "                return 0\n"
                "        else:\n"
                "            return 0\n"
                "    else:\n"
                "        return 0"
            ),
            rubric=(
                "Rewrites the function with guard clauses or a single combined condition, "
                "preserving the exact return behavior for all branches."
            ),
            keywords=("guard", "return", "nesting", "readability"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_008",
            category="coding",
            prompt=(
                "Write a SQL query against tables customers(id, name) and orders(id, "
                "customer_id, total, created_at) that returns the five customers with the "
                "highest lifetime order value in 2025, with their totals."
            ),
            rubric=(
                "Joins the tables, filters to 2025, groups by customer, sums totals, orders "
                "descending, and limits to five rows."
            ),
            keywords=("join", "group", "sum", "order", "limit"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_009",
            category="coding",
            prompt=(
                "Give me a bash one-liner that finds files larger than 100 MB modified in the "
                "last 7 days under /var/log, printing size and path, sorted largest first. "
                "Explain each flag."
            ),
            rubric=(
                "Uses find with -size and -mtime correctly, formats or pipes through sort, and "
                "explains the flags accurately."
            ),
            keywords=("find", "size", "mtime", "sort"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_010",
            category="coding",
            prompt=(
                "Write a debounce(fn, waitMs) function in plain JavaScript (no libraries) that "
                "delays calls to fn until waitMs of inactivity, preserving `this` and the most "
                "recent arguments. Show a usage example with a resize handler."
            ),
            rubric=(
                "Implements debounce with setTimeout/clearTimeout in a closure, applies fn with "
                "saved this and args, and shows a realistic usage example."
            ),
            keywords=("debounce", "settimeout", "cleartimeout", "closure", "apply"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_011",
            category="coding",
            prompt=(
                "This binary search sometimes loops forever. Find the bug and fix it:\n\n"
                "def search(a, target):\n"
                "    lo, hi = 0, len(a) - 1\n"
                "    while lo < hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if a[mid] < target:\n"
                "            lo = mid\n"
                "        else:\n"
                "            hi = mid\n"
                "    return lo if a and a[lo] == target else -1"
            ),
            rubric=(
                "Identifies that `lo = mid` fails to shrink the range when hi == lo + 1 and "
                "fixes it with `lo = mid + 1`, explaining the invariant."
            ),
            keywords=("binary", "midpoint", "infinite", "loop", "boundary"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_012",
            category="coding",
            prompt=(
                "Explain this Python error and how to fix it: UnicodeDecodeError: 'utf-8' codec "
                "can't decode byte 0xff in position 0: invalid start byte. It happens when I "
                "call open(path).read() on a file a customer uploaded."
            ),
            rubric=(
                "Explains the file is not valid UTF-8 (possibly binary or another encoding), "
                "and suggests opening in binary mode, detecting the encoding, or specifying "
                "errors/encoding explicitly."
            ),
            keywords=("encoding", "utf-8", "binary", "bytes", "latin-1"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_013",
            category="coding",
            prompt=(
                "Implement an LRU cache decorator in Python without using functools.lru_cache. "
                "It should take a max_size argument, evict the least recently used entry when "
                "full, and work for functions with hashable positional arguments."
            ),
            rubric=(
                "Implements a decorator using an OrderedDict or equivalent, moves hits to the "
                "end, evicts the oldest entry at capacity, and preserves the wrapped function."
            ),
            keywords=("ordereddict", "evict", "capacity", "decorator", "wraps"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_014",
            category="coding",
            prompt=(
                "Explain the difference between LEFT JOIN and INNER JOIN in SQL with a small "
                "concrete example using users and orders tables, including what happens to "
                "users who have no orders."
            ),
            rubric=(
                "States that INNER JOIN keeps only matching rows while LEFT JOIN keeps all left "
                "rows with NULLs for unmatched right columns, illustrated with example rows."
            ),
            keywords=("left", "inner", "null", "unmatched", "rows"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_015",
            category="coding",
            prompt=(
                "Write a pytest fixture that creates a temporary SQLite database with a "
                "users(id, email) table, yields a connection, and cleans up afterwards. Show "
                "one test that uses it to insert and read back a row."
            ),
            rubric=(
                "Defines a fixture using tmp_path or :memory:, creates the schema, yields the "
                "connection, closes it on teardown, and shows a passing example test."
            ),
            keywords=("fixture", "sqlite", "yield", "tmp_path", "teardown"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_016",
            category="coding",
            prompt=(
                "Write a regex that validates semantic version strings like 1.4.2 and "
                "2.0.0-rc.1, allowing an optional pre-release suffix after a hyphen, and "
                "rejecting leading zeros in the numeric parts. Explain the pieces."
            ),
            rubric=(
                "Provides an anchored regex with major, minor, and patch groups disallowing "
                "leading zeros and an optional pre-release part, with a short explanation."
            ),
            keywords=("regex", "major", "minor", "patch", "anchor"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_017",
            category="coding",
            prompt=(
                "Why does this print 4 4 4 instead of 0 2 4, and what are two ways to fix "
                "it?\n\n"
                "fns = []\n"
                "for i in range(3):\n"
                "    fns.append(lambda: i * 2)\n"
                "for f in fns:\n"
                "    print(f())"
            ),
            rubric=(
                "Explains late binding of the loop variable in closures and offers fixes such "
                "as a default argument (lambda i=i: ...) or functools.partial."
            ),
            keywords=("closure", "late", "binding", "default", "lambda"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_018",
            category="coding",
            prompt=(
                "Explain how `git bisect` works and walk through using it to find which commit "
                "broke a test, including how to automate it with `git bisect run`."
            ),
            rubric=(
                "Describes the binary search over commits with good/bad marks, gives the "
                "start/good/bad workflow, and shows automating with bisect run and a script's "
                "exit code."
            ),
            keywords=("bisect", "binary", "good", "bad", "regression"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_019",
            category="coding",
            prompt=(
                "Write a Python generator windows(iterable, n) that yields overlapping sliding "
                "windows of length n as tuples, working on any iterable (including generators) "
                "without loading it all into memory."
            ),
            rubric=(
                "Implements a lazy generator, typically with collections.deque(maxlen=n), that "
                "yields tuples and handles iterables shorter than n."
            ),
            keywords=("generator", "yield", "deque", "window", "lazy"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_020",
            category="coding",
            prompt=(
                "Refactor this to remove duplication and add one retry on failure:\n\n"
                "def get_user(uid):\n"
                "    r = requests.get(BASE + '/users/' + str(uid), timeout=5)\n"
                "    r.raise_for_status()\n"
                "    return r.json()\n\n"
                "def get_team(tid):\n"
                "    r = requests.get(BASE + '/teams/' + str(tid), timeout=5)\n"
                "    r.raise_for_status()\n"
                "    return r.json()"
            ),
            rubric=(
                "Extracts a shared helper for GET-and-parse with a single retry, keeps both "
                "public functions, and avoids changing their signatures."
            ),
            keywords=("helper", "duplication", "retry", "timeout"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_021",
            category="coding",
            prompt=(
                "In a Jupyter notebook, calling asyncio.run(main()) raises RuntimeError: "
                "asyncio.run() cannot be called from a running event loop. Explain why this "
                "happens in notebooks and give two ways to run the coroutine correctly."
            ),
            rubric=(
                "Explains that Jupyter already runs an event loop, and suggests awaiting the "
                "coroutine directly or using nest_asyncio / loop.create_task as alternatives."
            ),
            keywords=("event", "loop", "await", "jupyter", "nest_asyncio"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_022",
            category="coding",
            prompt=(
                "Explain what each part of `set -euo pipefail` does at the top of a bash "
                "script, and give one example per flag of a bug it catches."
            ),
            rubric=(
                "Explains -e (exit on error), -u (error on unset variables), and -o pipefail "
                "(fail a pipeline if any stage fails), each with a concrete example."
            ),
            keywords=("errexit", "nounset", "pipefail", "exit", "unset"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_023",
            category="coding",
            prompt=(
                "This JavaScript logs 'done' before any user is fetched. Explain why and fix "
                "it:\n\n"
                "async function loadAll(ids) {\n"
                "  ids.forEach(async (id) => {\n"
                "    const user = await fetchUser(id);\n"
                "    console.log(user.name);\n"
                "  });\n"
                "  console.log('done');\n"
                "}"
            ),
            rubric=(
                "Explains that forEach does not await async callbacks, and fixes it with a "
                "for...of loop with await or Promise.all over mapped promises."
            ),
            keywords=("foreach", "await", "promise", "for...of", "all"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_024",
            category="coding",
            prompt=(
                "Using a SQL window function, write a query over payments(account_id, paid_at, "
                "amount) that returns each payment with a running total per account ordered by "
                "paid_at. Briefly explain the OVER clause you used."
            ),
            rubric=(
                "Uses SUM(amount) OVER (PARTITION BY account_id ORDER BY paid_at) and explains "
                "partitioning and ordering within the window."
            ),
            keywords=("window", "over", "partition", "order", "sum"),
            expected_route_type="coding",
        ),
        QualityCase(
            case_id="q_code_025",
            category="coding",
            prompt=(
                "Write a Python function parse_duration(s: str) -> int that converts strings "
                "like '1h30m', '45s', or '2h' into total seconds, raising ValueError on inputs "
                "it cannot parse. Include three example calls with expected results."
            ),
            rubric=(
                "Parses hour, minute, and second components (commonly via regex), sums to "
                "seconds, raises ValueError on garbage input, and shows correct examples."
            ),
            keywords=("regex", "seconds", "parse", "valueerror"),
            expected_route_type="coding",
        ),
    ]


def _reasoning_cases() -> list[QualityCase]:
    return [
        QualityCase(
            case_id="q_reason_001",
            category="reasoning",
            prompt=(
                "We are building an order-management service: about 2,000 writes per second at "
                "peak, strong consistency required for inventory decrements, and the analytics "
                "team needs ad hoc SQL queries. Compare PostgreSQL and DynamoDB for this "
                "workload and recommend one."
            ),
            rubric=(
                "Weighs consistency, ad hoc query needs, scaling, and operational cost for both "
                "options, and makes a justified recommendation tied to the stated requirements."
            ),
            keywords=("consistency", "query", "scaling", "transactions", "cost"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_002",
            category="reasoning",
            prompt=(
                "A six-person startup with one product is debating splitting its Django "
                "monolith into microservices because deploys feel risky. Argue for and against, "
                "and suggest what they should actually do first."
            ),
            rubric=(
                "Explains the operational cost of microservices for a tiny team, identifies "
                "that deploy risk has cheaper fixes (CI, tests, smaller releases), and gives a "
                "pragmatic recommendation."
            ),
            keywords=("monolith", "microservices", "deploy", "complexity", "team"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_003",
            category="reasoning",
            prompt=(
                "Estimate how many application servers we need for an API expected to serve "
                "50,000 requests per second at peak, if one instance handles about 1,200 "
                "requests per second at 60 percent CPU. Walk through the math, headroom "
                "assumptions, and what else you would want to measure."
            ),
            rubric=(
                "Shows the arithmetic, adds explicit headroom and failure-domain margin, and "
                "notes caveats such as latency targets, traffic shape, and downstream limits."
            ),
            keywords=("throughput", "headroom", "instances", "latency", "peak"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_004",
            category="reasoning",
            prompt=(
                "Outline a zero-downtime migration plan from MySQL 5.7 to PostgreSQL 15 for a "
                "300 GB transactional database backing a web app. Cover data sync, cutover, "
                "rollback, and how you would validate correctness."
            ),
            rubric=(
                "Proposes phased replication or dual writes, a verification step comparing "
                "data, a controlled cutover with rollback plan, and schema/query compatibility "
                "work."
            ),
            keywords=("replication", "dual", "cutover", "rollback", "validation"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_005",
            category="reasoning",
            prompt=(
                "Our platform team is choosing between REST with JSON and gRPC for internal "
                "service-to-service APIs across about 30 services in Go and Python. Compare "
                "the two for this context and state when each wins."
            ),
            rubric=(
                "Compares schema enforcement, performance, streaming, debugging, and tooling "
                "maturity, and gives context-dependent guidance rather than a blanket answer."
            ),
            keywords=("grpc", "rest", "schema", "streaming", "tooling"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_006",
            category="reasoning",
            prompt=(
                "Design review: users upload images to object storage, an uploader writes a row "
                "to a 'pending' table, and a single worker polls that table every 5 seconds to "
                "generate thumbnails. Volume is growing 20 percent per month. List the main "
                "weaknesses and how you would evolve this design."
            ),
            rubric=(
                "Identifies the single worker bottleneck, polling inefficiency, missing retry "
                "and idempotency story, and suggests a queue with multiple consumers and "
                "visibility into failures."
            ),
            keywords=("queue", "polling", "retry", "idempotency", "bottleneck"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_007",
            category="reasoning",
            prompt=(
                "We cache product pages in Redis with a 1-hour TTL, but merchandisers complain "
                "that price changes take up to an hour to appear. Propose a better cache "
                "invalidation strategy and discuss its tradeoffs."
            ),
            rubric=(
                "Proposes event-driven invalidation or versioned keys on price change, "
                "discusses consistency versus complexity, and keeps TTL as a safety net."
            ),
            keywords=("invalidation", "ttl", "event", "stale", "versioned"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_008",
            category="reasoning",
            prompt=(
                "Compliance asked for a complete history of changes to customer accounts. The "
                "team is debating full event sourcing versus keeping CRUD and adding an "
                "append-only audit table. Compare the approaches for a mid-size team and "
                "recommend one."
            ),
            rubric=(
                "Contrasts event sourcing's replay power and complexity (projections, "
                "versioning) against the simplicity of an audit table, and recommends based on "
                "the stated need."
            ),
            keywords=("event", "audit", "replay", "complexity", "append-only"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_009",
            category="reasoning",
            prompt=(
                "We run in one cloud region and leadership wants to survive a full region "
                "outage with under 15 minutes of downtime and under 1 minute of data loss. "
                "Sketch the main options (active-passive, active-active) and what each costs "
                "us in complexity."
            ),
            rubric=(
                "Maps the requirements to RTO/RPO, compares warm standby with async replication "
                "versus active-active, and covers failover triggering, DNS/traffic shifting, "
                "and testing."
            ),
            keywords=("failover", "replication", "rto", "rpo", "active-active"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_010",
            category="reasoning",
            prompt=(
                "Choose between Kafka and RabbitMQ for this workload: clickstream events at "
                "around 80,000 messages per second, consumers that may be offline for hours and "
                "need to catch up, and strict per-user ordering. Justify the choice."
            ),
            rubric=(
                "Recommends Kafka, citing retention/replay for offline consumers, partition "
                "keys for per-user ordering, and throughput, while noting RabbitMQ's strengths "
                "elsewhere."
            ),
            keywords=("kafka", "retention", "partition", "ordering", "throughput"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_011",
            category="reasoning",
            prompt=(
                "Our public REST API needs a breaking change to its pagination format. Compare "
                "URL versioning (/v2/), header-based versioning, and per-field evolution, then "
                "propose a deprecation plan for the old behavior."
            ),
            rubric=(
                "Compares the versioning schemes' discoverability and operational cost, and "
                "lays out a deprecation timeline with announcements, headers or warnings, and a "
                "sunset date."
            ),
            keywords=("versioning", "header", "deprecation", "sunset", "compatibility"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_012",
            category="reasoning",
            prompt=(
                "A 2 TB PostgreSQL users table is hitting vertical scaling limits. Discuss "
                "sharding strategies (hash by user_id, range by signup date, directory-based), "
                "the hotspot and rebalancing risks of each, and what cross-shard queries would "
                "cost us."
            ),
            rubric=(
                "Compares the shard key options with their hotspot, rebalancing, and lookup "
                "implications, and flags cross-shard joins and transactions as the major cost."
            ),
            keywords=("shard", "key", "hotspot", "rebalancing", "cross-shard"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_013",
            category="reasoning",
            prompt=(
                "Design rate limiting for our API gateway: 100 requests per minute per API key, "
                "bursts of up to 20 allowed, running on 12 gateway instances. Compare token "
                "bucket and sliding window approaches and where the counters should live."
            ),
            rubric=(
                "Explains token bucket's burst handling versus sliding window's smoothness, and "
                "addresses shared state across instances, typically centralized counters in "
                "Redis with atomic operations."
            ),
            keywords=("token", "bucket", "window", "burst", "redis"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_014",
            category="reasoning",
            prompt=(
                "We have about 40 cron jobs on a single VM doing nightly ETL, with no retries "
                "and failures noticed days later. Plan a migration to a workflow orchestrator "
                "such as Airflow: ordering of work, biggest risks, and what to migrate first."
            ),
            rubric=(
                "Proposes an incremental migration starting with the highest-risk or most "
                "dependent jobs, covers retries, alerting, dependencies, and backfill, and "
                "warns against a big-bang switch."
            ),
            keywords=("orchestrator", "retries", "dependencies", "backfill", "incremental"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_015",
            category="reasoning",
            prompt=(
                "We need threaded comments up to arbitrary depth, with fast retrieval of a "
                "whole thread and occasional moves of a subtree. Compare adjacency list with "
                "recursive CTEs, materialized path, and closure table schemas for this in "
                "PostgreSQL."
            ),
            rubric=(
                "Describes how each schema handles deep reads and subtree moves, including the "
                "write amplification of closure tables and path updates, and recommends with "
                "justification."
            ),
            keywords=("adjacency", "recursive", "path", "closure", "subtree"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_016",
            category="reasoning",
            prompt=(
                "Our checkout flow reads from PostgreSQL read replicas, and users sometimes do "
                "not see the order they just placed because of replication lag. Discuss "
                "options for read-your-writes consistency here and their costs."
            ),
            rubric=(
                "Offers options such as routing post-write reads to the primary, session "
                "stickiness, or LSN/timestamp tokens, and weighs added primary load against "
                "consistency."
            ),
            keywords=("lag", "replica", "primary", "consistency", "session"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_017",
            category="reasoning",
            prompt=(
                "Engineering wants feature flags with percentage rollouts and user targeting. "
                "We have 25 engineers. Compare building a flag system in-house against buying "
                "a vendor product, including the hidden costs of each, and recommend."
            ),
            rubric=(
                "Covers build costs (targeting, SDKs, audit, UI, reliability) versus vendor "
                "cost, lock-in, and data sharing, and recommends proportionate to team size."
            ),
            keywords=("flags", "targeting", "vendor", "audit", "cost"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_018",
            category="reasoning",
            prompt=(
                "For a payments service where a bad deploy is very expensive, compare "
                "blue-green deployment with canary releases. Which would you pick, what "
                "metrics would gate promotion, and how does database schema change complicate "
                "both?"
            ),
            rubric=(
                "Contrasts instant-switch blue-green with gradual canary risk exposure, names "
                "concrete gating metrics (error rate, latency, payment success), and notes "
                "backward-compatible migrations are needed for both."
            ),
            keywords=("canary", "blue-green", "rollback", "metrics", "migration"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_019",
            category="reasoning",
            prompt=(
                "Our app needs product search over 3 million SKUs with typo tolerance and "
                "faceted filtering. We already run PostgreSQL. Compare staying with Postgres "
                "full-text search against adding Elasticsearch, including the data sync burden."
            ),
            rubric=(
                "Weighs Postgres FTS simplicity and its relevance/typo limits against "
                "Elasticsearch capability plus the operational and synchronization cost of a "
                "second datastore."
            ),
            keywords=("full-text", "elasticsearch", "relevance", "sync", "facets"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_020",
            category="reasoning",
            prompt=(
                "Design idempotency for a POST /charges payment endpoint so client retries "
                "never double-charge. Cover key generation, where to store keys, how long to "
                "keep them, and what to return when a duplicate arrives mid-flight."
            ),
            rubric=(
                "Specifies client-supplied idempotency keys stored with request fingerprint and "
                "result, a retention window, and defined behavior for concurrent duplicates "
                "(e.g., 409 or wait)."
            ),
            keywords=("idempotency", "key", "retry", "duplicate", "store"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_021",
            category="reasoning",
            prompt=(
                "We must honor GDPR deletion requests within 30 days. User data lives in "
                "PostgreSQL, a data warehouse, object storage, and 90-day database backups. "
                "Design the deletion pipeline and explain how you handle the backups problem."
            ),
            rubric=(
                "Designs propagated deletion across stores with tracking and verification, and "
                "addresses backups via expiry windows plus re-deletion on restore or crypto-"
                "shredding."
            ),
            keywords=("deletion", "backups", "propagation", "audit", "restore"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_022",
            category="reasoning",
            prompt=(
                "Estimate yearly storage for a telemetry pipeline ingesting 5,000 events per "
                "second averaging 600 bytes each, with 13 months retention. Show the math, "
                "then discuss how compression and downsampling change the cost picture."
            ),
            rubric=(
                "Computes raw volume correctly (order of 100 TB/year before compression), then "
                "discusses compression ratios, downsampling older data, and tiered storage."
            ),
            keywords=("retention", "compression", "downsampling", "ingest", "tiered"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_023",
            category="reasoning",
            prompt=(
                "Design review: one Redis instance is used simultaneously as a cache with "
                "allkeys-lru eviction, a job queue via lists, and for distributed locks. What "
                "can go wrong with this setup, and how would you restructure it?"
            ),
            rubric=(
                "Spots that LRU eviction can silently drop queue entries and locks, notes "
                "persistence and failover concerns for locks, and recommends separating "
                "concerns into distinct instances or tools."
            ),
            keywords=("eviction", "queue", "locks", "persistence", "isolation"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_024",
            category="reasoning",
            prompt=(
                "We inherited a legacy PHP monolith with no tests that still serves all "
                "traffic. Describe a strangler-fig migration to new services: how routing "
                "works, what to extract first, and how to avoid a multi-year rewrite that "
                "never ships."
            ),
            rubric=(
                "Describes routing through a proxy/facade, extracting seams incrementally "
                "starting with low-risk or high-change areas, keeping both systems live, and "
                "measuring progress."
            ),
            keywords=("strangler", "proxy", "incremental", "routing", "seam"),
            expected_route_type="reasoning",
        ),
        QualityCase(
            case_id="q_reason_025",
            category="reasoning",
            prompt=(
                "A report-generation workload is idle most of the day but spikes to hundreds "
                "of concurrent jobs for two hours each night, each job running 1-5 minutes. "
                "Compare serverless functions with a container service plus autoscaling for "
                "this, including cost and limits."
            ),
            rubric=(
                "Compares scale-to-zero economics and cold starts against container runtime "
                "limits and autoscaling lag, addressing function duration/concurrency limits, "
                "and recommends with reasoning."
            ),
            keywords=("serverless", "cold", "scaling", "cost", "concurrency"),
            expected_route_type="reasoning",
        ),
    ]


_SUM_PREFIX = "Summarize the following in 2-3 bullet points, using only facts from the text:\n\n"


def _summarization_cases() -> list[QualityCase]:
    return [
        QualityCase(
            case_id="q_sum_001",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Notes from the Nortia Labs Q3 roadmap meeting on 14 May. The team agreed to "
                "prioritize the self-serve onboarding flow over the analytics export feature, "
                "since trial-to-paid conversion dropped to 9 percent last quarter. Priya will "
                "own the onboarding redesign and aims to ship a beta by 7 July. The analytics "
                "export moves to Q4 unless two more enterprise customers escalate. Engineering "
                "flagged that the billing service rewrite is blocking the new trial logic, so "
                "Daniel's team will deliver the billing API changes by 20 June. Marketing "
                "requested two weeks notice before any pricing page change."
            ),
            rubric=(
                "Faithful 2-3 bullet summary with no invented facts, covering the onboarding "
                "priority and conversion drop, the owner and beta date, and the billing "
                "dependency."
            ),
            keywords=("onboarding", "conversion", "billing", "beta", "q4"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_002",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Brightline Analytics released version 4.2 of its dashboard product on 2 "
                "June. The update adds scheduled PDF reports, a dark theme, and a query "
                "builder that supports cross-source joins. Page load times for large "
                "workspaces dropped by roughly 40 percent after the team moved chart rendering "
                "to web workers. The legacy export API is deprecated and will be removed in "
                "version 5.0, planned for early next year. Customers on the Starter plan get "
                "scheduled reports limited to five per workspace. A migration guide for the "
                "export API is available in the help center."
            ),
            rubric=(
                "Faithful summary naming the 4.2 release features, the performance "
                "improvement, and the export API deprecation; no invented facts."
            ),
            keywords=("4.2", "scheduled", "reports", "deprecated", "export"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_003",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "On 28 May, the Cobalt Freight shipment-tracking API returned errors for 47 "
                "minutes, affecting about 12 percent of requests. The incident began at 09:14 "
                "UTC when a configuration deploy doubled the connection pool size and "
                "exhausted database connections. On-call engineers rolled back the change at "
                "09:42 and error rates returned to normal by 10:01. No shipment data was lost, "
                "but 3,200 webhook deliveries were delayed and later replayed. The team is "
                "adding a connection-count alert and a staged rollout step for configuration "
                "changes to prevent recurrence."
            ),
            rubric=(
                "Faithful summary covering the outage duration and cause, the rollback, and "
                "the follow-up actions; states no data was lost without adding new claims."
            ),
            keywords=("outage", "rollback", "connections", "webhooks", "configuration"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_004",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "We report a fluorinated ester electrolyte additive that improves the cycle "
                "life of lithium-ion cells operating at low temperature. In pouch cells cycled "
                "at minus 20 degrees Celsius, cells with 2 percent additive retained 91 "
                "percent of initial capacity after 500 cycles, compared with 64 percent for "
                "the control. Impedance spectroscopy suggests the additive forms a thinner, "
                "more conductive interphase layer on the graphite anode. The additive is "
                "inexpensive to synthesize and compatible with existing manufacturing lines. "
                "Limitations include reduced performance above 45 degrees Celsius and untested "
                "behavior at high charge rates."
            ),
            rubric=(
                "Faithful summary of the additive's capacity retention result, the proposed "
                "mechanism, and the stated limitations; numbers preserved accurately."
            ),
            keywords=("electrolyte", "additive", "capacity", "cycles", "anode"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_005",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Lakeshore Robotics, a warehouse automation startup based in Duluth, "
                "announced a 38 million dollar Series B round on 4 June, led by Harbor Crest "
                "Ventures. The company says its picking robots are deployed in eleven "
                "distribution centers and handled 40 million items in the past year. The new "
                "funding will go toward a second manufacturing line and a software team in "
                "Toronto. Lakeshore reported that revenue tripled year over year, although the "
                "company remains unprofitable. Two pilot programs with grocery chains are "
                "planned for the fall, with results expected before the next fundraise."
            ),
            rubric=(
                "Faithful summary covering the funding amount and lead investor, the "
                "deployment scale, and the planned use of funds; no invented details."
            ),
            keywords=("funding", "series", "warehouse", "robots", "toronto"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_006",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Hiring committee notes, 21 May. The backend engineer search has three "
                "finalists after onsite interviews. Rosa scored highest on system design but "
                "has no Go experience; the panel agreed the gap is teachable. The committee "
                "voted four to one to extend an offer to Rosa at the L4 band, pending "
                "reference checks that Tomas will complete by Friday. If Rosa declines, the "
                "team will reopen the requisition rather than make an offer to the other "
                "finalists. The data platform role remains paused until the Q3 budget review, "
                "and recruiting will stop sourcing for it this week."
            ),
            rubric=(
                "Faithful summary covering the decision to offer Rosa pending references, the "
                "fallback plan, and the paused data platform role."
            ),
            keywords=("offer", "rosa", "references", "requisition", "paused"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_007",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "FieldNotes 3.8 for iOS and Android introduces full offline mode, the most "
                "requested feature in last year's customer survey. Notes, checklists, and "
                "photo attachments now sync automatically when a connection returns, with "
                "conflicts resolved by a last-writer-wins rule and a recoverable conflict "
                "history. The update also reduces app size by 22 percent and adds Spanish and "
                "Portuguese localization. Offline mode is available on the Pro plan only; Free "
                "plan users can read but not edit while disconnected. The rollout begins 15 "
                "June and reaches all users within two weeks."
            ),
            rubric=(
                "Faithful summary covering offline mode and sync behavior, the Pro-only "
                "restriction, and the rollout timing; no invented features."
            ),
            keywords=("offline", "sync", "conflicts", "pro", "rollout"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_008",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Internal incident report, 9 May. A planned failover test of the orders "
                "database promoted the standby correctly, but application servers kept "
                "connecting to the old primary for 18 minutes because the connection pooler "
                "cached stale DNS entries. Roughly 5,400 write requests failed during the "
                "window and customers saw checkout errors. The pager fired within two minutes, "
                "and engineers fixed the issue by restarting the pooler fleet. Follow-up "
                "actions: lower the pooler DNS cache to 30 seconds, add a runbook step to "
                "verify writer identity after failover, and repeat the test within one month."
            ),
            rubric=(
                "Faithful summary of the stale-DNS cause, the customer impact, and the "
                "follow-up actions; does not overstate or invent impact."
            ),
            keywords=("failover", "dns", "pooler", "checkout", "runbook"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_009",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "This study examined whether a 90-minute afternoon nap improves recall of "
                "word pairs learned that morning. One hundred twenty-eight adults aged 18 to "
                "35 were randomly assigned to a nap or quiet-rest condition. The nap group "
                "recalled 23 percent more word pairs at evening testing, and a smaller "
                "advantage persisted at a one-week follow-up. Polysomnography showed that "
                "recall gains correlated with time spent in slow-wave sleep rather than total "
                "nap duration. The authors note the sample skewed toward university students "
                "and that effects on procedural memory tasks were not measured."
            ),
            rubric=(
                "Faithful summary covering the recall advantage, the slow-wave sleep "
                "correlation, and the stated limitations; preserves the numbers."
            ),
            keywords=("nap", "recall", "slow-wave", "memory", "follow-up"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_010",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Helio Foods opened its first commercial production plant in Reno on 30 May, "
                "a 140,000 square foot facility that will make the company's chickpea-based "
                "protein powder. The plant can produce 18,000 tons per year and is expected to "
                "employ about 220 people at full capacity. Helio signed supply agreements with "
                "two national grocery chains and says the plant will cut its unit production "
                "cost by roughly a third compared with contract manufacturing. The company "
                "delayed a planned second facility in Georgia, citing equipment lead times, "
                "and now expects that site to open in 2028."
            ),
            rubric=(
                "Faithful summary covering the plant opening and capacity, the grocery "
                "agreements or cost reduction, and the delayed Georgia facility."
            ),
            keywords=("reno", "plant", "chickpea", "protein", "georgia"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_011",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Security review notes, 5 June. The quarterly review covered the audit of "
                "third-party browser extensions and the pending SOC 2 renewal. Two extensions "
                "used by the support team requested overly broad permissions and will be "
                "removed by 12 June. The auditors flagged that database backup encryption keys "
                "had not been rotated in 19 months; the infrastructure team committed to "
                "rotation by end of month and an automated 90-day schedule afterward. The "
                "phishing-simulation click rate fell from 11 to 6 percent after training. The "
                "next review will focus on vendor access to the production VPN."
            ),
            rubric=(
                "Faithful summary covering the extension removals, the key rotation finding "
                "and commitment, and the phishing improvement."
            ),
            keywords=("extensions", "keys", "rotation", "phishing", "soc"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_012",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Starting 1 August, the Driftway public API moves from a flat limit of 600 "
                "requests per minute to tiered limits: 300 for Free, 1,200 for Team, and 5,000 "
                "for Enterprise keys. Responses will include rate-limit headers and a "
                "Retry-After value on 429 errors. Webhook deliveries and OAuth token refreshes "
                "are exempt from the new limits. According to the company, fewer than 4 "
                "percent of integrations currently exceed their new tier. Customers can "
                "request a temporary limit increase through the dashboard, and SDK versions "
                "released after June handle retries with exponential backoff automatically."
            ),
            rubric=(
                "Faithful summary covering the tiered limits and effective date, the "
                "exemptions or headers, and the low share of affected integrations."
            ),
            keywords=("rate", "limits", "tiers", "429", "backoff"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_013",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Incident summary: on 17 April the TLS certificate for the Meridian Pay API "
                "expired at 00:00 UTC, causing all API requests to fail the handshake for 71 "
                "minutes. The automated renewal job had been silently failing since March "
                "because a DNS provider API token was revoked during an unrelated cleanup. "
                "Monitoring alerted on elevated client errors but not on certificate age. "
                "Engineers issued a new certificate manually at 01:11 UTC. Remediations "
                "include a certificate-expiry alert at 21 days, a weekly check of the renewal "
                "job's logs, and moving token management into the secrets manager."
            ),
            rubric=(
                "Faithful summary covering the expired certificate and outage length, the "
                "revoked-token root cause, and the remediation steps."
            ),
            keywords=("certificate", "expired", "renewal", "token", "alert"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_014",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "We measured the cooling effect of reflective roof coatings across 42 "
                "commercial buildings in a mid-sized desert city over two summers. Coated "
                "roofs lowered peak indoor temperatures by an average of 2.8 degrees Celsius "
                "and reduced air-conditioning energy use by 11 percent compared with matched "
                "control buildings. Benefits were largest for single-story buildings with "
                "limited insulation. Surface reflectivity declined about 9 percent per year "
                "due to dust accumulation, suggesting periodic cleaning is needed to sustain "
                "savings. We estimate a payback period of four years at current energy prices, "
                "excluding maintenance costs."
            ),
            rubric=(
                "Faithful summary covering the temperature and energy reductions, the "
                "reflectivity decline caveat, and the payback estimate."
            ),
            keywords=("reflective", "roofs", "cooling", "energy", "payback"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_015",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Verdant Energy connected the final turbine of its Sable Ridge wind farm to "
                "the grid on 26 May, completing a 312 megawatt project two months ahead of "
                "schedule. The 52-turbine site in eastern Wyoming is expected to generate "
                "enough electricity for about 95,000 homes. Verdant signed a fifteen-year "
                "power purchase agreement with a regional utility covering 70 percent of "
                "output, with the remainder sold on the wholesale market. Construction "
                "employed roughly 400 workers; the operating site will retain 28 permanent "
                "staff. The company's next project, a solar and storage site in Nevada, begins "
                "permitting this summer."
            ),
            rubric=(
                "Faithful summary covering the completed wind farm and its capacity, the power "
                "purchase agreement, and the next project; figures preserved."
            ),
            keywords=("wind", "turbine", "megawatt", "wyoming", "agreement"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_016",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Customer advisory board notes, 19 March. Eight customers attended; the main "
                "topic was the new permissions model. Three enterprise customers said role "
                "inheritance is confusing and asked for an effective-permissions viewer before "
                "they will roll it out. Two customers requested SCIM provisioning, which is "
                "already on the Q3 roadmap. Attendees ranked audit log export as the highest "
                "priority gap, ahead of the mobile app. The product team committed to shipping "
                "the permissions viewer by the end of Q2 and to monthly office hours. The next "
                "board session is scheduled for 11 September."
            ),
            rubric=(
                "Faithful summary covering the permissions viewer request and commitment, the "
                "audit log export priority, and the SCIM/roadmap point."
            ),
            keywords=("permissions", "viewer", "audit", "scim", "roadmap"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_017",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Penbright announced pricing changes effective 1 October. The Starter plan "
                "price stays at 12 dollars per seat, but the included document limit drops "
                "from unlimited to 500 active documents. A new Growth tier at 22 dollars per "
                "seat adds workflow automation and priority support. Existing annual "
                "subscribers keep current terms until renewal, and anyone who subscribed "
                "before 2024 keeps unlimited documents permanently as a loyalty benefit. The "
                "Business tier gains SSO at no extra cost. The company says the changes affect "
                "about 7 percent of current customers and expects most to stay on Starter."
            ),
            rubric=(
                "Faithful summary covering the Starter document limit change, the new Growth "
                "tier, and the grandfathering terms; prices kept accurate."
            ),
            keywords=("pricing", "starter", "growth", "documents", "tier"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_018",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Incident report: between 2 and 4 June, a retry bug in the Lumera checkout "
                "service charged 1,841 customers twice for the same order. The bug was "
                "introduced when a timeout was lowered without making the charge call "
                "idempotent, so client retries created duplicate charges. Support tickets "
                "surfaced the pattern on the morning of 4 June and the deploy was reverted "
                "within an hour. All duplicate charges were refunded automatically by 6 June, "
                "and affected customers received a 10 dollar credit. The fix adds idempotency "
                "keys to every payment request and a daily duplicate-charge reconciliation "
                "report."
            ),
            rubric=(
                "Faithful summary covering the double-charge cause and scale, the refunds and "
                "credit, and the idempotency fix; numbers preserved."
            ),
            keywords=("duplicate", "charges", "refunded", "idempotency", "retry"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_019",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "We present a post-training quantization method that compresses transformer "
                "language models to 3-bit weights while preserving accuracy. The method groups "
                "weights by output channel and learns per-group scale offsets from 512 "
                "calibration examples, requiring no retraining. On a 7-billion-parameter "
                "model, our approach loses 0.4 points of average benchmark accuracy versus 2.1 "
                "points for the strongest prior 3-bit baseline, while reducing memory use by "
                "78 percent compared with 16-bit weights. Inference throughput on a single "
                "consumer GPU improves 2.3 times. Code and quantized checkpoints are released "
                "under an open license."
            ),
            rubric=(
                "Faithful summary covering the 3-bit quantization approach, the accuracy and "
                "memory results, and the open release; no invented claims."
            ),
            keywords=("quantization", "3-bit", "calibration", "accuracy", "memory"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_020",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "The city of Marquette and startup Atlas Transit launched a six-month "
                "on-demand shuttle pilot on 1 June, replacing two low-ridership bus routes on "
                "the north side. Riders book trips through an app or by phone, and software "
                "pools passengers heading the same direction. Fares match the regular bus at 2 "
                "dollars, with free transfers to fixed routes. The pilot runs eight vans on "
                "weekdays from 6 a.m. to 9 p.m. The city council will evaluate the program "
                "against targets of 250 daily riders and an average wait under 12 minutes "
                "before deciding on a permanent contract."
            ),
            rubric=(
                "Faithful summary covering the pilot's scope and booking model, the fare "
                "parity, and the evaluation targets."
            ),
            keywords=("shuttle", "pilot", "on-demand", "fares", "riders"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_021",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Notes from the engineering operations meeting, 27 May. The team reviewed "
                "four incidents from the past month and found that three shared a root cause "
                "category: untested configuration changes. Going forward, every configuration "
                "change to production must go through the staged rollout pipeline, with no "
                "direct edits permitted. The on-call rotation expands from six to nine "
                "engineers in July to reduce pager load, and secondary on-call becomes a paid "
                "role. Postmortems must now be filed within five business days, and action "
                "items get tracked in the weekly sync until closed."
            ),
            rubric=(
                "Faithful summary covering the configuration-change root cause and new rollout "
                "rule, the on-call expansion, and the postmortem deadline."
            ),
            keywords=("configuration", "rollout", "on-call", "postmortems", "incidents"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_022",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Quarry 2.5 ships a rebuilt search ranking system. Queries are now matched "
                "with a hybrid of keyword and semantic scoring, and results from the past 30 "
                "days get a recency boost. In offline evaluation, the new ranker improved "
                "click-through on the first result by 18 percent, and time-to-first-click "
                "fell by about a second in a beta cohort of 2,000 workspaces. Administrators "
                "can disable semantic matching per workspace for compliance reasons. Indexing "
                "latency is unchanged. The rollout completes by 30 June, and the old ranker "
                "will be removed in the following release."
            ),
            rubric=(
                "Faithful summary covering the hybrid ranking change, the measured "
                "improvements, and the admin opt-out or rollout timing."
            ),
            keywords=("search", "ranking", "semantic", "click-through", "rollout"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_023",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Internal report, 13 May: the metrics cluster came within four hours of "
                "filling its disks after a tenant enabled debug-level logging and ingest "
                "doubled overnight. An engineer noticed the disk usage trend during an "
                "unrelated investigation at 07:40 and applied an emergency retention cut from "
                "30 to 21 days, freeing 18 percent capacity. No data was lost and no customers "
                "were affected. The report notes that the existing disk alert fires at 90 "
                "percent, which would have left under two hours to respond. Actions: alert on "
                "usage trend, per-tenant ingest quotas, and a quarterly capacity review."
            ),
            rubric=(
                "Faithful summary covering the near-miss cause, the emergency retention cut, "
                "and the planned alerting and quota actions."
            ),
            keywords=("disk", "ingest", "retention", "alert", "quotas"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_024",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "This three-year field trial measured soil carbon changes across 24 paired "
                "plots on commercial grain farms practicing cover cropping versus winter "
                "fallow. Cover-cropped plots gained an average of 0.41 tons of carbon per "
                "hectare per year in the top 30 centimeters, with gains concentrated in the "
                "first two years. Yield in the cash crop was unchanged in 19 of 24 pairs and "
                "modestly lower in dry years. Measurement variability was high, and the "
                "authors caution that detecting change on a single farm requires dense "
                "sampling. Results support including cover cropping in carbon market protocols "
                "with conservative crediting."
            ),
            rubric=(
                "Faithful summary covering the measured carbon gain, the yield findings, and "
                "the variability caveat; figures preserved."
            ),
            keywords=("cover", "cropping", "carbon", "hectare", "yield"),
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_sum_025",
            category="summarization",
            prompt=(
                _SUM_PREFIX
                + "Quill Press, the subscription publisher of serialized fiction, reported "
                "passing 400,000 paying subscribers in May, up from 260,000 a year earlier. "
                "Growth was driven by its translated catalog, which now accounts for 31 "
                "percent of reading time, and a referral program launched in January. Average "
                "revenue per subscriber held steady at 6.40 dollars per month. The company "
                "signed 85 new authors in the quarter and raised its standard royalty from 60 "
                "to 65 percent. Quill plans an audio offering this winter and says it remains "
                "profitable, with margins reinvested in author advances."
            ),
            rubric=(
                "Faithful summary covering the subscriber growth, the translated catalog and "
                "referral drivers, and the royalty increase or audio plans."
            ),
            keywords=("subscribers", "translated", "referral", "royalty", "audio"),
            expected_route_type="local",
        ),
    ]


def _private_cases() -> list[QualityCase]:
    return [
        QualityCase(
            case_id="q_priv_001",
            category="private",
            prompt=(
                "My doctor's portal asks me to describe my symptoms before the visit. Rewrite "
                "this so it is clear and concise: 'I have had throbbing headaches on the left "
                "side maybe four times a week for two months, with nausea and light "
                "sensitivity, and ibuprofen barely helps anymore.'"
            ),
            rubric=(
                "Rewrites the symptom description clearly and concisely, preserving all stated "
                "symptoms, frequency, and duration without adding medical claims."
            ),
            keywords=("headaches", "nausea", "sensitivity", "ibuprofen"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_002",
            category="private",
            prompt=(
                "I was just diagnosed with type 2 diabetes and my A1C came back at 8.1. Draft "
                "a simple one-week dinner plan I can bring to my dietitian appointment, "
                "focused on low added sugar and moderate carbohydrates."
            ),
            rubric=(
                "Produces a concrete seven-dinner plan consistent with low added sugar and "
                "moderate carbs, and frames it as a draft to review with the dietitian."
            ),
            keywords=("dinner", "carbohydrates", "sugar", "dietitian", "plan"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_003",
            category="private",
            prompt=(
                "Here is my journal entry from last night: 'Could not sleep again, kept "
                "replaying the argument with my sister, heart racing, felt like I was failing "
                "everyone.' Summarize the feelings in two sentences I can read aloud to my "
                "therapist."
            ),
            rubric=(
                "Produces a two-sentence first-person summary that faithfully reflects the "
                "insomnia, rumination, physical anxiety, and feelings of inadequacy."
            ),
            keywords=("sleep", "argument", "anxiety", "therapist"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_004",
            category="private",
            prompt=(
                "Draft a short message to my psychiatrist: I have been on 50 mg of sertraline "
                "for six weeks, my mood is better but I am sleeping badly and grinding my "
                "teeth, and I want to discuss adjusting the dose at my next appointment."
            ),
            rubric=(
                "Drafts a polite, specific message including the medication, dose, duration, "
                "improvements, side effects, and the request to discuss adjustment."
            ),
            keywords=("sertraline", "sleep", "dose", "appointment"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_005",
            category="private",
            prompt=(
                "I currently earn 96,000 dollars and was offered 104,000 to stay after "
                "receiving a 118,000 offer elsewhere. Draft a polite email to my manager, "
                "Asha, asking whether the company can match the outside offer, without it "
                "reading like an ultimatum."
            ),
            rubric=(
                "Drafts a professional, non-threatening email that states the situation and "
                "the ask clearly while expressing a preference to stay."
            ),
            keywords=("offer", "salary", "match", "stay"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_006",
            category="private",
            prompt=(
                "I have 14,300 dollars of credit card debt across three cards at 19 to 27 "
                "percent APR and take home 4,100 dollars a month. Make a simple payoff plan "
                "that compares the avalanche and snowball approaches for my situation."
            ),
            rubric=(
                "Lays out both payoff strategies with the stated numbers, explains the "
                "interest-versus-motivation tradeoff, and suggests a workable monthly amount."
            ),
            keywords=("debt", "apr", "avalanche", "snowball", "payoff"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_007",
            category="private",
            prompt=(
                "Summarize these notes from my call with my divorce lawyer into a short list "
                "of next steps: gather two years of bank statements, do not move out before "
                "the custody evaluation, mediation is scheduled for 12 August, and keep all "
                "messages with my ex civil and in writing."
            ),
            rubric=(
                "Produces a clear action list faithfully reflecting all four points without "
                "adding legal advice beyond the notes."
            ),
            keywords=("custody", "mediation", "statements", "lawyer"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_008",
            category="private",
            prompt=(
                "My landlord kept my 1,800 dollar security deposit claiming carpet damage that "
                "was already there when I moved in, and I have dated photos proving it. Draft "
                "a firm but professional demand letter noting my documentation, without making "
                "threats I cannot back up."
            ),
            rubric=(
                "Drafts a professional demand letter stating the amount, the dispute, and the "
                "photographic evidence, requesting return of the deposit with a deadline."
            ),
            keywords=("deposit", "landlord", "photos", "letter", "deadline"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_009",
            category="private",
            prompt=(
                "Marco and I have argued about money every week since I lost my job, and I "
                "shut down whenever he brings up the budget. Help me draft a calm message "
                "asking him to set up a weekly 30-minute money conversation with some ground "
                "rules."
            ),
            rubric=(
                "Drafts a non-blaming message proposing the weekly conversation with concrete "
                "ground rules, acknowledging the sender's own pattern of shutting down."
            ),
            keywords=("budget", "weekly", "rules", "conversation"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_010",
            category="private",
            prompt=(
                "Rewrite this complaint so it is factual and professional before I send it to "
                "HR: 'My manager keeps commenting on my accent in meetings and joked about my "
                "green card status twice in front of the team. It is humiliating and I want it "
                "to stop.'"
            ),
            rubric=(
                "Rewrites the complaint in a factual, professional tone, preserving the "
                "specific behaviors and the requested outcome without exaggeration."
            ),
            keywords=("manager", "accent", "hr", "professional", "incidents"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_011",
            category="private",
            prompt=(
                "My biopsy result says 'benign fibroadenoma, no atypia, routine follow-up in "
                "12 months.' Write two sentences in plain language I can send to my family "
                "group chat so they stop worrying."
            ),
            rubric=(
                "Conveys in plain language that the result is benign and only routine "
                "follow-up is needed, reassuring without adding medical claims."
            ),
            keywords=("benign", "biopsy", "follow-up", "routine"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_012",
            category="private",
            prompt=(
                "My wallet and passport (number K8123456) were stolen while I was traveling. "
                "Make me a checklist for this week: freeze my credit, replace my driver's "
                "license, report and replace the passport, and watch for signs of identity "
                "theft."
            ),
            rubric=(
                "Produces an organized weekly checklist covering all four areas with sensible "
                "ordering and brief notes on how to do each step."
            ),
            keywords=("passport", "credit", "freeze", "identity", "checklist"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_013",
            category="private",
            prompt=(
                "We are starting our second IVF cycle in September after a failed transfer in "
                "March. Turn the clinic's schedule into a simple calendar plan: ten days of "
                "stimulation injections, monitoring visits every other day, and retrieval "
                "expected around day 12."
            ),
            rubric=(
                "Lays out a day-by-day or week-view plan matching the stated schedule, "
                "phrased supportively and without inventing medical specifics."
            ),
            keywords=("ivf", "injections", "monitoring", "retrieval", "calendar"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_014",
            category="private",
            prompt=(
                "I have had two panic attacks at work this month and had to leave a client "
                "meeting. Write a short script for telling my manager that I need occasional "
                "short breaks, without disclosing my full diagnosis."
            ),
            rubric=(
                "Provides a brief, professional script requesting short breaks that preserves "
                "privacy and avoids disclosing diagnostic details."
            ),
            keywords=("breaks", "manager", "script", "privacy"),
            expected_local_only=True,
            expected_route_type="local",
        ),
        QualityCase(
            case_id="q_priv_015",
            category="private",
            prompt=(
                "My mother left me 210,000 dollars in savings and left my brother the house, "
                "and he feels the split was unfair. Draft a warm letter to him proposing that "
                "we each get an independent appraisal of the house and talk it through before "
                "anyone involves lawyers."
            ),
            rubric=(
                "Drafts an empathetic letter acknowledging the brother's feelings and clearly "
                "proposing the appraisal-and-conversation step before legal escalation."
            ),
            keywords=("inheritance", "appraisal", "brother", "letter"),
            expected_local_only=True,
            expected_route_type="local",
        ),
    ]


def _grounding_cases() -> list[QualityCase]:
    return [
        QualityCase(
            case_id="q_ground_001",
            category="grounding",
            prompt="What time is it in Tokyo right now?",
            rubric=(
                "States the current time only if trusted grounding supplies it; otherwise "
                "says it cannot know the live time and offers the JST/UTC offset instead of "
                "guessing."
            ),
            keywords=("tokyo", "time", "jst", "utc"),
        ),
        QualityCase(
            case_id="q_ground_002",
            category="grounding",
            prompt="What is today's date?",
            rubric=(
                "Gives the date from trusted grounding context if present; otherwise states "
                "plainly that it cannot know the current date rather than inventing one."
            ),
            keywords=("date", "today", "current"),
        ),
        QualityCase(
            case_id="q_ground_003",
            category="grounding",
            prompt="What will the weather be like in Berlin this afternoon?",
            rubric=(
                "Provides a forecast only if live data is available; otherwise says it has no "
                "access to current weather and suggests checking a forecast service, with no "
                "fabricated temperatures."
            ),
            keywords=("berlin", "weather", "forecast", "temperature"),
        ),
        QualityCase(
            case_id="q_ground_004",
            category="grounding",
            prompt="What is Nvidia's stock price right now?",
            rubric=(
                "Quotes a price only from trusted live data; otherwise states it cannot access "
                "real-time market data and avoids stating a specific stale number as current."
            ),
            keywords=("nvidia", "stock", "price", "market"),
        ),
        QualityCase(
            case_id="q_ground_005",
            category="grounding",
            prompt="What is the most important AI news from this week?",
            rubric=(
                "Reports items only if grounded in live sources; otherwise explains its "
                "knowledge has a cutoff and does not present old news as this week's."
            ),
            keywords=("ai", "news", "week", "recent"),
        ),
        QualityCase(
            case_id="q_ground_006",
            category="grounding",
            prompt="How many days are left until next Monday?",
            rubric=(
                "Computes the answer only if the current date is known from grounding; "
                "otherwise asks for or notes the missing current date instead of guessing."
            ),
            keywords=("monday", "days", "today", "date"),
        ),
        QualityCase(
            case_id="q_ground_007",
            category="grounding",
            prompt="What time does the sun set in Oslo today?",
            rubric=(
                "Gives a sunset time only with grounded date and location data; otherwise "
                "explains the dependency on today's date and avoids a falsely precise time."
            ),
            keywords=("oslo", "sunset", "time", "today"),
        ),
        QualityCase(
            case_id="q_ground_008",
            category="grounding",
            prompt="What is the current USD to EUR exchange rate?",
            rubric=(
                "Provides a rate only from live data; otherwise states it cannot fetch current "
                "rates and may give an approximate historical range clearly labeled as such."
            ),
            keywords=("usd", "eur", "exchange", "rate"),
        ),
        QualityCase(
            case_id="q_ground_009",
            category="grounding",
            prompt="Has any major open-weight language model been released this month?",
            rubric=(
                "Answers from live sources if available; otherwise states its knowledge cutoff "
                "and does not assert releases it cannot verify happened this month."
            ),
            keywords=("model", "release", "open", "month"),
        ),
        QualityCase(
            case_id="q_ground_010",
            category="grounding",
            prompt="What day of the week is it today?",
            rubric=(
                "Answers from grounded date context if present; otherwise admits it does not "
                "know the current day rather than picking one."
            ),
            keywords=("day", "week", "today"),
        ),
    ]


def quality_cases() -> list[QualityCase]:
    """Return all 100 benchmark cases in a stable, deterministic order."""
    return [
        *_coding_cases(),
        *_reasoning_cases(),
        *_summarization_cases(),
        *_private_cases(),
        *_grounding_cases(),
    ]


def cases_by_category() -> dict[str, list[QualityCase]]:
    grouped: dict[str, list[QualityCase]] = {}
    for case in quality_cases():
        grouped.setdefault(case.category, []).append(case)
    return grouped
