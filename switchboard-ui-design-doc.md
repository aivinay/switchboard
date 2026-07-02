# Switchboard UI v2 — design doc (rev 2, post TA+PM review)

*July 2, 2026 · targets the post-0.3.0 codebase · locked decisions: sidebar = sessions + search; savings hero = premium-quota framing; feedback = record + visible loop (no NEW auto-retrain); upgrade = commands + opt-out daily check. Rev 2 incorporates the technical-architect and product-manager review findings; the decision log is at the end.*

## 1. Goals and non-goals

Fix the four tested defects (broken savings panel, feedback toggle behavior, dead "Lock Private" affordance, no upgrade path), add the app-shell users expect (sidebar: sessions + search), and do it without weakening the privacy invariant, adding dependencies, or exceeding what a solo maintainer can review — delivery is **three independently mergeable branches**, not one mega-branch.

Non-goals this round: pinned chats, projects UI, streaming, theming, multi-user, retrain-from-UI button, FTS5.

## 2. What's broken today — corrected root causes

**Savings panel**: the dashboard is injected as a fourth child into a three-row CSS grid (`.shell`, `styles.css:33–34`), so it collides with the message list — a layout-slot bug, not absolute positioning. The 7-day bars *are* normalized (`app.js:469–474`) but render unlabeled, unframed, and mid-collision, so they read as garbage. Fix is structural (a proper drawer outside the grid) plus labeling, not a normalization patch.

**Feedback** (`makeFeedbackControls`, `app.js:282`): no retraction (re-click re-sends), 👎 sets no pressed state and its follow-up strip can't be dismissed, "wrong model →" offers the backend that just answered, history doesn't return stored ratings so reloads forget state. Server reality that the redesign must respect: `add_feedback` is **append-only** and denormalizes onto the telemetry record; a thumbs-down *also* snapshots prompt/context/response into `FeedbackExampleRecord` (only when `store_feedback_examples: true`; **default false**), which feeds `unprocessed_wrong_model_examples()` and — when the threshold (5) is reached — an **existing auto-retrain path** (`maybe_trigger_retraining`, `ui.py`). The wire vocabulary already includes `good`, `bad`, `too-weak`, `wrong-route`.

**Lock Private** (`index.html:69`): a green span shown when private mode is on; looks like a button, does nothing, label reads as an action.

**Upgrades**: no `switchboard version` / `switchboard upgrade`. Note `switchboard/__init__.py` wraps `importlib.metadata.version` with a stale hardcoded fallback (`"0.2.3"`) that must be fixed alongside this work.

## 3. App shell — sidebar with sessions and search

Two-column shell; left sidebar 264px, collapsible to an icon rail (hamburger in topbar; state in localStorage); below ~760px it's an overlay drawer.

```
┌────────────┬──────────────────────────────────────────────┐
│ + New chat │  [☰]  [Auto ⌄]   Switchboard   [🛡] [Savings]│
│ ┌────────┐ │                                              │
│ │🔍 Search│ │   …messages…                                 │
│ └────────┘ │                                              │
│ TODAY      │                                              │
│ 🔒● Login p…│                                             │
│  ○ SSN red…│                                              │
│ YESTERDAY  │   ┌────────────────────────────────────┐     │
│  ◐ Quota q…│   │ Ask Switchboard…                    │     │
│            │   │           [🔒 Private chat] [↑]    │     │
│ ────────── │   └────────────────────────────────────┘     │
│ v0.3.0  ⬆ │                                              │
└────────────┴──────────────────────────────────────────────┘
```

**Which sessions appear (inclusion rule).** Add a nullable `origin` column to the session record, set by entry point (`"ui"` / `"cli"`). The sidebar lists `origin="ui"` sessions plus any session with a user-set title; bare CLI one-shots (every `switchboard ask` without `--session` creates a session) stay out of the sidebar but remain reachable via search, which spans everything. This prevents heavy CLI use from flooding the UI while keeping the cross-surface story intact.

**Session rows as a routing ledger (identity).** Each row carries Switchboard-specific metadata at a glance: a small green lock when the session is private-flagged (§6), and a backend dot — ● green local-only, ● violet premium-touched, ◐ mixed — computed in the list query from the messages' recorded backends. This is the cheap addition that makes the shell distinctly Switchboard instead of generic chat furniture.

**List mechanics.** `GET /api/sessions?limit=100&before=<updated_at cursor>`: id, title (stored title, else first user message truncated ~40 chars), message_count, updated_at, private, backend_summary — produced by **one** GROUP-BY/join query (no N+1). Sidebar groups by Today/Yesterday/date; "Load more" appears at the limit. The session `title` column already exists (`models/sessions.py:14`) — rename needs no migration; the new columns are exactly three: `private` (Track 2), `origin` and `deleted_at` (Track 3), each registered in the existing migration hook (`_migrate_sqlite`/`_add_missing_columns`, `db.py:19–70` — the same path that added `preferred_model`), since `create_all` won't upgrade existing databases.

**Search.** Instant client-side title filter; 300ms-debounced `GET /api/sessions/search?q=` matching message content + titles via SQLite `LIKE` with `%`/`_`/`\` **escaped and an `ESCAPE` clause** (a literal "100%" query must not match everything). Returns session id/title/~80-char snippet around the first match. Empty or 1-char `q` returns nothing.

**Rendering safety (hard rule).** Titles and snippets are user-controlled text: titles render via `textContent` only; snippets are escaped first (reuse `escapeHtml`, `app.js:37`), then the match is wrapped in `<mark>` by splitting on the escaped match — never raw `innerHTML` from stored strings. Acceptance test: a session titled `<img src=x onerror=alert(1)>` renders inert everywhere (list, header, search results).

**Delete = soft-delete with undo.** Hover ⋯ → Rename (inline input) / Delete. Delete tombstones the session (`deleted_at`), shows a ~10s "Session deleted — Undo" toast, and purges on expiry/next boot; **purge cascades to the session's `ChatMessageRecord`s, and search joins only non-tombstoned sessions** — deleted content must be unreachable from every surface, not just the list. For a local-first tool this chat is the user's only copy; hard delete behind a two-click confirm is not acceptable.

**Empty states.** No sessions: "Chats you start here and in the CLI (when titled) appear here." No search results: "No matches for 'q' — search covers all sessions, including CLI ones." First-run chat pane: a small welcome panel with three clickable demo prompts — one that routes local ("Summarize this paragraph…"), one that routes premium ("Design a caching strategy for…"), one that trips the privacy floor ("My SSN is 123-45-6789 — draft a letter…") — so the first minute of use visibly demonstrates routing chips, the premium/local split, and the privacy floor. Onboarding and differentiation in one panel.

**Footer**: version (from `GET /api/version`) + update pill when a newer release is cached (§7); pill opens a popover with the upgrade command + copy button.

## 4. Savings — a proper drawer, quota-framed, self-explanatory

Right-side slide-over drawer (overlay dims chat; ✕ / Esc / click-outside close), toggled by the topbar "Savings" button; the drawer lives **outside** the `.shell` grid, which structurally kills the overlap bug.

1. **Hero: "Premium calls avoided"** — week count, large; subtitle "31 of 42 requests handled locally or by tools this week." Stacked bar local/tools/premium with a plain-words legend directly beneath: *"local = your machine · tools = live-data lookups · premium = your Codex/Claude subscriptions."* No dollar figures anywhere.
2. **Quota meters**: when soft budgets are set, labeled progress bars (Codex 5h, Claude 7d), green→amber→red. When budgets are unset, show a one-line teaser in the same slot — "Set soft budgets in personal.yaml to watch your Codex 5-hour / Claude weekly windows" — the arbitrage story is the differentiator; most users must at least learn it exists.
3. **Tokens saved**: two stat cards (compression / local routing), one-line explanation each.
4. **7-day trend**: normalized flexbox bars (scaled to week max, fixed height), day labels beneath, premium overlay segment, values via `title`.
5. **Feedback quality strip**: "You rated 14 answers — 12 good, 2 corrected · 3 corrections pending" (pending count per §5; this is the persistent home for the pending number).
6. **Empty state**: "No activity yet — ask something and Switchboard starts counting what it saves you."

All numbers trace to `/api/dashboard` + `/api/quota` fields; the dashboard payload gains rated-answer counts. Any UI-side math that invents values is deleted. Acceptance: correct at 360/768/1440px with zero-data, partial-data, and full-data fixtures.

## 5. Feedback — explicit states, reversible, honest about the loop

State machine per message (mutually exclusive by construction):

```
neutral ──👍──► good ──👍──► neutral(retracted)
   │👎
   ▼
down-open (👎 filled, popover: [Bad answer] [wrong model → …] [✕])
   │pick                                  │✕ / Esc / outside / 👎 again
   ▼                                      ▼
corrected (chip "wrong model → Codex ✓")  neutral(retracted)
```

**Semantics split (quality vs routing), using existing wire vocabulary.** "Bad answer" sends `rating: "bad"` — a quality signal, **not** counted as a router correction. "Wrong model → X" sends `rating: "wrong-route"` with the corrected backend — the router-training signal. The chips exclude the backend that answered. `too-weak` remains accepted server-side as legacy input but the UI stops emitting it.

**Retraction is a real delete, not a fourth rating.** `DELETE /api/chat/feedback/{request_id}` removes, in one transaction: that request's `FeedbackRecord` rows, the denormalized `feedback_rating` on the telemetry record, and any `FeedbackExampleRecord` snapshot for it. Re-rating is an upsert per request_id (no double-counting in `feedback_summary` or `preferred_model_from_feedback`). Acceptance test: a retracted `wrong-route` appears in neither `unprocessed_wrong_model_examples()` nor pending counts nor routing bias.

**Persistence.** `corrected_backend` is stored on `FeedbackRecord.preferred_model` (field exists) so history restores the full state — history responses include rating + corrected backend — independent of the `store_feedback_examples` preference.

**Pending count + honest ack copy.** `GET /api/feedback/pending` returns `unprocessed_wrong_model_count()` (the mechanism that already exists — not a nonexistent weights timestamp). The post-correction ack adapts to the actual config. Two notes that keep the copy truthful: the auto-retrain path already exists (`maybe_trigger_retraining`, threshold = `preferences.feedback_retrain_threshold`, default 5) but has no off-switch — this design adds `preferences.feedback_auto_retrain: true` as one (an off-switch only, not a new loop); and a `wrong-route` rating *immediately* biases routing via `preferred_model_from_feedback` regardless of `store_feedback_examples`, so the default-state copy must not imply the correction did nothing:

- `store_feedback_examples: false` (default): "Saved — this correction immediately nudges routing preferences. To also retrain the classifier from your corrections, enable `store_feedback_examples`." (The enable-nudge shows at most once per session, not on every consecutive correction.)
- examples on, `feedback_auto_retrain: true`: "Saved — N of {threshold} corrections until the router retrains automatically." (threshold read from config, never hardcoded)
- examples on, `feedback_auto_retrain: false`: "Saved — N corrections pending. Run `switchboard train-router` to apply." (shown once N ≥ 3; a single example is training noise), with a copy-command button.

**Upsert cleanup edge**: re-rating away from `wrong-route` (e.g., to `good`) runs the same example-store cleanup as retraction — a withdrawn correction must not linger in pending counts or retraining input.

**UX rationale** (maintainer's question): one 👎 entry point, then a clean split between "answer was bad" and "model choice was wrong," every state reversible, and the loop finally visible — including telling the truth when the loop is switched off. 👍 stops being a dead end: good ratings feed the drawer's quality strip.

## 6. Privacy affordances — one status, one control, clearly different

1. **Status — "Privacy floor: always on."** Topbar 🛡 glyph + text, deliberately de-buttoned (no pill, no border, muted color — visually a label, not a control). It IS clickable, but as a status that explains itself: a plain-language popover — "Prompts containing personal or secret content never leave this machine, even when premium backends are selected. This cannot be turned off." No jargon.
2. **Control — 🔒 "Private chat" composer toggle** (`aria-pressed`, tinted composer border when on). When on, every send forces `backend: "ollama"`; messages show the Lock chip. **The toggle wins over the model picker**: while on, the picker is visually locked with an annotation ("Private chat forces Local"). **Persistence is server-side**: a `private` flag on the session (set via the sessions PATCH endpoint), with localStorage only as cache — cleared storage or a second browser must not silently revert a private chat to Auto. **Turning it off asks once**: "Earlier messages in this chat may be included as context for premium backends from now on. Continue?" — the honest handling of cross-context, not a silent switch.
3. Forced-local requests with Ollama down fail with the existing honest error — never fall through to premium (this is already how forced decisions behave in `switchboard_core.py`; a regression test pins it).

## 7. Version and upgrade UX

`switchboard version` / `--version`: installed version via the fixed `switchboard.__version__` (remove the stale `"0.2.3"` fallback — fall back to reading `pyproject.toml` or "unknown (source checkout)"), plus "latest on PyPI: X — run: switchboard upgrade" when the cache knows a newer one. `switchboard upgrade [--check]`: detect install method — pipx → `pipx upgrade switchboard-local`; uv tool → `uv tool upgrade switchboard-local`; plain venv → `python -m pip install -U switchboard-local`; editable/git checkout or PEP 668 externally-managed → **print** the right manual command instead of executing. Surface subprocess exit codes; never claim success on failure.

**Daily check, mechanically honest.** No daemon-thread-and-exit (it dies before the response arrives). Instead: a synchronous check with a hard ~1s budget, run **only** in `switchboard version`, `switchboard upgrade --check`, and `switchboard ui` startup, and only when the cache at `<config home>/update-check.json` is >24h stale — the path resolved via the existing config-home logic (`config.py:24–27`, honoring `SWITCHBOARD_CONFIG_HOME`/`XDG_CONFIG_HOME`), never a hardcoded `~/.config` (cache writes tolerate read-only filesystems silently). Governed by `preferences.update_check_enabled: true` + `SWITCHBOARD_UPDATE_CHECK=off`; skipped in CI; failures cached as "checked."

**Disclosure where people look.** First check prints a one-time notice: "Switchboard checks PyPI once a day for new versions. Disable: SWITCHBOARD_UPDATE_CHECK=off." The README privacy section and `docs/privacy.md` both document it. For a tool whose pitch is "nothing leaves your machine," disclosure only in a docs file is a trust landmine.

`GET /api/version` → `{installed, latest, update_available}` for the sidebar pill.

## 8. Trust model and security (new)

The UI server binds loopback by default and is unauthenticated; the new session endpoints make full chat content listable and deletable over HTTP, so: when bound to a non-loopback host, print a prominent warning and require an explicit env opt-in (`SWITCHBOARD_ALLOW_REMOTE_MUTATIONS=1`) for DELETE/PATCH session endpoints and feedback DELETE. **The guard ships in the same branch as the first mutation endpoints (Track 2, which adds sessions PATCH and feedback DELETE); Track 3 extends it** — the design's own rule must not depend on a later branch landing. Document the trust model ("the UI port is your machine's trust boundary") in `docs/privacy.md`. XSS rules per §3; search input length-capped; all new endpoints covered by the existing metadata-only telemetry rule.

## 9. Frontend architecture (new)

`app.js` is ~763 lines of globals; this design roughly doubles the UI. The no-build-step constraint stays, but the code may split into a few plain scripts loaded in order (`state.js`, `overlays.js`, `sidebar.js`, `drawer.js`, `feedback.js`, `app.js`) sharing one namespaced global (`window.SB`). Two shared utilities are mandatory, introduced before any new surface: a single app-state object (current session, sidebar state, open-overlay stack) and a **dismissable-stack** utility — one document-level Esc/click-outside handler that closes the top-most overlay (popover → drawer → sidebar-drawer) in order. Per-component keydown handlers are how "Esc closes things in reverse order" becomes buggy; one stack is how it stays correct. localStorage keys namespaced `switchboard.*`.

## 10. API additions (summary)

| Endpoint | Purpose | Notes |
|---|---|---|
| `GET /api/sessions?limit&before` | Sidebar list | one aggregate query; includes private + backend_summary; origin filter |
| `GET /api/sessions/search?q=` | Content search | LIKE-escaped; snippets escaped + `<mark>` |
| `PATCH /api/sessions/{id}` | Rename; set/unset `private` | title column exists |
| `DELETE /api/sessions/{id}` | Soft-delete (tombstone + delayed purge) | undo window ~10s |
| `POST /api/chat/feedback` | Upsert rating per request | UI emits `good`/`bad`/`wrong-route` |
| `DELETE /api/chat/feedback/{request_id}` | Transactional retraction across all three stores | |
| `GET /api/feedback/pending` | `unprocessed_wrong_model_count()` | |
| `GET /api/dashboard` (ext.) | + rated-answer counts | |
| `GET /api/version` | installed/latest/update_available | |

## 11. Definition of done — one pass/fail check per tested pain point

1. **Savings**: drawer renders correctly at 360/768/1440px against zero/partial/full data fixtures; closed drawer occludes nothing (the grid-slot bug is structurally impossible).
2. **Feedback**: no reachable state renders both thumbs active; every rating is retractable and stays retracted after reload; retracted corrections are absent from pending counts, training examples, and routing bias; **re-rating `wrong-route` → `good` scrubs the correction identically to retraction**; the ack copy matches the actual `store_feedback_examples`/`feedback_auto_retrain` configuration with the threshold read from config.
3. **Lock Private**: zero dead affordances — the status explains itself on click; the toggle forces local, survives browser-storage loss (server-side flag), wins over the picker, and fails closed with Ollama down.
4. **Upgrade**: on a machine with 0.3.0 installed via pip/pipx/uv, `switchboard upgrade` alone reaches the newer version; `switchboard version` shows installed + latest; the first-run notice printed exactly once; `SWITCHBOARD_UPDATE_CHECK=off` provably suppresses the network call.
5. **Empty screen**: first-run shows the three demo prompts; sidebar lists/searches/renames/soft-deletes sessions with undo; a `<img onerror>` title is inert; CLI one-shots don't flood the list.

The final walkthrough in each branch's validation section is this checklist, verbatim.

## 12. Delivery plan — three independently mergeable branches

1. **`feat/upgrade-cli`** — §7 (version/upgrade/daily check + disclosure + `__version__` fix). Zero UI risk, fixes pain #4 outright, ships first.
2. **`feat/ui-defect-fixes`** — §5 feedback machine + §6 privacy affordances + §4 savings drawer, plus the §9 utilities they need **and the §8 non-loopback mutation guard** (this branch introduces the first mutation endpoints). Fixes pains #1–#3.
3. **`feat/ui-shell`** — §3 sidebar/search/first-run + §8 trust-model guards + footer pill. Fixes pain #5.

Each branch: own tests, own click-through against the relevant Definition-of-done items, maintainer reviews and merges before the next lands. If branch 3 stalls, branches 1–2 still shipped.

## 13. Out of scope

Pinned chats and Projects (hooks left: session `pinned` flag would sit beside `private`; per-message `project` metadata exists for a future grouping), streaming, auth/multi-user, retrain-from-UI button, FTS5.

---

## Appendix — review decision log (rev 1 → rev 2)

Accepted in full: TA-1 (transactional DELETE retraction — was the blocker), TA-2 (pending = `unprocessed_wrong_model_count()`; ack copy acknowledges the existing auto-retrain; `preferred_model` persistence), TA-3 (origin column + inclusion rule + single aggregate query; title column already exists), TA-4 (XSS rules + acceptance test), TA-5 (toggle wins over picker; honest toggle-off confirm), TA-6 (trust model + non-loopback guards), TA-7 (file split + dismissable stack + shared state), TA-8 (state-dir cache, `__version__` reuse/fix, synchronous bounded check at three call sites), TA-9 (§2 root causes corrected; `bad` added to vocabulary), TA-10 (limit 100 + cursor + load-more), TA-11 (LIKE escaping), PM-1 (Definition of done, §11), PM-2 (three-branch delivery, §12), PM-3 (first-run demo prompts + all empty states), PM-4 ("Privacy floor: always on" as self-explaining de-buttoned status), PM-5 (first-run notice + README disclosure), PM-6 (soft-delete + undo), PM-7 (server-side private flag), PM-8 (plain-words legend), PM-9 (≥3 gating + pending's home in the drawer strip), PM-10 (session-row lock + backend dots), PM-11 (resolved via the `bad` vs `wrong-route` split — "Bad answer" is no longer counted as a router correction), PM-12 (quota teaser).

Reconciliations: PM-9/PM-11/TA-2 intersect at the ack copy — resolved with config-adaptive copy that tells the truth in all three states (examples off / auto-retrain / manual). The locked "no auto-retrain" decision is preserved as "no NEW auto-retrain": the existing threshold path is disclosed, not expanded, and gains an off-switch (`feedback_auto_retrain`).

Round-2 minors, applied: TA-r2-1 (`feedback_auto_retrain` preference added — the third ack state's knob now exists), TA-r2-2 (upsert-away-from-`wrong-route` scrubs the example store; added to DoD-2), TA-r2-3 ≡ PM-r2-1 (mutation guard moved into Track 2), TA-r2-4 (three new columns, not two; migration hook `_migrate_sqlite`/`_add_missing_columns` named), TA-r2-5 (purge cascades to messages; search excludes tombstoned sessions), TA-r2-6 (threshold and config-home path read from config, never hardcoded), PM-r2-2 (default-state ack acknowledges the immediate routing-preference effect; enable-nudge shown at most once per session).
