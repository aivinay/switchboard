const SB = (window.SB = window.SB || {});
const appState = SB.state;
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const messages = document.querySelector("#messages");
const sidebar = document.querySelector("#sidebar");
const sidebarScrim = document.querySelector("#sidebar-scrim");
const sidebarToggle = document.querySelector("#sidebar-toggle");
const sidebarCollapse = document.querySelector("#sidebar-collapse");
const sidebarNewChat = document.querySelector("#sidebar-new-chat");
const sessionSearch = document.querySelector("#session-search");
const sessionList = document.querySelector("#session-list");
const sessionLoadMore = document.querySelector("#session-load-more");
const sessionToast = document.querySelector("#session-toast");
const versionPill = document.querySelector("#version-pill");
const uiVersion = document.querySelector("#ui-version");
const updateBadge = document.querySelector("#update-badge");
const versionPopover = document.querySelector("#version-popover");
const versionCopy = document.querySelector("#version-copy");
const copyUpgradeCommand = document.querySelector("#copy-upgrade-command");
const modelButton = document.querySelector("#model-picker-button");
const selectedModel = document.querySelector("#selected-model");
const modelMenu = document.querySelector("#model-menu");
const quotaMeters = document.querySelector("#quota-meters");
const dashboard = document.querySelector("#dashboard");
const dashboardToggle = document.querySelector("#dashboard-toggle");
const dashboardClose = document.querySelector("#dashboard-close");
const dashboardScrim = document.querySelector("#dashboard-scrim");
const dashboardSubtitle = document.querySelector("#dashboard-subtitle");
const dashboardStack = document.querySelector("#dashboard-stack");
const dashboardEmpty = document.querySelector("#dashboard-empty");
const drawerQuotaMeters = document.querySelector("#drawer-quota-meters");
const quotaTeaser = document.querySelector("#quota-teaser");
const privacyFloor = document.querySelector("#privacy-floor");
const privacyFloorPopover = document.querySelector("#privacy-floor-popover");
const privateChatToggle = document.querySelector("#private-chat-toggle");
const modelLockNote = document.querySelector("#model-lock-note");
const metricPremiumAvoided = document.querySelector("#metric-premium-avoided");
const metricTokensCompression = document.querySelector("#metric-tokens-compression");
const metricTokensRouting = document.querySelector("#metric-tokens-routing");
const metricPremiumCalls = document.querySelector("#metric-premium-calls");
const feedbackQuality = document.querySelector("#feedback-quality");
const backendUsage = document.querySelector("#backend-usage");
const trend = document.querySelector("#trend");
const sessionStorageKey = SB.storageKeys.sessionId;

const fallbackModelOptions = [
  { value: "auto", label: "Auto", description: "Routes automatically", available: true },
  { value: "codex", label: "Codex", description: "Best for coding tasks", available: false },
  { value: "claude", label: "Claude", description: "Good for reasoning and design", available: false },
  { value: "ollama", label: "Ollama", description: "Runs locally", available: false },
];

let currentModel = appState.currentModel;
let isSending = appState.composer.isSending;
let sessionId = appState.sessionId;
let modelLabels = { auto: "Auto" };
let modelOptions = [];
let modelMenuOverlay = null;
let privacyFloorOverlay = null;
let dashboardOverlay = null;
let sidebarOverlay = null;
let versionOverlay = null;
let modelBeforePrivate = currentModel === "ollama" ? "auto" : currentModel;
let privateOffConfirmed = false;
let visibleSessions = [];
let sessionSearchTimer = null;
let sessionListBefore = null;
let sessionUndoTimer = null;

/* ------------------------------------------------------------------ */
/* Minimal safe markdown renderer (local-first: no CDN dependencies). */
/* ------------------------------------------------------------------ */

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderInline(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|\W)\*([^*\n]+)\*(?=\W|$)/g, "$1<em>$2</em>");
  html = html.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );
  return html;
}

function renderBlocks(text) {
  const lines = text.split("\n");
  const html = [];
  let list = null;

  function closeList() {
    if (list) {
      html.push(list === "ul" ? "</ul>" : "</ol>");
      list = null;
    }
  }

  for (const line of lines) {
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (heading) {
      closeList();
      const level = Math.min(heading[1].length + 2, 6);
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
    } else if (bullet) {
      if (list !== "ul") {
        closeList();
        html.push("<ul>");
        list = "ul";
      }
      html.push(`<li>${renderInline(bullet[1])}</li>`);
    } else if (ordered) {
      if (list !== "ol") {
        closeList();
        html.push("<ol>");
        list = "ol";
      }
      html.push(`<li>${renderInline(ordered[1])}</li>`);
    } else if (line.trim() === "") {
      closeList();
    } else {
      closeList();
      html.push(`<p>${renderInline(line)}</p>`);
    }
  }
  closeList();
  return html.join("");
}

function renderMarkdown(container, text) {
  container.textContent = "";
  const segments = text.split(/```([\w+-]*)\n?([\s\S]*?)```/g);
  // segments alternate: [text, lang, code, text, lang, code, ..., text]
  for (let i = 0; i < segments.length; i += 3) {
    const plain = segments[i];
    if (plain && plain.trim()) {
      const block = document.createElement("div");
      block.innerHTML = renderBlocks(plain);
      container.appendChild(block);
    }
    if (i + 2 < segments.length) {
      const lang = segments[i + 1] || "";
      const code = segments[i + 2] || "";
      container.appendChild(makeCodeBlock(code, lang));
    }
  }
}

function makeCodeBlock(code, lang) {
  const wrapper = document.createElement("div");
  wrapper.className = "code-block";
  const header = document.createElement("div");
  header.className = "code-header";
  const label = document.createElement("span");
  label.textContent = lang || "code";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "copy-button";
  copy.textContent = "Copy";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(code);
      copy.textContent = "Copied";
      setTimeout(() => (copy.textContent = "Copy"), 1500);
    } catch {
      copy.textContent = "Failed";
    }
  });
  header.appendChild(label);
  header.appendChild(copy);
  const pre = document.createElement("pre");
  const codeEl = document.createElement("code");
  codeEl.textContent = code.replace(/\n$/, "");
  pre.appendChild(codeEl);
  wrapper.appendChild(header);
  wrapper.appendChild(pre);
  return wrapper;
}

/* ------------------------------------------------------------------ */
/* Messages                                                            */
/* ------------------------------------------------------------------ */

function rememberSession(nextSessionId) {
  if (!nextSessionId) {
    return;
  }
  const changed = sessionId !== nextSessionId;
  sessionId = nextSessionId;
  appState.sessionId = nextSessionId;
  window.localStorage.setItem(sessionStorageKey, nextSessionId);
  if (changed && appState.composer.privateChat) {
    persistPrivateChat(true);
  }
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function clearWelcome() {
  const welcome = messages.querySelector(".welcome-panel");
  if (welcome) {
    welcome.remove();
  }
}

function showWelcomeIfEmpty() {
  if (messages.children.length > 0) {
    return;
  }
  const panel = document.createElement("div");
  panel.className = "welcome-panel";
  const title = document.createElement("h2");
  title.textContent = "Start with a route you can see";
  panel.appendChild(title);
  for (const prompt of [
    "Summarize this paragraph in three local-first bullets.",
    "Design a caching strategy for a busy API.",
    "My SSN is 123-45-6789 — draft a letter asking to correct a record.",
  ]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = prompt;
    button.addEventListener("click", () => {
      input.value = prompt;
      resizeInput();
      updateSendState();
      sendMessage();
    });
    panel.appendChild(button);
  }
  messages.appendChild(panel);
}

function addMessage(text, role) {
  clearWelcome();
  const item = document.createElement("div");
  item.className = `message ${role}`;
  const body = document.createElement("div");
  body.className = "message-text";
  body.textContent = text;
  item.appendChild(body);
  messages.appendChild(item);
  scrollToBottom();
  return item;
}

function addAssistantMessage({ markdown, displayModel, routing }) {
  clearWelcome();
  const item = document.createElement("div");
  item.className = "message assistant";
  const body = document.createElement("div");
  body.className = "message-text";
  if (markdown) {
    renderMarkdown(body, markdown);
  }
  item.appendChild(body);
  item.appendChild(makeMetaRow(item, displayModel, routing));
  messages.appendChild(item);
  scrollToBottom();
  return { item, body };
}

function makeChip(label, tone = "neutral") {
  const chip = document.createElement("span");
  chip.className = `route-chip ${tone}`;
  chip.textContent = label;
  return chip;
}

function appendRoutingChips(meta, displayModel, routing) {
  if (!routing) {
    return;
  }
  const chips = document.createElement("span");
  chips.className = "route-chips";
  chips.appendChild(makeChip(displayModel || routing.backend || "Switchboard", "backend"));
  if (routing.route_type) {
    chips.appendChild(makeChip(routing.route_type, "route"));
  }
  if (routing.private_chat || routing.privacy_floor) {
    chips.appendChild(makeChip("Lock", "privacy"));
  }
  if (routing.tool_grounded) {
    chips.appendChild(makeChip("Tool", "tool"));
  }
  if (routing.compressed) {
    const percent = routing.compression_percent;
    chips.appendChild(makeChip(percent ? `Compressed ${percent}%` : "Compressed", "compressed"));
  }
  if (routing.escalated) {
    chips.appendChild(makeChip("Escalated", "escalated"));
  }
  if (routing.quota) {
    chips.appendChild(makeChip("Quota", "quota"));
  }
  meta.appendChild(chips);
}

function makeMetaRow(item, displayModel, routing) {
  const meta = document.createElement("div");
  meta.className = "message-meta";

  appendRoutingChips(meta, displayModel, routing);

  if (routing && (routing.latency_ms || routing.cost_type)) {
    const facts = [];
    if (routing.latency_ms) {
      facts.push(`${(routing.latency_ms / 1000).toFixed(1)}s`);
    }
    if (routing.cost_type && routing.cost_type !== "unknown") {
      facts.push(routing.cost_type);
    }
    if (facts.length) {
      const factsEl = document.createElement("span");
      factsEl.className = "meta-facts";
      factsEl.textContent = facts.join(" · ");
      meta.appendChild(factsEl);
    }
  }

  if (routing && routing.routing_reason) {
    const whyButton = document.createElement("button");
    whyButton.type = "button";
    whyButton.className = "why-button";
    whyButton.textContent = "why?";
    const panel = document.createElement("div");
    panel.className = "why-panel";
    panel.hidden = true;
    const reason = document.createElement("p");
    reason.textContent = routing.routing_reason;
    panel.appendChild(reason);
    if (routing.selected_model) {
      const model = document.createElement("p");
      model.textContent = `Model: ${routing.selected_model}`;
      panel.appendChild(model);
    }
    whyButton.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
    });
    meta.appendChild(whyButton);
    item.appendChild(panel);
  }

  if (routing && routing.request_id) {
    meta.appendChild(
      makeFeedbackControls(routing.request_id, {
        answeredBackend: routing.backend,
        rating: routing.feedback_rating,
        correctedBackend: routing.corrected_backend,
      })
    );
  }
  return meta;
}

function backendLabel(backend) {
  return {
    ollama: "Ollama",
    codex: "Codex",
    "claude-code": "Claude",
  }[backend] || backend;
}

function makeFeedbackControls(requestId, initial = {}) {
  const group = document.createElement("span");
  group.className = "feedback-group";
  const up = document.createElement("button");
  up.type = "button";
  up.className = "feedback-button";
  up.textContent = "\u{1F44D}";
  up.title = "Good answer";
  up.setAttribute("aria-pressed", "false");
  const down = document.createElement("button");
  down.type = "button";
  down.className = "feedback-button";
  down.textContent = "\u{1F44E}";
  down.title = "Something was wrong";
  down.setAttribute("aria-pressed", "false");
  const popover = document.createElement("span");
  popover.className = "feedback-popover";
  popover.hidden = true;
  const status = document.createElement("span");
  status.className = "feedback-status";
  status.setAttribute("aria-live", "polite");

  let rating = initial.rating || null;
  let correctedBackend = initial.correctedBackend || null;
  let draftDown = false;
  let overlay = null;

  function setPressed() {
    const downActive = draftDown || rating === "bad" || rating === "wrong-route";
    up.classList.toggle("active", rating === "good");
    down.classList.toggle("active", downActive);
    up.setAttribute("aria-pressed", String(rating === "good"));
    down.setAttribute("aria-pressed", String(downActive));
  }

  function closePopover(fromStack = false) {
    popover.hidden = true;
    draftDown = false;
    setPressed();
    if (overlay && !fromStack) {
      SB.dismissableStack.remove(overlay);
    }
  }

  function renderStoredStatus() {
    status.textContent = "";
    if (rating === "wrong-route" && correctedBackend) {
      status.textContent = `wrong model \u2192 ${backendLabel(correctedBackend)} \u2713`;
    } else if (rating === "bad") {
      status.textContent = "bad answer \u2713";
    } else if (rating === "good") {
      status.textContent = "good \u2713";
    }
  }

  function showAck(payload) {
    status.textContent = "";
    const message = document.createElement("span");
    if (payload.nudge_enable_examples) {
      if (appState.feedback.enableNudgeShown) {
        message.textContent = "Saved.";
      } else {
        appState.feedback.enableNudgeShown = true;
        window.sessionStorage.setItem(SB.storageKeys.feedbackNudgeSeen, "1");
        message.textContent = payload.ack_message || "Saved.";
      }
    } else {
      message.textContent = payload.ack_message || "Saved.";
    }
    status.appendChild(message);
    if (payload.copy_command) {
      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "copy-button feedback-copy";
      copy.textContent = "Copy";
      copy.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(payload.copy_command);
          copy.textContent = "Copied";
        } catch {
          copy.textContent = "Failed";
        }
      });
      status.appendChild(copy);
    }
  }

  async function sendFeedback(payload) {
    try {
      const response = await fetch("/api/chat/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: requestId, ...payload }),
      });
      if (response.ok) {
        const result = await response.json();
        rating = result.rating;
        correctedBackend = result.preferred_model || null;
        draftDown = false;
        closePopover();
        setPressed();
        showAck(result);
      }
    } catch {
      /* feedback is best-effort */
    }
  }

  async function retractFeedback() {
    try {
      const response = await fetch(`/api/chat/feedback/${encodeURIComponent(requestId)}`, {
        method: "DELETE",
      });
      if (response.ok) {
        rating = null;
        correctedBackend = null;
        draftDown = false;
        closePopover();
        setPressed();
        status.textContent = "retracted \u2713";
      }
    } catch {
      /* feedback is best-effort */
    }
  }

  function popoverButton(label, payload) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "followup-button";
    button.textContent = label;
    button.addEventListener("click", () => {
      sendFeedback(payload);
    });
    return button;
  }

  function showPopover() {
    popover.replaceChildren();
    draftDown = true;
    setPressed();
    const close = document.createElement("button");
    close.type = "button";
    close.className = "feedback-close";
    close.textContent = "\u00d7";
    close.title = "Close feedback options";
    close.addEventListener("click", () => closePopover());
    popover.appendChild(close);
    popover.appendChild(popoverButton("Bad answer", { rating: "bad", detail: "bad_answer" }));
    const label = document.createElement("span");
    label.className = "followup-label";
    label.textContent = "wrong model \u2192";
    popover.appendChild(label);
    for (const [name, backend] of [
      ["Ollama", "ollama"],
      ["Codex", "codex"],
      ["Claude", "claude-code"],
    ]) {
      if (backend === initial.answeredBackend) {
        continue;
      }
      popover.appendChild(
        popoverButton(name, {
          rating: "wrong-route",
          detail: "wrong_model",
          corrected_backend: backend,
        })
      );
    }
    popover.hidden = false;
    if (!overlay) {
      overlay = SB.dismissableStack.register({
        id: `feedback-${requestId}`,
        element: popover,
        trigger: down,
        close: () => closePopover(true),
      });
    }
    SB.dismissableStack.open(overlay);
  }

  up.addEventListener("click", () => {
    if (rating === "good") {
      retractFeedback();
      return;
    }
    sendFeedback({ rating: "good" });
  });
  down.addEventListener("click", () => {
    if (!popover.hidden || rating === "bad" || rating === "wrong-route") {
      retractFeedback();
      return;
    }
    showPopover();
  });
  setPressed();
  renderStoredStatus();
  group.appendChild(up);
  group.appendChild(down);
  group.appendChild(popover);
  group.appendChild(status);
  return group;
}

/* ------------------------------------------------------------------ */
/* History                                                             */
/* ------------------------------------------------------------------ */

async function loadHistory() {
  if (!sessionId) {
    showWelcomeIfEmpty();
    return;
  }
  try {
    const response = await fetch(
      `/api/chat/history?session_id=${encodeURIComponent(sessionId)}`
    );
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    applyPrivateChatState(Boolean(payload.private));
    messages.textContent = "";
    for (const message of payload.messages || []) {
      if (message.role === "user") {
        addMessage(message.content, "user");
      } else {
        addAssistantMessage({
          markdown: message.content,
          displayModel: message.display_model,
          routing: message.request_id
            ? {
                request_id: message.request_id,
                backend: message.backend,
                feedback_rating: message.feedback_rating,
                corrected_backend: message.corrected_backend,
                ...(message.routing || {}),
              }
            : null,
        });
      }
    }
    showWelcomeIfEmpty();
  } catch {
    /* offline or first run: start with an empty thread */
    showWelcomeIfEmpty();
  }
}

function cachePrivateChat(enabled) {
  if (enabled) {
    window.localStorage.setItem(SB.storageKeys.privateChat, "1");
  } else {
    window.localStorage.removeItem(SB.storageKeys.privateChat);
  }
}

async function persistPrivateChat(enabled) {
  if (!sessionId) {
    return;
  }
  try {
    await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ private: enabled }),
    });
  } catch {
    /* Private-chat persistence is retried after the next successful send. */
  }
}

function applyPrivateChatState(enabled, options = {}) {
  const shouldPersist = options.persist || false;
  const restoreModel = options.restoreModel || false;
  appState.composer.privateChat = enabled;
  cachePrivateChat(enabled);
  if (privateChatToggle) {
    privateChatToggle.setAttribute("aria-pressed", String(enabled));
  }
  form.classList.toggle("private-chat-on", enabled);
  modelButton.classList.toggle("locked", enabled);
  modelButton.setAttribute("aria-disabled", String(enabled));
  if (modelLockNote) {
    modelLockNote.hidden = !enabled;
  }
  if (enabled) {
    if (currentModel !== "ollama") {
      modelBeforePrivate = currentModel;
    }
    chooseModel("ollama");
  } else if (restoreModel) {
    chooseModel(modelBeforePrivate || "auto");
  }
  if (shouldPersist) {
    persistPrivateChat(enabled);
  }
}

function startNewChat() {
  sessionId = null;
  appState.sessionId = null;
  privateOffConfirmed = false;
  applyPrivateChatState(false, { restoreModel: true });
  window.localStorage.removeItem(sessionStorageKey);
  messages.textContent = "";
  showWelcomeIfEmpty();
  input.focus();
}

/* ------------------------------------------------------------------ */
/* Sidebar sessions                                                    */
/* ------------------------------------------------------------------ */

function setSidebarCollapsed(collapsed) {
  appState.sidebar.collapsed = collapsed;
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  window.localStorage.setItem(SB.storageKeys.sidebarCollapsed, collapsed ? "1" : "0");
  if (sidebarCollapse) {
    sidebarCollapse.setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
  }
}

function setSidebarOpen(open, fromStack = false) {
  document.body.classList.toggle("sidebar-open", open);
  if (sidebarScrim) {
    sidebarScrim.hidden = !open;
  }
  if (!sidebarOverlay || fromStack) {
    return;
  }
  if (open) {
    SB.dismissableStack.open(sidebarOverlay);
  } else {
    SB.dismissableStack.remove(sidebarOverlay);
  }
}

function sessionGroupLabel(updatedAt) {
  const date = new Date(updatedAt);
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);
  if (date.toDateString() === today.toDateString()) {
    return "Today";
  }
  if (date.toDateString() === yesterday.toDateString()) {
    return "Yesterday";
  }
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function backendDot(summary) {
  const dot = document.createElement("span");
  dot.className = `backend-dot ${summary || "empty"}`;
  dot.title =
    summary === "mixed"
      ? "Mixed local and premium"
      : summary === "premium"
        ? "Premium touched"
        : summary === "local"
          ? "Local only"
          : "No assistant messages";
  return dot;
}

function appendHighlightedSnippet(container, text, query) {
  const value = String(text || "");
  const needle = query.trim().toLowerCase();
  if (!needle) {
    container.textContent = value;
    return;
  }
  const lower = value.toLowerCase();
  let cursor = 0;
  let match = lower.indexOf(needle);
  while (match >= 0) {
    if (match > cursor) {
      container.appendChild(document.createTextNode(value.slice(cursor, match)));
    }
    const mark = document.createElement("mark");
    mark.textContent = value.slice(match, match + needle.length);
    container.appendChild(mark);
    cursor = match + needle.length;
    match = lower.indexOf(needle, cursor);
  }
  if (cursor < value.length) {
    container.appendChild(document.createTextNode(value.slice(cursor)));
  }
}

function updateActiveSessionRows() {
  for (const row of sessionList?.querySelectorAll(".session-row") || []) {
    row.classList.toggle("active", row.dataset.sessionId === sessionId);
  }
}

function rowMeta(session) {
  const meta = document.createElement("span");
  meta.className = "session-meta";
  if (session.private) {
    const lock = document.createElement("span");
    lock.className = "session-lock";
    lock.textContent = "🔒";
    meta.appendChild(lock);
  }
  meta.appendChild(backendDot(session.backend_summary));
  return meta;
}

function renderSessionRow(session, query = "") {
  const row = document.createElement("div");
  row.tabIndex = 0;
  row.setAttribute("role", "button");
  row.className = "session-row";
  row.dataset.sessionId = session.session_id;
  row.addEventListener("click", () => openSession(session.session_id));
  row.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      openSession(session.session_id);
    }
  });
  const meta = rowMeta(session);
  const title = document.createElement("span");
  title.className = "session-title";
  title.textContent = session.title;
  row.appendChild(meta);
  row.appendChild(title);

  const actions = document.createElement("span");
  actions.className = "session-actions";
  const rename = document.createElement("button");
  rename.type = "button";
  rename.textContent = "Rename";
  rename.addEventListener("click", (event) => {
    event.stopPropagation();
    startRenameSession(session, row);
  });
  const del = document.createElement("button");
  del.type = "button";
  del.textContent = "Delete";
  del.addEventListener("click", (event) => {
    event.stopPropagation();
    deleteSession(session.session_id);
  });
  actions.appendChild(rename);
  actions.appendChild(del);
  row.appendChild(actions);

  if (query && session.snippet !== undefined) {
    const snippet = document.createElement("span");
    snippet.className = "session-snippet";
    appendHighlightedSnippet(snippet, session.snippet, query);
    row.appendChild(snippet);
  }
  return row;
}

function renderSessionList(sessions, { query = "", append = false, search = false } = {}) {
  if (!sessionList) {
    return;
  }
  if (!append) {
    sessionList.textContent = "";
  }
  if (!sessions.length && !append) {
    const empty = document.createElement("p");
    empty.className = "sidebar-empty";
    empty.textContent = query
      ? `No matches for '${query}' — search covers all sessions, including CLI ones.`
      : "Chats you start here and in the CLI (when titled) appear here.";
    sessionList.appendChild(empty);
    return;
  }
  let currentGroup = "";
  for (const session of sessions) {
    const group = search ? "Search results" : sessionGroupLabel(session.updated_at);
    if (group !== currentGroup) {
      currentGroup = group;
      const heading = document.createElement("div");
      heading.className = "session-group";
      heading.textContent = group;
      sessionList.appendChild(heading);
    }
    sessionList.appendChild(renderSessionRow(session, query));
  }
  updateActiveSessionRows();
}

async function loadSessions({ append = false } = {}) {
  if (!sessionList) {
    return;
  }
  const params = new URLSearchParams({ limit: "100" });
  if (append && sessionListBefore) {
    params.set("before", sessionListBefore);
  }
  const response = await fetch(`/api/sessions?${params.toString()}`);
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  const sessions = payload.sessions || [];
  visibleSessions = append ? [...visibleSessions, ...sessions] : sessions;
  renderSessionList(sessions, { append });
  sessionListBefore = sessions.length ? sessions[sessions.length - 1].updated_at : null;
  if (sessionLoadMore) {
    sessionLoadMore.hidden = sessions.length < 100;
  }
}

function filterVisibleSessions(query) {
  const needle = query.trim().toLowerCase();
  if (!needle) {
    renderSessionList(visibleSessions);
    return;
  }
  renderSessionList(
    visibleSessions.filter((session) => session.title.toLowerCase().includes(needle)),
    { query }
  );
}

async function searchSessions(query) {
  if (query.trim().length < 2) {
    filterVisibleSessions(query);
    return;
  }
  const response = await fetch(
    `/api/sessions/search?q=${encodeURIComponent(query.trim())}`
  );
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  renderSessionList(payload.results || [], { query, search: true });
}

function scheduleSessionSearch() {
  const query = sessionSearch?.value || "";
  filterVisibleSessions(query);
  if (sessionSearchTimer) {
    clearTimeout(sessionSearchTimer);
  }
  sessionSearchTimer = setTimeout(() => searchSessions(query), 300);
}

async function openSession(nextSessionId) {
  if (!nextSessionId) {
    return;
  }
  sessionId = nextSessionId;
  appState.sessionId = nextSessionId;
  window.localStorage.setItem(sessionStorageKey, nextSessionId);
  messages.textContent = "";
  updateActiveSessionRows();
  await loadHistory();
  setSidebarOpen(false);
}

function startRenameSession(session, row) {
  const inputEl = document.createElement("input");
  inputEl.className = "session-rename";
  inputEl.value = session.stored_title || session.title;
  row.replaceChildren(inputEl);
  inputEl.focus();
  inputEl.select();
  async function save() {
    const title = inputEl.value.trim();
    const response = await fetch(`/api/sessions/${encodeURIComponent(session.session_id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    if (response.ok) {
      loadSessions();
    }
  }
  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      save();
    } else if (event.key === "Escape") {
      loadSessions();
    }
  });
  inputEl.addEventListener("blur", save);
}

function showSessionToast(sessionIdForUndo) {
  if (!sessionToast) {
    return;
  }
  if (sessionUndoTimer) {
    clearTimeout(sessionUndoTimer);
  }
  sessionToast.textContent = "";
  const message = document.createElement("span");
  message.textContent = "Session deleted";
  const undo = document.createElement("button");
  undo.type = "button";
  undo.textContent = "Undo";
  undo.addEventListener("click", async () => {
    const response = await fetch(
      `/api/sessions/${encodeURIComponent(sessionIdForUndo)}/undo-delete`,
      { method: "POST" }
    );
    if (response.ok) {
      sessionToast.hidden = true;
      loadSessions();
    }
  });
  sessionToast.appendChild(message);
  sessionToast.appendChild(undo);
  sessionToast.hidden = false;
  sessionUndoTimer = setTimeout(() => {
    sessionToast.hidden = true;
  }, 10000);
}

async function deleteSession(sessionIdToDelete) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionIdToDelete)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    return;
  }
  if (sessionIdToDelete === sessionId) {
    startNewChat();
  }
  showSessionToast(sessionIdToDelete);
  loadSessions();
}

/* ------------------------------------------------------------------ */
/* Version footer                                                      */
/* ------------------------------------------------------------------ */

function setVersionPopoverOpen(open, fromStack = false) {
  if (!versionPopover || !versionPill) {
    return;
  }
  versionPopover.hidden = !open;
  versionPill.setAttribute("aria-expanded", String(open));
  if (!versionOverlay || fromStack) {
    return;
  }
  if (open) {
    SB.dismissableStack.open(versionOverlay);
  } else {
    SB.dismissableStack.remove(versionOverlay);
  }
}

async function loadVersionStatus() {
  if (!uiVersion) {
    return;
  }
  try {
    const response = await fetch("/api/version");
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    uiVersion.textContent = `v${payload.installed || "unknown"}`;
    const upgradeCommand = "switchboard upgrade";
    if (payload.update_available && payload.latest) {
      updateBadge.hidden = false;
      versionCopy.textContent = `Switchboard ${payload.latest} is available. Run ${upgradeCommand}.`;
      copyUpgradeCommand.hidden = false;
      copyUpgradeCommand.onclick = async () => {
        try {
          await navigator.clipboard.writeText(upgradeCommand);
          copyUpgradeCommand.textContent = "Copied";
        } catch {
          copyUpgradeCommand.textContent = "Failed";
        }
      };
    } else {
      updateBadge.hidden = true;
      versionCopy.textContent = "Switchboard is up to date.";
      copyUpgradeCommand.hidden = true;
    }
  } catch {
    uiVersion.textContent = "vunknown";
  }
}

/* ------------------------------------------------------------------ */
/* Dashboard                                                           */
/* ------------------------------------------------------------------ */

function formatNumber(value) {
  return new Intl.NumberFormat().format(value || 0);
}

function countLabel(value, singular, plural = `${singular}s`) {
  const count = Number(value || 0);
  return `${formatNumber(count)} ${count === 1 ? singular : plural}`;
}

function renderQuotaMeters(payload) {
  if (!quotaMeters || !payload || !payload.enabled) {
    if (quotaMeters) {
      quotaMeters.hidden = true;
    }
    return;
  }
  quotaMeters.textContent = "";
  for (const backend of ["codex", "claude-code"]) {
    const window = payload.windows?.[backend];
    if (!window || window.budget === null || window.budget === undefined) {
      continue;
    }
    const meter = document.createElement("div");
    meter.className = "quota-meter";
    const label = document.createElement("span");
    label.textContent = `${window.label} ${window.used}/${window.budget}`;
    const bar = document.createElement("span");
    bar.className = "quota-bar";
    const fill = document.createElement("span");
    fill.style.width = `${Math.min(100, Math.round((window.used / window.budget) * 100))}%`;
    if (window.constrained) {
      fill.classList.add("constrained");
    }
    bar.appendChild(fill);
    meter.appendChild(label);
    meter.appendChild(bar);
    quotaMeters.appendChild(meter);
  }
  quotaMeters.hidden = quotaMeters.children.length === 0;
}

function quotaFillClass(window) {
  if (window.constrained) {
    return "constrained";
  }
  if (!window.budget) {
    return "";
  }
  const ratio = window.used / window.budget;
  if (ratio >= 0.75) {
    return "warning";
  }
  return "";
}

function renderDashboardQuota(payload) {
  if (!drawerQuotaMeters || !quotaTeaser) {
    return;
  }
  drawerQuotaMeters.textContent = "";
  if (!payload || !payload.enabled) {
    quotaTeaser.hidden = false;
    return;
  }
  quotaTeaser.hidden = true;
  for (const backend of ["codex", "claude-code"]) {
    const window = payload.windows?.[backend];
    if (!window || window.budget === null || window.budget === undefined) {
      continue;
    }
    const meter = document.createElement("div");
    meter.className = "drawer-quota-meter";
    const header = document.createElement("div");
    const label = document.createElement("span");
    label.textContent = window.label;
    const value = document.createElement("strong");
    value.textContent = `${window.used}/${window.budget}`;
    header.appendChild(label);
    header.appendChild(value);
    const bar = document.createElement("span");
    bar.className = "quota-bar drawer-quota-bar";
    const fill = document.createElement("span");
    fill.style.width = `${Math.min(100, Math.round((window.used / window.budget) * 100))}%`;
    const tone = quotaFillClass(window);
    if (tone) {
      fill.classList.add(tone);
    }
    bar.appendChild(fill);
    meter.appendChild(header);
    meter.appendChild(bar);
    drawerQuotaMeters.appendChild(meter);
  }
  quotaTeaser.hidden = drawerQuotaMeters.children.length !== 0;
}

function renderStackedBar(handled, total) {
  if (!dashboardStack) {
    return;
  }
  dashboardStack.textContent = "";
  const segments = [
    ["local", handled.local || 0],
    ["tools", handled.tools || 0],
    ["premium", handled.premium || 0],
  ];
  const denominator = Math.max(1, total || segments.reduce((sum, item) => sum + item[1], 0));
  for (const [name, count] of segments) {
    const segment = document.createElement("span");
    segment.className = `stacked-segment ${name}`;
    segment.style.width = `${Math.round((count / denominator) * 100)}%`;
    segment.title = `${name}: ${count}`;
    dashboardStack.appendChild(segment);
  }
}

function renderTrend(days) {
  trend.textContent = "";
  const maxRequests = Math.max(1, ...((days || []).map((day) => day.requests)));
  for (const day of days || []) {
    const item = document.createElement("div");
    item.className = "trend-day";
    const bar = document.createElement("span");
    bar.className = "trend-bar";
    bar.style.height = `${Math.max(8, Math.round((day.requests / maxRequests) * 58))}px`;
    bar.title = `${day.date}: ${day.requests} requests, ${day.premium_calls} premium`;
    if (!day.requests) {
      bar.classList.add("empty");
    } else {
      const local = document.createElement("span");
      local.className = "trend-segment local";
      const tools = document.createElement("span");
      tools.className = "trend-segment tools";
      const premium = document.createElement("span");
      premium.className = "trend-segment premium";
      local.style.flexGrow = String(day.local_calls || 0);
      tools.style.flexGrow = String(day.tool_calls || 0);
      premium.style.flexGrow = String(day.premium_calls || 0);
      bar.appendChild(premium);
      bar.appendChild(tools);
      bar.appendChild(local);
    }
    const label = document.createElement("span");
    label.className = "trend-label";
    label.textContent = day.date.slice(5);
    item.appendChild(bar);
    item.appendChild(label);
    trend.appendChild(item);
  }
}

function renderFeedbackQuality(feedback) {
  if (!feedbackQuality) {
    return;
  }
  const total = feedback?.total || 0;
  if (!total) {
    feedbackQuality.textContent = "No ratings yet.";
    return;
  }
  feedbackQuality.textContent = `You rated ${countLabel(total, "answer")} — ${formatNumber(
    feedback.good
  )} good, ${countLabel(feedback.corrected, "correction")} · ${countLabel(
    feedback.pending_corrections,
    "correction"
  )} pending`;
}

function renderDashboard(payload) {
  if (!payload) {
    return;
  }
  const avoided = payload.premium_calls_avoided_vs_always_premium || 0;
  const total = payload.total_requests || 0;
  const handled = payload.handled_requests || { local: avoided, tools: 0, premium: 0 };
  metricPremiumAvoided.textContent = formatNumber(avoided);
  metricTokensCompression.textContent = formatNumber(payload.estimated_tokens_saved?.compression);
  metricTokensRouting.textContent = formatNumber(payload.estimated_tokens_saved?.routing);
  metricPremiumCalls.textContent = formatNumber(payload.premium_calls);
  dashboardSubtitle.textContent = `${formatNumber(avoided)} of ${formatNumber(
    total
  )} requests handled locally or by tools this week.`;
  dashboardEmpty.hidden = total !== 0;
  renderStackedBar(handled, total);
  renderTrend(payload.last_7_days || []);
  renderFeedbackQuality(payload.feedback || {});
}

async function loadQuotaStatus() {
  try {
    const response = await fetch("/api/quota");
    if (response.ok) {
      const payload = await response.json();
      renderQuotaMeters(payload);
      return payload;
    }
  } catch {
    if (quotaMeters) {
      quotaMeters.hidden = true;
    }
  }
  return null;
}

async function loadDashboardQuota() {
  renderDashboardQuota(await loadQuotaStatus());
}

async function loadDashboard() {
  try {
    const response = await fetch("/api/dashboard");
    if (response.ok) {
      renderDashboard(await response.json());
    }
  } catch {
    /* dashboard is best-effort */
  }
}

function toggleDashboard() {
  setDashboardOpen(dashboard.hidden);
  if (!dashboard.hidden) {
    loadDashboard();
    loadDashboardQuota();
  }
}

function setDashboardOpen(open, fromStack = false) {
  if (!dashboard) {
    return;
  }
  dashboard.hidden = !open;
  if (dashboardScrim) {
    dashboardScrim.hidden = !open;
  }
  dashboardToggle.setAttribute("aria-expanded", String(open));
  document.body.classList.toggle("dashboard-open", open);
  if (!dashboardOverlay || fromStack) {
    return;
  }
  if (open) {
    SB.dismissableStack.open(dashboardOverlay);
  } else {
    SB.dismissableStack.remove(dashboardOverlay);
  }
}

/* ------------------------------------------------------------------ */
/* Streaming                                                           */
/* ------------------------------------------------------------------ */

function handleStreamEvent(event, pending, state) {
  rememberSession(event.session_id);
  if (event.type === "metadata") {
    state.displayModel = event.display_model;
    state.routing = event;
    return;
  }
  if (event.type === "chunk") {
    state.raw += event.text || "";
    if (state.bodyEl) {
      state.bodyEl.textContent = state.raw;
    }
    scrollToBottom();
    return;
  }
  if (event.type === "done") {
    state.routing = { ...(state.routing || {}), ...event };
    return;
  }
  if (event.type === "error") {
    pending.remove();
    addMessage(event.message || "Something went wrong. Please try again.", "error");
    state.failed = true;
  }
}

async function streamAssistantResponse(response, pending, state) {
  if (!response.body) {
    throw new Error("Streaming is not available in this browser.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      handleStreamEvent(JSON.parse(line), pending, state);
      if (state.failed) {
        return;
      }
    }
  }
  if (buffer.trim() && !state.failed) {
    handleStreamEvent(JSON.parse(buffer), pending, state);
  }
}

async function sendMessage() {
  const message = input.value.trim();
  if (!message || isSending) {
    return;
  }

  addMessage(message, "user");
  input.value = "";
  resizeInput();
  isSending = true;
  appState.composer.isSending = true;
  updateSendState();
  const pending = addMessage("Thinking...", "pending");
  const state = { raw: "", bodyEl: null, displayModel: null, routing: null, failed: false };
  const privateChat = appState.composer.privateChat;
  const backend = privateChat ? "ollama" : currentModel;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, backend, session_id: sessionId, private: privateChat }),
    });
    if (!response.ok) {
      const payload = await response.json();
      pending.remove();
      addMessage(payload.detail?.message || "The request failed.", "error");
      return;
    }
    // Repurpose the pending bubble as the streaming assistant message.
    pending.className = "message assistant";
    pending.textContent = "";
    state.bodyEl = document.createElement("div");
    state.bodyEl.className = "message-text";
    pending.appendChild(state.bodyEl);

    await streamAssistantResponse(response, pending, state);

    if (!state.failed) {
      renderMarkdown(state.bodyEl, state.raw);
      pending.appendChild(makeMetaRow(pending, state.displayModel, state.routing));
      scrollToBottom();
    }
  } catch (error) {
    pending.remove();
    addMessage("Switchboard is not reachable. Is the UI server still running?", "error");
  } finally {
    isSending = false;
    appState.composer.isSending = false;
    updateSendState();
    loadBackendStatus();
    loadQuotaStatus();
    if (dashboard && !dashboard.hidden) {
      loadDashboard();
      loadDashboardQuota();
    }
    loadSessions();
    input.focus();
  }
}

/* ------------------------------------------------------------------ */
/* Controls                                                            */
/* ------------------------------------------------------------------ */

function setMenuOpen(open) {
  setMenuOpenFromStack(open, false);
}

function setMenuOpenFromStack(open, fromStack) {
  modelMenu.hidden = !open;
  modelButton.setAttribute("aria-expanded", String(open));
  if (!modelMenuOverlay || fromStack) {
    return;
  }
  if (open) {
    SB.dismissableStack.open(modelMenuOverlay);
  } else {
    SB.dismissableStack.remove(modelMenuOverlay);
  }
}

function renderModelOptions(options) {
  modelLabels = {};
  modelMenu.textContent = "";
  modelOptions = [];
  for (const option of options) {
    modelLabels[option.value] = option.label;
    const button = document.createElement("button");
    button.className = "model-option";
    button.type = "button";
    button.role = "option";
    button.dataset.model = option.value;
    button.setAttribute("aria-selected", String(option.value === currentModel));
    if (option.value === currentModel) {
      button.classList.add("selected");
    }
    const text = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = option.label;
    const detail = document.createElement("small");
    const hot = Array.isArray(option.hot_models) && option.hot_models.length
      ? ` Hot: ${option.hot_models.join(", ")}`
      : "";
    detail.textContent = `${option.description || ""}${hot}`;
    text.appendChild(title);
    text.appendChild(detail);
    const status = document.createElement("span");
    status.className = `availability-dot ${option.available ? "available" : "unavailable"}`;
    status.title = option.available ? "Available" : option.warning || "Unavailable";
    const check = document.createElement("span");
    check.className = "checkmark";
    check.setAttribute("aria-hidden", "true");
    check.textContent = "✓";
    const right = document.createElement("span");
    right.className = "option-state";
    right.appendChild(status);
    right.appendChild(check);
    button.appendChild(text);
    button.appendChild(right);
    button.addEventListener("click", () => chooseModel(option.value));
    modelMenu.appendChild(button);
    modelOptions.push(button);
  }
  selectedModel.textContent = modelLabels[currentModel] || "Auto";
}

async function loadBackendStatus() {
  try {
    const response = await fetch("/api/backends/status");
    if (!response.ok) {
      throw new Error("backend status unavailable");
    }
    const payload = await response.json();
    renderModelOptions(payload.options || fallbackModelOptions);
    if (appState.composer.privateChat) {
      applyPrivateChatState(true);
    }
  } catch {
    renderModelOptions(fallbackModelOptions);
    if (appState.composer.privateChat) {
      applyPrivateChatState(true);
    }
  }
}

function chooseModel(value) {
  if (appState.composer.privateChat && value !== "ollama") {
    return;
  }
  currentModel = value;
  appState.currentModel = value;
  selectedModel.textContent = modelLabels[value] || value;
  for (const option of modelOptions) {
    const isSelected = option.dataset.model === value;
    option.classList.toggle("selected", isSelected);
    option.setAttribute("aria-selected", String(isSelected));
  }
  setMenuOpen(false);
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function updateSendState() {
  send.disabled = isSending || input.value.trim().length === 0;
}

function setPrivacyFloorOpen(open, fromStack = false) {
  if (!privacyFloor || !privacyFloorPopover) {
    return;
  }
  privacyFloorPopover.hidden = !open;
  privacyFloor.setAttribute("aria-expanded", String(open));
  if (!privacyFloorOverlay || fromStack) {
    return;
  }
  if (open) {
    SB.dismissableStack.open(privacyFloorOverlay);
  } else {
    SB.dismissableStack.remove(privacyFloorOverlay);
  }
}

function togglePrivateChat() {
  if (appState.composer.privateChat) {
    if (
      !privateOffConfirmed &&
      !window.confirm(
        "Earlier messages in this chat may be included as context for premium backends from now on. Continue?"
      )
    ) {
      return;
    }
    privateOffConfirmed = true;
    applyPrivateChatState(false, { persist: true, restoreModel: true });
    return;
  }
  applyPrivateChatState(true, { persist: true });
}

modelButton.addEventListener("click", () => {
  if (appState.composer.privateChat) {
    return;
  }
  setMenuOpen(modelMenu.hidden);
});

modelMenuOverlay = SB.dismissableStack.register({
  id: "model-menu",
  element: modelMenu,
  trigger: modelButton,
  close: () => setMenuOpenFromStack(false, true),
});

if (dashboard && dashboardToggle) {
  dashboardOverlay = SB.dismissableStack.register({
    id: "dashboard",
    element: dashboard,
    trigger: dashboardToggle,
    close: () => setDashboardOpen(false, true),
  });
}

if (privacyFloor && privacyFloorPopover) {
  privacyFloorOverlay = SB.dismissableStack.register({
    id: "privacy-floor",
    element: privacyFloorPopover,
    trigger: privacyFloor,
    close: () => setPrivacyFloorOpen(false, true),
  });
  privacyFloor.addEventListener("click", () => {
    setPrivacyFloorOpen(privacyFloorPopover.hidden);
  });
}

if (sidebar && sidebarToggle) {
  sidebarOverlay = SB.dismissableStack.register({
    id: "sidebar",
    element: sidebar,
    trigger: sidebarToggle,
    close: () => setSidebarOpen(false, true),
  });
}

if (versionPill && versionPopover) {
  versionOverlay = SB.dismissableStack.register({
    id: "version",
    element: versionPopover,
    trigger: versionPill,
    close: () => setVersionPopoverOpen(false, true),
  });
  versionPill.addEventListener("click", () => {
    setVersionPopoverOpen(versionPopover.hidden);
  });
}

if (privateChatToggle) {
  privateChatToggle.addEventListener("click", togglePrivateChat);
}

if (dashboardToggle) {
  dashboardToggle.addEventListener("click", toggleDashboard);
}

if (dashboardClose) {
  dashboardClose.addEventListener("click", () => setDashboardOpen(false));
}

if (sidebarToggle) {
  sidebarToggle.addEventListener("click", () => setSidebarOpen(true));
}

if (sidebarCollapse) {
  sidebarCollapse.addEventListener("click", () => {
    setSidebarCollapsed(!appState.sidebar.collapsed);
  });
}

if (sidebarNewChat) {
  sidebarNewChat.addEventListener("click", () => {
    startNewChat();
    setSidebarOpen(false);
  });
}

if (sessionSearch) {
  sessionSearch.addEventListener("input", scheduleSessionSearch);
}

if (sessionLoadMore) {
  sessionLoadMore.addEventListener("click", () => loadSessions({ append: true }));
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});

input.addEventListener("input", () => {
  resizeInput();
  updateSendState();
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

renderModelOptions(fallbackModelOptions);
setSidebarCollapsed(appState.sidebar.collapsed);
applyPrivateChatState(appState.composer.privateChat);
resizeInput();
updateSendState();
loadSessions();
loadVersionStatus();
loadBackendStatus();
loadQuotaStatus();
loadHistory();
