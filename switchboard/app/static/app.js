const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const messages = document.querySelector("#messages");
const modelButton = document.querySelector("#model-picker-button");
const selectedModel = document.querySelector("#selected-model");
const modelMenu = document.querySelector("#model-menu");
const modelOptions = Array.from(document.querySelectorAll(".model-option"));
const newChatButton = document.querySelector("#new-chat");
const sessionStorageKey = "switchboard.session_id";

const modelLabels = {
  auto: "Auto",
  codex: "Codex",
  claude: "Claude",
  ollama: "Ollama",
};

let currentModel = "auto";
let isSending = false;
let sessionId = window.localStorage.getItem(sessionStorageKey) || null;

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
  sessionId = nextSessionId;
  window.localStorage.setItem(sessionStorageKey, nextSessionId);
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(text, role) {
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

function makeMetaRow(item, displayModel, routing) {
  const meta = document.createElement("div");
  meta.className = "message-meta";

  const label = document.createElement("span");
  label.className = "model-label";
  label.textContent = displayModel || "Switchboard";
  meta.appendChild(label);

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
    meta.appendChild(makeFeedbackControls(routing.request_id));
  }
  return meta;
}

function makeFeedbackControls(requestId) {
  const group = document.createElement("span");
  group.className = "feedback-group";
  const up = document.createElement("button");
  up.type = "button";
  up.className = "feedback-button";
  up.textContent = "\u{1F44D}";
  up.title = "Good answer";
  const down = document.createElement("button");
  down.type = "button";
  down.className = "feedback-button";
  down.textContent = "\u{1F44E}";
  down.title = "Something was wrong";
  const followup = document.createElement("span");
  followup.className = "feedback-followup";
  followup.hidden = true;

  async function sendFeedback(payload, active) {
    try {
      const response = await fetch("/api/chat/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: requestId, ...payload }),
      });
      if (response.ok) {
        up.classList.toggle("active", active === up);
        down.classList.toggle("active", active === down);
      }
    } catch {
      /* feedback is best-effort */
    }
  }

  function followupButton(label, payload) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "followup-button";
    button.textContent = label;
    button.addEventListener("click", () => {
      sendFeedback(payload, down);
      followup.replaceChildren();
      const ack = document.createElement("span");
      ack.className = "followup-ack";
      ack.textContent = "noted \u2713";
      followup.appendChild(ack);
      setTimeout(() => (followup.hidden = true), 1600);
    });
    return button;
  }

  function showFollowup() {
    followup.replaceChildren();
    followup.hidden = false;
    followup.appendChild(
      followupButton("Bad answer", { rating: "too-weak", detail: "bad_answer" })
    );
    const label = document.createElement("span");
    label.className = "followup-label";
    label.textContent = "wrong model \u2192";
    followup.appendChild(label);
    for (const [name, backend] of [
      ["Ollama", "ollama"],
      ["Codex", "codex"],
      ["Claude", "claude-code"],
    ]) {
      followup.appendChild(
        followupButton(name, {
          rating: "wrong-route",
          detail: "wrong_model",
          corrected_backend: backend,
        })
      );
    }
  }

  up.addEventListener("click", () => {
    followup.hidden = true;
    sendFeedback({ rating: "good" }, up);
  });
  down.addEventListener("click", showFollowup);
  group.appendChild(up);
  group.appendChild(down);
  group.appendChild(followup);
  return group;
}

/* ------------------------------------------------------------------ */
/* History                                                             */
/* ------------------------------------------------------------------ */

async function loadHistory() {
  if (!sessionId) {
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
    for (const message of payload.messages || []) {
      if (message.role === "user") {
        addMessage(message.content, "user");
      } else {
        addAssistantMessage({
          markdown: message.content,
          displayModel: message.display_model,
          routing: message.request_id ? { request_id: message.request_id } : null,
        });
      }
    }
  } catch {
    /* offline or first run: start with an empty thread */
  }
}

function startNewChat() {
  sessionId = null;
  window.localStorage.removeItem(sessionStorageKey);
  messages.textContent = "";
  input.focus();
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
  updateSendState();
  const pending = addMessage("Thinking...", "pending");
  const state = { raw: "", bodyEl: null, displayModel: null, routing: null, failed: false };

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, backend: currentModel, session_id: sessionId }),
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
    updateSendState();
    input.focus();
  }
}

/* ------------------------------------------------------------------ */
/* Controls                                                            */
/* ------------------------------------------------------------------ */

function setMenuOpen(open) {
  modelMenu.hidden = !open;
  modelButton.setAttribute("aria-expanded", String(open));
}

function chooseModel(value) {
  currentModel = value;
  selectedModel.textContent = modelLabels[value];
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

modelButton.addEventListener("click", () => {
  setMenuOpen(modelMenu.hidden);
});

for (const option of modelOptions) {
  option.addEventListener("click", () => {
    chooseModel(option.dataset.model);
  });
}

if (newChatButton) {
  newChatButton.addEventListener("click", startNewChat);
}

document.addEventListener("click", (event) => {
  if (!event.target.closest(".model-picker")) {
    setMenuOpen(false);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setMenuOpen(false);
  }
});

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

resizeInput();
updateSendState();
loadHistory();
