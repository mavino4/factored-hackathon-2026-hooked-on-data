// AI Assistant web UI. Plain JS, no build step.
// Security: every piece of server or model text is rendered with textContent, never as HTML.
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  config: null,
  auth: null,          // {header: {...}, user: "name"}
  conversations: [],
  current: null,       // {id, kind, title}
  busy: false,
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child != null) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// ---------------------------------------------------------------------------
// Auth: "dev" (X-User-Id header) or "oidc" (authorization code + PKCE)
// ---------------------------------------------------------------------------
const TOKEN_KEY = "aip.token";
const DEV_USER_KEY = "aip.devUser";
const PKCE_KEY = "aip.pkce";

function b64url(bytes) {
  return btoa(String.fromCharCode(...new Uint8Array(bytes)))
    .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function randomString(bytes = 32) {
  return b64url(crypto.getRandomValues(new Uint8Array(bytes)));
}

async function discovery() {
  const issuer = state.config.oidc_issuer.replace(/\/?$/, "/");
  const res = await fetch(`${issuer}.well-known/openid-configuration`);
  if (!res.ok) throw new Error(t("idp_unavailable"));
  return res.json();
}

function redirectUri() {
  return `${location.origin}/`;
}

async function startOidcLogin() {
  const meta = await discovery();
  const verifier = randomString(48);
  const challenge = b64url(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)));
  const oauthState = randomString(16);
  sessionStorage.setItem(PKCE_KEY, JSON.stringify({ verifier, state: oauthState }));
  const params = new URLSearchParams({
    response_type: "code",
    client_id: state.config.oidc_client_id,
    redirect_uri: redirectUri(),
    scope: "openid profile",
    audience: state.config.oidc_audience,  // Auth0 needs this to issue an API access token
    // Always ask for credentials: an existing provider session must never silently
    // sign in the previous person on a shared computer.
    prompt: "login",
    state: oauthState,
    code_challenge: challenge,
    code_challenge_method: "S256",
  });
  location.assign(`${meta.authorization_endpoint}?${params}`);
}

async function finishOidcLogin(code, returnedState) {
  const saved = JSON.parse(sessionStorage.getItem(PKCE_KEY) || "null");
  sessionStorage.removeItem(PKCE_KEY);
  history.replaceState(null, "", "/");
  if (!saved || saved.state !== returnedState) {
    throw new Error(t("login_failed", { detail: "state mismatch" }));
  }
  const meta = await discovery();
  const res = await fetch(meta.token_endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      code,
      redirect_uri: redirectUri(),
      client_id: state.config.oidc_client_id,
      code_verifier: saved.verifier,
    }),
  });
  if (!res.ok) throw new Error(t("login_failed", { detail: "no token" }));
  const token = await res.json();
  const stored = { access_token: token.access_token, expires_at: Date.now() + (token.expires_in || 3600) * 1000 };
  // Tab-scoped storage so a reload keeps you signed in; cleared when the tab closes.
  sessionStorage.setItem(TOKEN_KEY, JSON.stringify(stored));
  return stored;
}

function subjectOf(accessToken) {
  try {
    const payload = JSON.parse(atob(accessToken.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    return payload.name || payload.email || payload.sub || "signed in";
  } catch {
    return "signed in";
  }
}

async function initAuth() {
  if (state.config.auth_mode === "dev") {
    const user = sessionStorage.getItem(DEV_USER_KEY);
    return user ? { header: { "X-User-Id": user }, user } : null;
  }
  const params = new URLSearchParams(location.search);
  if (params.has("code")) {
    const token = await finishOidcLogin(params.get("code"), params.get("state"));
    return { header: { Authorization: `Bearer ${token.access_token}` }, user: subjectOf(token.access_token) };
  }
  if (params.has("error")) {
    history.replaceState(null, "", "/");
    throw new Error(t("login_failed", { detail: params.get("error_description") || params.get("error") }));
  }
  const saved = JSON.parse(sessionStorage.getItem(TOKEN_KEY) || "null");
  if (saved && saved.expires_at > Date.now() + 30_000) {
    return { header: { Authorization: `Bearer ${saved.access_token}` }, user: subjectOf(saved.access_token) };
  }
  return null;
}

// Sign out and discard everything from this user's session. The page is reloaded
// (or sent to the provider's logout) so no data stays in memory or on screen.
async function logout() {
  const wasOidc = state.config.auth_mode === "oidc" && sessionStorage.getItem(TOKEN_KEY);
  sessionStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(DEV_USER_KEY);
  sessionStorage.removeItem(PKCE_KEY);
  state.auth = null;
  state.current = null;
  state.conversations = [];
  resetView();
  if (wasOidc) {
    try {
      const meta = await discovery();
      if (meta.end_session_endpoint) {
        const params = new URLSearchParams({
          client_id: state.config.oidc_client_id,
          post_logout_redirect_uri: redirectUri(),
        });
        location.replace(`${meta.end_session_endpoint}?${params}`);
        return;
      }
    } catch { /* provider unreachable: still sign out locally */ }
  }
  location.replace("/");
}

// Remove every trace of the previous conversation and user from the page.
function resetView() {
  $("conversation-list").replaceChildren();
  $("chat-title").textContent = t("start");
  $("chat-kind").hidden = true;
  $("chat-kind").textContent = "";
  $("user-name").textContent = "";
  $("input").value = "";
  messagesEl().replaceChildren(el("div", { class: "empty muted" }, t("empty_state")));
  state.current = null;
  setBusy(false);
}

function showLogin(errorMessage) {
  $("app").hidden = true;
  $("login").hidden = false;
  const dev = state.config.auth_mode === "dev";
  $("dev-login").hidden = !dev;
  $("oidc-login").hidden = dev;
  $("login-hint").textContent = t(dev ? "login_hint_dev" : "login_hint_oidc");
  $("login-error").hidden = !errorMessage;
  $("login-error").textContent = errorMessage || "";
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------
class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function api(path, { method = "GET", body } = {}) {
  const headers = { ...state.auth.header };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (res.status === 401) {
    await logout();
    throw new ApiError(401, t("session_expired"));
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
    throw new ApiError(res.status, detail);
  }
  return res;
}

// POST and read Server-Sent Events from the response body (EventSource can't POST).
async function streamEvents(path, body, onEvent) {
  const res = await api(path, { method: "POST", body });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let name = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7);
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      onEvent(name, data.length ? JSON.parse(data.join("\n")) : {});
    }
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
const messagesEl = () => $("messages");

function scrollToBottom() {
  const box = messagesEl();
  box.scrollTop = box.scrollHeight;
}

function addBubble(role, text = "") {
  const bubble = el("div", { class: "bubble" }, text);
  messagesEl().append(el("div", { class: `msg ${role}` }, bubble));
  scrollToBottom();
  return bubble;
}

function addNote(text) {
  messagesEl().append(el("div", { class: "note" }, text));
  scrollToBottom();
}

function describeInput(input) {
  const json = JSON.stringify(input ?? {});
  return json.length > 160 ? `${json.slice(0, 160)}…` : json;
}

function addToolCall(name, input) {
  const node = el("div", { class: "tool" }, `${name}(${describeInput(input)})`);
  messagesEl().append(node);
  scrollToBottom();
  return node;
}

function addError(message, retry) {
  const box = el("div", { class: "error-box", role: "alert" }, el("span", {}, message));
  if (retry) {
    box.append(el("button", {
      onclick: () => { box.remove(); retry(); },
    }, t("retry")));
  }
  messagesEl().append(box);
  scrollToBottom();
}

function addApprovalCard(action) {
  const card = el("div", { class: "approval", "data-open": "" },
    el("strong", {}, t("approval_needed")),
    el("span", {}, t("wants_to_run", { tool: action.tool_name })),
    el("pre", {}, JSON.stringify(action.input, null, 2)));
  const approve = el("button", { class: "primary" }, t("approve"));
  const reject = el("button", {}, t("reject"));
  approve.disabled = reject.disabled = state.busy;
  const decide = (decision) => {
    if (state.busy) return;
    card.removeAttribute("data-open");
    approve.disabled = reject.disabled = true;
    card.append(el("span", { class: "muted" }, t(decision === "approve" ? "approved" : "rejected")));
    runStream(`/v1/conversations/${state.current.id}/actions/${action.action_id || action.id}`,
      { decision }, { startBubble: false });
  };
  approve.addEventListener("click", () => decide("approve"));
  reject.addEventListener("click", () => decide("reject"));
  card.append(el("div", { class: "actions" }, approve, reject));
  messagesEl().append(card);
  scrollToBottom();
}

// Render stored history: text, tool calls, approval notes. Tool results are summarized.
function renderHistory(messages, pendingActions) {
  const box = messagesEl();
  box.replaceChildren();
  for (const message of messages) {
    const content = message.content;
    if (message.role === "user") {
      if (typeof content === "string") {
        if (content.startsWith("[Approval]")) addNote(content.split("\n")[0].replace("[Approval] ", ""));
        else addBubble("user", content);
      } else {
        for (const block of content) {
          if (block.type === "text") addBubble("user", block.text);
          // tool_result blocks: already represented by their tool call chip
        }
      }
      continue;
    }
    const blocks = typeof content === "string" ? [{ type: "text", text: content }] : content;
    for (const block of blocks) {
      if (block.type === "text" && block.text.trim()) addBubble("assistant", block.text);
      else if (block.type === "tool_use") addToolCall(block.name, block.input);
    }
  }
  for (const action of pendingActions || []) addApprovalCard(action);
  if (!messages.length) {
    box.append(el("div", { class: "empty muted" },
      state.current.kind === "agent"
        ? t("empty_agent")
        : t("empty_chat")));
  }
  scrollToBottom();
}

function defaultTitle(kind) {
  return t(kind === "agent" ? "title_query" : "title_chat");
}

function renderConversationList() {
  const list = $("conversation-list");
  list.replaceChildren();
  for (const conv of state.conversations) {
    const button = el("button", { onclick: () => openConversation(conv.id) },
      el("span", { class: "title" }, conv.title || defaultTitle(conv.kind)),
      el("span", { class: "badge" }, t(conv.kind === "agent" ? "kind_agent" : "kind_chat")));
    if (state.current && state.current.id === conv.id) button.setAttribute("aria-current", "true");
    list.append(el("li", {}, button));
  }
}

function setBusy(busy) {
  state.busy = busy;
  $("new-chat").disabled = busy;
  $("new-agent").disabled = busy;
  // Approval buttons wait until the current reply has finished streaming.
  for (const button of document.querySelectorAll(".approval[data-open] button")) button.disabled = busy;
  const disabled = busy || !state.current;
  $("input").disabled = disabled;
  $("send").disabled = disabled;
  if (!disabled) $("input").focus();
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------
async function loadConversations() {
  const res = await api("/v1/conversations");
  state.conversations = (await res.json()).conversations;
  renderConversationList();
}

async function openConversation(id) {
  if (state.busy) return;  // don't switch away from a reply that is still streaming
  $("app").classList.remove("sidebar-open");
  setBusy(true);  // the composer must not target the previous conversation meanwhile
  let conv;
  try {
    conv = await (await api(`/v1/conversations/${id}`)).json();
  } finally {
    setBusy(false);
  }
  state.current = { id: conv.id, kind: conv.kind, title: conv.title };
  $("chat-title").textContent = conv.title || defaultTitle(conv.kind);
  $("chat-kind").hidden = false;
  $("chat-kind").textContent = t(conv.kind === "agent" ? "kind_agent" : "kind_chat");
  renderHistory(conv.messages, conv.pending_actions);
  renderConversationList();
  setBusy(false);
}

async function createConversation(kind) {
  if (state.busy) return;
  setBusy(true);
  let conv;
  try {
    conv = await (await api("/v1/conversations", { method: "POST", body: { kind } })).json();
  } finally {
    setBusy(false);
  }
  state.conversations.unshift(conv);
  await openConversation(conv.id);
}

// Stream one reply (chat message, regenerate, agent run or approval decision).
async function runStream(path, body, { startBubble = true } = {}) {
  setBusy(true);
  let bubble = startBubble ? addBubble("assistant") : null;
  const tools = new Map();
  let failed = false;
  try {
    await streamEvents(path, body, (name, data) => {
      switch (name) {
        case "delta":
          if (!bubble) bubble = addBubble("assistant");
          bubble.textContent += data.text;
          scrollToBottom();
          break;
        case "tool_call":
          if (bubble && !bubble.textContent) bubble.parentElement.remove();
          bubble = null;
          tools.set(data.id, addToolCall(data.name, data.input));
          break;
        case "tool_result": {
          const chip = tools.get(data.id) || addToolCall(data.name, {});
          chip.classList.add(data.is_error ? "fail" : "ok");
          chip.title = data.content;
          break;
        }
        case "approval_required":
          addApprovalCard(data);
          break;
        case "refusal":
          addNote(data.message);
          break;
        case "error":
          failed = true;
          addError(data.message, state.current.kind === "chat" && data.code !== "busy"
            ? () => runStream(`/v1/conversations/${state.current.id}/regenerate`)
            : null);
          break;
        case "done":
          break;
      }
    });
  } catch (err) {
    failed = true;
    addError(err.message || t("connection_lost"), state.current.kind === "chat"
      ? () => runStream(`/v1/conversations/${state.current.id}/regenerate`) : null);
  } finally {
    if (bubble && !bubble.textContent) bubble.parentElement.remove();
    setBusy(false);
    if (!failed) loadConversations().catch(() => {});  // refresh titles/order
  }
}

async function send(event) {
  event.preventDefault();
  const input = $("input");
  const text = input.value.trim();
  if (!text || state.busy || !state.current) return;
  input.value = "";
  autoResize();
  messagesEl().querySelector(".empty")?.remove();
  addBubble("user", text);
  const path = state.current.kind === "agent"
    ? `/v1/conversations/${state.current.id}/agent-runs`
    : `/v1/conversations/${state.current.id}/messages`;
  await runStream(path, { text });
}

function autoResize() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function showApp() {
  resetView();  // never show anything left over from another user
  $("login").hidden = true;
  $("app").hidden = false;
  $("user-name").textContent = state.auth.user;
  await loadConversations();
  if (state.conversations.length) await openConversation(state.conversations[0].id);
}

function guard(fn) {
  return (...args) => Promise.resolve(fn(...args)).catch((err) => {
    if (err instanceof ApiError && err.status === 401) return;  // already back at login
    addError(err.message || String(err));
    setBusy(false);
  });
}

async function boot() {
  applyTranslations();
  state.config = await (await fetch("/config.json")).json();

  $("dev-login").addEventListener("submit", (event) => {
    event.preventDefault();
    const user = $("dev-user").value.trim();
    sessionStorage.setItem(DEV_USER_KEY, user);
    state.auth = { header: { "X-User-Id": user }, user };
    showApp().catch((err) => showLogin(err.message));
  });
  $("oidc-login").addEventListener("click", () => startOidcLogin().catch((err) => showLogin(err.message)));
  $("logout").addEventListener("click", () => { logout(); });
  $("new-chat").addEventListener("click", guard(() => createConversation("chat")));
  $("new-agent").addEventListener("click", guard(() => createConversation("agent")));
  $("composer").addEventListener("submit", guard(send));
  $("input").addEventListener("input", autoResize);
  $("input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("toggle-sidebar").addEventListener("click", () => $("app").classList.toggle("sidebar-open"));
  // On small screens the menu overlays the chat: tapping outside it closes it.
  document.querySelector(".chat").addEventListener("click", (event) => {
    if (!$("toggle-sidebar").contains(event.target)) $("app").classList.remove("sidebar-open");
  });

  try {
    state.auth = await initAuth();
  } catch (err) {
    showLogin(err.message);
    return;
  }
  if (state.auth) await showApp();
  else showLogin();
}

boot().catch((err) => {
  document.body.replaceChildren(el("p", { class: "error boot-error" },
    t("could_not_start", { error: err.message })));
});
