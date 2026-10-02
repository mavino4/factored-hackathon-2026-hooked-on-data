// BankBot web UI. Plain JS, no build step.
// Security: every piece of server or model text is rendered with textContent, never as HTML.
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  config: null,
  auth: null,          // {header: {...}, user: "name"}
  me: null,            // {user_id, first_name} for the greeting
  conversations: [],
  current: null,       // {id, title, seen: messages rendered}
  draft: false,        // welcome view: a new query not saved until the first message
  busy: false,
  poll: null,          // timer while an advisor attends the conversation
};

// While an advisor attends a conversation, check for their replies this often.
const ADVISOR_POLL_MS = 5000;

// Most common questions (from the Datathon call transcripts), one click away.
const QUICK_ACTIONS = ["qa_card_balance", "qa_savings_balance", "qa_available", "qa_overdue",
  "qa_products"];

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
// Auth: "dev" (X-User-Id header), "password" (session cookie the page can't read) or
// "oidc" (authorization code + PKCE)
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
  if (state.config.auth_mode === "password") {
    const res = await fetch("/v1/me");  // is there a session cookie still valid?
    return res.ok ? { header: {}, user: (await res.json()).user_id } : null;
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
  if (state.config.auth_mode === "password") {
    try { await fetch("/v1/auth/logout", { method: "POST" }); } catch { /* offline: the session expires on its own */ }
  }
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
  stopPolling();
  $("chat-title").textContent = t("start");
  $("user-name").textContent = "";
  $("input").value = "";
  messagesEl().replaceChildren(el("div", { class: "empty muted" }, t("empty_state")));
  state.current = null;
  state.draft = false;
  state.me = null;
  setBusy(false);
}

function showLogin(errorMessage) {
  $("app").hidden = true;
  $("login").hidden = false;
  const mode = state.config.auth_mode;
  $("dev-login").hidden = mode !== "dev";
  $("password-login").hidden = mode !== "password";
  $("oidc-login").hidden = mode !== "oidc";
  $("login-password").value = "";
  $("login-hint").textContent = t(`login_hint_${mode}`);
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

// ---------------------------------------------------------------------------
// Markdown for the assistant's replies: a small subset (paragraphs, headings, lists,
// **bold**, *italic*, `code`), built as DOM nodes with text nodes only - never HTML.
// ---------------------------------------------------------------------------
const INLINE = /\*\*(.+?)\*\*|__(.+?)__|`([^`]+)`|(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])/g;

function inline(text) {
  const out = [];
  let last = 0;
  for (const m of text.matchAll(INLINE)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1] !== undefined || m[2] !== undefined) out.push(el("strong", {}, ...inline(m[1] ?? m[2])));
    else if (m[3] !== undefined) out.push(el("code", {}, m[3]));
    else out.push(el("em", {}, ...inline(m[4])));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function withBreaks(lines) {
  return lines.flatMap((line, i) => (i ? [el("br"), ...inline(line)] : inline(line)));
}

function renderMarkdown(text) {
  const frag = document.createDocumentFragment();
  let para = [];
  let list = null;
  const flush = () => {
    if (para.length) frag.append(el("p", {}, ...withBreaks(para)));
    para = [];
  };
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flush();
      const tag = bullet ? "ul" : "ol";
      if (!list || list.tagName.toLowerCase() !== tag) {
        list = el(tag);
        if (numbered && numbered[1] !== "1") list.setAttribute("start", numbered[1]);
        frag.append(list);
      }
      list.append(el("li", {}, ...inline(bullet ? bullet[1] : numbered[2])));
      continue;
    }
    if (!line.trim()) { flush(); list = null; continue; }
    list = null;
    if (heading) { flush(); frag.append(el(`h${Math.min(heading[1].length + 2, 6)}`, {}, ...inline(heading[2]))); continue; }
    para.push(line);
  }
  flush();
  return frag;
}

function addBubble(role, text = "") {
  const bubble = el("div", { class: "bubble" }, text);
  if (role === "assistant") setMarkdown(bubble, text);
  messagesEl().append(el("div", { class: `msg ${role}` }, bubble));
  scrollToBottom();
  return bubble;
}

// Assistant bubbles keep their raw text (streamed deltas are appended to it) and
// are re-rendered from it.
function setMarkdown(bubble, text) {
  bubble.dataset.raw = text;
  bubble.classList.add("md");
  bubble.replaceChildren(renderMarkdown(text));
}

function addNote(text) {
  messagesEl().append(el("div", { class: "note" }, text));
  scrollToBottom();
}

// Progress while BankBot works: an animated line ("Consultando sus productos..."),
// never the tools' names, arguments or raw results.
function showStatus(text) {
  let node = messagesEl().querySelector(".status");
  if (!node) {
    node = el("div", { class: "status", role: "status" },
      el("span", { class: "status-text" }),
      el("span", { class: "dots", "aria-hidden": "true" }, el("i"), el("i"), el("i")));
  }
  node.querySelector(".status-text").textContent = text;
  messagesEl().append(node);  // keep it as the last element
  scrollToBottom();
}

function hideStatus() {
  messagesEl().querySelector(".status")?.remove();
}

function statusFor(tool) {
  const key = `status_${tool}`;
  const text = t(key);
  return text === key ? t("status_consulting") : text;
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
      { decision });
  };
  approve.addEventListener("click", () => decide("approve"));
  reject.addEventListener("click", () => decide("reject"));
  card.append(el("div", { class: "actions" }, approve, reject));
  messagesEl().append(card);
  scrollToBottom();
}

// Offered when the customer insists without being resolved: talk to a human advisor?
function addHandoffOffer(handoffId) {
  const card = el("div", { class: "approval handoff", "data-open": "" },
    el("strong", {}, t("handoff_offer_title")),
    el("span", {}, t("handoff_offer_text")));
  const yes = el("button", { class: "primary" }, t("handoff_yes"));
  const no = el("button", {}, t("handoff_no"));
  yes.disabled = no.disabled = state.busy;
  const answer = guard(async (accept) => {
    if (state.busy) return;
    card.removeAttribute("data-open");
    yes.disabled = no.disabled = true;
    const conversationId = state.current.id;
    await api(`/v1/conversations/${conversationId}/handoff`, {
      method: "POST", body: { handoff_id: handoffId, accept, language: LANG } });
    if (accept) await openConversation(conversationId);  // shows the advisor mode
    else card.append(el("span", { class: "muted" }, t("handoff_declined")));
  });
  yes.addEventListener("click", () => answer(true));
  no.addEventListener("click", () => answer(false));
  card.append(el("div", { class: "actions" }, yes, no));
  messagesEl().append(card);
  scrollToBottom();
}

// A message written by a human advisor, labeled as such.
function addAdvisorBubble(text) {
  const bubble = addBubble("assistant", text);
  bubble.parentElement.classList.add("advisor");
  bubble.prepend(el("span", { class: "author" }, t("advisor")));
  return bubble;
}

// While an advisor attends the conversation, reload it to show their replies.
function startPolling() {
  if (state.poll || !state.current) return;
  const id = state.current.id;
  state.poll = setInterval(async () => {
    if (!state.current || state.current.id !== id) return stopPolling();
    if (state.busy) return;
    try {
      const conv = await (await api(`/v1/conversations/${id}`)).json();
      if (!state.current || state.current.id !== id || state.busy) return;
      if (conv.messages.length !== state.current.seen) {
        renderHistory(conv.messages, conv.pending_actions, conv.handoff);
        state.current.seen = conv.messages.length;
      }
      if (!conv.handoff || conv.handoff.status !== "open") stopPolling();
    } catch { /* keep trying on the next tick */ }
  }, ADVISOR_POLL_MS);
}

function stopPolling() {
  clearInterval(state.poll);
  state.poll = null;
}

// Render stored history: only what the customer said and BankBot (or an advisor)
// answered. Tool calls and their results stay behind the scenes.
function renderHistory(messages, pendingActions, handoff) {
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
        }
      }
      continue;
    }
    const blocks = typeof content === "string" ? [{ type: "text", text: content }] : content;
    for (const block of blocks) {
      if (block.type !== "text" || !block.text.trim()) continue;
      if (message.author === "operator") addAdvisorBubble(block.text);
      else addBubble("assistant", block.text);
    }
  }
  for (const action of pendingActions || []) addApprovalCard(action);
  if (handoff && handoff.status === "offered") addHandoffOffer(handoff.id);
  if (handoff && handoff.status === "open") {
    addNote(t("handoff_waiting"));
    startPolling();
  }
  if (!messages.length) box.append(el("div", { class: "empty muted" }, t("empty_agent")));
  scrollToBottom();
}

function defaultTitle() {
  return t("title_query");
}

function renderConversationList() {
  const list = $("conversation-list");
  list.replaceChildren();
  for (const conv of state.conversations) {
    const button = el("button", { onclick: () => openConversation(conv.id) },
      el("span", { class: "title" }, conv.title || defaultTitle()));
    if (state.current && state.current.id === conv.id) button.setAttribute("aria-current", "true");
    list.append(el("li", {}, button));
  }
}

function setBusy(busy) {
  state.busy = busy;
  $("new-agent").disabled = busy;
  // Approval and advisor-offer buttons wait until the current reply has finished streaming.
  for (const button of document.querySelectorAll(".approval[data-open] button")) button.disabled = busy;
  for (const button of document.querySelectorAll(".quick-actions button")) button.disabled = busy;
  const disabled = busy || (!state.current && !state.draft);
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
  stopPolling();
  state.current = { id: conv.id, title: conv.title, seen: conv.messages.length };
  state.draft = false;
  $("chat-title").textContent = conv.title || defaultTitle();
  renderHistory(conv.messages, conv.pending_actions, conv.handoff);
  renderConversationList();
  setBusy(false);
}

// The welcome view: a new query with BankBot's greeting and quick actions. Nothing is
// saved until the customer sends a message or picks an action.
function showWelcome() {
  if (state.busy) return;
  $("app").classList.remove("sidebar-open");
  stopPolling();
  state.current = null;
  state.draft = true;
  $("chat-title").textContent = t("title_query");
  messagesEl().replaceChildren();
  const name = state.me && state.me.first_name;
  addBubble("assistant greeting", name ? t("greeting", { name }) : t("greeting_anon"));
  const buttons = QUICK_ACTIONS.map((key) =>
    el("button", { class: "quick", type: "button", onclick: guard(() => sendText(t(key))) }, t(key)));
  messagesEl().append(el("div", { class: "quick-actions", role: "group",
    "aria-label": t("quick_title") }, ...buttons));
  renderConversationList();
  setBusy(false);
}

// Re-render texts that were built in JavaScript after a language change.
function refreshLanguage() {
  if (!$("login").hidden) {
    showLogin($("login-error").hidden ? undefined : $("login-error").textContent);
    return;
  }
  renderConversationList();
  if (state.draft) {
    showWelcome();
  } else if (state.current) {
    $("chat-title").textContent = state.current.title || defaultTitle();
  } else {
    $("chat-title").textContent = t("start");
  }
}

// Stream one reply (a message or an approval decision).
async function runStream(path, body) {
  setBusy(true);
  let bubble = null;
  let failed = false;
  showStatus(t("status_thinking"));
  try {
    // The assistant replies in the language the customer is looking at.
    await streamEvents(path, { ...body, language: LANG }, (name, data) => {
      switch (name) {
        case "delta":
          hideStatus();
          if (!bubble) bubble = addBubble("assistant");
          setMarkdown(bubble, bubble.dataset.raw + data.text);
          scrollToBottom();
          break;
        case "tool_call":
          if (bubble && !bubble.textContent) bubble.parentElement.remove();
          bubble = null;
          showStatus(statusFor(data.name));
          break;
        case "tool_result":
          break;  // progress only: the result reaches the customer through the answer
        case "approval_required":
          hideStatus();
          addApprovalCard(data);
          break;
        case "handoff_offer":
          hideStatus();
          addHandoffOffer(data.handoff_id);
          break;
        case "handoff":  // the reply above says an advisor will take over
        case "human_waiting":
          hideStatus();
          if (name === "human_waiting") addNote(t("handoff_waiting"));
          startPolling();
          break;
        case "refusal":
          hideStatus();
          addNote(data.message);
          break;
        case "error":
          hideStatus();
          failed = true;
          addError(data.message);
          break;
        case "done":
          break;
      }
    });
  } catch (err) {
    failed = true;
    addError(err.message || t("connection_lost"));
  } finally {
    hideStatus();
    if (bubble && !bubble.textContent) bubble.parentElement.remove();
    setBusy(false);
    if (state.current) state.current.seen = undefined;  // the next poll re-renders
    if (!failed) loadConversations().catch(() => {});  // refresh titles/order
  }
}

async function send(event) {
  event.preventDefault();
  const input = $("input");
  const text = input.value.trim();
  if (!text || state.busy || (!state.current && !state.draft)) return;
  input.value = "";
  autoResize();
  await sendText(text);
}

async function sendText(text) {
  if (!text || state.busy || (!state.current && !state.draft)) return;
  if (state.draft) {  // first message of a new query: create it now
    setBusy(true);
    let conv;
    try {
      conv = await (await api("/v1/conversations", { method: "POST", body: {} })).json();
    } finally {
      setBusy(false);
    }
    state.current = { id: conv.id, title: conv.title, seen: 0 };
    state.draft = false;
    state.conversations.unshift(conv);
    renderConversationList();
  }
  messagesEl().querySelector(".empty")?.remove();
  messagesEl().querySelector(".quick-actions")?.remove();
  addBubble("user", text);
  await runStream(`/v1/conversations/${state.current.id}/agent-runs`, { text });
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
  $("change-password").hidden = state.config.auth_mode !== "password";
  await loadConversations();
  try {
    state.me = await (await api("/v1/me")).json();
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return;  // already back at login
    state.me = null;  // greet without a name
  }
  showWelcome();  // every session starts with a new query and BankBot's greeting
}

async function passwordLogin() {
  const res = await fetch("/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: $("login-user").value, password: $("login-password").value }),
  });
  $("login-password").value = "";
  if (res.status === 429) throw new Error(t("too_many_attempts"));
  if (!res.ok) throw new Error(t("invalid_credentials"));
  state.auth = { header: {}, user: (await res.json()).user_id };
  await showApp();
}

async function changePassword() {
  const error = $("password-error");
  error.hidden = true;
  const res = await fetch("/v1/auth/password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ current_password: $("current-password").value,
                           new_password: $("new-password").value }),
  });
  if (res.ok) {
    $("password-dialog").close();
    $("password-form").reset();
    return;
  }
  const key = { 401: "session_expired", 403: "wrong_current_password", 422: "password_rule",
                429: "too_many_attempts" }[res.status] || "password_not_changed";
  error.textContent = t(key);
  error.hidden = false;
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
  $("password-login").addEventListener("submit", (event) => {
    event.preventDefault();
    passwordLogin().catch((err) => showLogin(err.message));
  });
  $("change-password").addEventListener("click", () => {
    $("password-form").reset();
    $("password-error").hidden = true;
    $("password-dialog").showModal();
  });
  $("password-cancel").addEventListener("click", () => $("password-dialog").close());
  $("password-form").addEventListener("submit", (event) => {
    event.preventDefault();
    changePassword().catch((err) => {
      $("password-error").textContent = err.message;
      $("password-error").hidden = false;
    });
  });
  $("oidc-login").addEventListener("click", () => startOidcLogin().catch((err) => showLogin(err.message)));
  $("logout").addEventListener("click", () => { logout(); });
  $("new-agent").addEventListener("click", guard(() => showWelcome()));
  for (const select of document.querySelectorAll("select.lang-select")) {
    select.value = LANG;
    select.addEventListener("change", () => {
      setLanguage(select.value);
      refreshLanguage();
    });
  }
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
