// UI translations. The language follows the browser: Spanish, Portuguese, else English.
"use strict";

const I18N = {
  es: {
    app_title: "Asistente del banco",
    login_hint: "Inicie sesión para continuar.",
    login_hint_dev: "Modo desarrollo local: elija cualquier usuario.",
    login_hint_oidc: "Inicie sesión con su cuenta para continuar.",
    username_label: "Usuario (modo desarrollo)",
    continue: "Continuar",
    sign_in: "Iniciar sesión",
    sign_out: "Cerrar sesión",
    conversations: "Conversaciones",
    new_query: "+ Nueva consulta",
    new_chat: "+ Pregunta general",
    title_query: "Nueva consulta",
    title_chat: "Pregunta general",
    kind_agent: "consulta",
    kind_chat: "general",
    start: "Inicie una conversación",
    empty_state: "Elija Nueva consulta para preguntar por los saldos de sus productos, o "
      + "Pregunta general para dudas generales.",
    empty_agent: "Pregunte por el saldo de sus cuentas, tarjetas o préstamos.",
    empty_chat: "Haga su pregunta.",
    placeholder: "Escriba su mensaje…",
    send: "Enviar",
    retry: "Reintentar",
    approval_needed: "Se necesita su aprobación",
    wants_to_run: "El asistente quiere ejecutar {tool} con:",
    approve: "Aprobar",
    reject: "Rechazar",
    approved: "Aprobado.",
    rejected: "Rechazado.",
    session_expired: "Su sesión expiró. Inicie sesión nuevamente.",
    connection_lost: "Se perdió la conexión.",
    could_not_start: "No se pudo iniciar: {error}",
    idp_unavailable: "El proveedor de identidad no está disponible",
    login_failed: "No se pudo iniciar sesión: {detail}",
  },
  pt: {
    app_title: "Assistente do banco",
    login_hint: "Entre para continuar.",
    login_hint_dev: "Modo de desenvolvimento local: escolha qualquer usuário.",
    login_hint_oidc: "Entre com a sua conta para continuar.",
    username_label: "Usuário (modo de desenvolvimento)",
    continue: "Continuar",
    sign_in: "Entrar",
    sign_out: "Sair",
    conversations: "Conversas",
    new_query: "+ Nova consulta",
    new_chat: "+ Pergunta geral",
    title_query: "Nova consulta",
    title_chat: "Pergunta geral",
    kind_agent: "consulta",
    kind_chat: "geral",
    start: "Inicie uma conversa",
    empty_state: "Escolha Nova consulta para perguntar sobre os saldos dos seus produtos, ou "
      + "Pergunta geral para dúvidas gerais.",
    empty_agent: "Pergunte sobre o saldo das suas contas, cartões ou empréstimos.",
    empty_chat: "Faça a sua pergunta.",
    placeholder: "Escreva a sua mensagem…",
    send: "Enviar",
    retry: "Tentar novamente",
    approval_needed: "É necessária a sua aprovação",
    wants_to_run: "O assistente quer executar {tool} com:",
    approve: "Aprovar",
    reject: "Rejeitar",
    approved: "Aprovado.",
    rejected: "Rejeitado.",
    session_expired: "A sua sessão expirou. Entre novamente.",
    connection_lost: "A conexão foi perdida.",
    could_not_start: "Não foi possível iniciar: {error}",
    idp_unavailable: "O provedor de identidade não está disponível",
    login_failed: "Não foi possível entrar: {detail}",
  },
  en: {
    app_title: "Bank assistant",
    login_hint: "Sign in to continue.",
    login_hint_dev: "Local development mode: pick any username.",
    login_hint_oidc: "Sign in with your account to continue.",
    username_label: "Username (dev mode)",
    continue: "Continue",
    sign_in: "Sign in",
    sign_out: "Sign out",
    conversations: "Conversations",
    new_query: "+ New query",
    new_chat: "+ General question",
    title_query: "New query",
    title_chat: "General question",
    kind_agent: "query",
    kind_chat: "general",
    start: "Start a conversation",
    empty_state: "Choose New query to ask about your product balances, or General question "
      + "for general questions.",
    empty_agent: "Ask about the balance of your accounts, cards or loans.",
    empty_chat: "Ask anything.",
    placeholder: "Send a message…",
    send: "Send",
    retry: "Retry",
    approval_needed: "Approval needed",
    wants_to_run: "The assistant wants to run {tool} with:",
    approve: "Approve",
    reject: "Reject",
    approved: "Approved.",
    rejected: "Rejected.",
    session_expired: "Your session expired. Please sign in again.",
    connection_lost: "Connection lost.",
    could_not_start: "Could not start: {error}",
    idp_unavailable: "Identity provider unavailable",
    login_failed: "Login failed: {detail}",
  },
};

const LANG = (() => {
  for (const tag of navigator.languages || [navigator.language || "en"]) {
    const base = tag.toLowerCase().split("-")[0];
    if (base in I18N) return base;
  }
  return "en";
})();

function t(key, vars = {}) {
  const text = I18N[LANG][key] ?? I18N.en[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (_, name) => String(vars[name] ?? ""));
}

// Static texts: data-i18n (text), data-i18n-placeholder, data-i18n-aria-label.
function applyTranslations(root = document) {
  document.documentElement.lang = LANG;
  document.title = t("app_title");
  for (const node of root.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  for (const node of root.querySelectorAll("[data-i18n-placeholder]")) {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  }
  for (const node of root.querySelectorAll("[data-i18n-aria-label]")) {
    node.setAttribute("aria-label", t(node.dataset.i18nAriaLabel));
  }
}
