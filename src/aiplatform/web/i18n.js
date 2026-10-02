// UI translations. The language follows the browser (Spanish, Portuguese, else English)
// until the user picks one in the selector; that choice is remembered in this browser.
"use strict";

const I18N = {
  es: {
    app_title: "BankBot",
    language: "Idioma",
    login_hint: "Inicie sesión para continuar.",
    login_hint_dev: "Modo desarrollo local: elija cualquier usuario.",
    login_hint_oidc: "Inicie sesión con su cuenta para continuar.",
    login_hint_password: "Ingrese su usuario y contraseña.",
    username: "Usuario",
    password: "Contraseña",
    invalid_credentials: "Usuario o contraseña incorrectos.",
    too_many_attempts: "Demasiados intentos. Espere un momento e intente de nuevo.",
    change_password: "Cambiar contraseña",
    current_password: "Contraseña actual",
    new_password: "Nueva contraseña",
    password_rule: "Al menos 12 caracteres, distinta de su usuario.",
    wrong_current_password: "La contraseña actual no es correcta.",
    password_not_changed: "No se pudo cambiar la contraseña.",
    cancel: "Cancelar",
    save: "Guardar",
    username_label: "Usuario (modo desarrollo)",
    continue: "Continuar",
    sign_in: "Iniciar sesión",
    sign_out: "Cerrar sesión",
    conversations: "Conversaciones",
    new_query: "+ Nueva consulta",
    title_query: "Nueva consulta",
    start: "Inicie una conversación",
    empty_state: "Elija Nueva consulta para preguntar por sus productos o por cualquier "
      + "duda bancaria.",
    empty_agent: "Pregunte por el saldo de sus cuentas, tarjetas o préstamos.",
    greeting: "¡Hola, {name}! Soy BankBot. ¿En qué puedo ayudarte?",
    greeting_anon: "¡Hola! Soy BankBot. ¿En qué puedo ayudarte?",
    quick_title: "Consultas frecuentes",
    qa_card_balance: "Saldo de mi tarjeta de crédito",
    qa_savings_balance: "Saldo de mi cuenta de ahorros",
    qa_available: "¿Cuánto crédito disponible tengo?",
    qa_overdue: "¿Tengo pagos atrasados?",
    qa_products: "¿Qué productos tengo?",
    status_thinking: "Procesando su consulta",
    status_consulting: "Consultando",
    status_get_products: "Consultando sus productos",
    status_get_customer_profile: "Consultando sus datos",
    placeholder: "Escriba su mensaje…",
    send: "Enviar",
    retry: "Reintentar",
    approval_needed: "Se necesita su aprobación",
    wants_to_run: "El asistente quiere ejecutar {tool} con:",
    approve: "Aprobar",
    reject: "Rechazar",
    approved: "Aprobado.",
    rejected: "Rechazado.",
    advisor: "Asesor",
    handoff_offer_title: "¿Desea hablar con un asesor?",
    handoff_offer_text: "Parece que no hemos podido resolver su solicitud. Un asesor puede "
      + "atenderle en esta misma conversación.",
    handoff_yes: "Sí, con un asesor",
    handoff_no: "No, gracias",
    handoff_declined: "Seguimos conversando con BankBot.",
    handoff_waiting: "Un asesor le responderá aquí. Puede dejar sus mensajes.",
    session_expired: "Su sesión expiró. Inicie sesión nuevamente.",
    connection_lost: "Se perdió la conexión.",
    could_not_start: "No se pudo iniciar: {error}",
    idp_unavailable: "El proveedor de identidad no está disponible",
    login_failed: "No se pudo iniciar sesión: {detail}",
  },
  pt: {
    app_title: "BankBot",
    language: "Idioma",
    login_hint: "Entre para continuar.",
    login_hint_dev: "Modo de desenvolvimento local: escolha qualquer usuário.",
    login_hint_oidc: "Entre com a sua conta para continuar.",
    login_hint_password: "Informe o seu usuário e a sua senha.",
    username: "Usuário",
    password: "Senha",
    invalid_credentials: "Usuário ou senha incorretos.",
    too_many_attempts: "Muitas tentativas. Aguarde um momento e tente novamente.",
    change_password: "Alterar senha",
    current_password: "Senha atual",
    new_password: "Nova senha",
    password_rule: "Pelo menos 12 caracteres, diferente do seu usuário.",
    wrong_current_password: "A senha atual não está correta.",
    password_not_changed: "Não foi possível alterar a senha.",
    cancel: "Cancelar",
    save: "Salvar",
    username_label: "Usuário (modo de desenvolvimento)",
    continue: "Continuar",
    sign_in: "Entrar",
    sign_out: "Sair",
    conversations: "Conversas",
    new_query: "+ Nova consulta",
    title_query: "Nova consulta",
    start: "Inicie uma conversa",
    empty_state: "Escolha Nova consulta para perguntar sobre os seus produtos ou qualquer "
      + "dúvida bancária.",
    empty_agent: "Pergunte sobre o saldo das suas contas, cartões ou empréstimos.",
    greeting: "Olá, {name}! Sou o BankBot. Como posso ajudar você?",
    greeting_anon: "Olá! Sou o BankBot. Como posso ajudar você?",
    quick_title: "Consultas frequentes",
    qa_card_balance: "Saldo do meu cartão de crédito",
    qa_savings_balance: "Saldo da minha conta poupança",
    qa_available: "Quanto crédito disponível eu tenho?",
    qa_overdue: "Tenho pagamentos em atraso?",
    qa_products: "Quais produtos eu tenho?",
    status_thinking: "Processando a sua consulta",
    status_consulting: "Consultando",
    status_get_products: "Consultando os seus produtos",
    status_get_customer_profile: "Consultando os seus dados",
    placeholder: "Escreva a sua mensagem…",
    send: "Enviar",
    retry: "Tentar novamente",
    approval_needed: "É necessária a sua aprovação",
    wants_to_run: "O assistente quer executar {tool} com:",
    approve: "Aprovar",
    reject: "Rejeitar",
    approved: "Aprovado.",
    rejected: "Rejeitado.",
    advisor: "Atendente",
    handoff_offer_title: "Deseja falar com um atendente?",
    handoff_offer_text: "Parece que não conseguimos resolver a sua solicitação. Um atendente "
      + "pode ajudar nesta mesma conversa.",
    handoff_yes: "Sim, com um atendente",
    handoff_no: "Não, obrigado",
    handoff_declined: "Continuamos a conversa com o BankBot.",
    handoff_waiting: "Um atendente responderá aqui. Pode deixar as suas mensagens.",
    session_expired: "A sua sessão expirou. Entre novamente.",
    connection_lost: "A conexão foi perdida.",
    could_not_start: "Não foi possível iniciar: {error}",
    idp_unavailable: "O provedor de identidade não está disponível",
    login_failed: "Não foi possível entrar: {detail}",
  },
  en: {
    app_title: "BankBot",
    language: "Language",
    login_hint: "Sign in to continue.",
    login_hint_dev: "Local development mode: pick any username.",
    login_hint_oidc: "Sign in with your account to continue.",
    login_hint_password: "Enter your username and password.",
    username: "Username",
    password: "Password",
    invalid_credentials: "Wrong username or password.",
    too_many_attempts: "Too many attempts. Wait a moment and try again.",
    change_password: "Change password",
    current_password: "Current password",
    new_password: "New password",
    password_rule: "At least 12 characters, different from your username.",
    wrong_current_password: "The current password is wrong.",
    password_not_changed: "The password could not be changed.",
    cancel: "Cancel",
    save: "Save",
    username_label: "Username (dev mode)",
    continue: "Continue",
    sign_in: "Sign in",
    sign_out: "Sign out",
    conversations: "Conversations",
    new_query: "+ New query",
    title_query: "New query",
    start: "Start a conversation",
    empty_state: "Choose New query to ask about your products or any banking question.",
    empty_agent: "Ask about the balance of your accounts, cards or loans.",
    greeting: "Hi, {name}! I'm BankBot. How can I help you?",
    greeting_anon: "Hi! I'm BankBot. How can I help you?",
    quick_title: "Frequent questions",
    qa_card_balance: "My credit card balance",
    qa_savings_balance: "My savings account balance",
    qa_available: "How much credit do I have available?",
    qa_overdue: "Do I have overdue payments?",
    qa_products: "What products do I have?",
    status_thinking: "Working on your request",
    status_consulting: "Checking",
    status_get_products: "Checking your products",
    status_get_customer_profile: "Checking your details",
    placeholder: "Send a message…",
    send: "Send",
    retry: "Retry",
    approval_needed: "Approval needed",
    wants_to_run: "The assistant wants to run {tool} with:",
    approve: "Approve",
    reject: "Reject",
    approved: "Approved.",
    rejected: "Rejected.",
    advisor: "Advisor",
    handoff_offer_title: "Would you like to talk to an advisor?",
    handoff_offer_text: "It looks like we couldn't resolve your request. An advisor can help "
      + "you in this same conversation.",
    handoff_yes: "Yes, an advisor",
    handoff_no: "No, thanks",
    handoff_declined: "We'll keep going with BankBot.",
    handoff_waiting: "An advisor will reply here. You can leave your messages.",
    session_expired: "Your session expired. Please sign in again.",
    connection_lost: "Connection lost.",
    could_not_start: "Could not start: {error}",
    idp_unavailable: "Identity provider unavailable",
    login_failed: "Login failed: {detail}",
  },
};

const LANGUAGES = { es: "Español", pt: "Português", en: "English" };
const LANG_KEY = "aip.lang";

function savedLanguage() {
  try {
    const lang = localStorage.getItem(LANG_KEY);
    return lang in I18N ? lang : null;
  } catch {
    return null;  // storage blocked (private mode, etc.): fall back to the browser
  }
}

function browserLanguage() {
  for (const tag of navigator.languages || [navigator.language || "en"]) {
    const base = tag.toLowerCase().split("-")[0];
    if (base in I18N) return base;
  }
  return "en";
}

let LANG = savedLanguage() || browserLanguage();

function t(key, vars = {}) {
  const text = I18N[LANG][key] ?? I18N.en[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (_, name) => String(vars[name] ?? ""));
}

function setLanguage(lang) {
  if (!(lang in I18N)) return;
  LANG = lang;
  try { localStorage.setItem(LANG_KEY, lang); } catch { /* not persisted */ }
  applyTranslations();
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
  for (const select of root.querySelectorAll("select.lang-select")) select.value = LANG;
}
