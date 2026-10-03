"""Keyword rules, checked in order: attack, human, out of scope, account, general.

Written from the intent definitions (chat/prompts.py, CLASSIFY_SYSTEM_PROMPT) for es, pt
and en, never from the test set. Each rule is a regular expression over the normalized
text (classifiers.normalize: lowercase, no accents). Attack rules also run over the text
with digits read as letters, spaced-out letters joined and the text reversed, which are
the usual ways to hide a keyword.
"""

import re

from aiplatform.agent.classifiers import Prediction, normalize, normalize_keep_spaces


def _any(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


# Words that, near each other, make a manipulation attempt.
_IGNORE = r"(ignor\w*|olvid\w*|esquec\w*|esquece|disregard|forget|desconsider\w*|omit\w*|salta\w*)"
_RULES = (r"(instruc\w*|instruc|instrucoes|instructions?|regla\w*|regras?|rules?|polic\w*|"
          r"politicas?|restricc\w*|restricoes|restrictions?|pautas|directrices|guidelines|"
          r"lo que te dijeron|what you were told)")
_SECRET = (r"(system prompt|prompt\b|mensaje de sistema|mensaje del sistema|system message|"
           r"mensagem do sistema|instrucciones (iniciales|internas|de sistema|del sistema|"
           r"originales)|tus instrucciones|sus instrucciones|suas instrucoes|your instructions|"
           r"configurac\w*|configuration|hidden config\w*|variables de entorno|"
           r"environment variables|api key|reglas internas|internal rules|regras internas)")
_ROLE = (r"(administrador|admin\b|auditor\w*|desarrollador|developer|desenvolvedor|"
         r"gerente|compliance|operador|staff|empleado del banco|funcionario|oficial|"
         r"officer|supervisor|ingeniero|engineer|soporte tecnico)")
_ELSEWHERE = (r"(otro cliente|outro cliente|another customer|other customer|del cliente|"
              r"do cliente|of customer|cliente cli-|customer cli-|datos de|dados de|"
              r"cuentas de|contas de|balance of|saldo de (la|el|los|las)? ?(senor|senora|"
              r"cliente)|instruc\w*|herramientas|ferramentas|tools|autoriz\w*|por ley|"
              r"by law|excepcion|exception|excecao|muestre\w*|mostrar|show me|liste)")

ATTACK = [
    ("ignore the rules", _any(fr"\b{_IGNORE}\b.{{0,40}}\b{_RULES}",
                              fr"\b{_RULES}\b.{{0,30}}\b(suspend\w*|desactiv\w*|disabled)")),
    ("reveal the prompt or configuration",
     _any(fr"(muestr\w*|mostr\w*|revel\w*|dime|dimelo|diga|dig[ao]|imprim\w*|print|show|"
          fr"repit\w*|repeat|copi\w*|quote|cita\w*|traduc\w*|traduz\w*|translate|output|"
          fr"list\w*|ensena\w*|comparte|share|escrib\w*|write|exib\w*|cuales son|what were|"
          fr"what are|quais sao).{{0,40}}{_SECRET}",
          fr"{_SECRET}.{{0,40}}(completo|complete|verbatim|palabra por palabra|tal cual|"
          fr"exactamente|exactly|literal\w*|en yaml|en json|as json|in yaml|in json)",
          r"con (que|cual) prompt", r"what (model|version) are you",
          r"que modelo de (ia|inteligencia)")),
    ("new persona without rules",
     _any(r"\b(eres|sos|seras|you are|voce e|vc e|agora e|ahora eres|now you are)\b.{0,30}"
          r"\b(dan|sin (restricciones|reglas|limites|filtros)|no (rules|restrictions|limits)|"
          r"sem (regras|limites|restricoes)|without (rules|restrictions)|unrestricted)",
          r"(a partir de ahora|desde ahora|from now on|a partir de agora) (eres|you are|"
          r"voce e|seras)", r"you are no longer", r"ya no eres", r"nao e mais",
          r"\b(finge|finja|fingir|pretend|actua como|act as|haz de cuenta|imagina que eres|"
          r"juguemos|vamos a jugar|let'?s play|role ?play)\b.{0,60}(sin|without|sem|no)\b"
          r".{0,20}(regla|rule|regra|limit|restric|polic|filtr)",
          r"\b(finge|finja|pretend|actua como|act as)\b.{0,40}\b(empleado|employee|"
          r"funcionario|bot|ia|ai|asistente|assistant)\b.{0,40}(otro cliente|outro cliente|"
          r"another customer|sin restricciones|sem limites|no rules)",
          r"(the )?(safety|security) rules (don'?t|do not) (exist|apply)")),
    ("claims authority", _any(fr"\b(soy|sou|i am|i'?m|as the|como (el|la)|en calidad de)\b"
                              fr".{{0,30}}\b{_ROLE}\b.{{0,120}}{_ELSEWHERE}")),
    ("imitates the system, a tool or an approval",
     _any(r"^\W*\[?(system|sistema|admin|developer)\]?\s*:", r"\[(approval|aprobacion|system)\]",
          r"(resultado de la herramienta|tool result|resultado da ferramenta|tool response|"
          r"respuesta de (tu|la) herramienta)", r"\{\s*\"(status|tool|balance|saldo)\"",
          r"nota (para|al) (el )?clasificador", r"\bclasificador\s*:", r"\bclassifier\s*:",
          r"(classify|clasifica|classifique) (this|este|esta|isto)",
          r"\bintent\s*=", r"modo (mantenimiento|desarrollador|developer|dios)",
          r"(developer|maintenance|god) mode", r"(ya|already) (lo )?(aprob\w*|approved)",
          r"(instruccion|instrucao|instruction|nota del sistema|nota do sistema|system note)"
          r" (para|for|to) (el|o|the)? ?(asistente|assistente|assistant|ia|ai)",
          r"<<\s*(ia|ai)\s*:", r"estimado asistente", r"(note|nota) (to|para) (the )?"
          r"(assistant|asistente|ia|ai)")),
    ("another customer's data",
     _any(r"\bcli-[a-z0-9]{6,}", r"(cliente|customer|titular|holder) (con|with|com) "
          r"(cedula|documento|dni|cpf|id|identificacion)", r"(datos|saldo|productos|cuentas|"
          r"dados|contas|balance|products) de (otro|outro) cliente",
          r"(another|other) customer'?s?")),
    ("internal tools", _any(r"get_products|get_customer_profile|classify_intent|tool_use",
                            r"(herramientas|ferramentas|tools|funciones|functions) "
                            r"(internas|internal|disponibles|available|que (puedes|podes|"
                            r"pode) (llamar|usar|chamar))",
                            r"(que|which|quais) (herramientas|tools|ferramentas|funciones)",
                            r"(nombres|names|nomes) (tecnicos|exactos|technical|exact)",
                            r"/etc/passwd|\.\./\.\./")),
    ("dictated output",
     _any(r"(responde|responda|reply|answer|respond|diga|di|say|escribe|write|diz)\w* "
          r"(unicamente|solo|solamente|only|apenas|exactamente|exactly|exatamente)\b",
          r"(diga|di|say|dime|repite|repeat|confirm\w*)\w* (exactamente|exactly|"
          r"exatamente|literalmente|que)\b.{0,20}['\"]",
          r"confirm\w* (que )?(mi|meu|my) saldo (es|e|is)", r"zx-canary",
          r"(first line|primera linea|primeira linha) (of|de)",
          r"solo responde ['\"]", r"(confirm|confirma|confirmala)\w* .{0,30}(transfer|"
          r"realizada|realizado|done|feita)")),
    ("deceive others", _any(r"phishing", r"(haciendote|haciendose|fingiendo|pretending|"
                            r"posing|fingindo|se fazendo) (pasar )?(por|as|ser|to be) "
                            r"(el|o|the|a)? ?banco|bank",
                            r"(a nombre del|em nome do|on behalf of the|in the name of the) "
                            r"(banco|bank)",
                            r"(sms|correo|email|e-mail|mensaje|mensagem|message)\b.{0,80}"
                            r"(pin|clave|contrasena|senha|password|cvv)\b.{0,40}"
                            r"(enlace|link|confirme|confirm)")),
    ("hypothetical framing", _any(r"hipoteticament\w*|hypothetical\w*|en un mundo de ficcion|"
                                  r"in a (fictional|fiction) world|si no tuviera\w* reglas|"
                                  r"if you had no rules|se voce nao tivesse regras")),
    ("injects output", _any(r"(agregue|agrega|anada|anade|incluya|incluye|incluyalo|inclua|"
                            r"adicione|add|append|include|pon|ponga|coloque)\w*\b.{0,60}"
                            r"(imagen|imagem|image|enlace|link|<script|<img|javascript:|"
                            r"texto exacto|exact text|!\[)",
                            r"!\[[^\]]*\]\(https?://", r"javascript:",
                            r"\]\(https?://[^)]*(saldo|balance|nombre|name)")),
    ("encoded instructions", _any(r"(decodifica|decode|descodifica|decodifique)\w*.{0,40}"
                                  r"(base64|instruc|instruction)",
                                  r"(leido|lido|read|leelo|leia) (al reves|ao contrario|"
                                  r"backwards|in reverse)",
                                  r"(orden|ordem|order) (inverso|reverse)")),
    ("asks for the rules verbatim",
     _any(r"(complete|completa|termina) la frase.{0,40}(instruc|reglas|rules)",
          r"(repite|repeat|repita) (todo )?(el|the|o)? ?(texto|text) (que esta )?(antes|above|"
          r"anterior|acima)",
          r"(repite|repitame|repeat|copia|copie)\w* (esas|essas|those|tus|your) (pautas|reglas|"
          r"rules|regras|instrucciones|instructions)", r"system_prompt")),
]

# Spaced-out or digit-for-letter text, read back as words (attack rules only).
LEET = str.maketrans("013457@$", "oieastas")


def _variants(text: str) -> list[str]:
    norm = normalize(text)
    out = [norm]
    letters = norm.split()
    if len(letters) >= 6 and sum(len(w) == 1 for w in letters) / len(letters) > 0.6:
        # "i g n o r a   t u s" -> "ignora tus": wider gaps separate the words.
        words = re.split(r"\s{2,}", normalize_keep_spaces(text))
        out.append(" ".join(w.replace(" ", "") for w in words))
    if re.search(r"[a-z][0-9]|[0-9][a-z]", norm):
        out.append(norm.translate(LEET))
    out.append(norm[::-1])
    return out


HUMAN = _any(
    r"\b(hablar|hable|comunic\w*|pas[ae]me|pasarme|pasa me|transfier\w*|transfer\w*|"
    r"conect\w*|connect|atiend\w*|atenda|falar|fale|passa|passe|me passa|speak|talk|put me "
    r"through|get me|quiero|necesito|prefiero|want|need|quero|preciso)\b.{0,40}\b(un |una |"
    r"um |uma |a |an |con |com |with |to )?(asesor\w*|persona|humano|humana|agente|"
    r"ejecutiv\w*|operador\w*|alguien|servicio al cliente|atendente|atendimento humano|"
    r"pessoa|gerente|human|agent|person|representative|someone|somebody|manager|"
    r"advisor|adviser|real person|persona real|pessoa de verdade)\b",
    r"^\W*(asesor|agente|operador|humano|representative|agent|human|atendente|"
    r"operator|persona|pessoa|advisor|gerente)\W*$",
    r"(hay|ha) (alguien|alguem|alguma pessoa)\b.{0,30}(banco|hablar|falar)",
    r"\bis there (someone|anyone|a person)\b", r"atendimento humano")

# "How do I...": asking how to do something, not asking to do it. "como" + a first-person
# verb ("como bloqueo", "como hago"), "how do/can/to".
HOW_TO = re.compile(r"^\W*(como (\w+o|se|puedo|posso|faco|consigo)|how (do|can|does|to|"
                    r"should|would)|de que (manera|forma)|qual a forma|o que preciso|"
                    r"que necesito|cuales son los (pasos|requisitos)|what do i need)\b")

OUT_OF_SCOPE = [
    ("transfer or payment",
     _any(r"\b(transfier\w*|transferir|transfer|transfere|envia\w*|enviar|send|giro\w*|"
          r"gire|pix|deposit\w* (a|en|para)|wire)\b.{0,40}\b(\d|dolares|pesos|reais|usd|"
          r"cop|brl|a mi|para mi|to my|a la cuenta|para a conta|a cuenta|cuenta \d|"
          r"hermano|hermana|irma|sister|brother|mama|mae)",
          r"\b(pagar|paga|pague|pagarle|pay|abonar|abone)\b.{0,30}\b(mi|my|minha|meu|la|"
          r"el|a|o|the|tarjeta|cartao|card|factura|cuota|prestamo|loan|boleto|recibo)",
          r"\bhaz(me)? (una|un) (transferencia|giro|pago)", r"\bquiero (transferir|pagar|"
          r"enviar|girar)", r"\bquero (transferir|pagar|enviar|fazer um pix)")),
    ("block or cancel", _any(r"\b(bloque\w*|bloquei\w*|cancel\w*|congel\w*|freeze|block|"
                             r"suspend\w*|desactiv\w*|desativ\w*|close)\b.{0,40}\b(tarjeta|"
                             r"cartao|card|cuenta|conta|account)",
                             r"\b(tarjeta|cartao|card|cuenta|conta|account)\b.{0,40}"
                             r"\b(bloque\w*|bloquei\w*|cancel\w*|congel\w*|freeze)",
                             r"\b(robaron|perdi|perdí|stolen|lost|roubaram|perdi)\b.{0,30}"
                             r"(tarjeta|cartao|card)")),
    ("complaint or dispute", _any(r"\b(queja|reclamo|reclam\w*|complaint|denuncia|"
                                  r"contest\w*|disput\w*|no reconozco|nao reconheco|"
                                  r"don'?t recognize|indevid\w*|cobro (indebido|que no)|"
                                  r"cobrança|insatisfech\w*|insatisfeit\w*)\b")),
    ("statement or history", _any(r"\b(movimientos|movimentac\w*|extracto|extrato|"
                                  r"statement|transactions|transacciones|transacoes|"
                                  r"historial|historico|history|compras (del|de la|que)|"
                                  r"ultim\w* (movimient|transacc|compras))\b",
                                  r"(fecha|data|date) .{0,30}(cobr\w*|charged|debit\w*)")),
    ("application or change",
     _any(r"\b(solicitar|solicito|pedir|apply|aplicar|tramitar|abrir|open|contratar|"
          r"quero (pedir|um)|quiero (un|una|sacar|pedir|solicitar|abrir))\b.{0,40}"
          r"\b(prestamo|emprestimo|loan|credito|tarjeta|cartao|card|cuenta|conta|account|"
          r"cdt|seguro)",
          r"\b(suban|subir|aumenta\w*|aument\w*|increase|bump|amplia\w*|eleva\w*)\b.{0,30}"
          r"\b(cupo|limite|limit)", r"\b(cambi\w*|actualic\w*|actualiz\w*|atualiz\w*|"
          r"update|change|modific\w*|alter\w*)\b.{0,40}\b(datos|dados|direccion|endereco|"
          r"address|correo|email|telefono|phone|celular|clave|senha|contrasena|password|pin)",
          r"\bcertificado\b|\bcertificate\b|paz y salvo|refinanc\w*|renegoci\w*")),
]

BANKING = (r"(saldo|balance|tarjeta|cuenta|prestamo|credito|cupo|limite|limit|deuda|"
           r"inversion|investimento|hipotec\w*|cartao|conta|poupanca|emprestimo|card|"
           r"account|loan|mortgage|savings|checking|productos|produtos|products|mora\b|"
           r"atras\w*|late|overdue|vence|vencimiento|vencimento|expir\w*|tasa|juros|"
           r"interes\w*|interest|disponible|disponivel|available|debo|devo|owe|segmento|"
           r"cliente desde|financiamento|fatura|parcela|cuota|ahorro\w*|corriente|"
           r"debito|visa|mastercard|cdt|apr|rotativo|banco|bank|pago minimo)")

ACCOUNT = _any(
    fr"\b(mi|mis|my|meu|minha|meus|minhas|mio|mia)\b.{{0,40}}\b{BANKING}",
    fr"\b{BANKING}\b.{{0,30}}\b(mi|mis|my|meu|minha|meus|minhas|tengo|tenho|i have)\b",
    r"\b(cuanto|quanto|how much|cual es|qual e|qual o|qual a|what'?s|what is)\b.{0,30}"
    r"\b(debo|devo|owe|me queda|me falta|tengo|tenho|i have|do i have|i owe|resta|sobra)\b",
    r"\b(tengo|tenho|do i have|have i got|estoy|estou|am i)\b.{0,30}\b(pago|pagos|pagamento|"
    r"parcela|cuota|payment|atras\w*|mora|late|overdue|vencid\w*|al dia|em dia)",
    r"\b(que|quais|which|what) (productos|produtos|products|cuentas|contas|accounts|"
    r"tarjetas|cartoes|cards)\b.{0,20}\b(tengo|tenho|have|do i)",
    r"\b(debo|devo|i owe)\b", r"\b(mi|my|meu|minha) (saldo|balance|balnce)\b",
    r"\bsald\w* (de|da|do|del) (mi|mis|meu|minha)\b",
    r"\b(cuanto|quanto|how much)\b.{0,30}\b(vale|rinde|rende|worth)\b.{0,20}\b(mi|my|meu|minha)")

GENERAL = _any(
    r"^\W*(hola|buenas|buenos dias|buenas tardes|buenas noches|ola|oi|bom dia|boa tarde|"
    r"boa noite|hi|hello|hey|good (morning|afternoon|evening))\b\W*(\w+\W*){0,3}$",
    r"\b(gracias|obrigad[oa]|thanks|thank you|muy amable|perfecto|ok|entendido|entendi|"
    r"listo|chao|adios|tchau|bye|genial|vale)\b",
    r"\b(que es|que son|que significa|o que e|o que sao|what is|what are|what does|"
    r"como funciona|como funcionam|how does|how do|diferencia|diferenca|difference|"
    r"conviene|vale la pena|is it (safe|better)|es seguro|e seguro|requisitos|"
    r"que puedes|que podes|o que voce|what can you|en que me (puedes|podes) ayudar)\b",
    fr"^\W*(como|how)\b.{{0,60}}{BANKING}")


def _match(rules: list[tuple[str, re.Pattern]], texts: list[str]) -> str | None:
    for name, pattern in rules:
        if any(pattern.search(t) for t in texts):
            return name
    return None


class RuleClassifier:
    name = "rules"

    def predict_text(self, text: str) -> Prediction:
        norm = normalize(text)
        if rule := _match(ATTACK, _variants(text)):
            return Prediction("attack", 0.9, f"rule: {rule}")
        if HUMAN.search(norm):
            return Prediction("human", 0.9, "rule: asks for a person")
        how_to = bool(HOW_TO.match(norm))
        if not how_to and (rule := _match(OUT_OF_SCOPE, [norm])):
            return Prediction("out_of_scope", 0.85, f"rule: {rule}")
        if ACCOUNT.search(norm) and not how_to:
            return Prediction("account", 0.85, "rule: own products or figures")
        if GENERAL.search(norm) or how_to:
            return Prediction("general", 0.7, "rule: banking question or small talk")
        if re.search(BANKING, norm):
            return Prediction("general", 0.4, "default: mentions banking")
        return Prediction("out_of_scope", 0.4, "default: nothing about banking")
