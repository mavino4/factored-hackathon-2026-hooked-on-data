"""Speech to text and text to speech with OpenAI (gpt-4o-mini-transcribe, gpt-4o-mini-tts),
through the official SDK. Needs OPENAI_API_KEY. The customer's audio is sent to OpenAI
and nowhere else: it is not stored, logged or traced (only its transcript is, like any
typed message).
"""

import time
from dataclasses import dataclass

STT_MODEL = "gpt-4o-mini-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
VOICE = "coral"

# List prices, USD. Transcription is billed per token (audio in, text out); speech returns
# no usage, so its cost is estimated from the text at ~15 characters per spoken second.
STT_USD_PER_MTOK = {"audio": 3.0, "text": 1.25, "output": 5.0}
TTS_USD_PER_MINUTE = 0.015
CHARS_PER_SECOND = 15

# Words a bank customer says that a general model may mishear.
VOCABULARY = {
    "es": "Consulta bancaria: saldo, cupo, tarjeta de crédito, CDT, cuenta de ahorros, "
          "extracto, crédito de libre inversión, asesor.",
    "pt": "Consulta bancária: saldo, limite, cartão de crédito, CDB, conta poupança, "
          "extrato, empréstimo, atendente.",
    "en": "Banking question: balance, credit limit, credit card, CD, savings account, "
          "statement, loan, advisor.",
}

TONE = ("Speak as a friendly, calm bank advisor: clear, warm and unhurried. Read amounts "
        "and dates naturally.")

# What the browser records: MIME type (without codec parameters) -> file name for the API.
AUDIO_TYPES = {
    "audio/webm": "speech.webm",
    "audio/ogg": "speech.ogg",
    "audio/mp4": "speech.mp4",
    "audio/x-m4a": "speech.m4a",
    "audio/aac": "speech.aac",
    "audio/mpeg": "speech.mp3",
    "audio/wav": "speech.wav",
    "audio/x-wav": "speech.wav",
}


@dataclass
class Transcript:
    text: str
    model: str
    seconds: float  # how long the API took
    usage: dict[str, int]  # tokens: audio and text in, text out
    cost_usd: float


@dataclass
class Speech:
    audio: bytes
    mime: str
    model: str
    chars: int
    seconds: float  # how long the API took
    cost_usd: float  # estimated (see TTS_USD_PER_MINUTE)


def base_type(content_type: str | None) -> str:
    return (content_type or "").split(";")[0].strip().lower()


def transcription_cost(usage: dict[str, int]) -> float:
    return (usage.get("audio", 0) * STT_USD_PER_MTOK["audio"]
            + usage.get("text", 0) * STT_USD_PER_MTOK["text"]
            + usage.get("output", 0) * STT_USD_PER_MTOK["output"]) / 1e6


def speech_cost(chars: int) -> float:
    return chars / CHARS_PER_SECOND / 60 * TTS_USD_PER_MINUTE


class SpeechClient:
    def __init__(self, api_key: str, *, stt_model: str = STT_MODEL,
                 tts_model: str = TTS_MODEL, voice: str = VOICE, timeout: float = 20.0,
                 attempts: int = 2, http_client=None):
        from openai import AsyncOpenAI

        # The SDK retries connection errors, 429 and 5xx itself, with backoff.
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout,
                                   max_retries=attempts - 1, http_client=http_client)
        self.stt_model, self.tts_model, self.voice = stt_model, tts_model, voice

    async def transcribe(self, audio: bytes, content_type: str,
                         language: str | None = None) -> Transcript:
        mime = base_type(content_type)
        extra = {"language": language, "prompt": VOCABULARY[language]} \
            if language in VOCABULARY else {}
        started = time.perf_counter()
        result = await self._client.audio.transcriptions.create(
            model=self.stt_model, file=(AUDIO_TYPES.get(mime, "speech.webm"), audio, mime),
            response_format="json", **extra)
        usage = _usage(getattr(result, "usage", None))
        return Transcript(text=result.text.strip(), model=self.stt_model,
                          seconds=time.perf_counter() - started, usage=usage,
                          cost_usd=transcription_cost(usage))

    async def synthesize(self, text: str, *, voice: str | None = None,
                         instructions: str = TONE) -> Speech:
        started = time.perf_counter()
        response = await self._client.audio.speech.create(
            model=self.tts_model, voice=voice or self.voice, input=text,
            instructions=instructions, response_format="mp3")
        return Speech(audio=response.content, mime="audio/mpeg", model=self.tts_model,
                      chars=len(text), seconds=time.perf_counter() - started,
                      cost_usd=speech_cost(len(text)))

    async def close(self) -> None:
        await self._client.close()


def _usage(usage) -> dict[str, int]:
    if usage is None or getattr(usage, "type", None) != "tokens":
        return {}
    details = usage.input_token_details
    audio = (details.audio_tokens or 0) if details else usage.input_tokens
    text = (details.text_tokens or 0) if details else 0
    return {"audio": audio, "text": text, "output": usage.output_tokens}
