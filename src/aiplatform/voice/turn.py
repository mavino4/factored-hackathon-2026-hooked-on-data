"""One voice turn: the customer's audio is transcribed, the transcript goes through the same
agent as a typed message (classifier, tools, approvals, handoff, tracing), and the answer
is spoken sentence by sentence while it is being written. The written answer keeps its
figures ("1.200.000,00 COP"); what is spoken has them in words (voice/numbers.py).

Events, in order: ``Transcribed`` (what was understood), then the agent's own events with
a ``SpokenAudio`` after each sentence as soon as it is synthesized (in order, a few at a
time), and the agent's ``AgentDone`` last, once every sentence has its audio. If nothing
was understood the turn stops with ``VoiceProblem("not_heard")`` and the agent is not
called. If speech fails, the written answer still arrives, with one ``speech_failed``.

The voice never approves anything: an action that needs approval is shown on screen, as
in the chat, and the voice asks the customer to confirm it there.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass

from aiplatform import metrics
from aiplatform.agent.events import AgentDone, AgentText, ApprovalRequired, HandoffOffered
from aiplatform.agent.loop import AgentRunner
from aiplatform.tracing import Tracing
from aiplatform.voice.numbers import spoken as spoken_figures
from aiplatform.voice.sentences import SentenceBuffer
from aiplatform.voice.speech import Speech, SpeechClient, Transcript

log = logging.getLogger(__name__)

# Sentences synthesized at once (order is kept when they are sent).
SPEECH_CONCURRENCY = 3

CONFIRM_ON_SCREEN = {
    "es": "Para continuar, confírmalo en la pantalla.",
    "pt": "Para continuar, confirme na tela.",
    "en": "To continue, please confirm on the screen.",
}
ADVISOR_OFFER = {
    "es": "¿Quieres que te comunique con un asesor? Responde en la pantalla.",
    "pt": "Quer falar com um atendente? Responda na tela.",
    "en": "Would you like to talk to an advisor? Please answer on the screen.",
}


@dataclass
class Transcribed:
    text: str


@dataclass
class SpokenAudio:
    seq: int
    mime: str
    audio: bytes


@dataclass
class VoiceProblem:
    code: str  # not_heard, transcription_failed, speech_failed


async def voice_turn(agent: AgentRunner, speech: SpeechClient, tracing: Tracing | None, *,
                     user_id: str, conversation_id: str, audio: bytes, content_type: str,
                     language: str | None = None,
                     customer_id: str | None = None) -> AsyncIterator:
    received = time.perf_counter()
    try:
        transcript = await speech.transcribe(audio, content_type, language)
    except Exception:
        log.exception("transcription failed")
        metrics.VOICE_ERRORS.labels("transcription").inc()
        yield VoiceProblem("transcription_failed")
        return
    metrics.VOICE_STT_SECONDS.observe(transcript.seconds)
    metrics.VOICE_COST.labels("transcription").inc(transcript.cost_usd)
    if not any(c.isalnum() for c in transcript.text):
        metrics.VOICE_ERRORS.labels("not_heard").inc()
        yield VoiceProblem("not_heard")
        return
    yield Transcribed(transcript.text)

    queue: asyncio.Queue = asyncio.Queue()
    gate = asyncio.Semaphore(SPEECH_CONCURRENCY)
    tasks: list[asyncio.Task] = []
    spoken: list[Speech] = []
    ref: dict = {}
    first_audio: float | None = None
    phrases = language if language in CONFIRM_ON_SCREEN else "es"

    async def speak(seq: int, text: str) -> None:
        result = None
        async with gate:
            try:
                result = await speech.synthesize(text)
            except Exception:
                log.exception("speech synthesis failed")
        await queue.put(("audio", seq, result))

    def say(texts: list[str]) -> None:
        # The screen keeps the figures; the voice says them as whole quantities.
        for text in texts:
            tasks.append(asyncio.create_task(speak(len(tasks), spoken_figures(text, language))))

    async def produce() -> None:
        try:
            async for event in agent.run(user_id, conversation_id, transcript.text, language,
                                         customer_id=customer_id, channel="voice", trace=ref):
                await queue.put(("agent", event))
        except Exception as exc:  # noqa: BLE001  (re-raised by the consumer below)
            await queue.put(("error", exc))
        finally:
            queue.put_nowait(("end",))

    producer = asyncio.create_task(produce())
    buffer, ready, sent, ended = SentenceBuffer(), {}, 0, False
    done: AgentDone | None = None
    speech_failed = False
    try:
        while not (ended and sent == len(tasks)):
            item = await queue.get()
            match item:
                case ("agent", AgentText(text=text) as event):
                    say(buffer.feed(text))
                    yield event
                case ("agent", AgentDone() as event):
                    say(buffer.flush())
                    done = event  # sent after the last audio
                case ("agent", event):
                    if isinstance(event, ApprovalRequired | HandoffOffered):
                        say(buffer.flush())
                        say([(CONFIRM_ON_SCREEN if isinstance(event, ApprovalRequired)
                              else ADVISOR_OFFER)[phrases]])
                    yield event
                case ("audio", seq, result):
                    ready[seq] = result
                    while sent in ready:
                        result = ready.pop(sent)
                        sent += 1
                        if result is None:
                            if not speech_failed:
                                speech_failed = True
                                metrics.VOICE_ERRORS.labels("speech").inc()
                                yield VoiceProblem("speech_failed")
                            continue
                        spoken.append(result)
                        if first_audio is None:
                            first_audio = time.perf_counter() - received
                            metrics.VOICE_FIRST_AUDIO.observe(first_audio)
                        yield SpokenAudio(sent - 1, result.mime, result.audio)
                case ("error", exc):
                    raise exc
                case ("end",):
                    ended = True
                    say(buffer.flush())
        if done is not None:
            yield done
    finally:
        for task in [producer, *tasks]:
            task.cancel()
        for task in [producer, *tasks]:
            with suppress(asyncio.CancelledError, Exception):
                await task
        _record(tracing, ref, transcript, spoken, len(tasks), first_audio, len(audio),
                content_type)


def _record(tracing: Tracing | None, ref: dict, transcript: Transcript, spoken: list[Speech],
            sentences: int, first_audio: float | None, audio_bytes: int,
            content_type: str) -> None:
    tts_cost = sum(s.cost_usd for s in spoken)
    metrics.VOICE_COST.labels("speech").inc(tts_cost)
    if tracing is None:
        return
    steps = [{"name": "voice.stt", "model": transcript.model, "usage": {
                  "input_audio": transcript.usage.get("audio", 0),
                  "input_text": transcript.usage.get("text", 0),
                  "output": transcript.usage.get("output", 0)},
              "cost_usd": transcript.cost_usd,
              "metadata": {"latency_s": round(transcript.seconds, 3), "bytes": audio_bytes,
                           "content_type": content_type}}]
    if spoken:
        steps.append({"name": "voice.tts", "model": spoken[0].model,
                      "usage": {"input": sum(s.chars for s in spoken)}, "cost_usd": tts_cost,
                      "metadata": {"sentences": sentences, "spoken": len(spoken),
                                   "latency_s_max": round(max(s.seconds for s in spoken), 3),
                                   "audio_bytes": sum(len(s.audio) for s in spoken)}})
    scores = {"voice_stt_s": transcript.seconds}
    if first_audio is not None:
        scores["voice_first_audio_s"] = first_audio
    try:
        tracing.record_voice(ref, steps, scores)
    except Exception:
        log.warning("could not trace the voice steps", exc_info=True)
