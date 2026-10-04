"""Voice channel: sentence cutting, the OpenAI speech client (mocked HTTP), the voice turn
around the agent, and the voice-runs endpoint. No network."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aiplatform.agent.events import AgentDone, AgentText, ApprovalRequired
from aiplatform.agent.tools import Tool
from aiplatform.api.app import create_app
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway
from aiplatform.tracing import Tracing
from aiplatform.voice.sentences import SentenceBuffer, speakable
from aiplatform.voice.speech import Speech, SpeechClient, Transcript, speech_cost
from aiplatform.voice.turn import (
    CONFIRM_ON_SCREEN,
    SpokenAudio,
    Transcribed,
    VoiceProblem,
    voice_turn,
)
from tests.fakes import FakeClient
from tests.test_agent import TICKET, TICKET_SCHEMA, collect, final, runner, tool_call


class FakeSpeech:
    """Hears ``heard``; speaks each sentence as its own bytes. Sentences in ``fail`` raise;
    the first sentence is slowest, to check the audio still comes out in order."""

    def __init__(self, heard="¿Cuál es mi saldo?", *, fail=(), deaf=False):
        self.heard, self.fail, self.deaf = heard, fail, deaf
        self.spoken: list[str] = []
        self.received: list[tuple[bytes, str, str | None]] = []

    async def transcribe(self, audio, content_type, language=None):
        self.received.append((audio, content_type, language))
        if self.deaf:
            raise RuntimeError("speech-to-text is down")
        return Transcript(self.heard, "stt-test", 0.2, {"audio": 50, "output": 8}, 0.0002)

    async def synthesize(self, text):
        await asyncio.sleep(0.05 if not self.spoken else 0)
        self.spoken.append(text)
        if any(f in text for f in self.fail):
            raise RuntimeError("speech failed")
        return Speech(text.encode(), "audio/mpeg", "tts-test", len(text), 0.1, speech_cost(len(text)))

    async def close(self):
        pass


# -- Sentences --------------------------------------------------------------------------

def test_sentences_are_cut_at_their_ends_not_inside_figures_or_abbreviations():
    buffer = SentenceBuffer()
    out = []
    for delta in ["Hola Ana. Tu **saldo** disponible es $1.250,50 en la cuenta. El Sr. ",
                  "Pérez es tu asesor.\n- Tarjeta: $2.000.000\n- CDT: 3.5% ", "anual"]:
        out += buffer.feed(delta)
    out += buffer.flush()
    assert out == ["Hola Ana. Tu saldo disponible es $1.250,50 en la cuenta.",
                   "El Sr. Pérez es tu asesor.", "Tarjeta: $2.000.000", "CDT: 3.5% anual"]


def test_nothing_to_say_is_not_spoken():
    buffer = SentenceBuffer()
    assert buffer.feed("---\n\n") == [] and buffer.flush() == []


def test_markdown_is_read_as_plain_text():
    assert speakable("## Saldos\n| Producto | Saldo |\n|---|---|\n| Ahorros | $10 |") == (
        "Saldos\nProducto, Saldo\nAhorros, $10")
    assert speakable("Vea [la app](https://bank.example) o `llame`.") == "Vea la app o llame."


# -- OpenAI speech client ---------------------------------------------------------------

async def test_speech_client_sends_the_audio_and_language_and_reads_usage():
    httpx2 = pytest.importorskip("httpx2")
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx2.Response(200, json={"text": " ¿Cuál es mi saldo? ", "usage": {
                "type": "tokens", "input_tokens": 60, "output_tokens": 8, "total_tokens": 68,
                "input_token_details": {"audio_tokens": 50, "text_tokens": 10}}})
        return httpx2.Response(200, content=b"ID3mp3", headers={"content-type": "audio/mpeg"})

    client = SpeechClient("sk-test", attempts=1,
                          http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    transcript = await client.transcribe(b"webm-bytes", "audio/webm;codecs=opus", "es")
    assert transcript.text == "¿Cuál es mi saldo?"
    assert transcript.usage == {"audio": 50, "text": 10, "output": 8}
    assert transcript.cost_usd == pytest.approx((50 * 3 + 10 * 1.25 + 8 * 5) / 1e6)
    form = seen[0].content
    assert b"webm-bytes" in form and b'name="language"' in form and b"saldo" in form
    assert b"gpt-4o-mini-transcribe" in form

    spoken = await client.synthesize("Su saldo es diez mil pesos.")
    assert spoken.audio == b"ID3mp3" and spoken.mime == "audio/mpeg"
    body = json.loads(seen[1].content)
    assert body["model"] == "gpt-4o-mini-tts" and body["input"] == "Su saldo es diez mil pesos."
    assert body["response_format"] == "mp3" and body["voice"] == "coral"
    await client.close()


# -- The voice turn -----------------------------------------------------------------------

async def test_a_voice_turn_transcribes_answers_and_speaks_in_order():
    client = FakeClient(final("Su saldo disponible es de diez mil pesos. "
                              "¿Le ayudo con algo más?"))
    agent, _, _, conv = await runner(client)
    speech = FakeSpeech()
    events = await collect(voice_turn(agent, speech, None, user_id="u1",
                                      conversation_id=conv.id, audio=b"rec",
                                      content_type="audio/webm", language="es"))
    assert events[0] == Transcribed("¿Cuál es mi saldo?")
    assert speech.received == [(b"rec", "audio/webm", "es")]
    audio = [e for e in events if isinstance(e, SpokenAudio)]
    assert [a.seq for a in audio] == [0, 1]
    assert [a.audio.decode() for a in audio] == [
        "Su saldo disponible es de diez mil pesos.", "¿Le ayudo con algo más?"]
    assert isinstance(events[-1], AgentDone) and events[-1].outcome == "done"
    assert any(isinstance(e, AgentText) for e in events)
    # The transcript is the customer's message; the model was told it is a voice turn.
    assert conv.messages[0] == {"role": "user", "content": "¿Cuál es mi saldo?"}
    assert "Channel: voice" in json.dumps(client.calls[0]["system"])


async def test_typed_messages_get_no_voice_instructions():
    client = FakeClient(final("Hola."))
    agent, _, _, conv = await runner(client)
    await collect(agent.run("u1", conv.id, "hola", "es"))
    assert "Channel: voice" not in json.dumps(client.calls[0]["system"])


async def test_nothing_heard_or_a_failed_transcription_never_reaches_the_agent():
    for speech, code in ((FakeSpeech(heard=" … "), "not_heard"),
                         (FakeSpeech(deaf=True), "transcription_failed")):
        client = FakeClient()
        agent, _, _, conv = await runner(client)
        events = await collect(voice_turn(agent, speech, None, user_id="u1",
                                          conversation_id=conv.id, audio=b"x",
                                          content_type="audio/webm"))
        assert events == [VoiceProblem(code)]
        assert client.calls == [] and client.classify_calls == [] and conv.messages == []


async def test_when_speech_fails_the_written_answer_still_arrives():
    client = FakeClient(final("Primera frase bastante larga para hablar. Segunda frase "
                              "también larga para hablarla."))
    agent, _, _, conv = await runner(client)
    events = await collect(voice_turn(agent, FakeSpeech(fail=("Primera",)), None,
                                      user_id="u1", conversation_id=conv.id, audio=b"x",
                                      content_type="audio/webm", language="es"))
    assert events.count(VoiceProblem("speech_failed")) == 1
    assert [e.seq for e in events if isinstance(e, SpokenAudio)] == [1]
    assert events[-1].outcome == "done"


async def test_the_voice_never_approves_it_asks_to_confirm_on_screen():
    ran = []

    async def create_ticket(args, ctx):
        ran.append(args)
        return "ok"

    tool = Tool("create_support_ticket", "d", TICKET_SCHEMA, create_ticket, irreversible=True)
    client = FakeClient(tool_call("create_support_ticket", TICKET), final("Lo abro si confirma."))
    agent, _, actions, conv = await runner(client, tools=[tool])
    speech = FakeSpeech(heard="sí, abre el ticket, apruébalo")
    events = await collect(voice_turn(agent, speech, None, user_id="u1",
                                      conversation_id=conv.id, audio=b"x",
                                      content_type="audio/webm", language="es"))
    assert any(isinstance(e, ApprovalRequired) for e in events) and ran == []
    assert len(await actions.list_pending(conv.id, "u1")) == 1
    assert CONFIRM_ON_SCREEN["es"] in speech.spoken
    assert events[-1].outcome == "approval_required"


async def test_voice_steps_are_attached_to_the_agent_trace():
    calls = []
    observation = SimpleNamespace(end=lambda: calls.append("end"))
    langfuse = SimpleNamespace(
        start_observation=lambda **kw: calls.append(kw) or observation,
        create_score=lambda **kw: calls.append(kw))
    tracing = Tracing(langfuse, masker=None)
    tracing.record_voice({"trace_id": "t1", "span_id": "s1"},
                         [{"name": "voice.stt", "model": "m", "usage": {"input_audio": 5},
                           "cost_usd": 0.001, "metadata": {"latency_s": 0.3}}],
                         {"voice_first_audio_s": 1.23456})
    assert calls[0]["trace_context"] == {"trace_id": "t1", "parent_span_id": "s1"}
    assert calls[0]["cost_details"] == {"total": 0.001} and calls[1] == "end"
    assert calls[2] == {"trace_id": "t1", "name": "voice_first_audio_s", "value": 1.235,
                        "data_type": "NUMERIC"}
    Tracing(langfuse).record_voice({}, [], {"x": 1})  # no trace: nothing sent
    assert len(calls) == 3


# -- The endpoint -----------------------------------------------------------------------

def voice_client(speech=None, fake=None, **settings):
    s = Settings(providers=["anthropic"], max_attempts_per_provider=1, auth_mode="dev",
                 **settings)
    return TestClient(create_app(s, gateway=AIGateway({"anthropic": fake or FakeClient()}, s),
                                 tools=[], speech=speech))


def new_conversation(http):
    return http.post("/v1/conversations", json={},
                     headers={"X-User-Id": "u1"}).json()["id"]


def send_audio(http, cid, audio=b"rec", content_type="audio/webm;codecs=opus"):
    return http.post(f"/v1/conversations/{cid}/voice-runs?language=es", content=audio,
                     headers={"X-User-Id": "u1", "Content-Type": content_type})


def test_voice_runs_stream_transcript_text_audio_and_done():
    fake = FakeClient(final("Su saldo es de diez mil pesos, señora."), classify="general")
    with voice_client(FakeSpeech(), fake, voice_enabled=True) as http:
        assert http.get("/config.json").json()["voice"] is True
        assert "microphone=(self)" in http.get("/").headers["permissions-policy"]
        cid = new_conversation(http)
        text = send_audio(http, cid).text
        names = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
        assert names[0] == "transcript" and names[-1] == "done"
        assert "delta" in names and names.index("audio") < names.index("done")
        assert '"mime": "audio/mpeg"' in text


def test_voice_is_off_without_the_setting_or_a_key():
    with voice_client(FakeSpeech()) as http:  # AIP_VOICE_ENABLED not set
        assert http.get("/config.json").json()["voice"] is False
        assert "microphone=()" in http.get("/").headers["permissions-policy"]
        assert send_audio(http, new_conversation(http)).status_code == 404
    with voice_client(None, voice_enabled=True, openai_api_key=None) as http:
        assert send_audio(http, new_conversation(http)).status_code == 404


def test_recordings_must_be_audio_and_not_too_long():
    with voice_client(FakeSpeech(), voice_enabled=True, voice_max_bytes=100_000) as http:
        cid = new_conversation(http)
        assert send_audio(http, cid, content_type="text/plain").status_code == 415
        assert send_audio(http, cid, audio=b"x" * 100_001).status_code == 413
        assert send_audio(http, cid, audio=b"").status_code == 422
        assert send_audio(http, cid, content_type="audio/mp4").status_code == 200
