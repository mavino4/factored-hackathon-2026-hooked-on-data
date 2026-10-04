"""How much does the intent classifier lose when customers speak instead of type?

    uv run python -m evals.voice_bench                  # 60 messages, cached after the first run
    uv run python -m evals.voice_bench --limit 120 --errors

A sample of the intent test set (evals/intents/test.jsonl; only messages a person could
say: no history, no typo copies, no code or markup) is read aloud with OpenAI's
text-to-speech, in several voices, then transcribed with the production speech-to-text
(voice/speech.py), and classified again. Reported per language: word and character error
rate of the transcripts, intent accuracy on the written text vs on the transcript (LLM,
Jev and Jev+LLM at the production threshold, as in AIP_INTENT_CLASSIFIER=jev_llm),
attack recall, speech latency and cost per 1,000 voice turns.

Everything is cached in evals/intents/.cache/voice/ (audio, transcripts, predictions), so
repeating a run neither pays nor waits. Needs OPENAI_API_KEY, ANTHROPIC_API_KEY and
TYPESAFE_API_KEY in .env. Results go to evals/results/voice-<UTC>.json.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("AIP_AUTH_MODE", "dev")

from aiplatform.config import get_settings
from aiplatform.voice.speech import SpeechClient, speech_cost
from evals.classify_bench import CACHE_DIR, jev_predictions, llm_predictions, percentile
from evals.intent_dataset import TEST, read

VOICE_DIR = CACHE_DIR / "voice"
RESULTS = Path(__file__).parent / "results"
# Customers, not advisors: several voices, read like someone talking to their bank.
VOICES = ("coral", "ash", "nova", "onyx")
CUSTOMER = ("Read this as a bank customer speaking to a voice assistant on their phone: "
            "natural, conversational pace, not a narrator.")
# Written-only messages: code, markup, invisible or look-alike characters.
UNSPEAKABLE = re.compile(r"[<>{}\[\]|\\`=_#~^]|[\u200b-\u200f\u2060\ufeff]|[^\x00-\u024f¿¡€£]")
# A typical spoken answer, to estimate the text-to-speech cost of a voice turn.
ANSWER_CHARS = 220


def speakable_cases(limit: int, seed: int) -> list[dict]:
    """Up to ``limit`` cases, spread over language and intent, same ones every time."""
    cases = [c for c in read(TEST) if not c.get("history") and "original" not in c
             and len(c["text"]) <= 300 and not UNSPEAKABLE.search(c["text"])]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cases:
        groups[(c["language"], c["intent"])].append(c)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    chosen: list[dict] = []
    while len(chosen) < limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen) < limit:
                chosen.append(groups[key].pop())
    return chosen


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", text)


def edit_distance(a: list, b: list) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def error_rate(reference: list, hypothesis: list) -> float:
    return edit_distance(reference, hypothesis) / max(len(reference), 1)


async def hear(speech: SpeechClient, case: dict, voice: str, gate: asyncio.Semaphore) -> dict:
    """Read the case aloud and transcribe it (both cached on disk)."""
    key = hashlib.sha256(f"{speech.tts_model}|{voice}|{speech.stt_model}|{case['text']}"
                         .encode()).hexdigest()[:20]
    meta, audio_path = VOICE_DIR / f"{key}.json", VOICE_DIR / f"{key}.mp3"
    if meta.exists():
        return json.loads(meta.read_text())
    async with gate:
        if audio_path.exists():
            audio, tts_s = audio_path.read_bytes(), None
        else:
            spoken = await speech.synthesize(case["text"], voice=voice, instructions=CUSTOMER)
            audio, tts_s = spoken.audio, round(spoken.seconds, 4)
            audio_path.write_bytes(audio)
        transcript = await speech.transcribe(audio, "audio/mpeg", case["language"])
    row = {"id": case["id"], "voice": voice, "transcript": transcript.text,
           "tts_s": tts_s, "stt_s": round(transcript.seconds, 4), "usage": transcript.usage,
           "stt_cost_usd": transcript.cost_usd, "tts_cost_usd": speech_cost(len(case["text"])),
           "audio_bytes": len(audio)}
    tmp = meta.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, ensure_ascii=False))
    tmp.replace(meta)
    return row


def combined(jev: dict, llm: dict, threshold: float) -> str:
    """Production AIP_INTENT_CLASSIFIER=jev_llm: Jev when sure, else the LLM."""
    return jev["intent"] if jev["confidence"] >= threshold else llm["intent"]


async def run(args) -> dict:
    settings = get_settings()
    if settings.openai_api_key is None:
        raise SystemExit("OPENAI_API_KEY is not set in .env")
    if settings.typesafe_api_key is None:
        raise SystemExit("TYPESAFE_API_KEY is not set in .env (Jev)")
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    cases = speakable_cases(args.limit, args.seed)
    print(f"{len(cases)} cases {dict(Counter(c['language'] for c in cases))}")

    speech = SpeechClient(settings.openai_api_key.get_secret_value(),
                          stt_model=settings.voice_stt_model,
                          tts_model=settings.voice_tts_model)
    gate = asyncio.Semaphore(4)
    try:
        heard = await asyncio.gather(*(hear(speech, c, VOICES[i % len(VOICES)], gate)
                                       for i, c in enumerate(cases)))
    finally:
        await speech.close()

    spoken = [{**c, "id": f"{c['id']}~{h['voice']}", "text": h["transcript"] or "…"}
              for c, h in zip(cases, heard, strict=True)]
    key = settings.typesafe_api_key.get_secret_value()
    text_llm = await llm_predictions(cases, False)
    text_jev = await jev_predictions(cases, False, key)
    voice_llm = await llm_predictions(spoken, False, VOICE_DIR / "llm_predictions.jsonl")
    voice_jev = await jev_predictions(spoken, False, key, VOICE_DIR / "jev_predictions.jsonl")

    rows = []
    for case, said, h in zip(cases, spoken, heard, strict=True):
        ref, hyp = words(case["text"]), words(h["transcript"])
        rows.append({
            "id": case["id"], "language": case["language"], "intent": case["intent"],
            "voice_name": h["voice"], "written": case["text"], "transcript": h["transcript"],
            "wer": error_rate(ref, hyp),
            "cer": error_rate(list(" ".join(ref)), list(" ".join(hyp))),
            "stt_s": h["stt_s"], "stt_cost_usd": h["stt_cost_usd"],
            "on_text": {"llm": text_llm[case["id"]]["intent"], "jev": text_jev[case["id"]]["intent"],
                     "jev_llm": combined(text_jev[case["id"]], text_llm[case["id"]],
                                         settings.jev_threshold)},
            "on_voice": {"llm": voice_llm[said["id"]]["intent"],
                      "jev": voice_jev[said["id"]]["intent"],
                      "jev_llm": combined(voice_jev[said["id"]], voice_llm[said["id"]],
                                          settings.jev_threshold)}})
    return summarize(rows, args)


def accuracy(rows: list[dict], channel: str, model: str) -> float | None:
    """``channel``: "on_text" (the written message) or "on_voice" (its transcript)."""
    return round(sum(r[channel][model] == r["intent"] for r in rows) / len(rows), 4) \
        if rows else None


def summarize(rows: list[dict], args) -> dict:
    groups = {"all": rows, **{lang: [r for r in rows if r["language"] == lang]
                              for lang in sorted({r["language"] for r in rows})}}
    attacks = [r for r in rows if r["intent"] == "attack"]
    stt_cost = sum(r["stt_cost_usd"] for r in rows) / max(len(rows), 1)
    return {
        "meta": {"timestamp": datetime.now(UTC).isoformat(), "limit": args.limit,
                 "seed": args.seed},
        "by_language": {
            name: {"n": len(g),
                   "wer": round(sum(r["wer"] for r in g) / len(g), 4) if g else None,
                   "cer": round(sum(r["cer"] for r in g) / len(g), 4) if g else None,
                   "exact_transcripts": round(sum(r["wer"] == 0 for r in g) / len(g), 4)
                   if g else None,
                   "accuracy": {m: {"text": accuracy(g, "on_text", m), "voice": accuracy(g, "on_voice", m)}
                                for m in ("llm", "jev", "jev_llm")}}
            for name, g in groups.items()},
        "attack_recall": {m: {"text": accuracy(attacks, "on_text", m),
                              "voice": accuracy(attacks, "on_voice", m)}
                          for m in ("llm", "jev", "jev_llm")},
        "latency": {"stt_p50_s": percentile([r["stt_s"] for r in rows], 0.5),
                    "stt_p95_s": percentile([r["stt_s"] for r in rows], 0.95)},
        "cost_per_1000_turns_usd": {
            "speech_to_text": round(stt_cost * 1000, 4),
            "text_to_speech_estimate": round(speech_cost(ANSWER_CHARS) * 1000, 4)},
        "changed": [r for r in rows if r["on_voice"]["jev_llm"] != r["on_text"]["jev_llm"]],
        "rows": rows,
    }


def print_report(s: dict, errors: bool) -> None:
    print("\nLanguage     n    WER    CER  exact  | accuracy text → voice: "
          "LLM            Jev            Jev+LLM")
    for name, v in s["by_language"].items():
        acc = "  ".join(f"{v['accuracy'][m]['text'] * 100:5.1f}% → "
                        f"{v['accuracy'][m]['voice'] * 100:5.1f}%" for m in ("llm", "jev", "jev_llm"))
        print(f"{name:<8}{v['n']:>5} {v['wer'] * 100:5.1f}% {v['cer'] * 100:5.1f}% "
              f"{v['exact_transcripts'] * 100:5.1f}%  |  {acc}")
    ar = s["attack_recall"]
    if ar["llm"]["text"] is not None:
        print("Attack recall text → voice: " + " · ".join(
            f"{m} {ar[m]['text'] * 100:.0f}% → {ar[m]['voice'] * 100:.0f}%" for m in ar))
    lat, cost = s["latency"], s["cost_per_1000_turns_usd"]
    print(f"Speech-to-text p50 {lat['stt_p50_s']:.2f}s · p95 {lat['stt_p95_s']:.2f}s · per 1,000 voice "
          f"turns: transcription ${cost['speech_to_text']:.2f}, spoken answers "
          f"~${cost['text_to_speech_estimate']:.2f} ({ANSWER_CHARS} characters each)")
    print(f"Jev+LLM changed its answer on {len(s['changed'])} of {len(s['rows'])} messages")
    if errors:
        for r in s["changed"]:
            print(f"  [{r['language']} {r['voice_name']}] {r['intent']}: "
                  f"{r['on_text']['jev_llm']} → {r['on_voice']['jev_llm']}\n"
                  f"     written: {r['written']}\n     heard:   {r['transcript']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--errors", action="store_true",
                        help="list the messages whose intent changed when spoken")
    args = parser.parse_args()
    summary = asyncio.run(run(args))
    print_report(summary, args.errors)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"voice-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
