"""Deduplicate the Datathon call transcripts into the few unique conversation lines.

    uv run python evals/import_transcripts.py ../data/call_transcripts

The 171k transcripts are generated from a handful of templates. This prints stats and
writes evals/transcripts_unique.json with the distinct customer openings, customer
follow-ups and agent lines (placeholders such as {monto} kept as-is), which
evals/build_banking_cases.py turns into eval cases.
"""

import argparse
import collections
import csv
import glob
import json
import sys
from pathlib import Path

OUT = Path(__file__).parent / "transcripts_unique.json"


def turns(full_text: str) -> list[tuple[str, str]]:
    result = []
    for line in full_text.splitlines():
        line = line.strip()
        for prefix, role in (("Cliente:", "customer"), ("Agente:", "agent")):
            if line.startswith(prefix):
                result.append((role, line[len(prefix):].strip()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="directory with call_transcripts CSVs")
    args = parser.parse_args()
    csv.field_size_limit(sys.maxsize)

    rows = 0
    texts: collections.Counter[str] = collections.Counter()
    openings: collections.Counter[str] = collections.Counter()
    followups: collections.Counter[str] = collections.Counter()
    agent: collections.Counter[str] = collections.Counter()
    languages: collections.Counter[str] = collections.Counter()
    for f in sorted(glob.glob(str(args.path / "**/*.csv"), recursive=True)):
        with open(f, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                rows += 1
                languages[row.get("detected_language", "")] += 1
                texts[row["full_text"].strip()] += 1
                customer_turns = 0
                for role, text in turns(row["full_text"]):
                    if role == "agent":
                        agent[text] += 1
                    elif customer_turns == 0:
                        openings[text] += 1
                        customer_turns += 1
                    else:
                        followups[text] += 1
                        customer_turns += 1

    summary = {
        "rows": rows,
        "unique_conversations": len(texts),
        "languages": dict(languages),
        "customer_openings": [t for t, _ in openings.most_common()],
        "customer_followups": [t for t, _ in followups.most_common()],
        "agent_lines": [t for t, _ in agent.most_common()],
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"{rows:,} transcripts, {len(texts)} unique texts, languages {dict(languages)}")
    print(f"{len(openings)} openings, {len(followups)} follow-ups, {len(agent)} agent lines")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
