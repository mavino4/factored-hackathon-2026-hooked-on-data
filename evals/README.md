# Evals

A small quality baseline for the `chat` and `agent` routes. Use it before changing a prompt, a
tool, or a route's model in `src/aiplatform/llm/models.py` (`ROUTES`), and after the change.

> **The dataset is a placeholder.** `dataset.jsonl` has 15 generic starter cases that test the
> plumbing and basic behaviour. Replace or extend them with **20–50 real questions from your
> users/domain** before trusting the score for decisions.

## Run

```bash
# Local Ollama (free):
AIP_PROVIDERS='["ollama"]' uv run python evals/run.py

# Claude API (needs ANTHROPIC_API_KEY; costs are printed per case):
AIP_PROVIDERS='["anthropic"]' uv run python evals/run.py

# Options
uv run python evals/run.py --judge                   # also grade answers with an LLM judge
uv run python evals/run.py --save-baseline           # write evals/baseline.json
uv run python evals/run.py --compare evals/baseline.json   # exit 1 if score drops > 5 points
```

Each run prints a per-case table (pass/fail, latency, tokens, cost) and writes the full results,
including answers and failure reasons, to `evals/results/<UTC timestamp>.json` (gitignored).

Compare like with like: a baseline made without `--judge` should be compared with runs without
`--judge`, and with the same provider/model.

## Add a case

One JSON object per line in `dataset.jsonl`:

```json
{"id": "refund-policy", "route": "chat",
 "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "input": "How long do I have to request a refund?",
 "checks": {"must_include_any": ["30 days"], "must_not_include": ["I don't know"], "max_words": 60},
 "rubric": "States the 30-day refund window and how to request it."}
```

| Field | Meaning |
|---|---|
| `id` | Unique name |
| `route` | `chat`, `agent` (the model with the tools from `agent/tools.py`, no classifier) or `assistant` (the whole assistant as the API runs it: intent classifier, then the agent graph; customer turns in `history` are sent first) |
| `history` | Optional earlier turns |
| `input` | The user message being evaluated |
| `checks.must_include_any` | Passes if the answer contains any of these (case-insensitive) |
| `checks.must_not_include` | Fails if the answer contains any of these |
| `checks.max_words` | Word limit |
| `checks.must_call_tool` / `must_not_call_tool` | Agent tool-use expectations |
| `checks.must_not_execute_tool` | The tool may be requested but must not run (approval-gated tools) |
| `checks.outcome` / `outcome_not` | `assistant` route: how the run must (not) end: `done`, `blocked` (manipulation attempt), `handoff`… |
| `rubric` | One sentence for the optional `--judge` |

Deterministic checks always run. Prefer them: they're free, fast and repeatable.

## The upgrade rule

A route in `ROUTES` may move to a different model (e.g. Haiku 4.5 → Sonnet 5) **only if**, on
the same dataset and settings, its eval score **improves**, or the score is **equal and the
cost drops**. Record the before/after result files in the change description.

## Caveats

- **LLM judges on small local models are unreliable.** In the first run with `llama3.2:3b`, the
  judge failed two correct answers while writing reasons that agreed with the rubric. Use
  `--judge` with a capable Claude model, and read the reasons, not just the score.
- The agent loop here is a simplified copy of the production loop: it doesn't persist
  history or execute approval-gated tools. It measures model behaviour, not the API.
- Scores from a 15-case set move a lot with a single case (6.7 points each).

## Security suite

`evals/security.jsonl` (`make eval-security`, needs `make bank-db`) runs 50 cases through the
whole assistant: manipulation attempts that must end `blocked` or at least leak nothing
(instructions, tool names, another customer's data, a figure the customer dictated), real
questions wrapped in junk and customers sharing a PIN or CVV, which must be answered, and
ordinary controls that must never be taken for attacks. Run it after any change to the
prompts in `chat/prompts.py` or to the classifier; a false positive on a control is as much
a regression as a missed attack.
