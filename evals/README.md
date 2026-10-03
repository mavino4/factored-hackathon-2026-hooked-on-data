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

## Trace reports (Langfuse)

The suites above run prepared cases. These two commands measure the **real turns already
traced** in the self-hosted Langfuse (`make langfuse-up`), one trace per customer turn.

```bash
make trace-report                              # last 7 days
make trace-report ARGS="--since 24h --check"   # exit 1 if a threshold in trace_slo.json breaks
make trace-report ARGS="--release 0fc4aaa --compare evals/results/traces-<earlier>.json"
make trace-report ARGS="--quick"               # only turns sent by a quick-action button
make trace-report ARGS="--routing intent"      # only turns routed by AIP_QUICK_ACTIONS=intent (or direct)

make trace-score ARGS="--dry-run"              # grade turns, write nothing
make trace-score                               # write the grades to Langfuse as scores
make trace-score ARGS="--judge --sample 50"    # also an LLM judge (spends API credits)
make trace-score ARGS="--judge-jev --sample 40"  # does the LLM agree with Jev's decisions?
```

`trace-report` (`evals/traces.py`, read-only) prints and saves to `results/traces-<UTC>.json`:

| Group | What it measures |
|---|---|
| Latency | p50 / p95 / p99 / max of the whole turn, the classifier, each model call, the first token and each tool. Below about 100 values (`n`) the p99 is close to the max |
| Cost | Total, per turn, per conversation, the classifier's share, tokens per turn, cache reads |
| Classifier | Who decided the intent (`jev`, `llm`, `llm_low_confidence`, `llm_jev_error`): turns, classification latency and cost, turn p95, Jev/LLM agreement |
| Intent | Share, p95 and p99 latency and mean cost of each classified intent |
| Routing | Per `AIP_QUICK_ACTIONS` mode (`model` = classifier, `intent`, `direct` = no model): turns, quick-action turns, latency, mean cost and model calls per turn. Compare modes on quick-action turns only (`--quick`): the questions are then the same |
| Outcome | How turns ended: `done`, `blocked`, `handoff`, `max_iterations`, `incomplete` (no final step: an error or the customer left) |
| Behaviour | Answered account questions that used a bank tool, iterations per turn, turns with an error or a retry |
| Quality | Pass rate of the scores below, once written |

`trace-score` (`evals/score_traces.py`) writes one boolean score per turn and check; a turn
that already has a score is skipped, so running it again only grades new turns.

| Score | Passes when |
|---|---|
| `language_match` | The answer is in the customer's UI language (es / pt; skipped for short answers) |
| `no_leak` | The answer names no tool or prompt section |
| `used_bank_tool` | An answered account question called a bank tool in that turn |
| `helpfulness` (`--judge`) | The model judges that the answer addresses the question, or declines clearly when out of scope |

The app itself adds the scores `intent`, `outcome` and `prompt_injection`, the time to first
token and the release (`AIP_RELEASE`, set to the git commit by `make lan`) to every new trace.
Older traces lack them: the report then reads intent and outcome from the steps.

**From a trace to a regression case:**
`make trace-report ARGS="--export-cases --failed used_bank_tool"` (or `--outcome incomplete`)
writes the selected turns as cases to `results/cases-<UTC>.jsonl`. Their text is masked
(`<NAME>`, `<AMOUNT>`...), so review each one and replace the placeholders before adding it to
a suite.

**Limits.** Traces are masked before they leave the app, so nothing here can check that a
balance is right: that stays with `banking.jsonl`. This also measures existing traffic; it is
not a load test.

**Langfuse MCP (optional).** `.mcp.json` points Claude Code at the server's own MCP endpoint
(observations, scores, metrics, datasets). It needs the project keys in the environment:
`eval "$(make -s langfuse-mcp-env)"` before starting Claude Code.

## Intent classifiers without an LLM

The LLM classifier (`agent/intent.py`) is ~45 % of the cost of a turn and ~1.5 s at p95.
`agent/classifiers/` holds the alternatives, measured **in isolation** (not wired into
the agent yet): keyword rules (`rules`), TF-IDF + logistic regression (`ml`), and sentence
embeddings with a classifier on top, named `<embeddings>-<model>`:

| Embeddings | Where | Needs |
|---|---|---|
| `bge-m3` (1024 dims) | local Ollama | `ollama pull bge-m3` |
| `openai-3s`: OpenAI text-embedding-3-small (1536 dims) | OpenAI API, $0.02 / M tokens | `OPENAI_API_KEY=sk-...` in `.env` (skipped without it); vectors cached in `intents/.cache/` (gitignored) |

`jev`: Jev, TypeSafe AI's decision model, answers a typed `choice` question with a
probability per intent and a confidence (`TYPESAFE_API_KEY` in `.env`; skipped without it;
answers cached in `intents/jev_predictions.jsonl`, `--refresh-jev` to ask again).

Models on the vectors: `logreg` (logistic regression), `knn` (nearest-neighbour vote),
`rf` (random forest) and `boost` (scikit-learn histogram gradient boosting), each tuned by
cross-validation on the training set only.

```bash
ollama pull bge-m3                      # once, for the embeddings
make intent-data                        # test set, generated training data (Qwen ~2 h)
make classify-bench                     # all classifiers vs the LLM on the test set
make classify-bench ARGS="--only rules openai-3s-logreg --errors"
```

| File | What it is |
|---|---|
| `intents/test.jsonl` | Frozen test set: the eval suites labelled by hand (`intents/labels.py`), `intents/handwritten.jsonl` and `intents/handwritten2.jsonl` (typos, slang, mixed languages, emojis, capitals, voice, long messages), plus a misspelt copy of each (`intents/typos.py`, source `typos`). Never trained on |
| `intents/generated-{qwen,claude}.jsonl` | Generated messages per intent, language and angle (Qwen 2.5 7B locally, a smaller group with Claude) |
| `intents/generated-{qwen,claude}-r2.jsonl` | Round 2 (`--round 2`): the same intents written the way people type (typos, slang, voice, emojis...) |
| `intents/train.jsonl` | All generated messages plus a misspelt copy of each, minus anything equal or too close (bge-m3 cosine ≥ 0.92) to a test message |
| `intents/llm_predictions.jsonl` | The production LLM classifier's answers on the test set (cached: it costs API credits) |

The bench reports accuracy, macro-F1, attack recall, false attacks (real requests taken for
attacks), F1 per intent, accuracy per source, language, tag and clean vs misspelt, latency p50/p95/p99 and cost
per 1k messages, and a confidence view: for thresholds 0.5 / 0.7 / 0.9, the share of
messages a classifier is that sure about and its accuracy on them. Rules are written from the intent definitions; tune them on the
generated data, never on the test set's errors, or the score stops meaning anything.
