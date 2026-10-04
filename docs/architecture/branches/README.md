# Branch architectures, side by side

Each branch below builds on the one before it (a stack on top of `main`). Every diagram is
generated with [Archify](https://github.com/tt-a1i/archify) from the code of its own branch
(each box lists its source lines at that commit) and uses the **same layout**: a component
keeps its place in every diagram, so what a branch adds shows up as new boxes in otherwise
empty spots. Open the `.html` files locally in a browser for the interactive view (source
links, path tracing, dark mode); the `compare-*.html` views highlight what changed between
two consecutive branches (added, removed, changed).

| Branch | What it changes | Diagram | Compared with the previous branch |
|---|---|---|---|
| `main` | Password sign-in, classify-first agent, human handoff, masked Langfuse tracing | [html](../bankbot-architecture.html) · [png](../bankbot-architecture.png) | (earlier layout) |
| `trace-metrics` | Intent / outcome scores and release on every trace; trace reports (p50/p95/p99, cost) and graders that write scores back | [html](trace-metrics.html) · [png](trace-metrics.png) | — |
| `quick-intent` | `AIP_QUICK_ACTIONS=intent`: a quick-action button gives the intent, the classifier is skipped, the model still answers | [html](quick-intent.html) · [png](quick-intent.png) | [compare](compare-trace-metrics-vs-quick-intent.html) |
| `quick-direct` | `AIP_QUICK_ACTIONS=direct`: a quick action is answered from `get_products` with a fixed text, no model call | [html](quick-direct.html) · [png](quick-direct.png) | [compare](compare-quick-intent-vs-quick-direct.html) |
| `intent-classifiers` | `AIP_INTENT_CLASSIFIER=jev_llm`: Jev (TypeSafe) classifies the masked transcript first, the LLM below 0.9 confidence; offline classifier bench; faster bank RLS | [html](intent-classifiers.html) · [png](intent-classifiers.png) | [compare](compare-quick-direct-vs-intent-classifiers.html) |
| `voice` | Push-to-talk: OpenAI speech to text, the same agent, spoken answers sentence by sentence; figures shown as numbers, spoken in words | [html](voice.html) · [png](voice.png) | [compare](compare-intent-classifiers-vs-voice.html) |

## What moves between branches

| | `trace-metrics` | `quick-intent` | `quick-direct` | `intent-classifiers` | `voice` |
|---|---|---|---|---|---|
| Who decides the intent | LLM classifier | button, else LLM | button, else LLM | Jev ≥ 0.9, else LLM | Jev ≥ 0.9, else LLM |
| Model calls for a quick action | classifier + agent | agent | none | agent (quick modes kept) | agent (quick modes kept) |
| New external services | — | — | — | Jev (TypeSafe API) | Jev, OpenAI audio |
| How the customer talks | typing, buttons (sent as text) | typing, buttons (key trusted) | typing, buttons | typing, buttons | typing, buttons, voice |
| Added to traces | intent, outcome, release | routing metadata | routing metadata | classifier, jev_confidence, jev_agrees_llm | channel, voice_stt_s, voice_first_audio_s |

Measured results behind these choices (Jev vs the LLM, voice vs text) are in
[`evals/README.md`](../../../evals/README.md).

## Regenerating

The specs (`<branch>.json`) are the Archify input. Each pins the commit it describes in
`meta.repository.revision`; after changing a branch, update the source lines and run
`archify finalize architecture <branch>.json <branch>.html --repo-root <checkout of that
branch> --quality showcase`, then `archify compare architecture <previous>.json
<branch>.json compare-<previous>-vs-<branch>.html` for the comparison view.
