---
name: llm-redteaming
description: Use when testing LLM applications or agentic systems for prompt injection, jailbreaks, tool abuse, data leakage, and eval regressions. Uses garak/promptfoo and the configured MCP servers; writes all artifacts under pentest/.
license: MIT
metadata:
  owner: mayflowergmbh/kali-ai-redteam
---

# LLM Red Teaming Runbook

## Inputs To Collect

- Target base URL / API endpoint(s)
- Auth method (key, cookie, OAuth), rate limits, and scope constraints
- Model/provider (if known) and tool/plugin surface (if any)

## Baseline

1. Run `/recon <target-url>` and save output under `pentest/recon/`.
2. Record request/response schema (including streaming) and any moderation behavior.

## Automated Coverage

- `garak` for probe-driven scanning; save to `pentest/scans/garak-*`.
- `promptfoo` for eval suites/redteam runs; save to `pentest/scans/promptfoo-*`.

## Manual Exploitation

- Use `/probe` for quick injections and `/extract` for system prompt extraction attempts.
- Use Playwright MCP for headless flows (login, multi-step prompt injection, indirect injection via UI content).
- Persist key artifacts in `pentest/prompts/` and `pentest/exploits/` (full payload + full response).

## Reporting

- Map findings to OWASP LLM Top 10 categories.
- Include exact repro steps and expected/actual behavior.

## Local notes (added for this repository; not part of the upstream skill)

Source: github.com/mayflower/kali-ai-redteam at commit 6c51a12e744d0ecf84ce052fa62c0bb985772a29 (MIT). Only this skill and the
`/recon`, `/probe`, `/extract` and `/jailbreak` commands were installed; paths were changed
from `/pentest/` to `pentest/` (git-ignored). The upstream container, its permission list,
its MCP servers and the `/scan` command were not: `garak` and `promptfoo` are not installed
on this machine, so "Automated Coverage" needs them first.

- **Scope**: only this project's own assistant: the LAN demo (`https://<this machine>`,
  `make lan`) or a local `make run`. Nothing else.
- **Target API**: `POST /v1/auth/login` (username + password, session cookie), then
  `POST /v1/conversations` and `POST /v1/conversations/{id}/agent-runs` with
  `{"text": ..., "language": "es"|"pt"|"en"}`; the reply is an SSE stream (`delta`,
  `tool_call`, `done` with `outcome`). TLS uses the private CA in `deploy/tls/ca.crt`.
- **Limits**: nginx allows 30 requests/min per IP (6/min on login) and the app 10/min per
  user; five failed logins lock an account for 15 minutes.
- **What "detected" means here**: `outcome: "blocked"` on `done`, and in Langfuse a
  `prompt_injection` score on the trace.
- **Keep what works**: a payload that gets through goes into `evals/security.jsonl`
  (`make eval-security`) so it becomes a regression test.
- Test accounts and their passwords are in `credentials/`; never copy a password or
  customer data into `pentest/` artifacts.
