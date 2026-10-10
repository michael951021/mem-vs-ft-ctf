# RUNLOG.md

Chronological record of what ran, when, and where the data landed. Newest at bottom.

---

### 2026-10-10 — Phase 0 stand-up begins
- **Host measured** → `docs/host_facts.md`. Key: 2×3090 (both ~90% used by Ollama), 122 GB RAM, 1.2 TB free, **Docker absent**, Python 3.14 (need 3.11/3.12 venv), no job scheduler.
- **Inference endpoint found live**: Ollama @ `127.0.0.1:11434`. Models: qwen3.8:27b-q8, mistral-small3.2:24b, llama3.3:70b, + an obliterated 27B. No 7–14B pulled yet.
- **Ollama smoke test** (`qwen-agent`, temp 0, 8 tok): cold load ~35 s, eval ~5.9 s. Empty visible response ⇒ reasoning model with `<think>` tokens; client must handle.
- **Repo skeleton created** at working root; `git init` done; `.gitignore` in place.
- **Decisions logged**: D1–D7 (see DECISIONS.md).

#### Pending (blockers surfaced to user)
- [ ] Docker install (user runs sudo commands) → unblocks all CTF benchmarks.
- [ ] GPU freeing / Ollama stop window → required only for QLoRA arms E/F (deferred).
- [ ] D4-final: single frozen base model for all arms (leaning Qwen2.5-7B/14B-Instruct or Llama-3.1-8B).

#### Next (no blockers)
- [ ] Model client (`harness/scaffold/llm_client.py`) with thinking-token handling + OpenAI-compat path.
- [ ] Per-run JSONL logging (schema in `docs/run_schema.md`).
- [ ] Arm-A ReAct scaffold (tool loop) — dry-run against a mock challenge before Docker exists.

### 2026-10-10 — Model client built + throughput finding
- `harness/scaffold/llm_client.py`: Ollama `/api/chat`, fixed/logged decoding, `<think>` splitter
  (visible vs thinking separated; thinking tokens still billed to output_tokens). Live test OK —
  correct answer, token accounting works.
- **THROUGHPUT RED FLAG:** `qwen3.8:27b-q8` generates at ~1.3 tok/s when `num_ctx=16384` is
  requested while GPUs are ~90% occupied — the KV cache spills to CPU. 200-tok generations exceed
  60 s each. At 30–40 turns/challenge this makes the full A–D × 100-task × 3-seed pilot infeasible
  on the 27B-Q8 while Ollama shares the cards with the current workload.
- **Implication (strengthens D4-final → small base):** adopt a 7–14B instruct base for ALL arms.
  At Q4/Q8 a 7–14B fits one 3090 with room for a long KV cache → tens of tok/s, and matches the
  brief's QLoRA hardware band so H3 compares the SAME base family. Blocked only on GPU headroom to
  pull/serve it (deferred with the other heavy items).

### 2026-10-10 — SESSION HANDOFF (read this first on resume)
A fresh session should start here. State of play:
- **Guardrails configured** in `.claude/settings.local.json` (deny-list + bwrap sandbox, D8).
  `socat` is now installed. These load ONLY when Claude is launched with this project as its
  workspace: `cd /home/kevin/Documents/coders/mem-vs-ft-ctf && claude`. Verify sandbox is live:
  `echo x > ~/Documents/coders/_x.txt` should be DENIED; `curl -s http://127.0.0.1:11434/api/tags`
  should work. (If a plain launch, the guardrails are NOT active.)
- **Built + committed (Gate 0 code):** llm_client.py, react_agent.py (Arms A scaffold), logger.py,
  offline smoke test (passes, no GPU). Commits b119ca9, 869b616.
- **Live inference:** Ollama @ 127.0.0.1:11434. Base model for A–D build = qwen3.8:27b (SLOW,
  ~1.3 tok/s under GPU contention — see throughput finding). D4-final (pick 7–14B base) still open.
- **Still pending (external):** Docker install (user sudo, for benchmarks); GPU window for QLoRA
  (arms E/F) + serving a faster 7–14B base.
- **NEXT ACTION (no blockers):** build Arm B (raw-ICL), C (retrieval episodic), D (ACE playbook)
  learning modules in `memory/`, each unit-testable offline with the mock harness pattern in
  `harness/scaffold/tests/test_smoke_loop.py`. Then the `queue/` runner for deferred heavy runs.

### 2026-10-10 — Phase 1 build-out + live pilot (this session)
Docker still absent; GPUs still ~90% held by Ollama (free ~1.5–2.4 GB/card). Did
NOT block: built the full A–D harness, validated it offline, and launched a REAL
pilot on the local suite against the resident model.

- **Gate 0 (offline pipeline)** PASS: `env/local_suite` (12 challenges, 5 parent
  mechanisms, sibling surfaces, ground-truth checker); arms B/C/D
  (`memory/arms.py`) with learn-on-verdict / transfer / redact / freeze /
  save-load tests; driver `harness/run_experiment.py` ran all 4 arms end to end
  with a mock oracle; `scripts/analyze.py` produced H1/H2 tables + bootstrap CIs
  on the mock runs. All unit tests green under `.venv` (py3.12).
- **Throughput decision (D10):** `think:false` cuts a turn from minutes to ~25s.
  Verified a single real challenge: arm A solved `enc-base64` in 3 turns / 76s.
- **Live pilot LAUNCHED** (`scripts/run_pilot.sh`, background): arms A–D × seeds
  0–2, local suite, think-off, writing incrementally to `runs/*-pilot`. Seed-major
  so a complete 4-arm comparison lands first (~3.3 h/seed under contention).
  Harvest any time: `analyze → make_figures → render_report --inject`.
- **Staged on the host GPU queue** (`jobqueue/`, D12): matched-base 7B A–D re-run
  → decontaminated+redacted SFT corpus → QLoRA arm E → import → eval E/F →
  re-analyze. `runner.py plan` = ~4.8 h; GPU-gated so nothing fights Ollama.
  `jobqueue/HUB.md` documents attaching to the-box hub.
- **Report:** `REPORT.md` written (RQ/method/ethics/limits); §5 numbers are
  auto-injected from `results/summary.json` — no hand-typed results, no placeholders.
- **Early observation / risk:** the 27B base solves the easy local challenges at
  ~100% even with no memory (arm A), which limits H1 headroom on this pilot suite.
  Watching the harder transfer surfaces; if near-ceiling it is reported as a clean
  (if uninteresting) negative, with the powered InterCode run staged for headroom.

#### Resume pointer (next session)
Pilot writes to `runs/*-pilot`. On resume: `tail /tmp/claude-1000/pilot.log`;
`.venv/bin/python -m scripts.analyze --glob '*pilot*'` then `make_figures` +
`render_report --inject`. If a GPU is free: `.venv/bin/python -m jobqueue.runner run`.
