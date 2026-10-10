# DECISIONS.md

Branch choices + rationale, dated. Newest first. (Brief §11 / §13.)

## 2026-10-10 — D7: No formal queue exists → build a minimal file-based one
There is no slurm/ts/at/ray on the host. Per the user's "queue the big runs for the end" instruction, `queue/` holds JSON job specs + a `runner.py` that executes them sequentially, checks GPU free-memory preconditions before each heavy job, and logs ETAs. Quick/CPU jobs run inline; heavy (GPU training, large eval sweeps) are staged, not auto-launched.

## 2026-10-10 — D6: Heavy runs deferred, not run, this session (user directive)
GPUs are committed to the live Ollama server. User directive: quick non-disruptive sanity tests now; build everything; stage big runs with ETAs for a final launch. Session goal = a fully-built, dry-validated harness + staged runs, not completed training. Honestly scoped per §14 + §11-B6.

## 2026-10-10 — D5: Inference = local Ollama endpoint (already live)
User chose local inference. Ollama is already serving at `127.0.0.1:11434`. Arms A–D call it over HTTP (OpenAI-compatible `/v1` or native `/api/generate`). No cloud API; satisfies §8 (only outbound = the LLM endpoint, and it's loopback).

## 2026-10-10 — D4: Base policy model — provisional `qwen3.8:27b` for A–D build/pilot; final base TBD at Phase 3
The resident model is 27B, outside the brief's 7–14B QLoRA band. For arms A–D (inference only) model size is not VRAM-constrained, so we build/pilot against what's loaded. BUT H3 (memory vs weights) requires the SAME base family for A–D and the QLoRA arms E/F. **Open decision (D4-final):** either (a) adopt a 7–14B instruct model (e.g. Qwen2.5-7B/14B-Instruct or Llama-3.1-8B-Instruct) as the single frozen base for ALL arms — requires pulling it into Ollama for A–D and HF safetensors for E/F — or (b) keep 27B for A–D and QLoRA the 27B via FSDP. (a) is cleaner and matches the brief's hardware guidance; leaning (a). Decide before Phase 1 headline runs.

## 2026-10-10 — D3: Docker install deferred to user (needs sudo)
Docker absent; install requires root. User opted to run install commands themselves. Harness code that doesn't need Docker proceeds; benchmark container bring-up gated on install.

## 2026-10-10 — D2: Repo root = /home/kevin/Documents/coders/mem-vs-ft-ctf
User choice.

## 2026-10-10 — D1: Pilot benchmark = InterCode-CTF (per brief §6)
Lowest-friction execution-feedback flag-scored testbed. Stand up here first for Gate 0/1.

## 2026-10-10 — D8: Host-containment guardrails for the agent (user request)
User asked to constrain Claude Code so it can't destroy the host. Configured in
`.claude/settings.local.json` (gitignored):
- `permissions.deny`: ~40 catastrophic command families (sudo, disk wipe, power, systemctl/
  service, pkill/killall, ollama rm/stop, destructive docker/git) + reads of ssh/aws/.env.
  Protects the live Ollama + GPU work from being killed.
- `disableBypassPermissionsMode` + `blockReadsOutsideWorkingDirectories`.
- `sandbox.enabled: true` (bubblewrap present) with writes confined to the project, scratchpad,
  ~/.cache, ~/.ollama; network egress limited to pip/github/hf/ollama + loopback.
Prereqs for the sandbox: `socat` (MISSING — must install) for the network proxy; takes effect
only after a Claude Code restart/reload (`.claude/` did not exist at session start).

## 2026-10-10 — D9: Learning-module design + heuristic Reflector (arms B/C/D)
Arms share the LearningModule protocol + freeze/save/load (for the transfer
protocol: build on X, freeze, eval on Y). Consolidation gates on
`verdict["flag_verified"]` only. Arm D's Reflector is a deterministic HEURISTIC
by default (distils the verified winning trajectory into one generalised bullet)
so consolidation costs ZERO extra model calls — essential under ~1 tok/s
contention. An LLM-reflector is injectable (`Playbook(reflector_llm=…)`) as a
GPU-window ablation. Curator does delta updates (add/reinforce/demote) + dedup +
per-family cap; hierarchical vs compress-everything is the B4 ablation knob.

## 2026-10-10 — D10: Reasoning think-mode OFF for the pilot
`qwen3.8:27b` is a reasoning model; think tokens made turns take minutes under
GPU contention. Setting Ollama `think:false` cuts turns to ~10–60 tokens (~25s).
This is a fixed, logged decoding choice applied IDENTICALLY to every arm, so
inter-arm comparisons stay fair. Logged in each run's config snapshot.

## 2026-10-10 — D11: Docker-free local CTF suite as the measurable pilot testbed
Docker is still absent (blocker for InterCode). Rather than block, built
`env/local_suite/`: 12 benign picoCTF-style challenges across 5 parent mechanisms
with sibling surfaces, ground-truth flag checking, runnable in the bwrap sandbox.
This yields REAL measured A–D data now and exercises the full transfer protocol.
InterCode adapter is built + staged (`env/intercode/adapter.py`) for when Docker
lands. Per §11-B6: a smaller finished study on a real testbed beats a half-run one.

## 2026-10-10 — D12: Hub enqueue mechanism = host GPU queue, not a pool sandbox loop
Investigated the-box hub (`~/Documents/coders/loop`). The pool auto-discovers
`loop/projects/<name>` (git + TODO.md, non-symlink) and runs SANDBOXED
Claude-on-Ollama loops (`/work` only, `--unshare-all`, inference via a bound
ollama socket). Real GPU TRAINING is host-only there (loopgym's GPU runs are
host-only; sandbox only dry-runs). So the study's GPU-heavy arms are staged in
`jobqueue/` and run on the host by `jobqueue/runner.py` with per-job GPU
free-memory preconditions. `jobqueue/HUB.md` documents attaching to the hub.
(Renamed `queue/`→`jobqueue/` to avoid shadowing Python's stdlib `queue`.)

## 2026-10-10 — D13: H3 base = Qwen2.5-7B-Instruct, with a matched-base re-run (resolves D4-final)
For a clean compute-matched H3 the staged queue FIRST re-runs A–D on a single
7–14B base (Qwen2.5-7B-Instruct, fits one 3090 in 4-bit), THEN QLoRAs that same
base, so every arm (A–F) shares one frozen base. A–D's 27B pilot stands as the
inference-only in-distribution/transfer result; H3 is reported from the matched
7B pipeline once a GPU frees.
