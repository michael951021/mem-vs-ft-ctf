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
