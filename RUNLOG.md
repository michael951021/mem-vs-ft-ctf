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
