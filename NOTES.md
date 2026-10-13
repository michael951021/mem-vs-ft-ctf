# NOTES — current state

## Task: fix-h3run-45f7fb3f (Fix h3run: OOM) — 2nd pass
h3run = `bash scripts/gpu_job.sh` (loop GPU job, gpu/mem:48/10h). 1st-pass fix
(split install + memstat/mem_guard + heartbeat) merged; job re-ran 3x on 10-13
(00:05, 02:15, 03:40) and jobtriage still reported "OOM" at 04:00.

### New root cause (2nd pass)
00:05 run passed pilot + SFT (12 records) + heavy install (5.8GB venv, 00:28),
then QLoRA died at its FIRST heavy import — before the 15GB HF weight download
(no 10-13 blobs in shared HF cache). Reproduced here:
`ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`
(venv held numpy 1.26.4 + pyarrow 26.0.0 — unpinned uv resolution; torch 2.4.1
needs numpy 1.x, datasets 2.21's pyarrow 26 needs numpy 2 → unsatisfiable pair).
Consequences: step-4b import check could never pass (uv saw pins "satisfied"),
train_qlora.py:122 `from datasets import load_dataset` → clean exit 2 every run.
The 02:15/03:40 runs died mid-pilot (before build_sft); job log is host-only
(unreadable here) → undiagnosable. "oom-kill, code TERM" may be host-wide
preemption (TERM ≠ OOM SIGKILL) or clean exits mislabeled by jobtriage.

### Fix (this pass)
- Shared .venv-gpu repaired: `uv pip install 'numpy<2' 'pyarrow<19'` →
  pyarrow 18.1.0, numpy stays 1.26.4.
- scripts/gpu_job.sh: explicit `NUMPY_PIN=numpy==1.26.4` +
  `PYARROW_PIN=pyarrow==18.1.0` in light+heavy install lines (fresh builds can't
  re-resolve the conflict); heavy import check is now the source of truth — on
  failure it dumps the traceback tail to the diag file and exits 2.
- Diagnosability: main() tees the FULL job log (memstat, heartbeat, child
  stderr) to shared bind `.tools/h3run.joblog` (mktemp fallback) — the next
  kill is readable from agent sandboxes; rc=$? on every FATAL line.
- 1st-pass design unchanged (split install, mem_guard 16GB/600s QLoRA,
  32GB/300s merge skip, 2-min heartbeat, oll unload verify).

### Verified here (no GPU/Ollama model in this sandbox)
- bash -n rc 0; script source-safe (BASH_SOURCE guard).
- Heavy 7-package import on REAL venv rc 0 (torch 2.4.1+cu121, datasets 2.21.0,
  pyarrow 18.1.0/numpy 1.26.4) — .agent-evidence/h3run-2ndpass-acceptance.log.
- Known answer: `load_dataset(json, finetune/data/sft_all.jsonl)` = 12 rows
  (= manifest n_records), same log.
- Script's exact pinned heavy-install line on real venv: no-op, rc 0.
- `python3 -m scripts.tests.test_gpu_job_guard` 6/6 PASS;
  `python3 -m memory.tests.test_arms` PASS.

### Next box run (watch)
- `.tools/h3run.joblog` now readable from sandbox → next kill is diagnosable;
  check it if the job dies again.
- Expect: pilot A-D → 12 SFT → heavy import passes → 15GB HF 7B download →
  adapter in finetune/adapters/E-qwen7b + train_meta.json (measured gpu-hours).
- If pilot-window kills recur with no memory fault in the tee log, file a
  SEPARATE ticket (loop-harness preemption), per ticket Stop-when.

## Conventions
- Loop jobs run "no agent" via lib/jobs.py; shared binds: .tools, .venv-gpu,
  finetune/adapters, finetune/data, ~/.cache/huggingface (53GB). Committed 27B
  *-pilot artifacts never touched. Job logs: host-only (one-way loop bind).
- Tests: plain python modules, `python3 -m <pkg>.tests.<name>` (no pytest).
- Ollama in agent sandbox: no model served; use $OLLAMA_BASE_URL or
  $OLLAMA_HOST. Direct curl to 127.0.0.1:11434 is hook-blocked.
