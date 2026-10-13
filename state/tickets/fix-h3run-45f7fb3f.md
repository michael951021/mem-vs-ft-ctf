# Fix h3run: OOM (ran out of memory)

## Issue
Compute job `h3run` was OOM-killed — it exceeded its 48.0 GB memory cap (systemd MemoryMax), so every ticket waiting on its results is blocked. This is fixable: either cut the peak memory (e.g. micro-batch the log-probs/backward so the full [rows, T, vocab] tensor is never held) or raise the job line's `(mem:)` cap if the peak is genuinely needed.

## Evidence
Exit: `oom-kill`, code TERM; memory cap 48.0 GB.
Failing command: `bash scripts/gpu_job.sh`
Last console output (the UI does not show this):
  froze memory -> runs/20261013-035328-B-local-suite-s1-h3base/memory_frozen.json
[20261013-035328-B-local-suite-s1-h3base] TEST in-distribution on 5: ['enc-hex', 'cls-caesar', 'xor-single', 'for-grep', 'rev-logic']
  indist enc-hex      solved=True turns=3
  indist cls-caesar   solved=False turns=15
  indist xor-single   solved=False turns=15
  indist for-grep     solved=False turns=15
  indist rev-logic    solved=True turns=4
[20261013-035328-B-local-suite-s1-h3base] TEST transfer on 5: ['enc-base32', 'cls-atbash', 'xor-repeat', 'for-split', 'rev-arith']
  transfer enc-base32   solved=True turns=4
  transfer cls-atbash   solved=True turns=3
  transfer xor-repeat   solved=False turns=15
  transfer for-split    solved=True turns=3
  transfer rev-arith    solved=False turns=15
[20261013-035328-B-local-suite-s1-h3base] DONE -> runs/20261013-035328-B-local-suite-s1-h3base
Full log: /home/kevin/Documents/coders/loop/logs/jobs/mem-vs-ft-ctf-h3run.log

## Done when
Peak RSS measured and recorded, and the run completes (or `--steps 2` finishes) within the job's `(mem:)` cap. If you raise the cap instead of cutting memory, say why and record the peak.

## Stop when
After two attempts with no new evidence, record the blocker and stop — the signature then escalates to a human
card. Reproduce cheaply first (tiny config / small input), not by re-running the whole job.

## Plan
1. Reproduce the failure with the smallest/fastest invocation.
2. Fix the cause (trim memory / fix the command); add a fast check.
3. Make sure the fix reaches the job's branch so its retry picks it up.

---

## Second pass (agent/w1, 2026-10-13) — fix did not resolve; new root cause found

### Question
Why is h3run still dying after the split-install fix? Hypothesis: the OOM is no longer in the
install window — a deterministic ImportError in the QLoRA step now blocks CORE regardless of RAM.
Refutation: if the shared venv's heavy stack imported cleanly, the block is elsewhere.

### Evidence (observed this run, from shared binds — job log remains host-only, unreadable here)
- Fixed job re-ran 3x: 00:05, ~02:15, ~03:40 (10-13); jobtriage re-queued 04:00 ("OOM").
- 00:05 run got far: .venv-gpu rebuilt 00:05, full heavy stack on disk 00:28 (5.8GB), SFT corpus
  written 00:28 (12 records) → pilot + SFT + heavy install all passed. No adapter, no 7B HF-weight
  blobs dated 10-13 in the shared HF cache → QLoRA never reached the model download.
- REPRODUCED HERE: `.venv-gpu/bin/python -c "import torch,transformers,peft,bitsandbytes,trl,datasets,accelerate"`
  → `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4` (torch/peft/bnb/trl import fine;
  the chain dies at datasets→pyarrow).
- Cause: the venv holds numpy 1.26.4 (required by torch 2.4.1, which breaks under numpy 2.x) but
  datasets 2.21.0 pulled pyarrow 26.0.0, which needs numpy ≥ 2.0. Unpinned `uv pip install` made an
  unsatisfiable pair. Consequences: the step-4b import check NEVER passes (uv re-install is a
  no-op, all pins "satisfied"), and train_qlora.py:122 `from datasets import load_dataset`
  ImportError → clean exit 2 every run → the CORE adapter can never land.
- 02:15/03:40 runs died BEFORE build_sft (SFT mtime unchanged) → killed mid-pilot; with host-only
  logs this is undiagnosable. The ticket's "oom-kill" may partly reflect those kills (or jobtriage
  reusing the 13:20 signature; the original was "code TERM", not the OOM killer's SIGKILL).

### Done when (second pass)
1. Heavy import check passes on the REAL shared venv: full 7-package import, rc 0.
2. Known-answer check on real input: `datasets.load_dataset` on finetune/data/sft_all.jsonl
   returns exactly 12 rows (the corpus count is known from its manifest, independent of my code).
3. Fresh-build reproducibility: the script's heavy install line, run on the real venv, leaves a
   compatible set (pins are explicit, so uv cannot re-resolve the conflict).
4. Diagnosability: the job tees its full log (incl. memstat + heartbeat + child rc on FATAL) to a
   shared bind, so the NEXT kill is readable from the agent sandbox.
5. Regression: existing `scripts.tests.test_gpu_job_guard` (6/6) and `memory.tests.test_arms` pass.

### Stop when
Scope: scripts/gpu_job.sh install lines + log tee, the shared venv, ticket/NOTES/tests. NOT
touching: torch/bnb pins (2.4.1/0.43.3 proven to import), train_qlora.py, run_pilot.sh, 27B
artifacts. If the pilot-window kills prove to be host-wide preemption (visible once the tee log
exists), file a separate ticket — that is loop-harness territory, not this script's.
