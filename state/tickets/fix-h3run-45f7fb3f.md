# Fix h3run: OOM (ran out of memory)

## Issue
The `h3run` job (`bash scripts/gpu_job.sh`, gpu/mem:48/10h) was OOM-killed and produced
no CORE artifacts (no `runs/*-h3base`, no SFT data, no adapter). The H3 matched-base result
is the deliverable this job exists to make, so an undiagnosable early death blocks it.

## Question
Why did the job die, and can the script be made OOM-robust so CORE lands even under memory
pressure? Hypothesis: it died in the first heavy step — the one-shot install of the full
torch+CUDA training stack — before the CORE 7B pilot ran, with no memory diagnostics in the
log. Refutation: if the job log/state showed artifacts after the pilot, the death was later.

## Evidence
Observed (job log itself unavailable — job worktree pruned; state from shared binds):
- `.venv-gpu` (shared): venv skeleton only, NO site-packages, `.lock` present → install never completed.
- `~/.cache/uv/.tmp{VNxBqw,a0viWR}` @ 10-12 13:20: partial torch (957M) + nvidia-cudnn (937M)
  extraction → install SIGKILLed mid-extraction; a retry's download cascade ran 13:21–13:36 (~15 min).
- `finetune/data`, `finetune/adapters`, `results/*h3*` all empty → never passed the pilot.
- MEASURED (this run): `uv pip install` of the exact stack → peak RSS 75.6 MB, 2m36s warm-cache
  (`.agent-evidence/h3run-install-rss.log`). The install process alone cannot OOM a 48 GB cgroup;
  the kill came from sandbox-wide pressure, and the script logs NO memory state, so it is undiagnosable.
Guess (untested): host Ollama + other sandboxes push total host RAM past 122 GB during the
15-min download/install window, or the loop preempts on the declared mem:48.

## Done when
1. `bash -n scripts/gpu_job.sh` → rc 0.
2. Real-input install split works: script's light step installs pyyaml/numpy/scipy/matplotlib into
   a scratch venv from real PyPI (rc 0, imports pass); heavy set already resolves to all wheels
   (dry-run evidence in the same log dir).
3. Known-answer guard check: the new mem-guard function, fed fabricated /proc/meminfo files,
   returns pass/fail exactly at the threshold (32 GB → pass boundary, 15 GB → fail) — tested in-repo.
4. Expected post-fix behavior (verified next box run, not here — no GPU/Ollama model in this sandbox):
   pilot CORE lands before the heavy install; memstat lines in the job log show MemAvailable before
   each heavy step; plausible peak ≤ 30 GB (QLoRA ~10 GB, merge ~20 GB, sequential) so a 32 GB guard
   is non-binding unless host pressure is external — which the new logs would then show.

## Stop when
Scope: scripts/gpu_job.sh (+ a test for its guard function only). NOT touching: run_pilot.sh semantics,
27B artifacts, TODO job spec, train_qlora.py, jobqueue/*.json. Stop/split: if a change needs train_qlora.py
or the loop harness (not in this repo), record it and file a separate ticket instead.

## Plan
1. Split install: light stack (pilot needs) → run CORE pilot → heavy stack (training) → SFT/QLoRA.
2. Add `memstat()` (logs MemAvailable + cgroup limit/current) at start and before each heavy step;
   EXIT trap also logs it (diagnosable OOMs).
3. Add testable `mem_guard MIN_GB MEMINFO_PATH` with bounded wait; gate QLoRA (16 GB, wait) and
   best-effort merge (32 GB, skip) on it; verify Ollama 7B actually unloaded before training.
4. Verify: bash -n, real light install into scratch venv, known-answer guard test; commit.
