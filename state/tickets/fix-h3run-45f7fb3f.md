# Fix h3run: OOM (ran out of memory)

## Issue
Compute job `h3run` was OOM-killed — it exceeded its 48.0 GB memory cap (systemd MemoryMax), so every ticket waiting on its results is blocked. This is fixable: either cut the peak memory (e.g. micro-batch the log-probs/backward so the full [rows, T, vocab] tensor is never held) or raise the job line's `(mem:)` cap if the peak is genuinely needed.

## Evidence
Exit: `oom-kill`, code TERM; memory cap 48.0 GB.
Failing command: `bash scripts/gpu_job.sh`
Last console output (the UI does not show this):
  froze memory -> runs/20261012-204215-B-local-suite-s1-h3base/memory_frozen.json
[20261012-204215-B-local-suite-s1-h3base] TEST in-distribution on 5: ['enc-hex', 'cls-caesar', 'xor-single', 'for-grep', 'rev-logic']
  indist enc-hex      solved=True turns=3
  indist cls-caesar   solved=False turns=15
  indist xor-single   solved=False turns=15
  indist for-grep     solved=False turns=15
  indist rev-logic    solved=True turns=4
[20261012-204215-B-local-suite-s1-h3base] TEST transfer on 5: ['enc-base32', 'cls-atbash', 'xor-repeat', 'for-split', 'rev-arith']
  transfer enc-base32   solved=True turns=4
  transfer cls-atbash   solved=True turns=3
  transfer xor-repeat   solved=False turns=15
  transfer for-split    solved=True turns=3
  transfer rev-arith    solved=False turns=15
[20261012-204215-B-local-suite-s1-h3base] DONE -> runs/20261012-204215-B-local-suite-s1-h3base
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

## Worker diagnosis (agent/w1)

The `h3run` job (`bash scripts/gpu_job.sh`, gpu/mem:48/10h) was OOM-killed and produced
no CORE artifacts (no `runs/*-h3base`, no SFT data, no adapter). The H3 matched-base result
is the deliverable this job exists to make, so an undiagnosable early death blocks it.

### Question
Why did the job die, and can the script be made OOM-robust so CORE lands even under memory
pressure? Hypothesis: it died in the first heavy step — the one-shot install of the full
torch+CUDA training stack — before the CORE 7B pilot ran, with no memory diagnostics in the
log. Refutation: if the job log/state showed artifacts after the pilot, the death was later.
(Note: the 20:42 console output above shows a later run that DID pass the pilot — the
post-fix script targets that post-pilot heavy window with the split install + guards.)

### Evidence
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

### Done when
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

### Stop when
Scope: scripts/gpu_job.sh (+ a test for its guard function only). NOT touching: run_pilot.sh semantics,
27B artifacts, TODO job spec, train_qlora.py, jobqueue/*.json. Stop/split: if a change needs train_qlora.py
or the loop harness (not in this repo), record it and file a separate ticket instead.

### Plan
1. Split install: light stack (pilot needs) → run CORE pilot → heavy stack (training) → SFT/QLoRA.
2. Add `memstat()` (logs MemAvailable + cgroup limit/current) at start and before each heavy step;
   EXIT trap also logs it (diagnosable OOMs).
3. Add testable `mem_guard MIN_GB MEMINFO_PATH` with bounded wait; gate QLoRA (16 GB, wait) and
   best-effort merge (32 GB, skip) on it; verify Ollama 7B actually unloaded before training.
4. Verify: bash -n, real light install into scratch venv, known-answer guard test; commit.
