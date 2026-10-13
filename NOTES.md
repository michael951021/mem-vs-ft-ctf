# NOTES — current state

## Task: fe0dada2d73a (Isolate pilot arms per-process + checkpoint per arm)
run_pilot.sh ALREADY ran each (arm,seed) cell in its own subprocess (old lines
30-38; also in ee1b774) — the missing parts were per-cell peak-RSS logging and
any skip/checkpoint, so a job killed mid-pilot re-ran finished cells (4 full
pilot passes in .tools/h3run.joblog 10-13; ~6 GPU re-runs burned).

### Changes
- scripts/cell_runner.py (new): runs one cell as a single child, streams its
  output, then prints `cell-peak arm=.. seed=.. pid=.. peak_rss_kb=.. rc=..`
  (ru_maxrss of the reaped child = exact per-cell high-water RSS).
- scripts/run_pilot.sh: each cell runs via cell_runner; on rc=0 the run dir
  (from the `DONE ->` line) is copied to the checkpoint store and a marker
  (`$key.done`: line 1 run-dir name, line 2 PARAMS fingerprint) is written.
  Re-run: marker+params match => restore run dir + `SKIP` (no re-run); stale
  params => WARN + re-run. Store = .tools/pilot_ckpt/$TAG (shared bind;
  runs/ is per-job and NOT shared). `CKPT_DIR=` (empty) disables
  (dash form `${CKPT_DIR-...}`, not colon). `MOCK=1` = offline oracle cells
  (no GPU/Ollama) for tests/dry-runs.

### Verified (no GPU)
- Pre-fix acceptance FAIL (9 failures, expected) →
  .agent-evidence/pilot-ckpt-acceptance-prefix.log.
- Post-fix `python3 -m scripts.tests.test_pilot_ckpt` PASS 0 failures →
  pilot-ckpt-acceptance-final.log; per-leg logs pilot-ckpt-run{1,2,3}.log:
  run1 = 2 cells, distinct pids, peak_rss_kb ~21MB, both checkpointed;
  run2 = 2x SKIP, 0 re-runs, deleted dir restored byte-identical;
  run3 (CKPT_DIR="") = 2 cells, 0 checkpointed (disable path clean).
- bash -n rc 0; `python3 -m scripts.tests.test_gpu_job_guard` 6/6 PASS.
- Mock cell peak ~21MB supports the hypothesis that the 47.6G cgroup OOM
  holder is NOT the pilot python cells (they are tiny HTTP-inference clients).

### Open (watch on next real h3run)
- Attribute the 47.6G OOM: read the per-cell `cell-peak` lines in
  .tools/h3run.joblog; mock says cells are ~21MB, so expect the holder to be
  elsewhere (Ollama server? SFT build? HF download?).
- h3run (separate ticket, merged 73f2097): est_gpu_hours placeholder still
  uncalibrated — needs first real train_meta.json.

## Task: fix-h3run-ef9fad52 (Fix h3run: non-zero exit)
3 of 4 runs 10-13 died T+16.3-18min at arm=C seed=1 start, NO exit trace
(SIGKILL skips bash EXIT traps); 13:05 died the same way under the 64G cap
(=> 48G-OOM theory refuted for it); output bytes/pace identical across runs
(=> output-watchdog refuted). Leading cause still monolithic-pilot RSS
accumulation (47.6G peak at arm C) — neutralized by per-cell isolation
(240dfe7) but never yet run in production.
- gpu_job.sh: `_on_exit` now captures rc=$? FIRST (pre-fix `kill $HB`
  clobbered it: log L412 "exit rc=0" after `exit 2`); new tree_rss_kb
  (sum VmRSS over /proc descendants, no ps) + heartbeat routed via hb_once
  appending `tree_rss_kb=` => next kill: ramping = memory killer, flat =
  external; memstat cgroup fallback `?` -> `absent`.
- Verified: pre-fix acceptance 5 FAIL (gpu-job-exit-acceptance-prefix.log);
  post-fix test_gpu_job_exit 6/6 + guard 6/6 + pilot_ckpt 0-fail
  (-final.log, -regressions.log); real exit-2 with live HB logs rc=2.
- Open: next real h3run is the true acceptance (no GPU in sandbox); watch
  cell-peak lines + heartbeat RSS trajectory in .tools/h3run.joblog.

## Conventions
- Loop jobs run "no agent" via lib/jobs.py; shared binds: .tools, .venv-gpu,
  finetune/adapters, finetune/data, ~/.cache/huggingface (53GB). Committed 27B
  *-pilot artifacts never touched. Job log tee'd to .tools/h3run.joblog.
- Tests: plain python modules, `python3 -m <pkg>.tests.<name>` (no pytest).
- Ollama in agent sandbox: no model served; use $OLLAMA_BASE_URL or
  $OLLAMA_HOST. Direct curl to 127.0.0.1:11434 is hook-blocked.
