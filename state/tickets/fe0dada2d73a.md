# fe0dada2d73a — Isolate pilot arms per-process + checkpoint per arm

## Issue
The h3run GPU job re-runs the full A-D x 2-seed pilot from scratch on every attempt
(4 passes in .tools/h3run.joblog on 10-13: 05:25, 07:40, 09:20, 13:0x). The 05:25 and
09:20 passes died mid-pilot, discarding finished A/B cells (~6 GPU re-runs burned).
Also: the 48G cgroup OOM (peak ~47.6G) died in the arm=C window with zero per-cell
memory data, so the RAM holder is unidentified.

## Question
Can the pilot driver (a) prove each (arm,seed) cell runs in a fresh process with its
peak RSS logged, and (b) skip already-checkpointed cells on re-run?
Refuting observation: a second invocation re-runs a cell whose marker exists, or a
per-cell `cell-peak` line is missing.

## Evidence
- scripts/run_pilot.sh:30-38 — the loop ALREADY spawns one `python -m
  harness.run_experiment` subprocess per cell (per-process premise holds; same in
  ee1b774 pre-failure). Missing: per-cell peak-RSS logging and ANY skip/checkpoint.
- .tools/h3run.joblog — 4 full pilot passes; cgroup reads "?" in-sandbox, so the
  47.6G attribution needs the per-cell RSS data this ticket adds.
- Hypothesis (unverified): 47.6G is NOT the pilot's python cells (each is a tiny
  HTTP-inference client). Per-cell peak RSS on next run will confirm/refute.
- Check: 2-arm MOCK pilot via scripts/cell_runner.py + skip logic; evidence in
  .agent-evidence/pilot-ckpt-*.log.

## Done when
`ARMS='A B' SEEDS='0' MOCK=1 TAG=ckpttest OUT=<tmp>/runs CKPT_DIR=<tmp>/ckpt \
PY=.venv-gpu/bin/python bash scripts/run_pilot.sh` (1) logs one
`cell-peak arm=.. seed=.. pid=.. peak_rss_kb=..` per cell with distinct pids;
(2) re-run after deleting one cell's run dir logs `SKIP` for it, restores the dir,
and does not re-run it. Regression: CKPT_DIR empty => old behavior (test in
scripts/tests/test_pilot_ckpt.py).

## Stop when
Scope: scripts/run_pilot.sh, scripts/cell_runner.py, scripts/tests/test_pilot_ckpt.py.
Stop/split if skip-restore semantics break analyze/build_sft on restored dirs
(they need attempts.jsonl + raw/ + memory_frozen.json). Two attempts w/o new
evidence => record blocker and stop.

## Plan
1. Write acceptance test; run now (expect FAIL: no cell_runner/SKIP yet).
2. scripts/cell_runner.py: run cell argv as one child, pass output+rc through,
   print its pid + ru_maxrss (Linux: KB).
3. run_pilot.sh: per cell — marker matches params => SKIP + restore run dir;
   else run via cell_runner (tee'd), on rc=0 + "DONE ->" copy run dir into
   checkpoint store and write marker.
4. Re-run acceptance (both legs) + existing tests; save evidence.
