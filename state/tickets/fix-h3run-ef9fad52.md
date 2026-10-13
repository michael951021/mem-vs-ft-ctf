# fix-h3run-ef9fad52: Fix h3run: non-zero exit (exit-code)

## Issue
h3run (bash scripts/gpu_job.sh; 64G cap, 16 cpus, 10h) exits non-zero. In
.tools/h3run.joblog, 3 of 4 runs on 2026-10-13 (05:25, 09:20, 13:05) die with
NO exit trace (bash EXIT traps don't run on untrapped SIGKILL/SIGTERM) at
T+16.3-18.0 min, exactly at arm=C seed=1 start (attempts.jsonl 0-1 of 15
lines). The 07:40 run passed the same point. Exit-code reporting in the log
is also broken (below).

## Question
Why does the job exit non-zero, and how do we make the exit code truthful and
the next kill diagnosable from the log?

## Evidence
- 13:05 run ran under the 64G cap (job/h3run merged master after 0ff74e2 at
  13:05:03) yet died identically -> the 48G cgroup-OOM theory is refuted for
  that run (it may still explain the two 48G-cap deaths).
- Output volume/pace identical across all 4 runs (8.6-8.7 KB by the C-s1
  banner, ~530 B/min) -> output-based stall-watchdog / tee-buffering
  theories refuted: no buffer size distinguishes the survivor from the dead.
- Leading story: monolithic pilot RSS accumulation peaked ~47.6G at arm C
  (fix-h3run-45f7fb3f); per-run variance decides who dies. Per-cell process
  isolation + checkpoint (240dfe7, merged 14:14) structurally neutralizes it
  but has NEVER run in production (no cell-peak lines in the log yet).
- _on_exit bug (gpu_job.sh:72-76): it runs `kill "$HB"` before reading `$?`,
  so the exit line always shows the kill's rc, not the job's (live in log
  line 412: "memstat[exit rc=0]" immediately after `exit 2`) -- the
  "exit-code" in this ticket's title is literally wrong in the log.

## Done when
- A non-zero job exit is logged with the TRUE rc: `memstat[exit rc=N]`
  reports N (test fails pre-fix, passes post-fix).
- Each 120s heartbeat line carries the process-tree RSS sample
  (`tree_rss_kb=`) so the next kill shows a memory ramp (memory killer) or a
  flat line (external) -- known-answer fake-/proc test.
- Regressions pass: bash -n, test_gpu_job_guard, test_pilot_ckpt.
- Honest limit: proving the job completes needs the next real GPU run (no
  GPU/Ollama in agent sandboxes).

## Stop when
Two attempts without new evidence, or the next real run lands cell-peak +
heartbeat-RSS lines contradicting this diagnosis.

## Plan
1. Acceptance test scripts/tests/test_gpu_job_exit.py; run pre-fix (FAIL).
2. gpu_job.sh: `_on_exit` captures rc=$? as its first line; add
   tree_rss_kb ROOT [PROC_DIR] (sum VmRSS over /proc descendants, no ps)
   and route the heartbeat through hb_once which appends tree_rss_kb=;
   memstat cgroup fallback "?" -> "absent".
3. Run test + regressions; commit; git merge master.
