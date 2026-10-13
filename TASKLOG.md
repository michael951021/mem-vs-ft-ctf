# Task log
## Fix h3run: OOM (ran out of memory)
- Goal: Fix h3run: OOM (ran out of memory)
- Approach: acceptance-check-first (known-answer mem_guard test on fabricated /proc/meminfo, run against the unfixed script and failing), then split the install around the CORE pilot and add bounded, testable memory guards with full diagnostics so the next kill is diagnosable and the CORE lands first.
- Blockers: none — remaining verification (pilot-before-heavy-install ordering, memstat lines, peak ≤30GB) happens on the next box run of the GPU job; expectations recorded in the ticket's Done-when and NOTES.md.
## Fix h3run: OOM (ran out of memory)
- Goal: Fix h3run: OOM (ran out of memory)
- Approach: Reconstructed the post-fix failure from shared-bind mtimes, reproduced the ImportError on the real venv, fixed venv + script pins, then verified with the script's own import check, a 12-row known-answer load_dataset on the real SFT corpus, a no-op run of the exact pinned install line, and the existing test suites.
- Blockers: Job logs are host-only (loop bind exposes only .git + one lib file), so the 02:15/03:40 mid-pilot kills remain unverifiable from this sandbox — the new tee log makes the next such kill diagnosable; if it proves host-wide preemption it needs a separate loop-harness ticket.
## Fix h3run: RuntimeError in train_qlora.py
- Goal: Fix h3run: RuntimeError in train_qlora.py
- Approach: Reproduced the job-log traceback via the script's real import block on the shared venv, installed the missing module, pinned it, and made the import check the actual training import chain; re-verified pre/post, bash -n, guard tests, and a no-op install.
- Blockers: none
## Isolate pilot arms per-process + checkpoint per arm
- Goal: Isolate pilot arms per-process + checkpoint per arm
- Approach: scripts/cell_runner.py runs each cell as one reaped child and prints cell-peak arm=.. seed=.. pid=.. peak_rss_kb=.. (exact ru_maxrss per cell — the data the 47.6G OOM attribution needs; mock cells peak ~21MB, pointing the OOM holder elsewhere). scripts/run_pilot.sh copies each rc=0 run dir to .tools/pilot_ckpt/$TAG (a shared bind; runs/ is per-job and ephemeral) with a params-fingerprinted marker; matching markers trigger SKIP + byte-identical restore, stale ones re-run. CKPT_DIR= disables (dash form), MOCK=1 enables offline oracle cells. Acceptance test scripts/tests/test_pilot_ckpt.py: 3 legs — fresh 2-arm run (2 distinct pids, RSS, checkpoints), delete-and-rerun (2x SKIP, restored byte-identical, 0 re-runs), legacy disable path (0 checkpointed, no pollution).
- Blockers: None. Evidence: .agent-evidence/pilot-ckpt-acceptance-prefix.log (pre-fix FAIL, 9 failures), pilot-ckpt-acceptance-final.log (PASS 0), pilot-ckpt-run{1,2,3}.log (per-leg). Regression: test_gpu_job_guard 6/6, bash -n rc 0. Open item for the next real h3run: read the cell-peak lines in .tools/h3run.joblog to attribute the 47.6G OOM. Committed 154882b; merge master already up to date. results/redaction_audit.json left unstaged (harvest() side effect, outside ticket scope).
## Fix h3run: non-zero exit (exit-code)
- Goal: Fix h3run: non-zero exit (exit-code)
- Approach: proved the kill timeline and refuted output/cap theories from .tools/h3run.joblog, wrote a failing acceptance test first (test_gpu_job_exit: true-rc trap test + known-answer fake-/proc tree-RSS test), then fixed gpu_job.sh and verified with regressions.
- Blockers: none (true acceptance that the job completes needs the next real GPU run — no GPU/Ollama in agent sandboxes; the already-merged per-cell isolation + 64G cap address the documented memory root cause and will emit the attribution lines on that run).
