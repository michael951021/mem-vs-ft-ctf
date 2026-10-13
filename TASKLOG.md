# Task log
## Fix h3run: OOM (ran out of memory)
- Goal: Fix h3run: OOM (ran out of memory)
- Approach: acceptance-check-first (known-answer mem_guard test on fabricated /proc/meminfo, run against the unfixed script and failing), then split the install around the CORE pilot and add bounded, testable memory guards with full diagnostics so the next kill is diagnosable and the CORE lands first.
- Blockers: none — remaining verification (pilot-before-heavy-install ordering, memstat lines, peak ≤30GB) happens on the next box run of the GPU job; expectations recorded in the ticket's Done-when and NOTES.md.
## Fix h3run: OOM (ran out of memory)
- Goal: Fix h3run: OOM (ran out of memory)
- Approach: Reconstructed the post-fix failure from shared-bind mtimes, reproduced the ImportError on the real venv, fixed venv + script pins, then verified with the script's own import check, a 12-row known-answer load_dataset on the real SFT corpus, a no-op run of the exact pinned install line, and the existing test suites.
- Blockers: Job logs are host-only (loop bind exposes only .git + one lib file), so the 02:15/03:40 mid-pilot kills remain unverifiable from this sandbox — the new tee log makes the next such kill diagnosable; if it proves host-wide preemption it needs a separate loop-harness ticket.
