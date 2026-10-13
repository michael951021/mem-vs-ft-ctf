# Task log
## Fix h3run: OOM (ran out of memory)
- Goal: Fix h3run: OOM (ran out of memory)
- Approach: acceptance-check-first (known-answer mem_guard test on fabricated /proc/meminfo, run against the unfixed script and failing), then split the install around the CORE pilot and add bounded, testable memory guards with full diagnostics so the next kill is diagnosable and the CORE lands first.
- Blockers: none — remaining verification (pilot-before-heavy-install ordering, memstat lines, peak ≤30GB) happens on the next box run of the GPU job; expectations recorded in the ticket's Done-when and NOTES.md.
