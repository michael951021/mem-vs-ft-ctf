# h3report — Write up H3 in REPORT.md

## Issue
The matched-base H3 run (7B base, arms A–F) is complete and analysed in
`results/summary_h3.json`, but REPORT.md has no H3 results section; §1 still says
H3 awaits the queued runs and §9 still says the base-model mismatch is unresolved.
Readers of the report cannot see the memory-vs-weights result at all.

## Question
Can every number in a new REPORT.md H3 section be traced to
`results/summary_h3.json` (plus the two named non-summary sources: E's
`finetune/adapters/E-qwen7b/train_meta.json` GPU-hours and
`runs_superseded/README.md` exclusion count)? Refutation: any number in the
section that does not equal its source value, or is invented.

## Evidence
- `results/summary_h3.json` (140 attempts; success_rate, H1/H1b/H2, efficiency,
  sample_efficiency_curve per arm·split).
- Cross-check already done by hand: raw `runs/20261013-17*..-h3base/attempts.jsonl`
  (A s0/s1 indist 2/5+2/5=0.4; D 3/5+2/5=0.5) and
  `runs/20261014-2305..-h3/attempts.jsonl` (E indist 4/5, transfer 3/5) match the
  summary exactly.
- Acceptance check (new): `scripts/tests/test_h3_report.py` — (a) recompute pooled
  per-arm·split success from raw `runs/` and compare to summary_h3.json; (b) assert
  every number used in the REPORT.md H3 section equals the value at its source key.
  Output: `.agent-evidence/h3report-acceptance.log`.

## Done when
`python3 -m scripts.tests.test_h3_report` passes (0 failures) and the H3 section
exists between the auto-generated block and §5.3 Findings; every number in the
section appears in the check's (value, source-key) table. Known-answer checks:
raw-attempt recomputation (hand-verified above) and the E 4/5, A 4/10 spot values.
Plausible range written pre-run: pooled rates within [0.2, 0.8] (observed 0.2–0.8);
E GPU-hours < 2 h (measured 0.4375 — the 1146.88 estimate is a known-bad
placeholder, not used).

## Stop when
Scope: add the H3 section + fix the 2–3 stale H3-specific sentences; no renumbering
of existing sections, no new figures, no touching §5 auto-block or 27B artifacts.
Stop/split: if a number in the section cannot be pinned to a source value after two
attempts, mark it out of the section rather than guess.

## Plan
1. Write this ticket; run the acceptance check (fails: no H3 section).
2. Add `### 5.4 H3 — memory vs. weights on the matched 7B base` with tables
   (success, H1 advantage, H2 retention, efficiency, H1b turns saved), interpretation,
   caveats (1 E/F seed, n per cell, 8 superseded tries excluded, 0.4375 GPU-h
   measured, protocol max_turns 15 vs 20).
3. Point the two stale H3 sentences (§5.1, §9.2, §7) at §5.4.
4. Re-run the check + render_report.py regression; commit.
