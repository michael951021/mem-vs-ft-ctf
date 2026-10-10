# Study Brief (operational condensation)

Originating spec condensed to its operational core. Full prior-art citations live in `REPORT.md §3`.

## Research question
Can an agent progressively abstract verified experiences into reusable knowledge that transfers
across related task distributions, and how does that compare with QLoRA under matched compute, in
CTF environments where flags/execution give ground-truth feedback?

## Hypotheses (falsifiable; a clean negative is publishable)
- **H1 (in-distribution):** D > C > B > A on held-out same-family tasks.
- **H2 (transfer — core claim):** trained on family X, evaluated on sibling family Y (shared
  mechanism, different surface), D retains more of its in-distribution advantage than B/C.
- **H3 (memory vs weights, compute-matched):** at equal GPU-hours, QLoRA (E) and memory (D) sit at
  different points on the capability/efficiency frontier; hybrid (F) may dominate at higher cost.
  Characterize the frontier; do not assume a winner.

## Novelty vs. CyberEvolver (arXiv:2605.26195)
(1) controlled **cross-mechanism** transfer on mechanism-grouped families (they said benchmarks
group too coarsely to isolate this); (2) head-to-head vs **compute-matched QLoRA** (they do not).
Build on CyberEvolver; do not re-sell what it showed.

## Arms — see README. Arm D is the centerpiece (ACE-style Generator/Reflector/Curator, delta
updates, no full rewrites; Reflector conditions on the verifiable verdict, never self-assessment).
- D ablation (B4): hierarchical-promote (default, keep exemplars) vs compress-everything.
- Feedback granularity (B5): subtask/milestone signal vs flag-only.

## Testbeds (brief §6)
- Pilot: **InterCode-CTF** (100 picoCTF, Docker, flag reward). Stand up first.
- Headline: **NYU-CTF** (200) + **Cybench** (40, 17 subtask-annotated).

## Mechanism families + transfer protocol
Tag each challenge by *vulnerability mechanism* (e.g. crypto/padding-oracle, pwn/ret2win,
web/SQLi-blind), not coarse category. Strong-LLM tagging pass + manual verification; log the map
(a data artifact). Sibling families = same parent mechanism, different surface. Protocol:
consolidate/fine-tune on X only → freeze → eval on held-out sibling Y.
**Transfer retention = (advantage over A on Y) ÷ (advantage over A on X).**
Leakage control: split by challenge; decontaminate any training corpus against all eval
challenges; log removals. Frozen memory/adapter built only from X's trajectories.

## Metrics (log all per run; schema docs/run_schema.md)
success & pass@k; subtask/milestone progress; sample-efficiency curve; steps-to-solve;
tokens/task, memory size over time, wall-clock + GPU-hours.

## Compute-matching (H3 crux)
ONE budget axis = **total GPU-hours on the 2×3090 rig**, held equal across compared arms. Report
every headline comparison at matched budget AND each arm's full budget.

## Statistics
≥3 seeds (more for pilot); paired per-challenge comparisons; clustered SEs (families/subtasks);
report CIs not just points; fix+log decoding params; don't lower temperature to fake low variance.
Power caveat: 40–200-task suites are below the ~1000-item rule for small effects — lean on paired
designs + resampling.

## Phases (each gate writes real data, commits, updates RUNLOG, then continues)
- P0 stand-up + Arm-A sanity on pilot slice (within tolerance of a published baseline).
- P1 arms B/C/D in-distribution → H1 table w/ CIs.
- P2 transfer protocol → retention table + sample-efficiency → H2.
- P3 QLoRA (E) + hybrid (F), compute-matched → H3 w/ GPU-hour accounting. (P0–P3 = min core.)
- P4 scale to NYU-CTF + Cybench, full transfer matrix.
- P5 analysis + REPORT.md.

## Ethics (non-negotiable, brief §8)
Only the benchmark's own sandboxed Docker challenges; no real-world targets; isolated bridge, no
WAN except the loopback LLM endpoint. Redact every flag from any SFT/memory text (regex + manual
spot-check ≥20%). Gate every memory write behind the verifiable verdict; keep provenance. Dual-use
statement in REPORT.md. No weaponized PoCs.

## Scope under budget pressure (B6)
Drop P4 scale-up and any 32B/70B branch first; NEVER drop seeds/stats or compute-matching. A
smaller finished study beats a half-run large one. Never report an unmeasured number.
