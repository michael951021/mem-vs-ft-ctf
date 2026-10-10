# Memory vs. Weights for CTF Agents under Matched Compute

Empirical study: can an LLM agent "learn" a CTF skill domain **without weight updates** — by
consolidating verified experience into a compact, structured, transferable memory — and how does
that compare, at **matched compute**, to **QLoRA** fine-tuning on the same verified experience?
Domain: cybersecurity CTF (flags/execution give non-gameable, ground-truth feedback).

See the full brief in `docs/brief.md`. Research question and hypotheses H1–H3 in `REPORT.md §3`.

## Status
Phase 1 (arms A–D built + offline-validated; live pilot running on the local CTF suite).
Arms E/F (QLoRA) + the Docker benchmark scale-up are staged on the host GPU queue
(`jobqueue/`, GPU-gated) because the cards are committed to a live inference server.
See `RUNLOG.md` for the live state, `DECISIONS.md` for branch choices, `REPORT.md` for the
write-up (its §5 numbers auto-fill from logged data — no placeholders). Host facts in
`docs/host_facts.md`.

## The six arms (shared scaffold, swappable learning module)
| Arm | Learning module | Between-task behavior |
|---|---|---|
| A | none | pure scaffold baseline |
| B | raw-example ICL | prepend k full prior trajectories verbatim |
| C | retrieval episodic | embed + top-k retrieve similar trajectories |
| D | **consolidated semantic (ours)** | ACE-style Generator→Reflector→Curator playbook, delta updates |
| E | QLoRA fine-tune | SFT on verified-success trajectories; frozen at eval |
| F | hybrid | E's adapter + D's playbook |

## Layout
```
env/local_suite/   12 benign picoCTF-style challenges + ground-truth TaskEnv (no Docker)
env/intercode/     InterCode-CTF Docker adapter (staged; same TaskEnv protocol)
harness/scaffold/  frozen base policy: LLM client, ReAct loop, logger
harness/run_experiment.py   train→freeze→test(in-dist)→test(transfer) driver
memory/arms.py     arms B (raw-ICL), C (episodic), D (ACE playbook) + redact.py
finetune/          SFT corpus builder + host-only QLoRA trainer (arms E/F)
jobqueue/          host GPU-job queue + specs + HUB.md (how to attach to the-box hub)
scripts/           analyze, make_figures, render_report, audit_redaction, run_pilot.sh
configs/           QLoRA config; runs/ JSONL logs; results/ CSVs+figures+summary.json
REPORT.md          the deliverable (§5 auto-filled from logged data)
```

## Reproduce
```bash
python3.12 -m venv .venv && .venv/bin/pip install pyyaml numpy scipy matplotlib
# offline validation (no GPU/Docker):
.venv/bin/python -m memory.tests.test_arms
.venv/bin/python -m harness.scaffold.tests.test_smoke_loop
.venv/bin/python -m env.local_suite.env
# real pilot against a local Ollama endpoint (think-off, seed-pinned):
OUT=runs SEEDS="0 1 2" ARMS="A B C D" bash scripts/run_pilot.sh
# analysis + figures + report numbers:
.venv/bin/python -m scripts.analyze --glob '*pilot*'
.venv/bin/python -m scripts.make_figures
.venv/bin/python -m scripts.render_report --inject
# staged GPU arms (when a card frees) — see jobqueue/HUB.md:
.venv/bin/python -m jobqueue.runner plan && .venv/bin/python -m jobqueue.runner run
```
Prereqs: an Ollama endpoint (loopback) for the inference arms; Docker (isolated bridge, no WAN)
only for the staged InterCode scale-up; a free GPU for the staged QLoRA arms.
