# Memory vs. Weights for CTF Agents

Can an agent get better at a skill *without* touching its weights, just by
writing down what worked? And if it can, how does that stack up against
fine-tuning on the exact same experience?

I tested this on capture-the-flag challenges, because a flag is either right or
wrong. That matters here. A memory is only ever written after a verified solve,
never because the agent *thinks* it did well, so there is no room for the agent
to fool itself (or me).

Six arms share one frozen ReAct agent, and only the "learning" part changes:
nothing (A), pasting raw past solves (B), retrieving similar past episodes (C),
an ACE-style playbook that consolidates lessons into short strategy notes (D),
QLoRA fine-tuning on the verified solves (E), and both E and D together (F).

## What came out

![The two regimes](results/figures/fig6_two_regimes.png)

**On a strong 27B base, memory saves a turn, but not the memory I expected.**
Every arm solves all 12 challenges, so success can't move. The only signal is how
fast they solve. Raw examples (B) and episodic retrieval (C) each save a full
turn in-distribution (CI [+0.20, +1.80], 15 paired attempts). My consolidated
playbook (D), the arm the study was built around, saves only +0.40 and its CI
touches zero. A concrete worked example beats an abstract strategy note, and none
of it survives the move to a sibling challenge.

**On a matched 7B base that actually fails tasks, nothing clearly wins.**
The baseline solves 40% in-distribution. Fine-tuned weights (E) hit 80%, the
biggest jump, after 0.44 GPU-hours of QLoRA, but that is n=5 with a CI of
[0, +0.8]. The memory arms land between −0.10 and +0.10 for zero training cost.
On transfer every arm ties or trails the baseline's 60%.

**Read E's number with care.** Three things inflate it, and the report
discloses all of them (§5.4):
- E/F ran with 20 turns and 512 output tokens per step, while A–D got 15 and 384.
- E's training data came from a single source challenge, 64 records.
- E/F have one seed, while A–D have two.

So the honest headline is: **at this scale, memory and weights are not
distinguishable on success, and structured memory lost to plain examples on
efficiency.** That's a negative for the arm I built, and a clear pointer to the
next experiment: an LLM reflector that writes concrete command recipes instead
of abstract bullets, and a bigger benchmark (the InterCode adapter is written,
but those runs never happened).

Everything is traceable. Every number in the report comes from per-attempt logs
in `runs/` through `scripts/analyze.py`. Flags are redacted from memory and from
the training data, which is decontaminated against the eval split
(`scripts/audit_redaction.py` is a hard gate; 0 leaks). The full write-up is
[REPORT.md](REPORT.md), and the decisions and dead ends along the way are in
[DECISIONS.md](DECISIONS.md) and [RUNLOG.md](RUNLOG.md).

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
jobqueue/          host GPU-job specs used to stage the 7B and QLoRA runs
scripts/           analyze, make_figures, plot_headline, render_report, audit_redaction, run_pilot.sh
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
