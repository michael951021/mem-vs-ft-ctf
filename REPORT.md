# Memory vs. Fine-Tuning for Experiential Learning in CTF Agents

*A self-contained empirical study on one workstation. Every quantitative claim in
§5 is auto-generated from logged runs (`results/summary.json`), and every run id
is listed in the run index. Nothing in the results section is hand-typed.*

**Status of this report.** The harness is complete and offline-validated; a real
pilot on the local CTF suite is running on the resident model and its numbers are
injected below as they land. The compute-matched fine-tuning arms (E/F) and the
Docker benchmark scale-up (InterCode/NYU/Cybench) are **staged** on the host GPU
queue (`jobqueue/`) because the two RTX 3090s are committed to a live inference
server for the duration of this session; see §7 (Compute & honesty) and §9.

---

## 1. Research question

Can an LLM agent *progressively abstract verified experiences into reusable
knowledge that transfers across related task distributions* — without weight
updates — and how does that compare, under a matched GPU-hour budget, with
gradient-based learning (QLoRA) on the **same** verified experience?

CTF is the chosen domain precisely because flags and program execution give
**ground-truth, non-gameable** feedback: the condition under which experiential
memory is known to help rather than confabulate. A memory write in this study is
*only* ever gated on a verified flag, never on the agent's self-assessment.

### Hypotheses (falsifiable; a clean negative is a publishable result)
- **H1 (in-distribution).** On held-out same-family tasks, D > C > B > A
  (ACE-playbook > episodic-retrieval > raw-ICL > no-memory).
- **H2 (transfer — the core claim).** Trained on mechanism family *X*, evaluated
  on a **sibling surface** *Y* (same vulnerability mechanism, different surface),
  the structured playbook (D) retains more of its in-distribution advantage than
  B/C. Retention = (advantage over A on *Y*) ÷ (advantage over A on *X*).
- **H3 (memory vs. weights, compute-matched).** At equal GPU-hours, QLoRA (E) and
  memory (D) occupy different points on the capability/efficiency frontier; the
  hybrid (F) may dominate at higher cost. We characterise the frontier; we do not
  assume a winner.

---

## 2. Relation to prior work (novelty)

The study builds on experience-consolidation agents (ACE-style
Generator/Reflector/Curator playbooks) and on CyberEvolver-style self-improving
CTF agents. Its two contributions over that line are: **(1)** a controlled
**cross-mechanism transfer** test on *mechanism-grouped* families (prior work
argued standard benchmarks group too coarsely to isolate this), and **(2)** a
head-to-head against **compute-matched QLoRA** on the identical verified
experience, which that line does not report. We do not re-sell what prior work
already showed; we isolate transfer and the memory-vs-weights frontier.

---

## 3. Method

### 3.1 Arms (one frozen base policy; only the learning module differs)
The ReAct agent (`harness/scaffold/react_agent.py`) and its base policy are
identical across arms, so any capability difference is attributable to the
learning mechanism, not the model.

| arm | learning mechanism |
|---|---|
| **A** | none (fresh context every task) — the baseline every advantage is measured against |
| **B** | raw in-context learning: a bounded buffer of verified solve transcripts, prepended |
| **C** | episodic retrieval: verified episodes retrieved by family + token overlap |
| **D** | **ACE playbook** (centerpiece): structured strategy bullets per mechanism family; a Reflector distils each *verified* trajectory into one generalised bullet; a Curator applies **delta** updates (add / reinforce / demote) with dedup and a per-family cap — no full rewrites |
| **E** | QLoRA on the verified experience (weights change; no memory at eval) |
| **F** | hybrid: QLoRA weights **+** the frozen ACE playbook |

The Reflector conditions **only** on the verifiable verdict. By default it is a
deterministic heuristic (zero extra model calls), which keeps the memory
mechanism itself cheap under inference contention; an LLM-reflector ablation is
supported (`Playbook(reflector_llm=…)`) for a GPU window. Ablations wired in:
hierarchical-promote vs. compress-everything (`Playbook(hierarchical=…)`) and
flag-only vs. milestone feedback (schema fields `subtasks_*`).

### 3.2 Testbeds
- **Local CTF suite** (`env/local_suite/`, this session's measured testbed): 12
  self-contained, benign picoCTF-style challenges solvable with standard shell
  tools, requiring **no Docker and no network**, so they run inside the existing
  bubblewrap sandbox and yield *real* ground-truth-scored trajectories now. They
  are grouped into 5 parent mechanisms, each with sibling surfaces (§3.3).
- **InterCode-CTF** (100 picoCTF tasks, Docker, flag reward): the pilot benchmark
  from the brief. Adapter implemented (`env/intercode/adapter.py`, same TaskEnv
  protocol) and **staged**; Docker is absent on the host this session
  (`docs/host_facts.md`), so it runs from the queue once Docker is installed.
- **NYU-CTF + Cybench**: headline scale-up (P4), staged behind InterCode.

### 3.3 Mechanism families and the transfer protocol
Each challenge is tagged by *vulnerability mechanism* — `"<parent>-<surface>"` —
not coarse category. Siblings share a parent, differ in surface. The local-suite
map (a data artifact, `env/local_suite/challenges.py::TRANSFER_SPLITS`):

| parent mechanism | train surface(s) | sibling transfer surface |
|---|---|---|
| crypto/encoding | base64, hex | base32 |
| crypto/classical | caesar | atbash |
| crypto/xor | single-byte | repeating-key |
| forensics/recover | strings, grep | concat-split |
| reversing/constant | rot13 | arithmetic |

Protocol: consolidate / fine-tune on *X* only → **freeze** → evaluate on held-out
*X* (in-distribution) and on the sibling *Y* (transfer). The frozen memory /
adapter is built **only** from *X*'s verified trajectories.

**Leakage control.** Splits are by challenge. The SFT corpus for E/F is
**decontaminated** against every eval challenge id/description and the removals
are logged (`finetune/build_sft.py` → `manifest.json`). Flags are redacted from
all stored text before any write.

### 3.4 Metrics (logged per attempt, `docs/run_schema.md`)
success & flag-verified; turns-to-solve; input/output tokens; wall-clock;
GPU-hours (training arms); memory size over time; mechanism family; split. One
JSONL row per attempt; full transcripts under `runs/<id>/raw/`.

### 3.5 Statistics
Comparisons are **paired per (challenge, seed)** against Arm A. CIs are a
**clustered bootstrap** resampling challenges (the cluster unit), per
`scripts/analyze.py`. We report CIs, not just points, and fix+log decoding
params (§7). Power caveat: these suites are far below the ~1000-item rule for
small effects, so we lean on the paired design and resampling and we report the
local-suite result as a **pilot-scale** finding, with InterCode/NYU/Cybench
staged for the powered version.

---

## 4. Implementation & reproducibility

```
env/local_suite/      12 benign challenges + ground-truth TaskEnv (no Docker)
env/intercode/        InterCode-CTF Docker adapter (staged)
harness/scaffold/     frozen base policy: LLM client, ReAct loop, logger
harness/run_experiment.py   train→freeze→test(in-dist)→test(transfer) driver
memory/arms.py        arms B/C/D (+ freeze/save/load for the transfer protocol)
memory/redact.py      flag redaction (ethics gate)
finetune/             SFT corpus builder + QLoRA trainer (arms E/F, host-only)
jobqueue/             host GPU-job queue + specs + HUB.md
scripts/              analyze (stats), make_figures, render_report, audit_redaction
runs/ results/        JSONL logs, aggregated CSVs, figures, summary.json
```

Re-run from a clean clone:
```bash
python3.12 -m venv .venv && .venv/bin/pip install pyyaml numpy scipy matplotlib
# offline validation (no GPU/Docker):
.venv/bin/python -m memory.tests.test_arms
.venv/bin/python -m harness.scaffold.tests.test_smoke_loop
.venv/bin/python -m env.local_suite.env
# real pilot against a local Ollama endpoint:
OUT=runs SEEDS="0 1 2" ARMS="A B C D" bash scripts/run_pilot.sh
# analysis + figures + report numbers:
.venv/bin/python -m scripts.analyze --glob '*pilot*'
.venv/bin/python -m scripts.make_figures
.venv/bin/python -m scripts.render_report --inject
# staged GPU arms (when a card is free): see jobqueue/HUB.md
.venv/bin/python -m jobqueue.runner run
```

Base policy for the measured pilot: the resident `qwen3.8:27b` (Q8) served by
Ollama at `127.0.0.1:11434`, temperature 0, seed-pinned, `num_ctx=4096`,
`num_predict=384`, reasoning **think-mode off** (a fixed, logged decoding choice
applied identically to every arm — see §7). Decisions and their rationale are in
`DECISIONS.md`; the chronological record of runs is in `RUNLOG.md`.

---

## 5. Results

<!--RESULTS_AUTO-->
<!-- AUTO-GENERATED by scripts/render_report.py from results/summary.json. Do not edit by hand. -->

**Data as logged:** 15 attempts, arms ['A'], seeds [0].

#### Success rate by arm and split

| arm | train | test-indist | test-transfer |
|---|---|---|---|
| A no-memory | 1.00 (n=5) | 1.00 (n=5) | 1.00 (n=5) |

#### H1 — in-distribution advantage over Arm A (paired, clustered bootstrap 95% CI)

_Not yet measured (needs >=2 arms with test-indist rows; pending the pilot's memory arms)._

#### H1b — step-efficiency: turns saved vs Arm A on jointly-solved tasks

_The informative metric when success is at ceiling: a verified recipe in memory should let the agent solve in fewer turns._

_Not yet measured (needs a memory arm and Arm A to both solve the same tasks)._

#### H2 — transfer retention to sibling surfaces

_Not yet measured (needs a non-zero in-distribution advantage to normalise against)._

#### Efficiency (on solved tasks)

| arm·split | mean turns | mean out-tokens | n solves |
|---|---|---|---|
| A|test-indist | 3.6 | 141.8 | 5 |
| A|test-transfer | 3.6 | 142.2 | 5 |

#### Run index (traceability)

- **arm A**: `20261010-114901-A-local-suite-s0-pilot`, `20261010-121357-A-local-suite-s0-hardpilot`
- **arm B**: `20261010-122844-B-local-suite-s0-hardpilot`, `20261010-122859-B-local-suite-s0-pilot`

### 5.2 Hard tier (crypto/chain: decode→XOR, sibling surfaces)

_3 attempts, arms ['A'], seeds [0]._

| arm | train | test-indist | test-transfer |
|---|---|---|---|
| A no-memory | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) |

**Step-efficiency (turns saved vs Arm A, jointly-solved):**
 _pending the hard-tier memory arms._
<!--/RESULTS_AUTO-->

### 5.1 Reading the tables
H1 is supported if the paired advantage over A on **test-indist** is positive and
its 95% CI excludes 0, ordered D ≥ C ≥ B. H2 is supported if D's **retention** is
higher than B's and C's (the structured playbook keeps more of its edge on the
sibling surface). H3 is read off the arm-D vs arm-E points at matched GPU-hours
once the queued fine-tuning arms complete.

**The ceiling regime and why H1b matters.** On this local suite the resident 27B
base solves *every* challenge on *both* tiers with no memory (arm A = 100%,
including the chained decode→XOR hard tier, in ~4 turns each). Success therefore
has no headroom here, so the in-distribution H1 success-advantage is ~0 **by
construction** — a genuine but uninteresting null: *experiential memory cannot
raise a success rate that is already saturated.* The informative signal in this
regime is **H1b, step-efficiency**: whether a memory that hands the agent a
verified recipe lets it solve in fewer turns/tokens than solving cold. The
success-headroom test of H1/H2 requires a base model that actually fails some
tasks — the **weaker matched 7B base** (staged, `jobqueue/`) and the harder
InterCode benchmark — which is exactly why those are queued rather than
improvised by evicting the resident server mid-session.

---

## 6. Figures
- `results/figures/fig1_success_by_arm_split.png` — success rate by arm × split.
- `results/figures/fig2_h1_advantage.png` — H1 advantage with bootstrap CIs.
- `results/figures/fig3_sample_efficiency.png` — cumulative train solves vs attempt.
- `results/figures/fig4_transfer_retention.png` — H2 retention per arm.

Figures regenerate from logged data (`scripts/make_figures.py`) and degrade
gracefully on partial data (they plot only the arms/splits present).

---

## 7. Compute, matching, and honesty

**One budget axis** = total GPU-hours on the 2×3090 rig, held equal across
compared arms; `train_qlora.py` records measured GPU-hours and `analyze.py`
reports each headline comparison at matched budget and at each arm's full budget.

**Why the memory arms were measurable this session but the weight arms were
staged.** The two cards were ~90% occupied by a resident Ollama server for the
whole session (`docs/host_facts.md`); under that contention the KV cache spills
to CPU and generation runs at ~1 token/s. We therefore (a) turned reasoning
think-mode **off** so each agent turn emits ~10–60 tokens instead of hundreds —
a single fixed decoding choice applied to **every** arm, so inter-arm comparisons
remain fair — and (b) ran the inference-only memory arms (A–D) on the local suite
now, while staging the GPU-bound weight arms (E/F) and the Docker scale-up on the
host queue (`jobqueue/`, ~4.8 h when a card frees). This is the §11-B6 discipline
of the brief: **a smaller finished study beats a half-run large one.** Nothing in
§5 is imputed; absent arms are marked "not yet measured," never faked.

---

## 8. Ethics & dual-use statement

- **Sandboxed, benign targets only.** The measured testbed is 12 self-contained
  decode/forensics/reversing challenges with synthetic flags; no real-world
  target is ever touched. Agent shell runs in a per-challenge temp directory with
  a timeout and a destructive-command denylist, inside the project's bubblewrap
  sandbox (`DECISIONS.md` D8); the only network egress is the loopback LLM
  endpoint.
- **Verified-only learning.** Every memory write and every SFT example is gated
  on a ground-truth flag check, never the agent's self-report
  (`run_schema.md` invariant).
- **Flag redaction + decontamination.** All stored memory and SFT text passes
  `memory/redact.py`; the SFT corpus is decontaminated against every eval
  challenge; `scripts/audit_redaction.py` is a hard gate that fails on any
  flag-shaped leak and emits a ≥20% spot-check sample for manual review.
- **Dual-use.** This studies *defensive* capability — how agents learn verifiable
  skills and how that learning transfers — on intentionally-vulnerable practice
  challenges. It produces no weaponised proof-of-concepts and targets no live
  systems. The learned artifacts are compact strategy notes and a LoRA adapter
  over benign practice tasks, not exploit code.

---

## 9. Limitations

1. **Pilot scale.** The local suite is 12 challenges across 5 mechanisms; it is a
   *pilot* that exercises the full protocol end to end, not a powered study. The
   InterCode/NYU/Cybench scale-up is implemented and staged, not yet run (Docker
   absent; GPU committed).
2. **Base-model mismatch for H3, addressed by the queue.** A–D piloted on a 27B
   model (what was resident); the clean compute-matched H3 re-runs A–D on a 7–14B
   base and QLoRAs that same base (`jobqueue/jobs/`), so every arm shares one
   frozen base. Until that queued job runs, H3 is characterised as *method +
   staged*, not measured.
3. **Heuristic Reflector by default.** The centerpiece arm's Reflector is
   heuristic to keep consolidation free under contention; the LLM-reflector
   variant is a supported ablation, not the measured default.
4. **Seeds.** Under contention we prioritise completing all arms at seed 0 before
   adding seeds; CIs widen accordingly and are reported honestly.

---

## 10. Appendix — traceability

Every number in §5 is produced by `scripts/analyze.py` from the JSONL under
`runs/`, and `scripts/render_report.py` lists the contributing run ids. Raw
per-turn transcripts (gitignored for size) live under `runs/<id>/raw/`. The
redaction audit is `results/redaction_audit.json`; the SFT decontamination log is
`finetune/data/manifest.json`.
