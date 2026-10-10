# Memory vs. Weights for CTF Agents under Matched Compute

Empirical study: can an LLM agent "learn" a CTF skill domain **without weight updates** — by
consolidating verified experience into a compact, structured, transferable memory — and how does
that compare, at **matched compute**, to **QLoRA** fine-tuning on the same verified experience?
Domain: cybersecurity CTF (flags/execution give non-gameable, ground-truth feedback).

See the full brief in `docs/brief.md`. Research question and hypotheses H1–H3 in `REPORT.md §3`.

## Status
Phase 0 (stand-up). See `RUNLOG.md` for the live state and `DECISIONS.md` for branch choices.
Host facts (measured) in `docs/host_facts.md`.

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
env/         pinned deps, Dockerfiles, env setup
harness/     agent scaffold (scaffold/) + pluggable arms (arms/)
memory/      playbook (D), retrieval (C), episodic (B)
finetune/    QLoRA/FSDP configs, corpus build, flag-redaction, decontam
benchmarks/  adapters (InterCode-CTF, NYU-CTF, Cybench) + family-tag map
configs/     per-run YAML
runs/        per-run JSONL logs (schema: docs/run_schema.md)
results/     aggregated CSVs, figures/, stats/
queue/       file-based job queue + runner for deferred heavy runs
REPORT.md    the deliverable
```

## Reproduce (filled in as phases land)
Prereqs: Docker (isolated bridge, no WAN), an Ollama endpoint, a Python 3.11/3.12 venv.
See `env/` and per-phase commands below. (TBD as gates complete — no placeholders in results.)
