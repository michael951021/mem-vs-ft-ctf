# mem-vs-ft-ctf — H3 matched-base GPU run

> why: produce the memory-vs-weights (H3) result on one shared 7B base — the matched-base
> pilot (arms A-D, which breaks the 27B success ceiling so H1/H2 are testable as success
> hypotheses) plus QLoRA-E weight training with measured GPU-hours. Isolated to
> results/summary_h3.json; the committed 27B *-pilot artifacts are never touched.
>
> Compute jobs: (run: `CMD`) (gpu) (mem: GB) (cpus: N) (timeout: H) — lib/jobs.py runs them, no agent.

- [ ] **H3 matched-base GPU run** — 7B pilot A-D + QLoRA-E, best-effort E/F serving; writes results/summary_h3.json (id: h3run) (run: `bash scripts/gpu_job.sh`) (gpu) (mem: 64) (cpus: 16) (timeout: 10h)
- [x] **Fix h3run: OOM (ran out of memory)** — auto-filed from the job log (id: fix-h3run-45f7fb3f) (after: -)
- [x] **Fix h3run: OOM (ran out of memory)** — auto-filed from the job log (previous fix did not resolve it) (id: fix-h3run-45f7fb3f) (after: -)
- [x] **Fix h3run: RuntimeError in train_qlora.py** — auto-filed from the job log (id: fix-h3run-29326f13) (after: -)
- [x] **Isolate pilot arms per-process + checkpoint per arm** — run_experiment/run_pilot runs arms A-D (x seeds) in one process; a 48G OOM at arm C (peak ~47.6G, host had 90G free) points at memory not being released between arms. Run each (arm,seed) in its own subprocess so its memory is reclaimed before the next, and checkpoint completed (arm,seed) cells so a retry skips finished work instead of re-running arms A/B (which burned ~6 GPU re-runs). Done when: a 2-arm pilot runs each cell in a fresh process (peak RSS logged per cell), and a re-run with one cell already present skips it. (after: -)
