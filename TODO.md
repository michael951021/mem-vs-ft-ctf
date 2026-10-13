# mem-vs-ft-ctf — H3 matched-base GPU run

> why: produce the memory-vs-weights (H3) result on one shared 7B base — the matched-base
> pilot (arms A-D, which breaks the 27B success ceiling so H1/H2 are testable as success
> hypotheses) plus QLoRA-E weight training with measured GPU-hours. Isolated to
> results/summary_h3.json; the committed 27B *-pilot artifacts are never touched.
>
> Compute jobs: (run: `CMD`) (gpu) (mem: GB) (cpus: N) (timeout: H) — lib/jobs.py runs them, no agent.

- [ ] **H3 matched-base GPU run** — 7B pilot A-D + QLoRA-E, best-effort E/F serving; writes results/summary_h3.json (id: h3run) (run: `bash scripts/gpu_job.sh`) (gpu) (mem: 48) (cpus: 16) (timeout: 10h)
- [x] **Fix h3run: OOM (ran out of memory)** — auto-filed from the job log (id: fix-h3run-45f7fb3f) (after: -)
- [x] **Fix h3run: OOM (ran out of memory)** — auto-filed from the job log (previous fix did not resolve it) (id: fix-h3run-45f7fb3f) (after: -)
- [ ] **Fix h3run: RuntimeError in train_qlora.py** — auto-filed from the job log (id: fix-h3run-29326f13) (after: -)
