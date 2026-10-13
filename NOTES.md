# NOTES — current state

## Task: fix-h3run-45f7fb3f (Fix h3run: OOM)
h3run = `bash scripts/gpu_job.sh` (loop GPU job: gpu/mem:48/10h). It was OOM-killed
before producing any CORE artifact (no `runs/*-h3base`, no SFT data, no adapter).

### Diagnosis (evidence in state/tickets/fix-h3run-45f7fb3f.md)
- Shared `.venv-gpu` had only the venv skeleton + `.lock`; uv cache held partial
  torch/cudnn extractions (SIGKILL mid-install), then a ~15-min re-download cascade.
- The pre-fix script installed the FULL ~4.5GB torch+CUDA stack first, before the
  CORE pilot. Measured: that install peaks at 75.6 MB RSS — it cannot OOM a 48GB
  cgroup by itself, so the kill was host-wide pressure/preemption during the long
  install window. The script logged zero memory state → undiagnosable.

### Fix (scripts/gpu_job.sh)
- Split install: light deps (pyyaml/numpy/scipy/matplotlib) before the CORE pilot;
  heavy training stack (torch 2.4.1 + CUDA wheels + transformers/peft/bnb/trl/
  datasets/accelerate) only after pilot + SFT are on disk. CORE lands first.
- `memstat [tag]`: logs MemAvailable + cgroup max/current at start, post-pilot,
  before/after heavy install, before QLoRA, before merge, and on exit (EXIT trap).
- `mem_guard MIN_GB WAIT [MEMINFO]`: bounded wait until host MemAvailable >= MIN_GB.
  Gates: 8GB/15min (warn only) before heavy install; 16GB/10min (FATAL exit 3)
  before QLoRA (~10GB CPU peak); 32GB/5min (skip best-effort merge tail, ~20GB
  CPU peak). All pre-existing tags/isolation/exit semantics unchanged.
- Script is source-safe (main() guarded) so tests can call the helpers.
- Ollama unload now verified with `oll ps` (warn if 7B still loaded pre-training).

### Merge with master (49dca96)
Master had edited the same two files in parallel (main-checkout edits, 3de2bd8).
All of both sides kept:
- Heartbeat (2-min "job alive" line) vs the loop's 15-min stall watchdog — moved
  into main() (was top-level on master) so _on_exit kills it and sourcing for
  tests stays side-effect free.
- Import-based venv guard — applied to the LIGHT set at step 0 and the HEAVY set
  at step 4b (master's single full-stack guard didn't fit the split design;
  uv re-extracts exactly what is missing on a partial install).
- `rm -rf .venv-gpu` before rebuild; `-q` dropped from uv installs (visible
  install progress in the job log).
- Ticket file: master's auto-filed ticket kept canonical; w1's diagnosis
  appended under a provenance header (both verbatim).

### Verified here (no GPU/Ollama model in this sandbox)
- `bash -n` rc 0.
- `python3 -m scripts.tests.test_gpu_job_guard`: 6/6 PASS (known answers: pass at
  >= threshold, fail below; fabricated meminfo). Failed 3/5 pre-fix (mem_guard
  absent) — acceptance check ran first.
- Real light-stack install from PyPI into scratch venv: rc 0, imports OK.
  Heavy set previously resolved/installed to all wheels (exit 0) — evidence in
  `.agent-evidence/h3run-install-rss.log`.
- `python3 -m memory.tests.test_arms` still PASS.

### Next box run (what to watch)
- Job log should show memstat lines + periodic heartbeat lines; pilot completes
  before the heavy download starts; plausible peak <= 30GB, so the 32GB merge
  guard is non-binding unless external host pressure appears — which the logs
  would now show.

## Conventions
- Loop jobs run "no agent" via lib/jobs.py; shared binds: .tools, .venv-gpu,
  finetune/adapters, finetune/data. Committed 27B artifacts must never be touched.
- Tests: plain python modules, `python3 -m <pkg>.tests.<name>` (no pytest).
- Ollama in the agent sandbox: no model served; use $OLLAMA_BASE_URL (OpenAI /v1)
  or $OLLAMA_HOST (/api/tags). Direct curl to 127.0.0.1:11434 is hook-blocked.
