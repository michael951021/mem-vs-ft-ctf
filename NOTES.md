# NOTES — current state

## Task: fix-h3run-29326f13 (Fix h3run: RuntimeError in train_qlora.py) — 3rd pass
h3run = `bash scripts/gpu_job.sh` (loop GPU job, gpu/mem:48/10h). 1st pass (split
install + memstat/mem_guard + heartbeat + full-log tee to .tools/h3run.joblog)
and 2nd pass (numpy==1.26.4/pyarrow==18.1.0 pins) merged. 08:01 run on 10-13
passed pilot A-D (seeds 0+1) + SFT (40 records) + heavy install (no-op);
QLoRA then died at train_qlora.py:126 `from trl import SFTTrainer` with
`RuntimeError ... No module named 'rich'` (rc=1, .tools/h3run.joblog).

### Root cause (3rd pass)
trl 0.9.6 is a lazy module: bare `import trl` passes, so the script's
`import torch,...,trl,...` check passed. But trl.trainer.sft_trainer →
trl/trainer/utils.py:26 imports `rich`, and trl does NOT declare rich in its
metadata (checked trl-0.9.6.dist-info/METADATA) → uv never installs it → first
real training import dies.

### Fix (3rd pass)
- .venv-gpu: `uv pip install rich` → rich 15.0.0 (+ pygments/mdurl/markdown-it-py).
- scripts/gpu_job.sh: `rich==15.0.0` added to heavy install line; BOTH import
  checks now run TRAIN_IMPORTS = train_qlora.py's real training-path imports
  (torch, bitsandbytes, datasets.load_dataset, transformers.{AutoModelForCausalLM,
  AutoTokenizer, BitsAndBytesConfig, TrainingArguments}, peft.{LoraConfig,
  get_peft_model, prepare_model_for_kbit_training}, trl.SFTTrainer) — a fresh
  build installs rich; a broken venv gets re-extracted.

### Verified here (no GPU)
- Pre-fix acceptance: real import block on .venv-gpu rc=1, traceback identical to
  joblog → .agent-evidence/h3run-rich-import-pre.log (also the regression proof
  that the check fails when rich is absent).
- Post-fix: rc 0 "IMPORTS OK" → h3run-rich-import-post.log.
- Script's own TRAIN_IMPORTS (sed-extracted from gpu_job.sh) rc 0 →
  h3run-train-imports-script.log (bnb "no GPU" warning expected in sandbox).
- bash -n rc 0; `python3 -m scripts.tests.test_gpu_job_guard` 6/6 PASS.
- Script's exact pinned install line on real venv: no-op, rc 0 →
  h3run-reinstall-noop.log.

### Next box run (watch)
- Expect: pilot → 40 SFT → heavy import (now strict) → 15GB HF 7B download →
  train → finetune/adapters/E-qwen7b + train_meta.json (measured gpu-hours).
- If it dies again, read .tools/h3run.joblog (tee'd; rc on every FATAL line).
- KNOWN-BUGGY: `est_gpu_hours_single_3090` in the plan output (716.8 for 40 ex)
  is a placeholder (train_qlora.py estimate_gpu_hours, factor_s=3.0e-9, docstring
  says calibrate on first run); hand math says real is ~0.1-0.3 gpu-h. Calibrate
  from measured train_meta.json after the first real run — separate ticket.

## Conventions
- Loop jobs run "no agent" via lib/jobs.py; shared binds: .tools, .venv-gpu,
  finetune/adapters, finetune/data, ~/.cache/huggingface (53GB). Committed 27B
  *-pilot artifacts never touched. Job log tee'd to .tools/h3run.joblog.
- Tests: plain python modules, `python3 -m <pkg>.tests.<name>` (no pytest).
- Ollama in agent sandbox: no model served; use $OLLAMA_BASE_URL or
  $OLLAMA_HOST. Direct curl to 127.0.0.1:11434 is hook-blocked.
