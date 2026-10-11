#!/usr/bin/env bash
# Loop GPU-job entry for the H3 matched-base study.
#
# Runs INSIDE the loop's job sandbox (agent cmd_job): cwd=/work (this project's
# worktree), both GPUs dev-bound, host network shared (NET=on) so 127.0.0.1:11434
# is the host Ollama daemon. During a GPU job the shared *model server* is stopped,
# but the Ollama daemon stays up, so we serve our own 7B on the now-free GPU.
#
# Contract: the high-value CORE must land -- the matched-base 7B pilot (arms A-D,
# which tests H1/H2 at a non-ceiling model) and the QLoRA-E training with MEASURED
# GPU-hours (the compute datum for H3). The E/F Ollama-serving tail is best-effort;
# failing it never discards the core. Results isolate under tags h3base/h3 and
# results/summary_h3.json -- the committed 27B *-pilot artifacts are never touched.
set -u
unset OLLAMA_HOST OLLAMA_BASE_URL ANTHROPIC_BASE_URL   # ignore the dead proxy; harness defaults to 127.0.0.1:11434
export CTF_WORKROOT="$(mktemp -d)"
OLL=127.0.0.1:11434
B=qwen2.5:7b-instruct
PY=.venv-gpu/bin/python
UV=.tools/bin/uv
log(){ echo "[gpu_job $(date +%T)] $*"; }
oll(){ OLLAMA_HOST=$OLL ollama "$@"; }

# 0) fresh venv built inside the worktree (path-correct for the sandbox)
if [ ! -x "$PY" ]; then
  log "building .venv-gpu (uv)"
  "$UV" venv --python 3.12 .venv-gpu || { log "FATAL uv venv"; exit 2; }
  "$UV" pip install -q --python "$PY" pyyaml numpy scipy matplotlib \
      torch==2.4.1 transformers==4.44.2 peft==0.12.0 bitsandbytes==0.43.3 \
      trl==0.9.6 datasets==2.21.0 accelerate==0.33.0 || { log "FATAL deps"; exit 2; }
fi
log "venv ready"

# 1) base 7B present on host Ollama (pull over shared-net if missing)
oll list 2>/dev/null | grep -q "$B" || { log "pulling $B"; oll pull "$B" || { log "FATAL pull"; exit 2; }; }
log "base model present"

# 2) CORE: matched-base 7B pilot A-D (inference via host Ollama)
PY="$PY" MODEL="$B" TAG=h3base OUT=runs SEEDS='0 1' ARMS='A B C D' \
  GLOB='*-h3base' SUMMARY=results/summary_h3base.json \
  bash scripts/run_pilot.sh || log "pilot had per-arm failures (partial runs kept)"
log "pilot done"

# 3) free the 7B from the GPU before training
curl -s "$OLL/api/generate" -d "{\"model\":\"$B\",\"keep_alive\":0}" >/dev/null 2>&1 || true
oll stop "$B" 2>/dev/null || true

# 4) CORE: SFT corpus from the 7B TRAIN solves only (*-h3base)
"$PY" -m finetune.build_sft --runs runs --glob '*-h3base' --split-by-family --out finetune/data \
  || { log "FATAL build_sft"; exit 2; }
log "sft built"

# 5) CORE: QLoRA arm E (GPU training; writes measured gpu_hours to train_meta.json)
"$PY" -m finetune.train_qlora --config finetune/configs/qlora_qwen7b.toml \
    --data finetune/data/sft_all.jsonl --arm E --out finetune/adapters/E-qwen7b --min-free-mb 15000 \
  || { log "FATAL qlora train"; exit 2; }
log "qlora trained (gpu-hours in finetune/adapters/E-qwen7b/train_meta.json)"

# 6) BEST-EFFORT: merge adapter -> serve via Ollama -> eval E (weights) and F (weights+playbook)
{
  MERGED=finetune/adapters/E-qwen7b-merged
  "$PY" - finetune/adapters/E-qwen7b "$MERGED" <<'PYMERGE'
import sys
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
adapter, out = sys.argv[1], sys.argv[2]
m = AutoPeftModelForCausalLM.from_pretrained(adapter, device_map="cpu").merge_and_unload()
m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
print("merged ->", out)
PYMERGE
  printf 'FROM %s\nPARAMETER temperature 0\nPARAMETER num_ctx 4096\n' "$MERGED" > finetune/adapters/E.Modelfile
  oll create qwen7b-ftE -f finetune/adapters/E.Modelfile
  "$PY" -m harness.run_experiment --arm A --arm-label E --eval-only --model qwen7b-ftE --seed 0 --out runs --tag h3
  F="$(ls -d runs/*-D-local-suite-s0-h3base 2>/dev/null | tail -1)/memory_frozen.json"
  test -f "$F" && "$PY" -m harness.run_experiment --arm D --arm-label F --eval-only \
       --load-memory "$F" --model qwen7b-ftE --seed 0 --out runs --tag h3
  log "E/F served + evaluated"
} || log "E/F serving/eval incomplete -- adapter + gpu-hours still recorded; serving needs import follow-up"

# 7) aggregate whatever landed into the H3 summary (never clobbers the 27B summary.json)
"$PY" -m scripts.analyze --glob '*-h3*' --summary-out results/summary_h3.json || log "analyze issue"
"$PY" -m scripts.audit_redaction || log "audit issue"
log "H3 aggregate -> results/summary_h3.json ; gpu_job complete"
exit 0
