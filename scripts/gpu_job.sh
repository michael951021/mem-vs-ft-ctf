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
#
# OOM-robust design (fix-h3run-45f7fb3f): the pre-fix script installed the full
# ~4.5GB torch+CUDA stack before anything ran, got SIGKILLed mid-install under
# host-wide memory pressure, and logged no memory state. Now: (a) only the light
# deps (pilot/analysis need) are installed before the CORE pilot, so the pilot
# lands before the heavy window; (b) memstat lines record MemAvailable + cgroup
# state before every heavy step and at exit; (c) mem_guard waits (bounded) for
# host RAM before QLoRA (~10GB CPU) and the bf16 merge (~20GB CPU), instead of
# letting the kernel pick the victim.
set -u
unset OLLAMA_HOST OLLAMA_BASE_URL ANTHROPIC_BASE_URL   # ignore the dead proxy; harness defaults to 127.0.0.1:11434
CTF_WORKROOT=""
export CTF_WORKROOT
OLL=127.0.0.1:11434
B=qwen2.5:7b-instruct
PY=.venv-gpu/bin/python
UV=.tools/bin/uv

log(){ echo "[gpu_job $(date +%T)] $*"; }
oll(){ OLLAMA_HOST=$OLL ollama "$@"; }

# Diagnostics: MemAvailable + cgroup limit/current, so an OOM kill is diagnosable.
memstat(){
  local tag="${1:-}" avail_kb max_kb cur_kb
  avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  max_kb=$(cat /sys/fs/cgroup/memory.max 2>/dev/null \
           || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "?")
  cur_kb=$(cat /sys/fs/cgroup/memory.current 2>/dev/null \
           || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo "?")
  log "memstat[$tag] MemAvailable=${avail_kb:-0}kB cgroup_max=${max_kb} cgroup_current=${cur_kb}"
}

# mem_guard MIN_GB WAIT_SECS [MEMINFO_FILE]
# Return 0 once MemAvailable >= MIN_GB, waiting up to WAIT_SECS (60s polls).
# Return 1 if still below at the deadline. MEMINFO_FILE injectable for tests.
mem_guard(){
  local min_gb="${1:-0}" wait_secs="${2:-0}" meminfo="${3:-/proc/meminfo}"
  local deadline=$(( $(date +%s) + wait_secs )) avail_kb
  while :; do
    avail_kb=$(awk '/^MemAvailable:/{print $2}' "$meminfo")
    if [ "${avail_kb:-0}" -ge $((min_gb*1024*1024)) ]; then
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      return 1
    fi
    sleep 60
  done
}

_on_exit(){
  memstat "exit rc=$?"
  [ -n "$CTF_WORKROOT" ] && rm -rf "$CTF_WORKROOT"
}

main(){
  CTF_WORKROOT="$(mktemp -d)"
  export CTF_WORKROOT
  trap _on_exit EXIT
  memstat start

  # 0) venv + LIGHT deps only (pilot/analysis need pyyaml+numpy+matplotlib).
  #    Keep the pre-CORE window short: no multi-GB download before the pilot.
  if [ ! -x "$PY" ]; then
    log "building .venv-gpu (uv)"
    "$UV" venv --python 3.12 .venv-gpu \
      || { rm -rf .venv-gpu; "$UV" venv --python 3.12 .venv-gpu \
      || { log "FATAL uv venv"; exit 2; }; }
  fi
  "$UV" pip install -q --python "$PY" pyyaml numpy scipy matplotlib \
    || { log "FATAL light deps"; exit 2; }
  log "light deps ready"

  # 1) base 7B present on host Ollama (pull over shared-net if missing)
  oll list 2>/dev/null | grep -q "$B" || { log "pulling $B"; oll pull "$B" || { log "FATAL pull"; exit 2; }; }
  log "base model present"

  # 2) CORE: matched-base 7B pilot A-D (inference via host Ollama)
  PY="$PY" MODEL="$B" TAG=h3base OUT=runs SEEDS='0 1' ARMS='A B C D' \
    GLOB='*-h3base' SUMMARY=results/summary_h3base.json \
    bash scripts/run_pilot.sh || log "pilot had per-arm failures (partial runs kept)"
  memstat post-pilot
  log "pilot done"

  # 3) free the 7B from the GPU before training; verify it actually unloaded
  curl -s "$OLL/api/generate" -d "{\"model\":\"$B\",\"keep_alive\":0}" >/dev/null 2>&1 || true
  oll stop "$B" 2>/dev/null || true
  sleep 2
  if oll ps 2>/dev/null | grep -q "$B"; then
    log "WARN $B still loaded after stop -- training may OOM the GPU"
  else
    log "base model unloaded"
  fi

  # 4) CORE: SFT corpus from the 7B TRAIN solves only (*-h3base)
  "$PY" -m finetune.build_sft --runs runs --glob '*-h3base' --split-by-family --out finetune/data \
    || { log "FATAL build_sft"; exit 2; }
  log "sft built"

  # 4b) HEAVY training deps -- only now, with CORE pilot + SFT already on disk.
  #     The install itself is light (~80MB RSS); the guard just avoids starting
  #     the long multi-GB download window while the host is under memory pressure
  #     (the pre-fix job was killed exactly there).
  memstat before-heavy-install
  mem_guard 8 900 || log "WARN host MemAvailable < 8GB after 15min wait -- proceeding (install is light)"
  "$UV" pip install -q --python "$PY" \
      torch==2.4.1 transformers==4.44.2 peft==0.12.0 bitsandbytes==0.43.3 \
      trl==0.9.6 datasets==2.21.0 accelerate==0.33.0 \
    || { log "FATAL heavy deps"; exit 2; }
  memstat after-heavy-install
  log "heavy deps ready"

  # 5) CORE: QLoRA arm E (GPU training; writes measured gpu_hours to train_meta.json).
  #     CPU side of bf16 QLoRA peaks ~10GB -- wait for headroom instead of
  #     letting the kernel OOM-kill us mid-epoch.
  memstat before-qlora
  mem_guard 16 600 || { log "FATAL host MemAvailable < 16GB before QLoRA after 10min wait (see memstat lines)"; exit 3; }
  "$PY" -m finetune.train_qlora --config finetune/configs/qlora_qwen7b.toml \
      --data finetune/data/sft_all.jsonl --arm E --out finetune/adapters/E-qwen7b --min-free-mb 15000 \
    || { log "FATAL qlora train"; exit 2; }
  log "qlora trained (gpu-hours in finetune/adapters/E-qwen7b/train_meta.json)"

  # 6) BEST-EFFORT: merge adapter -> serve via Ollama -> eval E (weights) and F (weights+playbook).
  #    The bf16 CPU merge of the 7B peaks ~20GB; skip the whole tail if there is
  #    no headroom -- the adapter + gpu-hours (the core) are already safe.
  memstat before-merge
  if mem_guard 32 300; then
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
  else
    log "skip E/F merge+serve: host MemAvailable < 32GB after 5min wait (merge peaks ~20GB); core intact"
  fi

  # 7) aggregate whatever landed into the H3 summary (never clobbers the 27B summary.json)
  "$PY" -m scripts.analyze --glob '*-h3*' --summary-out results/summary_h3.json || log "analyze issue"
  "$PY" -m scripts.audit_redaction || log "audit issue"
  log "H3 aggregate -> results/summary_h3.json ; gpu_job complete"
  exit 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
