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
HB=""   # heartbeat subshell pid, set in main, cleared in _on_exit
# Pinned for a consistent CPU stack: torch 2.4.1 is built against the numpy 1.x ABI
# (it fails to import under numpy 2), and datasets' pyarrow must match that numpy.
# The first h3run venv got numpy 1.26.4 + pyarrow 26 (which needs numpy>=2) from
# unpinned resolution -> `import datasets` ImportError -> QLoRA dead on arrival,
# and the step-4b import check could never pass (uv saw all pins "satisfied").
NUMPY_PIN=numpy==1.26.4
PYARROW_PIN=pyarrow==18.1.0

log(){ echo "[gpu_job $(date +%T)] $*"; }
oll(){ OLLAMA_HOST=$OLL ollama "$@"; }

# Diagnostics: MemAvailable + cgroup limit/current, so an OOM kill is diagnosable.
memstat(){
  local tag="${1:-}" avail_kb max_kb cur_kb
  avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  # "absent" (not "?"): cgroupfs is not mounted in the job sandbox, so a
  # cgroup limit can't be the killer there; on the host the real value prints.
  max_kb=$(cat /sys/fs/cgroup/memory.max 2>/dev/null \
           || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo absent)
  cur_kb=$(cat /sys/fs/cgroup/memory.current 2>/dev/null \
           || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo absent)
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

# Sum VmRSS (kB) over the process tree rooted at ROOT_PID, using only /proc
# (the job sandbox has no ps). This is the memory trajectory that attributes
# the next kill: samples ramping toward the cap across heartbeats point at the
# memory killer; flat samples point at an external killer. PROC_DIR is
# injectable for tests. ppid is field 4 of /proc/<pid>/stat, but field 2
# (comm) may contain spaces/parens, so strip up to the LAST ')' first.
tree_rss_kb(){
  local root="$1" proc="${2:-/proc}"
  local -A children=()
  local d pid ppid
  for d in "$proc"/[0-9]*/stat; do
    [ -r "$d" ] || continue
    pid="${d#"$proc"/}"; pid="${pid%%/*}"
    ppid=$(sed 's/.*) //' "$d" 2>/dev/null | awk '{print $2}') || continue
    [ -n "$ppid" ] || continue
    children[$ppid]="${children[$ppid]:-} $pid"
  done
  local -a queue=("$root")
  local -A seen=()
  local total=0 p c rss
  while [ ${#queue[@]} -gt 0 ]; do
    p="${queue[0]}"; queue=("${queue[@]:1}")
    [ -n "${seen[$p]:-}" ] && continue
    seen[$p]=1
    rss=$(awk '/^VmRSS:/{print $2}' "$proc/$p/status" 2>/dev/null)
    [ -n "$rss" ] && total=$((total + rss))
    for c in ${children[$p]:-}; do queue+=("$c"); done
  done
  echo "$total"
}

# One heartbeat: the 120s "job alive" line plus the process-tree RSS sample,
# so a kill between steps is diagnosable from the trajectory in the job log.
hb_once(){
  local rss
  rss=$(tree_rss_kb $$ 2>/dev/null)
  echo "[gpu_job $(date +%T)] heartbeat (job alive) tree_rss_kb=${rss:-0}"
}

_on_exit(){
  # Capture the exit code FIRST: anything run before reading $? (like killing
  # the heartbeat) clobbers it, so the exit line used to report the kill's rc.
  local rc=$?
  [ -n "$HB" ] && kill "$HB" 2>/dev/null
  memstat "exit rc=$rc"
  [ -n "$CTF_WORKROOT" ] && rm -rf "$CTF_WORKROOT"
}

main(){
  CTF_WORKROOT="$(mktemp -d)"
  export CTF_WORKROOT
  trap _on_exit EXIT

  # The job's own stdout lives only in the host loop log, which agent sandboxes
  # cannot read -- an OOM-kill here was undiagnosable for a week. Mirror the full
  # log (memstat, heartbeat, child stderr/tracebacks) into the shared .tools bind.
  DIAG=.tools/h3run.joblog
  touch "$DIAG" 2>/dev/null || DIAG="$(mktemp)"
  exec > >(tee -a "$DIAG") 2>&1
  log "=== h3run job start pid=$$ diag=$DIAG ==="

  # The loop's GPU stall watchdog (JOB_STALL_GPU_MIN, default 15) kills a GPU job after
  # 15 min of no log output. Several steps here are legitimately silent for longer than
  # that -- the torch/bitsandbytes install, a long pilot arm, a QLoRA epoch. Emit a
  # job-wide heartbeat every 2 min so "slow but working" is never mistaken for wedged.
  ( while :; do sleep 120; hb_once; done ) & HB=$!

  memstat start

  # 0) venv + LIGHT deps only (pilot/analysis need pyyaml+numpy+matplotlib).
  #    Keep the pre-CORE window short: no multi-GB download before the pilot.
  #    Guard on an actual import, not just that python exists: a run killed
  #    mid-install leaves a partial venv (bin/python present, packages half-written)
  #    that must be rebuilt, not reused.
  if ! "$PY" -c "import yaml, numpy, scipy, matplotlib" 2>/dev/null; then
    log "building .venv-gpu (uv)"
    rm -rf .venv-gpu
    "$UV" venv --python 3.12 .venv-gpu \
      || { rm -rf .venv-gpu; "$UV" venv --python 3.12 .venv-gpu \
      || { log "FATAL uv venv rc=$?"; exit 2; }; }
  fi
  "$UV" pip install --python "$PY" pyyaml "$NUMPY_PIN" scipy matplotlib \
    || { log "FATAL light deps rc=$?"; exit 2; }
  log "light deps ready"

  # 1) base 7B present on host Ollama (pull over shared-net if missing)
  oll list 2>/dev/null | grep -q "$B" || { log "pulling $B"; oll pull "$B" || { log "FATAL pull rc=$?"; exit 2; }; }
  log "base model present"

  # 2) CORE: matched-base 7B pilot A-D (inference via host Ollama)
  memstat pre-pilot
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
    || { log "FATAL build_sft rc=$?"; exit 2; }
  log "sft built"

  # 4b) HEAVY training deps -- only now, with CORE pilot + SFT already on disk.
  #     The install itself is light (~80MB RSS); the guard just avoids starting
  #     the long multi-GB download window while the host is under memory pressure
  #     (the pre-fix job was killed exactly there).
  memstat before-heavy-install
  mem_guard 8 900 || log "WARN host MemAvailable < 8GB after 15min wait -- proceeding (install is light)"
  # The import check is the source of truth, not the install: a satisfied install
  # with an incompatible numpy/pyarrow pair would loop forever otherwise. It must
  # exercise train_qlora.py's REAL training-path imports, not bare `import trl`:
  # trl 0.9.6 is a lazy module (bare import passes) but its SFTTrainer chain
  # imports `rich` in trainer/utils.py WITHOUT declaring it in trl's metadata,
  # so uv never installs it and `from trl import SFTTrainer` dies at train time.
  # (+ bitsandbytes: imported at runtime by transformers' load_in_4bit path.)
  TRAIN_IMPORTS='import torch, bitsandbytes
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer'
  # Re-run the pinned install only if the heavy stack does not import: a run killed
  # mid-install leaves partial wheels; uv re-extracts exactly what is missing.
  if ! "$PY" -c "$TRAIN_IMPORTS" 2>/dev/null; then
    "$UV" pip install --python "$PY" \
      torch==2.4.1 transformers==4.44.2 peft==0.12.0 bitsandbytes==0.43.3 \
      trl==0.9.6 datasets==2.21.0 accelerate==0.33.0 rich==15.0.0 \
      "$NUMPY_PIN" "$PYARROW_PIN" \
    || { log "FATAL heavy deps rc=$?"; exit 2; }
  fi
  if ! "$PY" -c "$TRAIN_IMPORTS" 2>"$DIAG.import-err"; then
    { echo "FATAL heavy import check still failing after install:"; tail -5 "$DIAG.import-err"; } >>"$DIAG" 2>/dev/null
    log "FATAL heavy import check (see $DIAG tail)"
    exit 2
  fi
  memstat after-heavy-install
  log "heavy deps ready"

  # 5) CORE: QLoRA arm E (GPU training; writes measured gpu_hours to train_meta.json).
  #     CPU side of bf16 QLoRA peaks ~10GB -- wait for headroom instead of
  #     letting the kernel OOM-kill us mid-epoch.
  memstat before-qlora
  mem_guard 16 600 || { log "FATAL host MemAvailable < 16GB before QLoRA after 10min wait (see memstat lines)"; exit 3; }
  "$PY" -m finetune.train_qlora --config finetune/configs/qlora_qwen7b.toml \
      --data finetune/data/sft_all.jsonl --arm E --out finetune/adapters/E-qwen7b --min-free-mb 15000 \
    || { log "FATAL qlora train rc=$?"; exit 2; }
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
    } || log "E/F serving/eval incomplete rc=$? -- adapter + gpu-hours still recorded; serving needs import follow-up"
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
