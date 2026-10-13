#!/usr/bin/env bash
# Local-suite pilot: all arms x seeds, real Ollama inference (think-off).
# Seed-major so a COMPLETE 4-arm comparison (H1/H2) lands after the first pass.
# Writes incrementally to runs/; safe to harvest partial data at any time.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
OUT="${OUT:-runs}"
MODEL="${MODEL:-qwen-agent:latest}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-A B C D}"
# TAG isolates a run family in runs/ and its summary file. Default 'pilot' keeps
# the canonical 27B behaviour (summary.json + REPORT inject) byte-for-byte; a
# different TAG (e.g. h3base for the matched 7B H3 re-run) writes its OWN
# summary_<TAG>.json and never touches the committed 27B artifacts.
TAG="${TAG:-pilot}"
SUMMARY="${SUMMARY:-results/summary.json}"
GLOB="${GLOB:-*-$TAG}"
# MOCK=1 runs the offline oracle model (run_experiment --mock): no GPU/Ollama
# needed; used by the checkpoint acceptance test and for pipeline dry-runs.
MOCK="${MOCK:-0}"
# Checkpoint store for completed (arm,seed) cells. The pilot worktree's runs/
# is NOT a shared bind, so a job killed mid-pilot loses finished cells and the
# next pass re-runs them from scratch (the 10-13 h3run log shows 4 full pilot
# passes). Completed cells are copied here (a shared bind, ~20KB each) and a
# re-run restores + skips them. Set CKPT_DIR= (empty) to disable.
# Note the dash (not colon): CKPT_DIR= (empty) deliberately disables
# checkpointing, so ${CKPT_DIR-...} only defaults when unset.
CKPT_DIR="${CKPT_DIR-.tools/pilot_ckpt/$TAG}"
export CTF_WORKROOT="${CTF_WORKROOT:-$(mktemp -d)}"
echo "pilot start $(date -Is)  model=$MODEL seeds='$SEEDS' arms='$ARMS' out=$OUT tag=$TAG summary=$SUMMARY workroot=$CTF_WORKROOT"
harvest() {  # aggregate this run family; only the canonical 27B pilot touches REPORT/figures/hard tier
  $PY -m scripts.analyze --glob "$GLOB" --summary-out "$SUMMARY" >/dev/null 2>&1 || true
  if [[ "$TAG" == pilot ]]; then
    $PY -m scripts.analyze --glob '*-hardpilot' --summary-out results/summary_hard.json >/dev/null 2>&1 || true
    $PY -m scripts.make_figures >/dev/null 2>&1 || true
    $PY -m scripts.render_report --inject >/dev/null 2>&1 || true
  fi
  $PY -m scripts.audit_redaction >/dev/null 2>&1 || true
}
# Fingerprint of the parameters that must match for a checkpoint to apply
# (mirror cell_argv below): a different model / limits / mock flag changes the
# results, so an old cell must not be reused.
PARAMS="$TAG|$MODEL|suite=easy|max_turns=15|num_ctx=4096|num_predict=384|benchmark=local-suite|mock=$MOCK"

cell_argv(){  # $1=seed $2=arm -> global CELL array
  CELL=("$PY" -m harness.run_experiment --arm "$2" --seed "$1" --model "$MODEL" \
        --out "$OUT" --max-turns 15 --num-ctx 4096 --num-predict 384 --tag "$TAG")
  [ "$MOCK" = 1 ] && CELL+=(--mock)
}

for s in $SEEDS; do
  for a in $ARMS; do
    cell_argv "$s" "$a"
    key="$s-$a"
    if [ -n "$CKPT_DIR" ]; then
      mark="$CKPT_DIR/$key.done"
      name=""
      [ -f "$mark" ] && name=$(head -n1 "$mark")
      if [ -n "$name" ] && [ "$(sed -n 2p "$mark")" = "$PARAMS" ] \
         && [ -d "$CKPT_DIR/$key/run" ]; then
        [ -d "$OUT" ] || mkdir -p "$OUT"
        [ -d "$OUT/$name" ] || cp -r "$CKPT_DIR/$key/run" "$OUT/$name"
        echo "SKIP arm=$a seed=$s (checkpointed run=$name)"
        harvest
        continue
      fi
      [ -f "$mark" ] && echo "WARN arm=$a seed=$s checkpoint stale (params changed) -- re-running"
    fi
    echo "=== $(date -Is) arm=$a seed=$s ==="
    # Each cell runs in its own fresh process (cell_runner reaps it, so its
    # memory is reclaimed before the next cell and its peak RSS is logged).
    cell_log="$CTF_WORKROOT/cell-$key.log"
    "$PY" scripts/cell_runner.py --arm "$a" --seed "$s" -- "${CELL[@]}" \
      2>&1 | tee "$cell_log"
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] || echo "!! arm=$a seed=$s FAILED rc=$rc (continuing)"
    if [ -n "$CKPT_DIR" ] && [ "$rc" -eq 0 ]; then
      rundir=$(grep -o 'DONE -> .*' "$cell_log" | tail -n1 | sed 's/^DONE -> //')
      if [ -n "$rundir" ] && [ -d "$rundir" ]; then
        name=$(basename "$rundir")
        rm -rf "$CKPT_DIR/$key"
        mkdir -p "$CKPT_DIR/$key"
        cp -r "$rundir" "$CKPT_DIR/$key/run"
        printf '%s\n%s\n' "$name" "$PARAMS" > "$mark"
        echo "checkpointed arm=$a seed=$s -> $CKPT_DIR/$key (run=$name)"
      fi
    fi
    harvest   # keep results/ + REPORT.md §5 current after every arm
  done
done
echo "pilot done $(date -Is)"
