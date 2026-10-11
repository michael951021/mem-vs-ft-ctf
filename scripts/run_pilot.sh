#!/usr/bin/env bash
# Local-suite pilot: all arms x seeds, real Ollama inference (think-off).
# Seed-major so a COMPLETE 4-arm comparison (H1/H2) lands after the first pass.
# Writes incrementally to runs/; safe to harvest partial data at any time.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
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
for s in $SEEDS; do
  for a in $ARMS; do
    echo "=== $(date -Is) arm=$a seed=$s ==="
    $PY -m harness.run_experiment --arm "$a" --seed "$s" --model "$MODEL" \
        --out "$OUT" --max-turns 15 --num-ctx 4096 --num-predict 384 --tag "$TAG" \
        || echo "!! arm=$a seed=$s FAILED (continuing)"
    harvest   # keep results/ + REPORT.md §5 current after every arm
  done
done
echo "pilot done $(date -Is)"
