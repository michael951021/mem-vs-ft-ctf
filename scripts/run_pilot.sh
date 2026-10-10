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
export CTF_WORKROOT="${CTF_WORKROOT:-$(mktemp -d)}"
echo "pilot start $(date -Is)  model=$MODEL seeds='$SEEDS' arms='$ARMS' out=$OUT workroot=$CTF_WORKROOT"
harvest() {  # aggregate (per tier) + figures + inject report numbers from logged data
  $PY -m scripts.analyze --glob '*-pilot' --summary-out results/summary.json >/dev/null 2>&1 || true
  $PY -m scripts.analyze --glob '*-hardpilot' --summary-out results/summary_hard.json >/dev/null 2>&1 || true
  $PY -m scripts.make_figures >/dev/null 2>&1 || true
  $PY -m scripts.render_report --inject >/dev/null 2>&1 || true
  $PY -m scripts.audit_redaction >/dev/null 2>&1 || true
}
for s in $SEEDS; do
  for a in $ARMS; do
    echo "=== $(date -Is) arm=$a seed=$s ==="
    $PY -m harness.run_experiment --arm "$a" --seed "$s" --model "$MODEL" \
        --out "$OUT" --max-turns 15 --num-ctx 4096 --num-predict 384 --tag pilot \
        || echo "!! arm=$a seed=$s FAILED (continuing)"
    harvest   # keep results/ + REPORT.md §5 current after every arm
  done
done
echo "pilot done $(date -Is)"
