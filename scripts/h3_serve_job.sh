#!/usr/bin/env bash
# Loop GPU job `h3serve`: h3run's step 6 (serve the QLoRA-E model, eval E and F) redone strictly.
# h3run trained + merged E, but `ollama create` got `FROM finetune/adapters/...` (relative), which Ollama
# parsed as a registry host "finetune"; E/F evals then all 404'd and the best-effort tail swallowed it.
# Here every step must succeed or the job fails (and the queue's guard reports it).
# Matched comparison: E is quantized to Q4_K_M with the base qwen2.5:7b-instruct's chat template, like
# the A-D arms' base model; the harness sets temperature/num_ctx per request for every arm.
set -euo pipefail
unset OLLAMA_HOST OLLAMA_BASE_URL ANTHROPIC_BASE_URL
OLL=127.0.0.1:11434; PY=.venv-gpu/bin/python; BASE=qwen2.5:7b-instruct; E=qwen7b-ftE
oll(){ OLLAMA_HOST=$OLL ollama "$@"; }
log(){ echo "[h3serve $(date +%T)] $*"; }

if ! oll show "$E" >/dev/null 2>&1; then
  M="$PWD/finetune/adapters/E-qwen7b-merged"
  test -f "$M/model.safetensors.index.json"
  { echo "FROM $M"; oll show "$BASE" --modelfile | awk '/^TEMPLATE/{t=1} /^LICENSE/{t=0} t'; } > finetune/adapters/E.Modelfile
  log "creating $E (q4_K_M) from $M"
  oll create "$E" -q q4_K_M -f finetune/adapters/E.Modelfile
fi
oll show "$E" | sed -n 1,12p
log "smoke: one chat turn"
curl -sf "http://$OLL/api/chat" -d "{\"model\":\"$E\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}]}" \
  | "$PY" -c 'import json,sys; r=json.load(sys.stdin)["message"]["content"]; print("reply:", r[:80]); assert r.strip()'

rm -rf runs/*-[EF]-local-suite-s0-h3
log "E: weights only (fine-tuned model, no memory), eval-only"
"$PY" -m harness.run_experiment --arm A --arm-label E --eval-only --model "$E" --seed 0 --out runs --tag h3
F_MEM="$(ls -d runs/*-D-local-suite-s0-h3base | tail -1)/memory_frozen.json"
test -f "$F_MEM"
log "F: weights + D's frozen playbook ($F_MEM), eval-only"
"$PY" -m harness.run_experiment --arm D --arm-label F --eval-only --load-memory "$F_MEM" --model "$E" --seed 0 --out runs --tag h3

for arm in E F; do   # a run that only errored writes no attempts: fail loudly instead of aggregating nothing
  d="$(ls -d runs/*-$arm-local-suite-s0-h3 | tail -1)"
  n=$(grep -c . "$d/attempts.jsonl" 2>/dev/null || echo 0)
  [ "$n" -gt 0 ] || { log "arm $arm wrote no attempts ($d)"; exit 1; }
  log "arm $arm: $n attempts in $d"
done

"$PY" -m scripts.analyze --glob '*-h3*' --summary-out results/summary_h3.json
"$PY" -m scripts.audit_redaction
log "done: results/summary_h3.json"
