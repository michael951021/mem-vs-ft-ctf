#!/usr/bin/env bash
# Merge a QLoRA adapter into its base and import the result into Ollama as TAG,
# so arm-E/F evaluation uses the identical ReAct agent + Ollama path as A-D.
# HOST-ONLY (needs the training venv + a GPU window). Usage: import_adapter.sh ADAPTER_DIR TAG
set -euo pipefail
ADAPTER="${1:?adapter dir}"; TAG="${2:?ollama tag}"
cd "$(dirname "$0")/.."
MERGED="finetune/adapters/$(basename "$ADAPTER")-merged"
.venv/bin/python - "$ADAPTER" "$MERGED" <<'PY'
import sys
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
adapter, out = sys.argv[1], sys.argv[2]
m = AutoPeftModelForCausalLM.from_pretrained(adapter, device_map="cpu")
m = m.merge_and_unload()
m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
print("merged ->", out)
PY
# GGUF convert + ollama create would follow here; left to the GPU-window runbook
# (queue/HUB.md) since it depends on llama.cpp being built on the host.
cat > "finetune/adapters/$(basename "$ADAPTER").Modelfile" <<MF
FROM $MERGED
PARAMETER temperature 0
PARAMETER num_ctx 4096
MF
echo "wrote Modelfile; run: ollama create $TAG -f finetune/adapters/$(basename "$ADAPTER").Modelfile"
