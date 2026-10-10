# Host Facts (measured, not assumed)

Measured 2026-10-10 on the target workstation. Supersedes the §0 placeholders in the brief.

| Fact | Value | Source |
|---|---|---|
| GPUs | 2× NVIDIA RTX 3090, 24 GB each, no NVLink | `nvidia-smi` |
| GPU free (at measurement) | ~3.2 GB / ~2.7 GB — both cards ~90% used | `nvidia-smi` |
| GPU occupant | Ollama `llama-server` holding ~20.5 GB (serving `qwen-agent`) + desktop (gnome-remote-desktop, sunshine) ~0.8 GB | `nvidia-smi --query-compute-apps` |
| System RAM | 122 GB total, ~80 GB available | `free -g` |
| Disk (root fs) | 1.8 TB total, 1.2 TB free | `df -h` |
| Docker | NOT installed / daemon not running — **blocker for all CTF benchmarks** | `docker version` |
| System Python | 3.14.4 — too new for torch/bnb/unsloth/vLLM; use a 3.11/3.12 venv | `python3 --version` |
| Job queue / scheduler | NONE (no slurm/ts/at/ray). We build our own file-based queue in `queue/`. | PATH probe |

## Inference endpoint (already live)
- **Ollama** at `http://127.0.0.1:11434`, systemd `ollama.service` active.
- Models pulled locally:
  | tag | params | quant | size |
  |---|---|---|---|
  | `qwen-agent:latest` (alias of `qwen3.8:27b-q8_0`) | 27.3B | Q8_0 | 30.0 GB |
  | `qwen3.8:27b-q8_0` | 27.3B | Q8_0 | 30.0 GB |
  | `hf.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED:Q4_K_M` | 27.3B | Q4_K_M | 17.7 GB |
  | `mistral-small3.2:24b` | 24.0B | Q4_K_M | 15.2 GB |
  | `llama3.3:70b` | 70.6B | Q4_K_M | 42.5 GB |
- Smoke test (`qwen-agent`, temp 0): cold load ~35 s, eval ~5.9 s for 8 tokens.
- **Caveat:** `qwen3.8:27b` appears to be a reasoning model — an 8-token generation returned empty visible text (tokens spent inside a `<think>` block). The model client must handle thinking tokens.
- **Gap for QLoRA (arms E/F):** no 7–14B model is pulled. The brief wants a 7–14B QLoRA base that fits one 24 GB card in 4-bit. QLoRA trains from HF safetensors (not Ollama GGUF), so this is a separate HF download handled at Phase 3.

## Consequences for the plan
1. **Memory arms (A–D) need only inference** → runnable now against the live Ollama endpoint; no extra VRAM.
2. **QLoRA arms (E/F) need free VRAM** → cannot coexist with Ollama's ~20 GB/card. These are the deferred "big runs," launched at the end once the GPUs are freed (or Ollama is temporarily stopped).
3. All heavy runs are **staged in `queue/`** with time estimates; nothing heavy launches without the GPUs being free.
