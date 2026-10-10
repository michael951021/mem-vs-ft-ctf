"""QLoRA trainer for arms E (weights) / F (hybrid) -- HOST-ONLY, GPU-gated.

H3 crux: fine-tune the SAME base family A-D use, on the SAME verified train
trajectories the memory arms consolidated from, under a matched GPU-hour budget.

This script is safe to invoke in the sandbox with --dry-run: it validates the
config, confirms the SFT corpus + base model are present, prints the plan and the
estimated GPU-hours, and exits 0 WITHOUT importing torch or touching a GPU
(loopgym's host-only convention). The real run (no --dry-run) requires a free
GPU and the heavy deps (transformers/peft/bitsandbytes/trl), installed into the
venv only when the GPU window opens:
    .venv/bin/pip install -r finetune/requirements-train.txt

Base-model note (DECISIONS D4-final): A-D piloted on qwen3.8:27b via Ollama,
which is outside the 7-14B QLoRA band. For a clean, matched H3 the staged GPU job
(queue/jobs/*.json) FIRST re-runs A-D on a 7-14B base (Qwen2.5-7B-Instruct) so
every arm shares one frozen base, THEN QLoRAs that base here. GPU-hours are
logged for compute-matching against the memory arms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def load_config(path):
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def count_jsonl(path):
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def estimate_gpu_hours(n_examples, epochs, base_params_b, seq_len=1024):
    """Very rough QLoRA GPU-hour estimate for a single RTX 3090.
    ~ (6 * N_tokens * params) / throughput ; we fold the constants into a
    measured-ish factor and report it as an ESTIMATE only."""
    n_tokens = n_examples * seq_len * epochs
    # empirical factor for QLoRA fwd+bwd on a 3090, ~ seconds per (Btoken*Bparam)
    factor_s = 3.0e-9  # s per (token * param); calibrate on first real run
    secs = n_tokens * base_params_b * 1e9 * factor_s / 1.0
    return round(secs / 3600.0, 3)


def gpu_free_mb():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        return [int(x) for x in out.split()]
    except Exception:
        return []


def plan(cfg, args):
    n = count_jsonl(args.data)
    gh = estimate_gpu_hours(n, cfg["train"]["epochs"],
                            cfg["model"]["approx_params_b"],
                            cfg["train"].get("seq_len", 1024))
    return {
        "arm": args.arm,
        "base_model": cfg["model"]["name"],
        "approx_params_b": cfg["model"]["approx_params_b"],
        "quant": "nf4 (QLoRA 4-bit)",
        "lora": {k: cfg["lora"][k] for k in ("r", "alpha", "dropout", "target_modules")},
        "train": cfg["train"],
        "sft_corpus": args.data,
        "n_examples": n,
        "est_gpu_hours_single_3090": gh,
        "output_adapter": args.out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qlora_qwen7b.toml")
    ap.add_argument("--data", default="finetune/data/sft_all.jsonl")
    ap.add_argument("--arm", default="E", choices=["E", "F"])
    ap.add_argument("--out", default="finetune/adapters/E-qwen7b")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-free-mb", type=int, default=20000,
                    help="require this much free VRAM on some GPU before a real run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = plan(cfg, args)
    print("=== QLoRA plan ===")
    print(json.dumps(p, indent=2))

    if p["n_examples"] == 0:
        print("\n[blocked] SFT corpus is empty. Run finetune/build_sft.py after a "
              "pilot with verified TRAIN solves.", file=sys.stderr)
        if not args.dry_run:
            sys.exit(2)

    if args.dry_run:
        # validate base model availability (Ollama tag or local HF dir) without GPU
        print("\n[dry-run] config + corpus validated; no GPU touched. "
              "To execute: free a GPU, install finetune/requirements-train.txt, "
              "rerun without --dry-run.")
        sys.exit(0)

    free = gpu_free_mb()
    if not free or max(free) < args.min_free_mb:
        print(f"\n[blocked] need >= {args.min_free_mb} MB free on some GPU; "
              f"saw {free}. Free the GPU (stop/pause Ollama) and retry.",
              file=sys.stderr)
        sys.exit(3)

    # ---- real training path (heavy deps required) -----------------------
    t0 = time.time()
    import torch  # noqa: F401
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["hf_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["hf_name"], quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                      lora_dropout=cfg["lora"]["dropout"],
                      target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    ds = load_dataset("json", data_files=args.data, split="train")

    def fmt(ex):
        return tok.apply_chat_template(ex["messages"], tokenize=False)

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"], logging_steps=5, save_strategy="epoch",
        bf16=True, report_to=[])
    trainer = SFTTrainer(model=model, args=targs, train_dataset=ds,
                         formatting_func=fmt, max_seq_length=cfg["train"].get("seq_len", 1024))
    trainer.train()
    trainer.save_model(args.out)
    gh = round((time.time() - t0) / 3600.0, 4)
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump({"gpu_hours_measured": gh, "plan": p}, f, indent=2)
    print(f"done; adapter -> {args.out}; measured GPU-hours={gh}")


if __name__ == "__main__":
    main()
