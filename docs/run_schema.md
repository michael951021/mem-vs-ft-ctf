# Per-run JSONL schema

One JSON object per task attempt, appended to `runs/<run_id>/attempts.jsonl`.
A config snapshot is saved to `runs/<run_id>/config.yaml` alongside it.

```jsonc
{
  "run_id": "string",            // e.g. 20261010-A-intercode-s0
  "arm": "A|B|C|D|E|F",
  "model": "string",             // base model tag
  "seed": 0,
  "benchmark": "intercode-ctf|nyu-ctf|cybench",
  "challenge_id": "string",
  "mechanism_family": "string|null", // e.g. crypto/padding-oracle
  "split": "train|test-indist|test-transfer",
  "turns_used": 0,
  "success": false,              // final flag match / task solved
  "subtasks_passed": 0,          // Cybench-style; 0 if n/a
  "subtasks_total": 0,
  "flag_verified": false,        // ground-truth flag check (not self-report)
  "input_tokens": 0,
  "output_tokens": 0,
  "wall_clock_s": 0.0,
  "gpu_hours": 0.0,              // training arms; inference ~0 (shared server)
  "memory_size_tokens": 0,       // playbook/episodic size at attempt time
  "adapter_id": null,            // QLoRA adapter id or null
  "notes": "string"
}
```

### Invariants
- `flag_verified` is set by the harness's ground-truth checker, NEVER by the agent's self-assessment. Memory writes (arm D Reflector) gate on this field.
- `gpu_hours` for inference arms is attributed as shared-server time only if measured; otherwise 0 with a note (the Ollama server is always-on).
- Every quantitative claim in REPORT.md must cite a `run_id` present under `runs/`.
