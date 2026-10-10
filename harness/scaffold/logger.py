"""Per-run JSONL logging (schema: docs/run_schema.md).

One RunLogger per run_id. Appends one attempt object per task attempt and saves
a config snapshot alongside. Append-only + flush so a crash mid-run keeps data.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field

SCHEMA_FIELDS = [
    "run_id", "arm", "model", "seed", "benchmark", "challenge_id",
    "mechanism_family", "split", "turns_used", "success",
    "subtasks_passed", "subtasks_total", "flag_verified",
    "input_tokens", "output_tokens", "wall_clock_s", "gpu_hours",
    "memory_size_tokens", "adapter_id", "notes",
]


@dataclass
class Attempt:
    run_id: str
    arm: str
    model: str
    seed: int
    benchmark: str
    challenge_id: str
    split: str
    mechanism_family: str | None = None
    turns_used: int = 0
    success: bool = False
    subtasks_passed: int = 0
    subtasks_total: int = 0
    flag_verified: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    wall_clock_s: float = 0.0
    gpu_hours: float = 0.0
    memory_size_tokens: int = 0
    adapter_id: str | None = None
    notes: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        return {k: d[k] for k in SCHEMA_FIELDS}


class RunLogger:
    def __init__(self, runs_dir: str, run_id: str, config: dict):
        self.run_dir = os.path.join(runs_dir, run_id)
        os.makedirs(os.path.join(self.run_dir, "raw"), exist_ok=True)
        self.attempts_path = os.path.join(self.run_dir, "attempts.jsonl")
        self._save_config(config)

    def _save_config(self, config: dict):
        path = os.path.join(self.run_dir, "config.yaml")
        try:
            import yaml  # optional; fall back to json
            with open(path, "w") as f:
                yaml.safe_dump(config, f, sort_keys=False)
        except Exception:
            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

    def log(self, attempt: Attempt):
        with open(self.attempts_path, "a") as f:
            f.write(json.dumps(attempt.to_row()) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def save_raw(self, challenge_id: str, seed: int, transcript: dict):
        """Full turn-by-turn transcript for audit/provenance (gitignored)."""
        path = os.path.join(self.run_dir, "raw", f"{challenge_id}_s{seed}.json")
        with open(path, "w") as f:
            json.dump(transcript, f, indent=2)
