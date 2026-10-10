"""Ethics gate (brief 8): verify NO flag-shaped text survives in any stored
memory artifact or SFT corpus, and emit a >=20% spot-check sample for manual
review. Exits non-zero if any leak is found, so it can gate a phase.

Scans:
  runs/*/memory_frozen.json         (arms B/C/D persisted memories)
  finetune/data/*.jsonl             (arms E/F SFT corpus)
Writes results/redaction_audit.json (leak count + the sampled records).
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys

from memory.redact import contains_flagish


def scan_text_blobs():
    blobs = []
    for p in glob.glob("runs/*/memory_frozen.json"):
        with open(p) as f:
            blobs.append((p, f.read()))
    for p in glob.glob("finetune/data/*.jsonl"):
        with open(p) as f:
            for i, line in enumerate(f):
                blobs.append((f"{p}:{i}", line))
    return blobs


def main():
    blobs = scan_text_blobs()
    leaks = [(src, b) for src, b in blobs if contains_flagish(b)]
    rng = random.Random(0)
    k = max(1, int(0.2 * len(blobs))) if blobs else 0
    sample = rng.sample(blobs, min(k, len(blobs))) if blobs else []
    report = {
        "n_artifacts_scanned": len(blobs),
        "n_leaks": len(leaks),
        "leak_sources": [s for s, _ in leaks][:50],
        "spot_check_fraction": round(k / len(blobs), 3) if blobs else 0,
        "spot_check_sample": [{"src": s, "preview": b[:300]} for s, b in sample],
    }
    os.makedirs("results", exist_ok=True)
    with open("results/redaction_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"scanned {len(blobs)} artifacts; leaks={len(leaks)}; "
          f"spot-check {report['spot_check_fraction']*100:.0f}% "
          f"-> results/redaction_audit.json")
    if leaks:
        print("FLAG LEAK DETECTED in:", report["leak_sources"], file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
