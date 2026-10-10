"""File-based job queue for the GPU-heavy runs (DECISIONS D7).

The hub's pool runs *sandboxed Claude-on-Ollama* loops; actual GPU training is a
HOST operation (loopgym's own GPU runs are host-only). So the heavy arms of this
study -- the matched-base A-D re-run on a 7-14B model and the QLoRA arms E/F --
are staged here as JSON job specs and executed on the host by this runner when
GPU headroom exists. Nothing launches while the GPUs are busy.

A job spec (queue/jobs/*.json):
  {
    "id": "...", "needs_gpu_free_mb": 20000, "est_minutes": 90,
    "depends_on": ["<other job id>"],          # optional
    "cmd": ["...argv..."],                      # run on the host
    "cwd": ".", "note": "..."
  }

Usage:
  python -m queue.runner list                 # show staged jobs + readiness
  python -m queue.runner plan                 # topological order + ETAs
  python -m queue.runner run [--force]        # run ready jobs until none left
                                              # (--force skips the GPU precheck)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

JOBS_DIR = os.path.join(os.path.dirname(__file__), "jobs")
STATE = os.path.join(os.path.dirname(__file__), "state.json")


def gpu_free_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        return [int(x) for x in out.split()]
    except Exception:
        return []


def load_jobs():
    jobs = []
    for p in sorted(glob.glob(os.path.join(JOBS_DIR, "*.json"))):
        with open(p) as f:
            j = json.load(f)
        j["_path"] = p
        jobs.append(j)
    return jobs


def load_state():
    if os.path.isfile(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"done": [], "log": []}


def save_state(s):
    with open(STATE, "w") as f:
        json.dump(s, f, indent=2)


def topo(jobs):
    by_id = {j["id"]: j for j in jobs}
    order, seen = [], set()

    def visit(j, stack):
        if j["id"] in seen:
            return
        for dep in j.get("depends_on", []):
            if dep in by_id and dep not in stack:
                visit(by_id[dep], stack | {j["id"]})
        seen.add(j["id"])
        order.append(j)

    for j in jobs:
        visit(j, set())
    return order


def ready(job, state, free):
    if job["id"] in state["done"]:
        return (False, "done")
    for dep in job.get("depends_on", []):
        if dep not in state["done"]:
            return (False, f"waiting on {dep}")
    need = job.get("needs_gpu_free_mb", 0)
    if need and (not free or max(free) < need):
        return (False, f"needs {need}MB free, have {max(free) if free else 0}MB")
    return (True, "ready")


def cmd_list(jobs, state):
    free = gpu_free_mb()
    print(f"GPU free: {free} MB | {len(jobs)} staged jobs")
    for j in topo(jobs):
        ok, why = ready(j, state, free)
        mark = "DONE" if j["id"] in state["done"] else ("READY" if ok else "wait")
        print(f"  [{mark:5}] {j['id']:28s} ~{j.get('est_minutes','?')}min  {why}")
        print(f"            {j.get('note','')}")


def cmd_plan(jobs):
    total = 0
    print("Execution order (topological):")
    for i, j in enumerate(topo(jobs), 1):
        total += int(j.get("est_minutes", 0) or 0)
        print(f"  {i}. {j['id']}  (~{j.get('est_minutes','?')}min)"
              f"  deps={j.get('depends_on', [])}")
    print(f"Estimated total if run sequentially: ~{total} min (~{total/60:.1f} h)")


def cmd_run(jobs, state, force):
    for j in topo(jobs):
        free = gpu_free_mb()
        ok, why = ready(j, state, free)
        if j["id"] in state["done"]:
            continue
        if not ok and not force:
            print(f"skip {j['id']}: {why}")
            continue
        print(f"\n=== running {j['id']} (~{j.get('est_minutes','?')}min) ===")
        t0 = time.time()
        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                           j.get("cwd", ".")))
        rc = subprocess.call(j["cmd"], cwd=cwd)
        dt = round((time.time() - t0) / 60.0, 1)
        entry = {"id": j["id"], "rc": rc, "minutes": dt, "when": time.strftime("%FT%T")}
        state["log"].append(entry)
        if rc == 0:
            state["done"].append(j["id"])
            print(f"  OK in {dt}min")
        else:
            print(f"  FAILED rc={rc} after {dt}min -- stopping (fix + rerun)")
            save_state(state)
            sys.exit(rc)
        save_state(state)
    print("\nall ready jobs complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["list", "plan", "run"], default="list", nargs="?")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    jobs, state = load_jobs(), load_state()
    if not jobs:
        print("no jobs staged in", JOBS_DIR); return
    if args.action == "list":
        cmd_list(jobs, state)
    elif args.action == "plan":
        cmd_plan(jobs)
    else:
        cmd_run(jobs, state, args.force)


if __name__ == "__main__":
    main()
