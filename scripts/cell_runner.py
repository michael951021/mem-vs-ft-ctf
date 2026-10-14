"""Run one pilot cell in a fresh child process and report its peak RSS.

Usage: cell_runner.py --arm A --seed 0 -- <cmd> [args...]

Streams the child's stdout+stderr to our stdout (so the job log still shows
every attempt), exits with the child's exit code, then prints one final line:
  cell-peak arm=A seed=0 pid=<child pid> peak_rss_kb=<N> rc=<rc>

peak_rss_kb is the child's high-water RSS (ru_maxrss; KB on Linux). Each cell
getting its own child is what reclaims the cell's memory before the next cell
starts; the logged peak is the per-cell attribution the 47.6G cgroup OOM
investigation needs.
"""
from __future__ import annotations

import argparse
import resource
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("cell_runner: no command after --", file=sys.stderr)
        return 2

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    p.wait()
    # cell_runner reaps exactly one child (the cell), so RUSAGE_CHILDREN's
    # max is that cell's peak RSS.
    peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(f"cell-peak arm={args.arm} seed={args.seed} pid={p.pid} "
          f"peak_rss_kb={peak_kb} rc={p.returncode}", flush=True)
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
