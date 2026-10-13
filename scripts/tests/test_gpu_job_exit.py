"""Offline tests for scripts/gpu_job.sh exit-code + memory-trajectory fixes.

Checks (known answers, no GPU needed):
1. `bash -n` parses the script.
2. The EXIT trap reports the script's TRUE exit code: sourcing the script and
   exiting 2 must log `memstat[exit rc=2]`. Pre-fix, `_on_exit` runs
   `kill "$HB"` before reading `$?`, so it reports the kill's rc (1 when
   HB is empty), never the job's.
3. `tree_rss_kb ROOT PROC_DIR` sums VmRSS over ROOT's process-tree
   descendants using only /proc (no ps) -- verified against a fabricated
   proc tree with known RSS values.
4. `hb_once` (the 120s heartbeat body) appends a `tree_rss_kb=` sample so a
   future kill can be attributed from the log: a ramping sample means the
   memory killer, a flat one an external killer.

The script must stay source-safe: sourcing defines the helpers without
running the pipeline (main is guarded by BASH_SOURCE == $0).

Run: python3 -m scripts.tests.test_gpu_job_exit
"""
import os
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO, "scripts", "gpu_job.sh")


def sh(c):
    r = subprocess.run(["bash", "-c", c], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def true_exit_rc(want_rc=2):
    """Source the script, trap _on_exit, exit `want_rc`; look for the rc in the log line."""
    _, out, _ = sh(f'source "{SCRIPT}" >/dev/null 2>&1; trap _on_exit EXIT; '
                   f'exit {want_rc}')
    return "exit rc=%d" % want_rc in out


def tree_rss(root, proc):
    _, out, err = sh(f'source "{SCRIPT}" >/dev/null 2>&1; '
                     f'tree_rss_kb {root} "{proc}"')
    return out.strip(), err


def make_fake_proc(d):
    # pid (comm) state ppid ... ; status VmRSS in kB.
    # tree: 1 -> 10 -> 20 ; 1 -> 30 (not under 10)
    procs = {
        "1":  ("1 (init) S 0 1 1 0 -1 4194560 0 0 0 0 0 0 0 20 0 1 0 0 0 0", None),
        "10": ("10 (a) S 1 10 10 0 -1 4194560 0 0 0 0 0 0 0 20 0 1 0 0 0 0", 1000),
        "20": ("20 (b) S 10 10 10 0 -1 4194560 0 0 0 0 0 0 0 20 0 1 0 0 0 0", 3000),
        "30": ("30 (c) S 1 1 1 0 -1 4194560 0 0 0 0 0 0 0 20 0 1 0 0 0 0", 5000),
    }
    for pid, (stat, rss) in procs.items():
        os.makedirs(os.path.join(d, pid))
        with open(os.path.join(d, pid, "stat"), "w") as f:
            f.write(stat + "\n")
        with open(os.path.join(d, pid, "status"), "w") as f:
            f.write("Name:\t%s\n" % pid)
            if rss is not None:
                f.write("VmRSS:\t%d kB\n" % rss)


def main():
    failures = 0

    r = subprocess.run(["bash", "-n", SCRIPT], capture_output=True, text=True)
    ok = r.returncode == 0
    failures += (not ok)
    print(f"{'PASS' if ok else 'FAIL'} bash -n gpu_job.sh rc={r.returncode}")
    if not ok:
        print(r.stderr)

    ok = true_exit_rc(2)
    failures += (not ok)
    print(f"{'PASS' if ok else 'FAIL'} _on_exit reports true rc=2 on exit 2 "
          f"(pre-fix: 'kill $HB' clobbers $? before memstat reads it)")

    ok = true_exit_rc(3)
    failures += (not ok)
    print(f"{'PASS' if ok else 'FAIL'} _on_exit reports true rc=3 on exit 3")

    with tempfile.TemporaryDirectory() as d:
        make_fake_proc(d)
        want = "4000"  # root 10 (1000kB) + child 20 (3000kB); 30 (5000kB) excluded
        got, _ = tree_rss(10, d)
        ok = got == want
        failures += (not ok)
        print(f"{'PASS' if ok else 'FAIL'} tree_rss_kb(10) over fake proc = "
              f"{got!r} want={want!r} (descendants only)")
        got, _ = tree_rss(1, d)
        ok = got == "9000"  # 1000 + 3000 + 5000
        failures += (not ok)
        print(f"{'PASS' if ok else 'FAIL'} tree_rss_kb(1) over fake proc = "
              f"{got!r} want='9000' (whole tree)")

    _, out, _ = sh(f'source "{SCRIPT}" >/dev/null 2>&1; hb_once')
    ok = "heartbeat" in out and "tree_rss_kb=" in out
    failures += (not ok)
    print(f"{'PASS' if ok else 'FAIL'} hb_once emits heartbeat line with "
          f"tree_rss_kb= sample (got: {out.strip()!r})")

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
