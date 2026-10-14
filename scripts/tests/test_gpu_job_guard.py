"""Offline tests for scripts/gpu_job.sh memory guard (no GPU/Docker needed).

Checks that `mem_guard MIN_GB WAIT MEMINFO` pass/fails exactly at the
MemAvailable threshold, using fabricated /proc/meminfo files (known answers),
and that the whole script parses. The script must be source-safe: sourcing it
must define the helpers without running the pipeline.

Run: python3 -m scripts.tests.test_gpu_job_guard
"""
import os
import signal
import subprocess
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO, "scripts", "gpu_job.sh")


def run_guard(min_gb, avail_gb):
    with tempfile.NamedTemporaryFile("w", suffix=".meminfo", delete=False) as f:
        kb = avail_gb * 1024 * 1024
        f.write(f"MemTotal: {kb} kB\nMemFree: {kb} kB\nMemAvailable: {kb} kB\n")
        path = f.name
    # Run in its own process group: a non-source-safe script (pre-fix) starts
    # the real pipeline on source, so a timeout must kill the whole tree.
    proc = subprocess.Popen(
        ["bash", "-c",
         f'source "{SCRIPT}" >/dev/null 2>&1; mem_guard {min_gb} 0 "{path}" '
         f'&& echo PASS || echo FAIL'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        preexec_fn=os.setsid)
    try:
        out, _ = proc.communicate(timeout=60)
        return 0 if out.strip().endswith("PASS") else 1
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        return 1
    finally:
        os.unlink(path)


CASES = [
    # (min_gb, avail_gb, expected_rc)
    (32, 40, 0),   # well above -> pass
    (32, 32, 0),   # exactly at threshold -> pass (>= semantics)
    (32, 31, 1),   # 1 GB short -> fail
    (16, 16, 0),   # exactly at threshold -> pass
    (16, 15, 1),   # 1 GB short -> fail
]


def main():
    failures = 0
    r = subprocess.run(["bash", "-n", SCRIPT], capture_output=True, text=True)
    ok = r.returncode == 0
    failures += (not ok)
    print(f"{'PASS' if ok else 'FAIL'} bash -n gpu_job.sh rc={r.returncode}")
    if not ok:
        print(r.stderr)

    for min_gb, avail_gb, want in CASES:
        got = run_guard(min_gb, avail_gb)
        ok = got == want
        failures += (not ok)
        print(f"{'PASS' if ok else 'FAIL'} mem_guard(min={min_gb}GB, avail={avail_gb}GB) "
              f"rc={got} want={want}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
