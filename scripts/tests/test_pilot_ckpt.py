"""Offline acceptance for run_pilot.sh: per-cell fresh process + checkpoint.

Runs a 2-arm MOCK pilot (no GPU/Ollama) twice. Proves (1) each (arm,seed) cell
runs in its own fresh process with its peak RSS logged, and (2) a re-run skips
checkpointed cells and restores a deleted cell's run dir instead of re-running.

Known answers: a mock cell executes exactly 15 attempts (5 train + 5 indist +
5 transfer), so a correctly restored run dir has 15 attempts.jsonl lines.

Run: python3 -m scripts.tests.test_pilot_ckpt
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO, "scripts", "run_pilot.sh")
PY = os.environ.get("PY") or sys.executable  # the interpreter running this test, not a host-only venv
N_ATTEMPTS = 15  # 5 train + 5 test-indist + 5 test-transfer (easy suite)


def run_pilot(out_dir, ckpt_dir, log_path, timeout=900):
    env = dict(os.environ)
    env.update({
        "PY": PY, "MODEL": "mock-model", "TAG": "ckpttest",
        "SEEDS": "0", "ARMS": "A B", "MOCK": "1", "OUT": out_dir,
        "CKPT_DIR": ckpt_dir, "GLOB": "*-ckpttest",
        "SUMMARY": os.path.join(out_dir, "..", "summary.json"),
        "CTF_WORKROOT": tempfile.mkdtemp(prefix="pilot-ckpt-test-"),
    })
    with open(log_path, "w") as log:
        try:
            rc = subprocess.call(["bash", SCRIPT], env=env, cwd=REPO,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = 124
    with open(log_path) as f:
        return rc, f.read()


def cell_dirs(out_dir):
    """arm -> (run_dir_name, attempts_line_count)"""
    res = {}
    if not os.path.isdir(out_dir):
        return res
    for name in os.listdir(out_dir):
        m = re.fullmatch(r"(\d{8}-\d{6})-([AB])-local-suite-s0-ckpttest", name)
        if m:
            p = os.path.join(out_dir, name, "attempts.jsonl")
            n = sum(1 for _ in open(p)) if os.path.isfile(p) else -1
            res[m.group(2)] = (name, n)
    return res


def main():
    fails = []
    tmp = tempfile.mkdtemp(prefix="pilot-ckpt-acc-")
    out, ckpt = os.path.join(tmp, "runs"), os.path.join(tmp, "ckpt")
    ev = os.path.join(REPO, ".agent-evidence")
    os.makedirs(ev, exist_ok=True)
    try:
        # --- leg 1: fresh 2-arm mock pilot -> per-cell process + RSS + checkpoint
        rc, out1 = run_pilot(out, ckpt, os.path.join(ev, "pilot-ckpt-run1.log"))
        if rc != 0:
            fails.append(f"run1 rc={rc} (expected 0)")
        peaks = re.findall(
            r"cell-peak arm=([AB]) seed=0 pid=(\d+) peak_rss_kb=(\d+) rc=0", out1)
        if len(peaks) != 2:
            fails.append(f"run1: expected 2 cell-peak lines, got {len(peaks)}")
        else:
            arms = {p[0] for p in peaks}
            pids = [p[1] for p in peaks]
            if arms != {"A", "B"}:
                fails.append(f"run1: cell-peak arms={arms} (expected {{A,B}})")
            if len(set(pids)) != 2:
                fails.append(f"run1: cells shared a pid {pids} (not fresh processes)")
            if any(int(p[2]) < 5000 for p in peaks):
                fails.append(f"run1: implausible peak_rss_kb {[p[2] for p in peaks]}")
        dirs1 = cell_dirs(out)
        if set(dirs1) != {"A", "B"}:
            fails.append(f"run1: run dirs={dirs1} (expected A and B)")
        for a, (name, n) in dirs1.items():
            if n != N_ATTEMPTS:
                fails.append(f"run1: arm {a} attempts={n} (expected {N_ATTEMPTS})")
        for a in "AB":
            mark = os.path.join(ckpt, f"0-{a}.done")
            if not os.path.isfile(mark):
                fails.append(f"run1: missing checkpoint marker {mark}")
            elif not os.path.isfile(os.path.join(ckpt, f"0-{a}", "run", "attempts.jsonl")):
                fails.append(f"run1: checkpoint store for 0-{a} lacks run/attempts.jsonl")

        # --- leg 2: delete one cell's run dir, re-run -> both skipped, B restored
        if "B" in dirs1:
            shutil.rmtree(os.path.join(out, dirs1["B"][0]))
        rc, out2 = run_pilot(out, ckpt, os.path.join(ev, "pilot-ckpt-run2.log"))
        if rc != 0:
            fails.append(f"run2 rc={rc} (expected 0)")
        if re.search(r"cell-peak arm=[AB] seed=0", out2):
            fails.append("run2: re-ran a cell instead of skipping")
        for a in "AB":
            if f"SKIP arm={a} seed=0" not in out2:
                fails.append(f"run2: no 'SKIP arm={a} seed=0' line")
        dirs2 = cell_dirs(out)
        if set(dirs2) != {"A", "B"}:
            fails.append(f"run2: run dirs after restore={dirs2} (expected A and B)")
        elif dirs2["B"][1] != N_ATTEMPTS:
            fails.append(f"run2: restored B attempts={dirs2['B'][1]} (expected {N_ATTEMPTS})")
        # restored B must be byte-identical to the checkpointed copy
        bname = dirs2.get("B", ("",))[0]
        b_ckpt = os.path.join(ckpt, "0-B", "run", "attempts.jsonl")
        b_out = os.path.join(out, bname, "attempts.jsonl") if bname else ""
        if not (os.path.isfile(b_ckpt) and os.path.isfile(b_out)):
            fails.append("run2: cannot compare restored B (checkpoint or run dir missing)")
        else:
            if open(b_ckpt, "rb").read() != open(b_out, "rb").read():
                fails.append("run2: restored B attempts.jsonl differs from checkpoint")

        # --- leg 3: legacy path (CKPT_DIR empty) -> runs cells, no skip/store
        tmp3 = tempfile.mkdtemp(prefix="pilot-ckpt-legacy-")
        out3 = os.path.join(tmp3, "runs")
        rc, out3log = run_pilot(out3, "", os.path.join(ev, "pilot-ckpt-run3.log"))
        if rc != 0:
            fails.append(f"run3 (legacy) rc={rc} (expected 0)")
        if len(re.findall(r"cell-peak arm=[AB] seed=0", out3log)) != 2:
            fails.append("run3 (legacy): expected 2 cell-peak lines")
        if "SKIP arm=" in out3log:
            fails.append("run3 (legacy): unexpected SKIP with CKPT_DIR empty")

        shutil.rmtree(tmp3, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for f in fails:
        print("FAIL:", f)
    print(f"{'PASS' if not fails else 'FAIL'}: test_pilot_ckpt ({len(fails)} failure(s))")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
