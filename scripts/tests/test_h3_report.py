"""Acceptance for the H3 report section (REPORT.md, id: h3report).

(A) Known-answer: recompute pooled per-arm, per-split success from the raw
    attempts.jsonl of the canonical h3 runs and compare with
    results/summary_h3.json. Hand anchors: A test-indist 4/10 = 0.4 (2/5 per
    seed), E test-indist 4/5 = 0.8, E test-transfer 3/5 = 0.6.
(B) Traceability: every table row of the REPORT.md H3 section (### 5.4) must
    appear verbatim in the section, and every number in it must equal the
    value at its source (summary_h3.json, train_meta.json, manifest.json,
    run config.yaml). A final sweep requires every numeric token in the
    section to be traced to one of those sources or to the structural set
    (section refs, run-id/date digits, suite/seeds counts, model id).

Run: python3 -m scripts.tests.test_h3_report
"""
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY = os.path.join(REPO, "results", "summary_h3.json")
REPORT = os.path.join(REPO, "REPORT.md")
TRAIN_META = os.path.join(REPO, "results", "h3_train_meta.json")  # tracked copy of the adapter's train_meta.json
MANIFEST = os.path.join(REPO, "finetune", "data", "manifest.json")
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)


def load(p):
    with open(os.path.join(REPO, p)) as f:
        return json.load(f)


# canonical run dirs (excluded superseded tries: runs_superseded/README.md)
CANON = {
    ("A", 0): "20261013-175510-A-local-suite-s0-h3base",
    ("B", 0): "20261013-175730-B-local-suite-s0-h3base",
    ("C", 0): "20261013-180033-C-local-suite-s0-h3base",
    ("D", 0): "20261013-180248-D-local-suite-s0-h3base",
    ("A", 1): "20261013-180507-A-local-suite-s1-h3base",
    ("B", 1): "20261013-180719-B-local-suite-s1-h3base",
    ("C", 1): "20261013-180900-C-local-suite-s1-h3base",
    ("D", 1): "20261013-181036-D-local-suite-s1-h3base",
    ("E", 0): "20261014-230503-E-local-suite-s0-h3",
    ("F", 0): "20261014-230534-F-local-suite-s0-h3",
}


def part_a():
    S = load("results/summary_h3.json")
    pooled = {}
    for (arm, seed), d in CANON.items():
        m = glob.glob(os.path.join(REPO, "runs", d))
        check(len(m) == 1, f"A: expected one run dir for {arm} s{seed}: {m}")
        if len(m) != 1:
            continue
        for line in open(os.path.join(m[0], "attempts.jsonl")):
            x = json.loads(line)
            solved, n = pooled.get((arm, x["split"]), (0, 0))
            pooled[(arm, x["split"])] = (solved + (1 if x.get("success") else 0), n + 1)
    for arm in "ABCDEF":
        for split in ("train", "test-indist", "test-transfer"):
            solved, n = pooled.get((arm, split), (0, 0))
            src = S["success_rate"][f"{arm}|{split}"]
            rate = solved / n if n else 0.0
            check(n == src["n"], f"A: {arm}|{split} n raw={n} summary={src['n']}")
            check(abs(rate - src["rate"]) < 1e-9,
                  f"A: {arm}|{split} rate raw={rate} summary={src['rate']}")
    check(S["n_attempts"] == sum(n for _, n in pooled.values()),
          f"A: n_attempts {S['n_attempts']} vs raw {sum(n for _, n in pooled.values())}")


def f2(x):
    return f"{x:+.2f}"


def f2u(x):
    return f"{x:.2f}"


def h3_section():
    txt = open(REPORT).read()
    m = re.search(r"^### 5\.4 .*?$(.*?)(?=^### |\Z)", txt, re.M | re.S)
    return m.group(1) if m else None


T = load("results/summary_h3.json")
TM = load(TRAIN_META)
MF = load(MANIFEST)
CFG_A = open(os.path.join(REPO, "runs/20261013-175510-A-local-suite-s0-h3base/config.yaml")).read()
CFG_E = open(os.path.join(REPO, "runs/20261014-230503-E-local-suite-s0-h3/config.yaml")).read()


ROWS = []


def row(label, s, src):
    """Expected table row `s`, built from source values; must appear verbatim
    in the section (a hand-edited number breaks the match)."""
    ROWS.append((label, s, src))
for arm in "ABCD":
    row(f"rate {arm}",
        f"| {arm} | {f2u(T['success_rate'][arm + '|train']['rate'])} (n={T['success_rate'][arm + '|train']['n']}) "
        f"| {f2u(T['success_rate'][arm + '|test-indist']['rate'])} (n={T['success_rate'][arm + '|test-indist']['n']}) "
        f"| {f2u(T['success_rate'][arm + '|test-transfer']['rate'])} (n={T['success_rate'][arm + '|test-transfer']['n']}) |",
        f"summary success_rate {arm}|*")
for arm in "EF":
    row(f"rate {arm}",
        f"| {arm} | n={T['success_rate'][arm + '|train']['n']} "
        f"| {f2u(T['success_rate'][arm + '|test-indist']['rate'])} (n={T['success_rate'][arm + '|test-indist']['n']}) "
        f"| {f2u(T['success_rate'][arm + '|test-transfer']['rate'])} (n={T['success_rate'][arm + '|test-transfer']['n']}) |",
        f"summary success_rate {arm}|* (train n=0: learned offline)")
for arm in "BCDEF":
    h = T["H1_advantage_over_A"][arm]
    row(f"H1 {arm}",
        f"| {arm} | {f2(h['advantage'])} | [{f2(h['ci_lo'])}, {f2(h['ci_hi'])}] | {h['n_pairs']} |",
        f"summary H1_advantage_over_A {arm}")
for arm in "BCDEF":
    r = T["H2_transfer_retention"][arm]
    row(f"H2 {arm}",
        f"| {arm} | {f2(r['adv_indist'])} | {f2(r['adv_transfer'])} | {r['retention']:.1f} |",
        f"summary H2_transfer_retention {arm}")
for arm in "BCDEF":
    for split in ("test-indist", "test-transfer"):
        b = T["H1b_turns_saved_vs_A"][f"{arm}|{split}"]
        row(f"H1b {arm}|{split}",
            f"| {arm}|{split} | {f2(b['turns_saved_vs_A'])} | [{f2(b['ci_lo'])}, {f2(b['ci_hi'])}] | {b['n_pairs']} |",
            f"summary H1b_turns_saved_vs_A {arm}|{split} (rounded to 2dp)")
for arm in "ABCDEF":
    for split in ("test-indist", "test-transfer"):
        e = T["efficiency"][f"{arm}|{split}"]
        row(f"eff {arm}|{split}",
            f"| {arm}|{split} | {e['mean_turns_to_solve']:.2f} | {e['mean_out_tokens']:.1f} | {e['n_solves']} |",
            f"summary efficiency {arm}|{split}")
row("n_attempts", str(T["n_attempts"]), "summary n_attempts")
row("gpu_hours", f"{TM['gpu_hours_measured']}", "train_meta gpu_hours_measured")
row("gpu_hours_est", f"{TM['plan']['est_gpu_hours_single_3090']}",
    "train_meta plan.est_gpu_hours_single_3090 (uncalibrated placeholder, disclosed, not used)")
row("sft records", str(MF["n_records"]), "manifest n_records")
row("sft dropped", str(len(MF["decontamination"]["dropped"])), "manifest decontamination.dropped")
row("lora", f"r={TM['plan']['lora']['r']}", "train_meta lora.r")
row("epochs", f"{TM['plan']['train']['epochs']} epochs, batch {TM['plan']['train']['grad_accum']}",
    "train_meta train.epochs/grad_accum")
row("lr", f"lr {TM['plan']['train']['lr']}", "train_meta train.lr")
row("seq", f"seq {TM['plan']['train']['seq_len']}", "train_meta train.seq_len")
row("cfg A", f"max_turns {int(re.search('max_turns: (\\d+)', CFG_A).group(1))} / num_predict {int(re.search('num_predict: (\\d+)', CFG_A).group(1))}",
    "A config.yaml")
row("cfg E", f"max_turns {int(re.search('max_turns: (\\d+)', CFG_E).group(1))} / num_predict {int(re.search('num_predict: (\\d+)', CFG_E).group(1))}",
    "E config.yaml")
for arm in "ABCD":
    c = T["sample_efficiency_curve"][arm]
    row(f"curve {arm}", f"{arm} {c[-1]:.1f}", f"summary sample_efficiency_curve {arm} final")

# structural: section refs, run ids/dates, suite seeds splits, model id, CI level
STRUCT = {"1", "2", "3", "4", "5", "10", "0", "6", "7", "8", "9", "2.5", "5.1", "5.2",
          "5.3", "5.4", "2026", "12", "13", "175510", "181036", "20261013", "20261014",
          "230503", "230534", "95"}


def part_b():
    sec = h3_section()
    check(sec is not None, "B: REPORT.md has no '### 5.4' H3 section")
    if sec is None:
        return
    for label, s, src in ROWS:
        check(s in sec, f"B: {label}: source row {s!r} not verbatim in section ({src})")
    tokens = set(re.findall(r"\d+(?:\.\d+)?", sec))
    allowed = set(STRUCT)
    for _, s, _ in ROWS:
        allowed |= set(re.findall(r"\d+(?:\.\d+)?", s))
    untraced = tokens - allowed
    check(not untraced, f"B: numeric tokens without a traced source: {sorted(untraced)}")


if __name__ == "__main__":
    part_a()
    part_b()
    print(f"{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failure(s)")
    raise SystemExit(1 if FAILS else 0)
