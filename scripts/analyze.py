"""Aggregate run logs into H1/H2 tables with paired, clustered bootstrap CIs.

Reads every runs/*/attempts.jsonl, builds a tidy table, and computes:
  * per-arm success rate by split (train / test-indist / test-transfer)
  * H1: paired advantage over Arm A on test-indist, bootstrap CI over challenges
  * H2: transfer retention = adv_over_A(transfer) / adv_over_A(indist), per arm,
        with a bootstrap CI (resampling challenges within each split)
  * efficiency: mean turns + output tokens per solve
  * sample-efficiency curve: cumulative train solves vs attempt index (per arm)

Handles PARTIAL data: it uses whatever arms/seeds/splits have landed, and every
printed number is computed from logged rows only (never imputed). Writes CSVs to
results/aggregated/ and a machine-readable results/summary.json for the report.

Paired + clustered: comparisons are paired per (challenge, seed) against Arm A,
and the bootstrap resamples challenges (the cluster unit), per the brief's stats.

Usage: python -m scripts.analyze [--runs runs] [--glob '*pilot*']
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from collections import defaultdict

ARMS_ORDER = ["A", "B", "C", "D", "E", "F"]
SPLITS = ["train", "test-indist", "test-transfer"]


def load_rows(runs_dir: str, pat: str) -> list[dict]:
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, pat))):
        ap = os.path.join(run_dir, "attempts.jsonl")
        if not os.path.isfile(ap):
            continue
        with open(ap) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def success_rate(rows, arm, split):
    sel = [r for r in rows if r["arm"] == arm and r["split"] == split]
    return _mean(r["success"] for r in sel), len(sel)


def _paired_table(rows, arm, split):
    """{(challenge, seed): success} for arm and for A, over the same keys."""
    def tab(a):
        return {(r["challenge_id"], r["seed"]): bool(r["success"])
                for r in rows if r["arm"] == a and r["split"] == split}
    ta, tx = tab("A"), tab(arm)
    keys = sorted(set(ta) & set(tx))
    return [(k, tx[k], ta[k]) for k in keys]


def bootstrap_ci(values, stat, n=2000, alpha=0.05, seed=0):
    """Bootstrap CI for `stat` over clusters. `values` is a list; resampled with
    replacement. Returns (point, lo, hi)."""
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    point = stat(values)
    boots = []
    N = len(values)
    for _ in range(n):
        sample = [values[rng.randrange(N)] for _ in range(N)]
        boots.append(stat(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n)]
    hi = boots[int((1 - alpha / 2) * n) - 1]
    return (point, lo, hi)


def paired_advantage(rows, arm, split):
    """Mean of (arm_success - A_success) paired per challenge, clustered CI."""
    pairs = _paired_table(rows, arm, split)
    # cluster unit = challenge: average the per-(challenge,seed) diffs by challenge
    by_ch = defaultdict(list)
    for (ch, seed), sx, sa in pairs:
        by_ch[ch].append(int(sx) - int(sa))
    clusters = [(_mean(v)) for v in by_ch.values()]  # one value per challenge
    point, lo, hi = bootstrap_ci(clusters, _mean, seed=hash((arm, split)) & 0xffff)
    return {"n_pairs": len(pairs), "n_clusters": len(clusters),
            "advantage": round(point, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}


def paired_turns_advantage(rows, arm, split):
    """Mean reduction in turns-to-solve vs Arm A on challenges BOTH solved
    (paired per challenge, clustered bootstrap). Positive = arm is more
    step-efficient than A. This is the informative metric when success is at
    ceiling: memory that hands the agent a verified recipe should cut steps."""
    def solved_turns(a):
        return {(r["challenge_id"], r["seed"]): r["turns_used"]
                for r in rows if r["arm"] == a and r["split"] == split and r["success"]}
    ta, tx = solved_turns("A"), solved_turns(arm)
    keys = sorted(set(ta) & set(tx))  # only challenges both solved
    by_ch = defaultdict(list)
    for k in keys:
        ch = k[0]
        by_ch[ch].append(ta[k] - tx[k])  # A's turns minus arm's turns
    clusters = [_mean(v) for v in by_ch.values()]
    point, lo, hi = bootstrap_ci(clusters, _mean, seed=hash((arm, split, "turns")) & 0xffff)
    return {"n_pairs": len(keys), "n_clusters": len(clusters),
            "turns_saved_vs_A": round(point, 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)}


def transfer_retention(rows, arm):
    adv_in = paired_advantage(rows, arm, "test-indist")["advantage"]
    adv_tr = paired_advantage(rows, arm, "test-transfer")["advantage"]
    ret = (adv_tr / adv_in) if abs(adv_in) > 1e-9 else float("nan")
    return {"adv_indist": adv_in, "adv_transfer": adv_tr,
            "retention": round(ret, 3) if ret == ret else None}


def sample_efficiency(rows, arm):
    """Cumulative solves vs training-attempt index, per seed, averaged."""
    by_seed = defaultdict(list)
    for r in rows:
        if r["arm"] == arm and r["split"] == "train":
            by_seed[r["seed"]].append(r)
    curves = []
    for seed, rs in by_seed.items():
        rs = rs  # train order is the logged order
        cum, c = [], 0
        for i, r in enumerate(rs, 1):
            c += int(r["success"])
            cum.append(c)
        curves.append(cum)
    if not curves:
        return []
    L = min(len(c) for c in curves)
    return [round(_mean(c[i] for c in curves), 3) for i in range(L)]


def efficiency(rows, arm, split):
    sel = [r for r in rows if r["arm"] == arm and r["split"] == split and r["success"]]
    return {"mean_turns_to_solve": round(_mean(r["turns_used"] for r in sel), 2),
            "mean_out_tokens": round(_mean(r["output_tokens"] for r in sel), 1),
            "n_solves": len(sel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--glob", default="*")
    ap.add_argument("--out", default="results/aggregated")
    ap.add_argument("--summary-out", default="results/summary.json")
    args = ap.parse_args()

    rows = load_rows(args.runs, args.glob)
    if not rows:
        print("no rows found under", args.runs, "matching", args.glob)
        return
    arms = [a for a in ARMS_ORDER if any(r["arm"] == a for r in rows)]
    seeds = sorted({r["seed"] for r in rows})
    os.makedirs(args.out, exist_ok=True)

    print(f"Loaded {len(rows)} attempts | arms={arms} seeds={seeds}")
    print(f"{'arm':>4} | " + " | ".join(f"{s:>13}" for s in SPLITS))
    srate = {}
    for a in arms:
        cells = []
        for s in SPLITS:
            r, n = success_rate(rows, a, s)
            srate[(a, s)] = (r, n)
            cells.append(f"{r:5.2f} (n={n:>2})")
        print(f"{a:>4} | " + " | ".join(f"{c:>13}" for c in cells))

    summary = {"n_attempts": len(rows), "arms": arms, "seeds": seeds,
               "success_rate": {f"{a}|{s}": {"rate": srate[(a, s)][0],
                                             "n": srate[(a, s)][1]}
                                for a in arms for s in SPLITS},
               "H1_advantage_over_A": {}, "H2_transfer_retention": {},
               "H1b_turns_saved_vs_A": {},
               "efficiency": {}, "sample_efficiency_curve": {}}

    print("\nH1 — paired advantage over Arm A (test-indist), clustered bootstrap 95% CI:")
    for a in arms:
        if a == "A":
            continue
        adv = paired_advantage(rows, a, "test-indist")
        summary["H1_advantage_over_A"][a] = adv
        print(f"  D>C>B? {a}: Δ={adv['advantage']:+.3f}  "
              f"CI[{adv['ci_lo']:+.3f},{adv['ci_hi']:+.3f}]  "
              f"(pairs={adv['n_pairs']}, clusters={adv['n_clusters']})")

    print("\nH1b — turns saved vs Arm A on jointly-solved tasks (step-efficiency):")
    for a in arms:
        if a == "A":
            continue
        for s in ("test-indist", "test-transfer"):
            ta = paired_turns_advantage(rows, a, s)
            summary["H1b_turns_saved_vs_A"][f"{a}|{s}"] = ta
            if ta["n_pairs"]:
                print(f"  {a} {s}: {ta['turns_saved_vs_A']:+.2f} turns "
                      f"CI[{ta['ci_lo']:+.2f},{ta['ci_hi']:+.2f}] (pairs={ta['n_pairs']})")

    print("\nH2 — transfer retention (adv_transfer / adv_indist):")
    for a in arms:
        if a == "A":
            continue
        tr = transfer_retention(rows, a)
        summary["H2_transfer_retention"][a] = tr
        print(f"  {a}: adv_indist={tr['adv_indist']:+.3f} "
              f"adv_transfer={tr['adv_transfer']:+.3f} retention={tr['retention']}")

    print("\nEfficiency (on solves):")
    for a in arms:
        for s in ("test-indist", "test-transfer"):
            e = efficiency(rows, a, s)
            summary["efficiency"][f"{a}|{s}"] = e
        summary["sample_efficiency_curve"][a] = sample_efficiency(rows, a)

    # CSVs
    with open(os.path.join(args.out, "success_by_arm_split.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "split", "success_rate", "n"])
        for a in arms:
            for s in SPLITS:
                w.writerow([a, s, f"{srate[(a, s)][0]:.4f}", srate[(a, s)][1]])
    with open(os.path.join(args.out, "h1_advantage.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "advantage_over_A", "ci_lo", "ci_hi", "n_clusters"])
        for a, adv in summary["H1_advantage_over_A"].items():
            w.writerow([a, adv["advantage"], adv["ci_lo"], adv["ci_hi"], adv["n_clusters"]])

    os.makedirs("results", exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}/*.csv and results/summary.json")


if __name__ == "__main__":
    main()
