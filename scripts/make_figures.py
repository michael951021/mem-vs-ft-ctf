"""Generate report figures from logged runs (no numbers invented here).

Reads runs/*/attempts.jsonl (via scripts.analyze) and results/summary.json and
writes PNGs to results/figures/:
  fig1_success_by_arm_split.png   grouped bars, success rate by arm x split
  fig2_h1_advantage.png           advantage over A on test-indist, bootstrap CI bars
  fig3_sample_efficiency.png      cumulative train solves vs attempt index, per arm
  fig4_transfer_retention.png     retention = adv_transfer / adv_indist, per arm

Every figure degrades gracefully on partial data: it plots whatever arms/splits
are present and annotates n. Palette is colorblind-safe and fixed per arm.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analyze import (load_rows, success_rate, paired_advantage,
                             paired_turns_advantage,
                             transfer_retention, sample_efficiency, SPLITS,
                             ARMS_ORDER)

OUT = "results/figures"
# fixed, colorblind-safe per-arm palette (Okabe-Ito subset)
ARM_COLOR = {"A": "#999999", "B": "#0072B2", "C": "#009E73",
             "D": "#D55E00", "E": "#CC79A7", "F": "#E69F00"}
SPLIT_LABEL = {"train": "train", "test-indist": "in-dist (held-out)",
               "test-transfer": "transfer (sibling)"}


def _arms(rows):
    return [a for a in ARMS_ORDER if any(r["arm"] == a for r in rows)]


def fig_success(rows):
    arms = _arms(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.8 / len(SPLITS)
    for si, s in enumerate(SPLITS):
        xs = [i + si * w for i in range(len(arms))]
        ys, ns = [], []
        for a in arms:
            r, n = success_rate(rows, a, s)
            ys.append(r); ns.append(n)
        bars = ax.bar(xs, ys, width=w, label=SPLIT_LABEL[s],
                      color=plt.cm.Greys(0.3 + 0.25 * si), edgecolor="white")
        for x, y, n in zip(xs, ys, ns):
            if n:
                ax.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks([i + w for i in range(len(arms))])
    ax.set_xticklabels([f"Arm {a}" for a in arms])
    ax.set_ylabel("success rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Success rate by arm and split (local CTF suite)")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_success_by_arm_split.png", dpi=150)
    plt.close(fig)


def fig_h1(rows):
    arms = [a for a in _arms(rows) if a != "A"]
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    pts, los, his = [], [], []
    for a in arms:
        adv = paired_advantage(rows, a, "test-indist")
        pts.append(adv["advantage"])
        los.append(adv["advantage"] - adv["ci_lo"])
        his.append(adv["ci_hi"] - adv["advantage"])
    ax.bar(range(len(arms)), pts, color=[ARM_COLOR.get(a, "#555") for a in arms],
           yerr=[los, his], capsize=5, edgecolor="white")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([f"Arm {a}" for a in arms])
    ax.set_ylabel("Δ success vs Arm A (paired)")
    ax.set_title("H1: in-distribution advantage over no-memory baseline\n"
                 "(clustered bootstrap 95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_h1_advantage.png", dpi=150)
    plt.close(fig)


def fig_efficiency(rows):
    arms = _arms(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for a in arms:
        curve = sample_efficiency(rows, a)
        if curve:
            ax.plot(range(1, len(curve) + 1), curve, marker="o", ms=4,
                    color=ARM_COLOR.get(a, "#555"), label=f"Arm {a}")
            plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_xlabel("training attempt #"); ax.set_ylabel("cumulative solves")
    ax.set_title("Sample efficiency on the training stream")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_sample_efficiency.png", dpi=150)
    plt.close(fig)


def fig_retention(rows):
    arms = [a for a in _arms(rows) if a != "A"]
    vals = []
    for a in arms:
        tr = transfer_retention(rows, a)
        vals.append((a, tr["retention"]))
    vals = [(a, v) for a, v in vals if v is not None]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([a for a, _ in vals], [v for _, v in vals],
           color=[ARM_COLOR.get(a, "#555") for a, _ in vals], edgecolor="white")
    ax.axhline(1.0, color="#333", lw=0.8, ls="--", label="full retention")
    ax.set_ylabel("transfer retention"); ax.set_title("H2: transfer retention to sibling surfaces")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_transfer_retention.png", dpi=150)
    plt.close(fig)


def fig_turns_saved(rows):
    """The informative figure in the ceiling regime: turns saved vs A on
    jointly-solved tasks, per arm, for in-dist and transfer, with bootstrap CIs."""
    arms = [a for a in _arms(rows) if a != "A"]
    splits = ["test-indist", "test-transfer"]
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.8 / len(splits)
    any_pair = False
    for si, s in enumerate(splits):
        xs, ys, los, his = [], [], [], []
        for i, a in enumerate(arms):
            ta = paired_turns_advantage(rows, a, s)
            if not ta["n_pairs"]:
                continue
            any_pair = True
            xs.append(i + si * w); ys.append(ta["turns_saved_vs_A"])
            los.append(ta["turns_saved_vs_A"] - ta["ci_lo"])
            his.append(ta["ci_hi"] - ta["turns_saved_vs_A"])
        if xs:
            ax.bar(xs, ys, width=w, yerr=[los, his], capsize=4,
                   color=plt.cm.Greys(0.35 + 0.3 * si), edgecolor="white",
                   label={"test-indist": "in-dist", "test-transfer": "transfer"}[s])
    if not any_pair:
        plt.close(fig); return
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks([i + w / 2 for i in range(len(arms))])
    ax.set_xticklabels([f"Arm {a}" for a in arms])
    ax.set_ylabel("turns saved vs Arm A (paired)")
    ax.set_title("Step-efficiency: turns saved vs no-memory baseline\n"
                 "(jointly-solved tasks; clustered bootstrap 95% CI)")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_turns_saved.png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load_rows("runs", "*")
    if not rows:
        print("no data yet; figures skipped"); return
    fig_success(rows); fig_h1(rows); fig_efficiency(rows); fig_retention(rows)
    fig_turns_saved(rows)
    print(f"wrote figures to {OUT}/ from {len(rows)} attempts")


if __name__ == "__main__":
    main()
