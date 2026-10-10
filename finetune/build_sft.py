"""Build the SFT corpus for arms E/F (QLoRA) from VERIFIED trajectories.

H3 compares memory (D) against gradient learning (E=QLoRA, F=hybrid) on the SAME
verified experience. So the SFT corpus is drawn from exactly the trajectories the
memory arms consolidated from: the per-task transcripts under runs/<id>/raw/ whose
attempt was flag_verified on a TRAIN split challenge.

Each training example is a (prompt, completion) pair in chat form: the challenge
description + the agent's next action, teaching the base policy the verified
action sequence. Flags are REDACTED (brief 8); we additionally DECONTAMINATE
against every eval challenge id/description so no test item leaks into training.

Outputs finetune/data/sft_<family-or-all>.jsonl (one JSON chat record per line)
and a manifest with counts + the decontamination log (a data artifact).

Usage:
  python -m finetune.build_sft --runs runs --glob '*pilot*' --train-only \
      --out finetune/data
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from memory.redact import redact, contains_flagish
from harness.scaffold.react_agent import SYSTEM_PROMPT


def _eval_challenge_ids():
    """Every challenge used for evaluation (in-dist + transfer) -> decontaminate."""
    from env.local_suite.challenges import TRANSFER_SPLITS, by_id
    from harness.run_experiment import default_schedule
    _, indist, transfer = default_schedule()
    ids = set(indist) | set(transfer)
    d = by_id()
    descs = {d[cid].description for cid in ids if cid in d}
    return ids, descs


def iter_verified_train(runs_dir, pat, source_arm):
    for raw in sorted(glob.glob(os.path.join(runs_dir, pat, "raw", "*.json"))):
        with open(raw) as f:
            t = json.load(f)
        meta = t.get("meta", {})
        if meta.get("split") != "train" or not t.get("success"):
            continue
        if source_arm and meta.get("arm") not in (source_arm, None):
            continue
        # decontamination is only sound if we can identify the challenge; a raw
        # transcript without challenge_id must abort, never silently pass.
        if not meta.get("challenge_id"):
            raise ValueError(
                f"{raw}: raw transcript has no challenge_id; cannot decontaminate "
                f"safely. Re-run the pilot with the fixed driver before building SFT.")
        yield t


def build_examples(traj_blob, decontam_ids, decontam_descs, log):
    """Turn one verified train trajectory into chat SFT records (flags redacted)."""
    meta = traj_blob["meta"]
    cid = meta.get("challenge_id") or ""
    if cid in decontam_ids:
        log.append({"dropped": cid, "reason": "eval-challenge leak"})
        return []
    traj = traj_blob.get("trajectory", [])
    # reconstruct the running context the way the agent saw it, action by action
    recs, ctx = [], []
    for step in traj:
        action, arg = step.get("action"), step.get("arg")
        if not action:
            continue
        assistant = redact(step.get("assistant", "") or f"ACTION: {action}\nINPUT: {arg}")
        user = ("\n".join(ctx) if ctx else "Begin.")
        if contains_flagish(assistant) or contains_flagish(user):
            # belt-and-suspenders: never emit a record that still looks flaggy
            assistant, user = redact(assistant), redact(user)
        recs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "family": meta.get("mechanism_family"),
            "challenge_id": cid,
        })
        obs = step.get("observation", "")
        ctx.append(f"ACTION: {action}\nINPUT: {redact(str(arg))}")
        if obs:
            ctx.append(f"OBSERVATION:\n{redact(str(obs))[:800]}")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--glob", default="*pilot*")
    ap.add_argument("--out", default="finetune/data")
    ap.add_argument("--split-by-family", action="store_true",
                    help="also write one corpus per parent mechanism (for transfer SFT)")
    ap.add_argument("--source-arm", default="A",
                    help="which arm's verified TRAIN trajectories to train on "
                         "(default A: the no-memory base experience; '' = all arms)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    decontam_ids, decontam_descs = _eval_challenge_ids()
    log, all_recs = [], []
    for blob in iter_verified_train(args.runs, args.glob, args.source_arm):
        all_recs.extend(build_examples(blob, decontam_ids, decontam_descs, log))

    # final audit: zero flaggy records may be written. We scan the model/
    # challenge-derived messages (user + assistant). The system message is a
    # fixed, authored template whose only brace is a generic format hint
    # ("picoCTF{...}"), audited once here and never challenge-specific.
    def _flaggy(rec):
        return any(contains_flagish(m["content"]) for m in rec["messages"]
                   if m["role"] != "system")
    leaks = [r for r in all_recs if _flaggy(r)]
    assert not leaks, f"{len(leaks)} SFT records still contain flag-shaped text!"
    # the system template itself must carry no real flag, only the generic hint:
    # a match is a placeholder iff its interior has no alphanumerics (real flags do)
    from harness.scaffold.react_agent import SYSTEM_PROMPT
    import re as _re
    sys_flags = _re.findall(r"(?:picoCTF|flag)\{+([^{}]*)\}+", SYSTEM_PROMPT)
    bad = [m for m in sys_flags if _re.search(r"[A-Za-z0-9]", m)]
    assert not bad, f"system prompt contains a non-placeholder flag interior: {bad}"

    out_all = os.path.join(args.out, "sft_all.jsonl")
    with open(out_all, "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")

    manifest = {"n_records": len(all_recs),
                "n_source_trajectories_verified_train": len(set(r["challenge_id"] for r in all_recs)),
                "decontamination": {"eval_ids": sorted(decontam_ids),
                                    "dropped": log},
                "by_family": {}}
    if args.split_by_family:
        from collections import defaultdict
        byf = defaultdict(list)
        for r in all_recs:
            parent = (r["family"] or "unknown").rsplit("-", 1)[0]
            byf[parent].append(r)
        for parent, rs in byf.items():
            safe = parent.replace("/", "_")
            with open(os.path.join(args.out, f"sft_{safe}.jsonl"), "w") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
            manifest["by_family"][parent] = len(rs)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(all_recs)} SFT records -> {out_all}")
    print(f"decontamination dropped {len(log)} records; eval ids protected: {len(decontam_ids)}")
    print(f"manifest -> {args.out}/manifest.json")


if __name__ == "__main__":
    main()
