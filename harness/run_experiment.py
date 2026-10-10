"""Experiment driver: one arm x one seed over a challenge schedule.

Implements the brief's protocol end to end:
  TRAIN   (memory learns online on surface X)  -> FREEZE -> SAVE
  TEST in-distribution (held-out same-family)   [frozen]
  TEST transfer        (sibling surface Y)       [frozen]

Arm A (NoMemory) runs the identical schedule with a no-op memory, giving the
baseline every other arm's advantage is measured against. Every attempt is
logged per docs/run_schema.md; full transcripts go to runs/<id>/raw/.

Usage:
  python -m harness.run_experiment --arm D --seed 0 --benchmark local-suite \
      --model qwen-agent:latest --out runs
  python -m harness.run_experiment --arm A --seed 0 --mock      # offline, no GPU
"""
from __future__ import annotations

import argparse
import random
import time

from harness.scaffold.react_agent import ReActAgent, NoMemory
from harness.scaffold.llm_client import OllamaClient, LLMConfig
from harness.scaffold.logger import RunLogger, Attempt
from memory.arms import RawICL, Episodic, Playbook

ARM_FACTORY = {
    "A": lambda **k: NoMemory(),
    "B": lambda **k: RawICL(),
    "C": lambda **k: Episodic(),
    "D": lambda **k: Playbook(**({"reflector_llm": k["reflector_llm"]}
                                 if k.get("reflector_llm") else {})),
}


# --------------------------------------------------------------- mock LLM (offline)
class _MockResp:
    def __init__(self, text):
        self.text = text; self.thinking = ""
        self.input_tokens = 40; self.output_tokens = 15
        self.wall_clock_s = 0.0; self.load_s = 0.0; self.truncated = False


class MockSolver:
    """An offline oracle 'model': solves each local challenge with a short,
    deterministic action path. Used only to validate the driver without GPU.
    It reads the ground-truth flag from the env (NOT available to a real model)
    so the pipeline can be exercised end to end offline."""
    def __init__(self, env):
        self.env = env; self.i = 0
        self.script = self._script_for(env)

    def _script_for(self, env):
        flag = env.challenge.flag
        return [
            "Let me look.\nACTION: shell\nINPUT: ls",
            f"I will submit.\nACTION: submit\nINPUT: {flag}",
        ]

    def chat(self, messages, stop=None):
        t = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return _MockResp(t)


# --------------------------------------------------------------- env factory
def build_local_envs(challenge_ids, workroot):
    from env.local_suite.challenges import by_id
    from env.local_suite.env import LocalEnv
    d = by_id()
    return [LocalEnv(d[cid], workroot=workroot) for cid in challenge_ids]


def default_schedule():
    """Return (train_ids, test_indist_ids, transfer_ids) for the local suite.
    Train on the first surface(s) of each parent; hold out one same-family item
    for in-distribution test where available, and the sibling surface for
    transfer."""
    from env.local_suite.challenges import TRANSFER_SPLITS
    train, indist, transfer = [], [], []
    for parent, sp in TRANSFER_SPLITS.items():
        tr = sp["train"]
        # in-distribution test = last train item held out if >1, else reuse
        if len(tr) > 1:
            train += tr[:-1]; indist += tr[-1:]
        else:
            train += tr; indist += tr  # tiny suite: eval on the trained surface
        transfer += sp["transfer"]
    return train, indist, transfer


# --------------------------------------------------------------- one attempt
def run_attempt(agent, env, memory, logger, meta, frozen):
    if frozen and hasattr(memory, "freeze"):
        memory.freeze()
    mem_before = memory.size_tokens()
    res = agent.run(env, memory)
    attempt = Attempt(
        run_id=meta["run_id"], arm=meta["arm"], model=meta["model"],
        seed=meta["seed"], benchmark=meta["benchmark"],
        challenge_id=env.challenge_id,
        mechanism_family=getattr(env, "mechanism_family", None),
        split=meta["split"], turns_used=res.turns_used, success=res.success,
        flag_verified=res.flag_verified, input_tokens=res.input_tokens,
        output_tokens=res.output_tokens, wall_clock_s=round(res.wall_clock_s, 2),
        memory_size_tokens=mem_before,
        notes=meta.get("notes", ""),
    )
    logger.log(attempt)
    raw_meta = {**meta, "challenge_id": env.challenge_id,
                "mechanism_family": getattr(env, "mechanism_family", None)}
    logger.save_raw(env.challenge_id, meta["seed"],
                    {"meta": raw_meta, "trajectory": res.trajectory,
                     "submitted": res.submitted_flags, "success": res.success},
                    split=meta["split"])
    if hasattr(env, "cleanup"):
        env.cleanup()
    return res


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_FACTORY))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", default="local-suite")
    ap.add_argument("--model", default="qwen-agent:latest")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--mock", action="store_true", help="offline oracle model, no GPU")
    ap.add_argument("--workroot", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--arm-label", default=None,
                    help="override the logged arm string (e.g. E/F for the QLoRA arms)")
    ap.add_argument("--load-memory", default=None,
                    help="path to a frozen memory json to evaluate (skips its training)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip the TRAIN phase; evaluate the (loaded) frozen memory")
    args = ap.parse_args()
    arm_label = args.arm_label or args.arm

    run_id = (f"{time.strftime('%Y%m%d-%H%M%S')}-{arm_label}-{args.benchmark}"
              f"-s{args.seed}{('-' + args.tag) if args.tag else ''}")
    random.seed(args.seed)

    train_ids, indist_ids, transfer_ids = default_schedule()
    rng = random.Random(args.seed)
    rng.shuffle(train_ids)  # seed controls training order (sample-efficiency curve)

    cfg = LLMConfig(model=args.model, seed=args.seed, num_ctx=args.num_ctx,
                    num_predict=args.num_predict)
    config_snapshot = {
        "arm": arm_label, "base_arm": args.arm, "eval_only": args.eval_only,
        "load_memory": args.load_memory,
        "seed": args.seed, "benchmark": args.benchmark,
        "model": args.model, "mock": args.mock, "max_turns": args.max_turns,
        "llm": {"num_ctx": args.num_ctx, "num_predict": args.num_predict,
                "temperature": cfg.temperature, "top_p": cfg.top_p},
        "schedule": {"train": train_ids, "test_indist": indist_ids,
                     "transfer": transfer_ids},
    }
    logger = RunLogger(args.out, run_id, config_snapshot)
    if args.load_memory:
        from memory.arms import RawICL, Episodic, Playbook
        loader = {"B": RawICL, "C": Episodic, "D": Playbook}[args.arm]
        memory = loader.load(args.load_memory)
        memory.freeze()
    else:
        memory = ARM_FACTORY[args.arm](reflector_llm=None)
    workroot = args.workroot

    def make_agent(env):
        llm = MockSolver(env) if args.mock else OllamaClient(cfg)
        return ReActAgent(llm, max_turns=args.max_turns)

    meta_base = {"run_id": run_id, "arm": arm_label, "model": args.model,
                 "seed": args.seed, "benchmark": args.benchmark}

    if args.eval_only:
        print(f"[{run_id}] EVAL-ONLY (frozen memory={args.load_memory})")
    else:
        print(f"[{run_id}] TRAIN on {len(train_ids)}: {train_ids}")
        for env in build_local_envs(train_ids, workroot):
            meta = {**meta_base, "split": "train"}
            res = run_attempt(make_agent(env), env, memory, logger, meta, frozen=False)
            print(f"  train {env.challenge_id:12s} solved={res.success} "
                  f"turns={res.turns_used} mem_tok={memory.size_tokens()}")

    # freeze + persist the learned memory (the frozen artifact for transfer)
    if not args.eval_only and hasattr(memory, "save"):
        mem_path = f"{logger.run_dir}/memory_frozen.json"
        if hasattr(memory, "freeze"):
            memory.freeze()
        try:
            memory.save(mem_path); print(f"  froze memory -> {mem_path}")
        except Exception as e:  # NoMemory has no save
            print(f"  (no memory to save: {e})")

    print(f"[{run_id}] TEST in-distribution on {len(indist_ids)}: {indist_ids}")
    for env in build_local_envs(indist_ids, workroot):
        meta = {**meta_base, "split": "test-indist"}
        res = run_attempt(make_agent(env), env, memory, logger, meta, frozen=True)
        print(f"  indist {env.challenge_id:12s} solved={res.success} turns={res.turns_used}")

    print(f"[{run_id}] TEST transfer on {len(transfer_ids)}: {transfer_ids}")
    for env in build_local_envs(transfer_ids, workroot):
        meta = {**meta_base, "split": "test-transfer"}
        res = run_attempt(make_agent(env), env, memory, logger, meta, frozen=True)
        print(f"  transfer {env.challenge_id:12s} solved={res.success} turns={res.turns_used}")

    print(f"[{run_id}] DONE -> {logger.run_dir}")
    return logger.run_dir


if __name__ == "__main__":
    main()
