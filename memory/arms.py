"""Learning modules for arms B, C, D (arm A = NoMemory lives in the scaffold).

All three implement the LearningModule protocol (preamble / size_tokens /
consolidate) and add save()/load()/freeze() so a memory can be BUILT on a
training surface, FROZEN, and then evaluated on a held-out sibling surface --
the transfer protocol for H2. Consolidation always gates on the ground-truth
verdict (verdict["flag_verified"]); nothing is learned from a self-reported or
failed attempt.

Design choices (see DECISIONS.md D9):
  * Token budget is the shared currency. est_tokens = len(text)//4.
  * The Reflector (arm D) is HEURISTIC by default -- it distills the verified
    winning trajectory into one compact, generalized strategy bullet -- so that
    consolidation costs ZERO extra GPU calls. Under the current ~1.2 tok/s
    contention an LLM-reflector per task would dominate wall-clock; a heuristic
    reflector keeps the memory mechanism itself cheap and deterministic while
    still conditioning only on the verifiable verdict. An LLM reflector can be
    injected (reflector_llm=) for a later ablation when GPU is free.
  * Flags are REDACTED from every stored trajectory (ethics, brief 8) via
    memory.redact before anything is written.

Arms:
  B  RawICL       bounded buffer of verified solve transcripts, prepended verbatim
  C  Episodic     retrieval over verified episodes; top-k by family + token overlap
  D  Playbook     ACE-style: structured strategy bullets keyed by mechanism family,
                  delta updates (add / reinforce / demote), no full rewrites, capped
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict

from .redact import redact


def est_tokens(text: str) -> int:
    return max(0, len(text) // 4)


def _actions_from_traj(trajectory: list[dict]) -> list[tuple[str, str]]:
    """Extract the (action, arg) sequence actually taken, flags redacted."""
    out = []
    for step in trajectory:
        a, arg = step.get("action"), step.get("arg")
        if a and arg is not None:
            out.append((a, redact(str(arg))))
    return out


def _winning_actions(trajectory: list[dict]) -> list[tuple[str, str]]:
    """The non-redundant action path of a solve (drop no-op/failed submits)."""
    acts = []
    for step in trajectory:
        a, arg = step.get("action"), step.get("arg")
        if not a:
            continue
        if a == "submit" and step.get("flag_ok") is False:
            continue  # a wrong submit is not part of the solution
        acts.append((a, redact(str(arg)) if arg else ""))
    return acts


# --------------------------------------------------------------------- Arm B
class RawICL:
    """Prepend up to `k` verified solve transcripts (trimmed) as in-context demos."""
    name = "raw-icl"

    def __init__(self, k: int = 3, max_steps_per_demo: int = 8,
                 max_tokens: int = 2200):
        self.k = k
        self.max_steps_per_demo = max_steps_per_demo
        self.max_tokens = max_tokens
        self.demos: list[dict] = []  # {family, challenge_id, lines:[...]}
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def _render(self, demo: dict) -> str:
        head = f"# Worked example ({demo['family']}):"
        body = "\n".join(demo["lines"])
        return f"{head}\n{body}"

    def _selected(self, env):
        fam = getattr(env, "mechanism_family", None)
        parent = getattr(env, "parent_mechanism", None)
        # prefer same-family, then same-parent, then recent
        def score(d):
            if d["family"] == fam:
                return 2
            if parent and d["family"].rsplit("-", 1)[0] == parent:
                return 1
            return 0
        return sorted(self.demos, key=score, reverse=True)[: self.k]

    def preamble(self, env) -> str:
        sel = self._selected(env)
        if not sel:
            return ""
        blocks, used = [], 0
        for d in sel:
            txt = self._render(d)
            t = est_tokens(txt)
            if used + t > self.max_tokens:
                break
            blocks.append(txt)
            used += t
        if not blocks:
            return ""
        return ("Here are worked examples of solving related challenges. Adapt "
                "their approach; do not copy flags.\n\n" + "\n\n".join(blocks))

    def size_tokens(self) -> int:
        return sum(est_tokens(self._render(d)) for d in self.demos)

    def consolidate(self, env, trajectory, verdict) -> None:
        if self.frozen or not verdict.get("flag_verified"):
            return
        acts = _winning_actions(trajectory)[: self.max_steps_per_demo]
        if not acts:
            return
        lines = [f"- {a}: {arg}" for a, arg in acts]
        self.demos.append({
            "family": getattr(env, "mechanism_family", "unknown"),
            "challenge_id": env.challenge_id,
            "lines": lines,
        })

    # persistence
    def state(self) -> dict:
        return {"name": self.name, "k": self.k, "demos": self.demos}

    def save(self, path): _save(path, self.state())

    @classmethod
    def load(cls, path, **kw):
        s = _load(path); m = cls(k=s.get("k", 3), **kw); m.demos = s["demos"]; return m


# --------------------------------------------------------------------- Arm C
class Episodic:
    """Retrieval memory: store verified episodes, retrieve top-k for the task."""
    name = "episodic"

    def __init__(self, k: int = 3, max_tokens: int = 2000):
        self.k = k
        self.max_tokens = max_tokens
        self.episodes: list[dict] = []  # {family, challenge_id, desc, summary}
        self.frozen = False

    def freeze(self):
        self.frozen = True

    @staticmethod
    def _tokens(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    def _retrieve(self, env):
        fam = getattr(env, "mechanism_family", None)
        parent = getattr(env, "parent_mechanism", None)
        q = self._tokens(env.description())
        scored = []
        for e in self.episodes:
            overlap = len(q & self._tokens(e["desc"] + " " + e["summary"]))
            fam_bonus = 3 if e["family"] == fam else (
                1 if parent and e["family"].rsplit("-", 1)[0] == parent else 0)
            scored.append((fam_bonus * 2 + overlap, e))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [e for s, e in scored[: self.k] if s > 0]

    def preamble(self, env) -> str:
        hits = self._retrieve(env)
        if not hits:
            return ""
        used, blocks = 0, []
        for e in hits:
            txt = f"# Recalled approach ({e['family']}):\n{e['summary']}"
            t = est_tokens(txt)
            if used + t > self.max_tokens:
                break
            blocks.append(txt); used += t
        if not blocks:
            return ""
        return ("Relevant past solutions retrieved from episodic memory:\n\n"
                + "\n\n".join(blocks))

    def size_tokens(self) -> int:
        return sum(est_tokens(e["summary"]) for e in self.episodes)

    def _summarize(self, env, trajectory) -> str:
        acts = _winning_actions(trajectory)
        steps = "; ".join(f"{a}({arg})" if arg else a for a, arg in acts if a)
        return f"Solved by: {steps}" if steps else "Solved (no recorded steps)."

    def consolidate(self, env, trajectory, verdict) -> None:
        if self.frozen or not verdict.get("flag_verified"):
            return
        self.episodes.append({
            "family": getattr(env, "mechanism_family", "unknown"),
            "challenge_id": env.challenge_id,
            "desc": redact(env.description()),
            "summary": redact(self._summarize(env, trajectory)),
        })

    def state(self) -> dict:
        return {"name": self.name, "k": self.k, "episodes": self.episodes}

    def save(self, path): _save(path, self.state())

    @classmethod
    def load(cls, path, **kw):
        s = _load(path); m = cls(k=s.get("k", 3), **kw); m.episodes = s["episodes"]; return m


# --------------------------------------------------------------------- Arm D
@dataclass
class Bullet:
    text: str
    family: str
    uses: int = 0          # times retrieved into a context that then solved
    wins: int = 0          # verified solves after this bullet was in context
    created_from: str = ""  # challenge id of origin

    @property
    def score(self) -> float:
        # Laplace-smoothed win rate, tie-broken by exposure
        return (self.wins + 1) / (self.uses + 2)


class Playbook:
    """ACE-style structured playbook (Generator/Reflector/Curator).

    * Generator = the frozen base policy (the ReActAgent), guided by preamble().
    * Reflector = distills a verified trajectory into a single generalized
      strategy bullet (heuristic by default; conditions ONLY on the verdict).
    * Curator  = applies DELTA updates: add new bullet, or reinforce/dedup an
      existing one; demotes bullets that were present on failures; never
      rewrites the whole book; caps per-family bullets.
    """
    name = "playbook-ace"

    def __init__(self, max_bullets_per_family: int = 6, max_tokens: int = 1800,
                 reflector_llm=None, hierarchical: bool = True):
        self.bullets: list[Bullet] = []
        self.max_per_family = max_bullets_per_family
        self.max_tokens = max_tokens
        self.reflector_llm = reflector_llm   # optional; None => heuristic
        self.hierarchical = hierarchical     # True: keep exemplars; False: compress
        self.frozen = False
        self._last_preamble_ids: list[int] = []  # bullets shown last attempt

    def freeze(self):
        self.frozen = True

    # -- Reflector ----------------------------------------------------------
    def _reflect_heuristic(self, env, trajectory) -> str:
        acts = _winning_actions(trajectory)
        tools = [a for a, _ in acts]
        fam = getattr(env, "mechanism_family", "unknown")
        # generalize: name the tool chain, not the literal args
        chain = " -> ".join(dict.fromkeys(tools))  # unique, order-preserving
        verb = {
            "crypto/encoding": "identify the encoding and decode it",
            "crypto/classical": "treat as a classical substitution; brute force the key/shift",
            "crypto/xor": "use known-plaintext ('picoCTF{') to recover the XOR key",
            "forensics/recover": "search the artifact for printable flag text",
            "reversing/constant": "read the source and invert the per-byte transform",
        }.get(fam.rsplit("-", 1)[0], "inspect inputs, derive, then submit")
        return f"For {fam.rsplit('-',1)[0]}: {verb}. Typical tool path: {chain or 'shell -> submit'}."

    def _reflect(self, env, trajectory) -> str:
        if self.reflector_llm is not None:
            try:
                return self._reflect_llm(env, trajectory)
            except Exception:
                pass
        return self._reflect_heuristic(env, trajectory)

    def _reflect_llm(self, env, trajectory) -> str:
        acts = _winning_actions(trajectory)
        steps = "\n".join(f"{a}: {arg}" for a, arg in acts)
        prompt = [
            {"role": "system", "content": "You distill a solved challenge into ONE reusable, "
             "generalized strategy sentence. No flags, no challenge-specific literals."},
            {"role": "user", "content": f"Mechanism family: {env.mechanism_family}\n"
             f"Verified winning steps:\n{steps}\n\nOne-sentence general strategy:"},
        ]
        r = self.reflector_llm.chat(prompt)
        return redact(r.text.strip().split("\n")[0])[:240]

    # -- Curator ------------------------------------------------------------
    def _family_bullets(self, family: str) -> list[Bullet]:
        parent = family.rsplit("-", 1)[0]
        return [b for b in self.bullets if b.family.rsplit("-", 1)[0] == parent]

    def _curate_add(self, text: str, family: str, origin: str):
        text = redact(text)
        # dedup: if a near-identical bullet exists in the parent, reinforce it
        for b in self._family_bullets(family):
            if _similar(b.text, text):
                b.wins += 1
                return
        self.bullets.append(Bullet(text=text, family=family, created_from=origin))
        # cap per parent mechanism: keep highest-scoring (hierarchical) or newest
        fb = self._family_bullets(family)
        if len(fb) > self.max_per_family:
            if self.hierarchical:
                worst = min(fb, key=lambda b: b.score)
            else:  # compress-everything: drop oldest to force consolidation
                worst = fb[0]
            self.bullets.remove(worst)

    # -- preamble (Generator guidance) --------------------------------------
    def preamble(self, env) -> str:
        fam = getattr(env, "mechanism_family", None)
        fb = sorted(self._family_bullets(fam or ""), key=lambda b: b.score, reverse=True)
        self._last_preamble_ids = [id(b) for b in fb]
        if not fb:
            return ""
        used, lines = 0, []
        for b in fb:
            line = f"- {b.text}"
            t = est_tokens(line)
            if used + t > self.max_tokens:
                break
            lines.append(line); used += t
            b.uses += 1
        if not lines:
            return ""
        return ("PLAYBOOK (verified strategies for this mechanism; apply them):\n"
                + "\n".join(lines))

    def size_tokens(self) -> int:
        return sum(est_tokens(b.text) for b in self.bullets)

    def consolidate(self, env, trajectory, verdict) -> None:
        if self.frozen:
            return
        shown = [b for b in self.bullets if id(b) in self._last_preamble_ids]
        if verdict.get("flag_verified"):
            for b in shown:
                b.wins += 1
            bullet = self._reflect(env, trajectory)
            self._curate_add(bullet, getattr(env, "mechanism_family", "unknown"),
                             env.challenge_id)
        else:
            # demote bullets that were present on a failure (soft signal)
            for b in shown:
                b.uses += 1  # exposure without a win lowers score
        self._last_preamble_ids = []

    # persistence
    def state(self) -> dict:
        return {"name": self.name, "hierarchical": self.hierarchical,
                "bullets": [asdict(b) for b in self.bullets]}

    def save(self, path): _save(path, self.state())

    @classmethod
    def load(cls, path, **kw):
        s = _load(path)
        m = cls(hierarchical=s.get("hierarchical", True), **kw)
        m.bullets = [Bullet(**b) for b in s["bullets"]]
        return m


# --------------------------------------------------------------------- helpers
def _similar(a: str, b: str) -> bool:
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return False
    j = len(ta & tb) / len(ta | tb)
    return j >= 0.6


def _save(path: str, state: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


ARMS = {"B": RawICL, "C": Episodic, "D": Playbook}
