"""Offline tests for arms B/C/D: learning gates on the verdict, grows with
verified solves, redacts flags, freezes, and round-trips through save/load.
Run: python3 -m memory.tests.test_arms  (no GPU/Docker)."""
import os
import tempfile

from memory.arms import RawICL, Episodic, Playbook
from memory.redact import redact, contains_flagish


class FakeEnv:
    def __init__(self, cid, family):
        self.challenge_id = cid
        self.mechanism_family = family
        self.parent_mechanism = family.rsplit("-", 1)[0]
    def description(self):
        return f"challenge {self.challenge_id} in {self.mechanism_family}: decode the file"


def _solve_traj(flag="picoCTF{secret}"):
    return [
        {"turn": 1, "action": "shell", "arg": "ls"},
        {"turn": 2, "action": "read", "arg": "message.txt"},
        {"turn": 3, "action": "shell", "arg": "base64 -d message.txt"},
        {"turn": 4, "action": "submit", "arg": flag, "flag_ok": True},
    ]


VERDICT_WIN = {"flag_verified": True, "success": True}
VERDICT_LOSS = {"flag_verified": False, "success": False}


def _check_arm_basic(mod_factory):
    env = FakeEnv("enc-base64", "crypto/encoding-base64")
    m = mod_factory()
    # 1. no learning from a failed attempt
    m.consolidate(env, _solve_traj(), VERDICT_LOSS)
    assert m.size_tokens() == 0, "learned from a failure!"
    # 2. learns from a verified solve
    m.consolidate(env, _solve_traj(), VERDICT_WIN)
    assert m.size_tokens() > 0, "did not learn from a verified solve"
    # 3. preamble fires on a same-family sibling and contains NO flag
    sib = FakeEnv("enc-base32", "crypto/encoding-base32")
    pre = m.preamble(sib)
    assert pre, "no transfer preamble to sibling surface"
    assert not contains_flagish(pre), "flag leaked into preamble!"
    assert "picoCTF{secret}" not in pre
    # 4. freeze stops learning
    m.freeze()
    before = m.size_tokens()
    m.consolidate(FakeEnv("x", "crypto/xor-single"), _solve_traj(), VERDICT_WIN)
    assert m.size_tokens() == before, "learned while frozen"
    # 5. save/load round-trip preserves size
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "mem.json")
        m.save(p)
        m2 = type(m).load(p)
        assert m2.size_tokens() == before
    return True


def test_raw_icl():
    assert _check_arm_basic(lambda: RawICL(k=3))


def test_episodic():
    assert _check_arm_basic(lambda: Episodic(k=3))


def test_playbook():
    assert _check_arm_basic(lambda: Playbook())


def test_playbook_dedup_and_cap():
    m = Playbook(max_bullets_per_family=3)
    env = FakeEnv("enc-base64", "crypto/encoding-base64")
    for i in range(10):
        m.consolidate(FakeEnv(f"enc-{i}", "crypto/encoding-base64"),
                      _solve_traj(), VERDICT_WIN)
    fam = m._family_bullets("crypto/encoding-base64")
    assert len(fam) <= 3, f"cap exceeded: {len(fam)}"
    # heuristic reflector yields identical bullets -> dedup should keep it tiny
    assert len(fam) == 1, f"dedup failed, got {len(fam)} distinct bullets"


def test_redaction_on_store():
    # a trajectory whose arg contains the flag must not leave it in memory
    m = RawICL()
    traj = [{"turn": 1, "action": "submit", "arg": "picoCTF{leak_me}", "flag_ok": True}]
    m.consolidate(FakeEnv("c", "crypto/encoding-base64"),
                  [{"turn": 1, "action": "shell", "arg": "echo picoCTF{leak_me}"}] + traj,
                  VERDICT_WIN)
    blob = str(m.state())
    assert "leak_me" not in blob, "flag leaked into stored demo"
    assert "[REDACTED_FLAG]" in blob


if __name__ == "__main__":
    test_raw_icl(); test_episodic(); test_playbook()
    test_playbook_dedup_and_cap(); test_redaction_on_store()
    print("PASS: arms B/C/D learn-on-verdict, transfer, redact, freeze, persist")
