"""Offline smoke test: ReAct loop + logger with a mock LLM and mock env.
Run: python3 -m harness.scaffold.tests.test_smoke_loop  (no GPU/Docker needed)."""
from harness.scaffold.react_agent import ReActAgent, NoMemory
from harness.scaffold.logger import RunLogger, Attempt
import base64, tempfile, os


class MockEnv:
    challenge_id = "mock-crypto-01"
    _flag = "picoCTF{m0ck_fl4g}"
    def description(self): return "secret.txt holds the flag, base64-encoded."
    def exec(self, cmd):
        if "base64" in cmd and "-d" in cmd: return self._flag
        if cmd.strip().startswith("ls"): return "secret.txt"
        return "(no output)"
    def read_file(self, path):
        if "secret" in path: return base64.b64encode(self._flag.encode()).decode()
        raise FileNotFoundError(path)
    def check_flag(self, flag): return flag.strip() == self._flag


class MockResp:
    def __init__(self, t): self.text=t; self.thinking=""; self.input_tokens=50
    output_tokens=20; wall_clock_s=0.01; load_s=0.0; truncated=False


class MockLLM:
    script=["ACTION: shell\nINPUT: ls",
            "ACTION: read\nINPUT: secret.txt",
            "ACTION: shell\nINPUT: echo cGljb0NURnttMGNrX2ZsNGd9 | base64 -d",
            "ACTION: submit\nINPUT: picoCTF{m0ck_fl4g}"]
    def __init__(self): self.i=0
    def chat(self, messages, stop=None):
        r=MockResp(self.script[min(self.i,len(self.script)-1)]); self.i+=1; return r


def test_loop_solves_and_logs():
    res = ReActAgent(MockLLM(), max_turns=10).run(MockEnv(), NoMemory())
    assert res.success and res.flag_verified and res.turns_used == 4
    with tempfile.TemporaryDirectory() as d:
        lg = RunLogger(d, "smoke", {"arm": "A", "model": "mock"})
        lg.log(Attempt(run_id="smoke", arm="A", model="mock", seed=0, benchmark="mock",
                       challenge_id="mock-crypto-01", split="test-indist",
                       turns_used=res.turns_used, success=res.success,
                       flag_verified=res.flag_verified))
        assert os.path.getsize(lg.attempts_path) > 0
    print("PASS: loop solved in", res.turns_used, "turns; logging OK")


if __name__ == "__main__":
    test_loop_solves_and_logs()
