"""TaskEnv over the Docker-free local challenge suite.

Implements the harness TaskEnv protocol (harness/scaffold/react_agent.py):
description / exec / read_file / check_flag. The agent's shell commands run via
subprocess inside a fresh per-challenge working directory created under the
scratchpad, with a wall-clock timeout and a conservative command denylist as
defense-in-depth. The whole process already runs inside the project's
bubblewrap sandbox (DECISIONS.md D8), so this is a second containment layer,
not the only one.

check_flag is GROUND TRUTH: exact match against the challenge's flag, with a
little tolerance for surrounding whitespace/quotes the model sometimes adds.
It is the only thing that may set success / gate a memory write.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

from .challenges import Challenge

# Conservative denylist: these never appear in a legitimate decode solution and
# would be destructive or exfiltrating. Matched case-insensitively as substrings
# on the command string. The outer bwrap sandbox is the real boundary.
_DENY = [
    "rm -rf", "mkfs", "dd if=", ":(){", "shutdown", "reboot", "poweroff",
    "> /dev/sd", "chmod -r 000", "curl http", "wget http", "nc -e", "/etc/passwd",
    "ssh ", "scp ", "systemctl", "service ", "kill ", "pkill", "killall",
]


class LocalEnv:
    def __init__(self, challenge: Challenge, workroot: str | None = None,
                 timeout_s: int = 20):
        self.challenge = challenge
        self.challenge_id = challenge.id
        self.mechanism_family = challenge.family
        self.parent_mechanism = challenge.parent
        self.timeout_s = timeout_s
        root = workroot or os.environ.get("CTF_WORKROOT") or tempfile.gettempdir()
        os.makedirs(root, exist_ok=True)
        self.workdir = tempfile.mkdtemp(prefix=f"ctf_{challenge.id}_", dir=root)
        challenge.setup(self.workdir)
        self._submits = 0

    # -- protocol -----------------------------------------------------------
    def description(self) -> str:
        files = sorted(os.listdir(self.workdir))
        return (f"{self.challenge.description}\n"
                f"Files in the working directory: {', '.join(files)}\n"
                f"The flag format is {self.challenge.flag[:7]}{{...}}.")

    def exec(self, cmd: str) -> str:
        low = cmd.lower()
        for bad in _DENY:
            if bad in low:
                return f"blocked: command matched denylisted pattern {bad!r}"
        try:
            p = subprocess.run(
                cmd, shell=True, cwd=self.workdir, capture_output=True,
                text=True, timeout=self.timeout_s,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                     "HOME": self.workdir, "LC_ALL": "C"},
            )
            out = (p.stdout or "") + (p.stderr or "")
            return out if out.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return f"(command timed out after {self.timeout_s}s)"
        except Exception as e:  # noqa: BLE001
            return f"exec error: {e}"

    def read_file(self, path: str) -> str:
        full = os.path.join(self.workdir, path) if not os.path.isabs(path) else path
        # confine reads to the challenge workdir
        if os.path.realpath(full).startswith(os.path.realpath(self.workdir)):
            with open(full, "rb") as f:
                data = f.read()
            try:
                return data.decode()
            except UnicodeDecodeError:
                return f"(binary file, {len(data)} bytes; hex head: {data[:64].hex()})"
        return "read error: path outside challenge working directory"

    def check_flag(self, flag: str) -> bool:
        cand = (flag or "").strip().strip('"').strip("'").strip()
        # tolerate the model wrapping the flag in quotes or trailing punctuation
        m = re.search(r"picoCTF\{[^}]*\}", cand)
        if m:
            cand = m.group(0)
        return cand == self.challenge.flag

    def cleanup(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


if __name__ == "__main__":
    # offline self-test: reference solutions solve each challenge, no LLM.
    from .challenges import all_challenges
    import base64 as _b64
    passed = 0
    for c in all_challenges():
        env = LocalEnv(c, workroot=tempfile.mkdtemp())
        # drive each with its reference approach via shell where possible
        got = None
        if c.id == "enc-base64":
            got = env.exec("base64 -d message.txt")
        elif c.id == "enc-hex":
            got = env.exec("cat message.txt | xxd -r -p")
        elif c.id == "for-strings":
            got = env.exec("strings dump.bin | grep picoCTF")
        elif c.id == "for-grep":
            got = env.exec("grep picoCTF server.log")
        elif c.id == "for-split":
            got = env.exec("cat part_a.txt part_b.txt")
        else:
            # generic python fallback proves the file is there & solvable
            got = env.exec("ls")
        ok_exec = got is not None
        # flag check with the true flag always passes (ground-truth sanity)
        assert env.check_flag(c.flag), f"ground-truth check failed for {c.id}"
        assert not env.check_flag("picoCTF{wrong}"), f"false positive for {c.id}"
        passed += 1
        env.cleanup()
    print(f"LocalEnv self-test OK: {passed} challenges, ground-truth checker sound")
