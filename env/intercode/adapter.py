"""InterCode-CTF TaskEnv adapter (STAGED -- requires Docker).

Implements the same TaskEnv protocol as env/local_suite so arms A-F run on
InterCode-CTF unchanged once Docker is available on the host (currently absent:
docs/host_facts.md). InterCode-CTF ships 100 picoCTF tasks with a bash tool and
a flag-match reward; we wrap its per-task container so the ReAct agent's shell/
read/submit map onto the container, and check_flag stays GROUND TRUTH.

Bring-up (when Docker is installed), see env/intercode/README.md:
  1. git clone https://github.com/princeton-nlp/intercode  (into a fresh dir)
  2. docker build the ctf image per their Dockerfile
  3. the task data (queries + gold flags) loads from their ctf dataset json

This module is import-safe without Docker: it only touches Docker inside
connect(), so the rest of the harness imports it freely. The mechanism-family
tagging pass (env/intercode/tag_families.py) annotates each task before use.
"""
from __future__ import annotations

import os
import shlex
import subprocess


class InterCodeEnv:
    """One InterCode-CTF task as a TaskEnv. Requires a running task container."""

    def __init__(self, task: dict, image: str = "intercode-ctf",
                 timeout_s: int = 30):
        # task: {"task_id", "query"/"description", "gold" (flag), "files_dir",
        #        "mechanism_family" (from the tagging pass)}
        self.challenge_id = str(task["task_id"])
        self.mechanism_family = task.get("mechanism_family")
        self.parent_mechanism = (self.mechanism_family or "").rsplit("-", 1)[0] or None
        self._desc = task.get("description") or task.get("query", "")
        self._gold = task["gold"]
        self.image = image
        self.timeout_s = timeout_s
        self._cid = None  # running container id

    def connect(self):
        """Start the task container (lazy; Docker required)."""
        files = os.path.abspath(self._task_files)  # mount read-only
        out = subprocess.check_output(
            ["docker", "run", "-d", "--rm", "--network", "none",
             "-v", f"{files}:/ctf:ro", self.image, "sleep", "3600"],
            text=True)
        self._cid = out.strip()
        return self

    # -- protocol -----------------------------------------------------------
    def description(self) -> str:
        return self._desc

    def exec(self, cmd: str) -> str:
        if not self._cid:
            raise RuntimeError("container not connected; call connect() first")
        p = subprocess.run(
            ["docker", "exec", self._cid, "bash", "-lc", cmd],
            capture_output=True, text=True, timeout=self.timeout_s)
        return ((p.stdout or "") + (p.stderr or "")) or "(no output)"

    def read_file(self, path: str) -> str:
        return self.exec(f"cat {shlex.quote(path)}")

    def check_flag(self, flag: str) -> bool:
        import re
        cand = (flag or "").strip().strip('"').strip("'")
        m = re.search(r"picoCTF\{[^}]*\}|flag\{[^}]*\}", cand)
        if m:
            cand = m.group(0)
        return cand == self._gold

    def cleanup(self):
        if self._cid:
            subprocess.run(["docker", "kill", self._cid], capture_output=True)
            self._cid = None

    @property
    def _task_files(self):
        return os.environ.get("INTERCODE_FILES", "./env/intercode/files")


def docker_available() -> bool:
    try:
        subprocess.run(["docker", "version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False
