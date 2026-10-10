"""Flag redaction (brief 8, non-negotiable).

Every string written into any persistent memory (arms B/C/D) or any SFT corpus
(arms E/F) passes through redact() first, which replaces anything shaped like a
CTF flag with a placeholder. A manual spot-check of >=20% of stored records is
part of the Phase-1 gate (scripts/audit_redaction.py).

We redact the common flag shells (picoCTF{...}, flag{...}, CTF{...}, and the
bare {...} that follows a known prefix) rather than specific known flags, so the
filter also catches flags from challenges we have not enumerated.
"""
from __future__ import annotations

import re

_FLAG_RES = [
    re.compile(r"\b(?:picoCTF|flag|FLAG|CTF|ctf|HTB|htb)\{[^}]*\}"),
    re.compile(r"\bpico\{[^}]*\}"),
]
PLACEHOLDER = "[REDACTED_FLAG]"


def redact(text: str) -> str:
    if not text:
        return text
    out = text
    for rx in _FLAG_RES:
        out = rx.sub(PLACEHOLDER, out)
    return out


def contains_flagish(text: str) -> bool:
    return any(rx.search(text or "") for rx in _FLAG_RES)


if __name__ == "__main__":
    samples = [
        ("the answer is picoCTF{s3cr3t_v4lue}", "the answer is [REDACTED_FLAG]"),
        ("flag{abc} and CTF{xyz}", "[REDACTED_FLAG] and [REDACTED_FLAG]"),
        ("no flag here", "no flag here"),
    ]
    for inp, want in samples:
        got = redact(inp)
        assert got == want, f"{inp!r} -> {got!r} != {want!r}"
    assert contains_flagish("picoCTF{x}") and not contains_flagish("plain text")
    print("redact self-test OK")
