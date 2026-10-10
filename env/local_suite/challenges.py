"""Docker-free local CTF challenge suite.

Each challenge is a self-contained, benign, picoCTF-style task whose flag is
recovered with standard shell tools (base64, xxd, tr, python3, strings, grep).
It needs no Docker and no network, so it runs inside the existing bubblewrap
sandbox and yields *real* ground-truth-scored agent trajectories right now,
while the heavier InterCode-CTF Docker benchmark is staged for the hub queue.

Why this exists (brief 11-B6 scope control): a smaller finished study on a
real, verifiable testbed beats a half-run large one. These challenges are
tagged by *vulnerability mechanism* (not coarse category) and grouped into
parent mechanisms with sibling surfaces, so the SAME transfer protocol (train
on surface X, eval on sibling surface Y) that we will run on InterCode/NYU-CTF
is exercised here end to end.

A challenge spec:
  id            : unique challenge id
  family        : mechanism family  "<parent>/<surface>"  (parent before the last '-')
  parent        : parent mechanism (siblings share it) -- derived if omitted
  split_group   : challenges with the same split_group are the same *item* across
                  surfaces; used only for bookkeeping, not required
  description   : the prompt shown to the agent (NEVER contains the flag)
  setup(dir)    : writes the challenge's files into a fresh working directory
  flag          : ground-truth flag string (NEVER shown to the agent)
  solvable_hint : reference solution path (for our own difficulty bookkeeping)
"""
from __future__ import annotations

import base64
import codecs
import os
from dataclasses import dataclass
from typing import Callable

FLAG_PREFIX = "picoCTF"


@dataclass
class Challenge:
    id: str
    family: str
    description: str
    setup: Callable[[str], None]
    flag: str
    solvable_hint: str = ""

    @property
    def parent(self) -> str:
        # parent mechanism = family with the trailing surface stripped
        return self.family.rsplit("-", 1)[0]


def _w(path: str, data, mode="w"):
    with open(path, mode) as f:
        f.write(data)


# ----------------------------------------------------------------- crypto/encoding
def _c_base64() -> Challenge:
    flag = f"{FLAG_PREFIX}{{b4se64_i5_n0t_encryp710n}}"
    def setup(d):
        _w(os.path.join(d, "message.txt"),
           base64.b64encode(flag.encode()).decode() + "\n")
    return Challenge(
        "enc-base64", "crypto/encoding-base64",
        "message.txt contains an encoded string. Decode it to recover the flag.",
        setup, flag, "base64 -d message.txt")


def _c_hex() -> Challenge:
    flag = f"{FLAG_PREFIX}{{hex_dump_rever5ed}}"
    def setup(d):
        _w(os.path.join(d, "message.txt"), flag.encode().hex() + "\n")
    return Challenge(
        "enc-hex", "crypto/encoding-hex",
        "message.txt contains a hex-encoded string. Decode it to recover the flag.",
        setup, flag, "xxd -r -p message.txt")


def _c_base32() -> Challenge:
    flag = f"{FLAG_PREFIX}{{b4se32_pads_w1th_equals}}"
    def setup(d):
        _w(os.path.join(d, "message.txt"),
           base64.b32encode(flag.encode()).decode() + "\n")
    return Challenge(
        "enc-base32", "crypto/encoding-base32",
        "message.txt contains a base32-encoded string. Decode it to recover the flag.",
        setup, flag, "base32 -d message.txt (or python base64.b32decode)")


# ----------------------------------------------------------------- crypto/classical
def _rot(s: str, n: int) -> str:
    out = []
    for ch in s:
        if ch.isupper():
            out.append(chr((ord(ch) - 65 + n) % 26 + 65))
        elif ch.islower():
            out.append(chr((ord(ch) - 97 + n) % 26 + 97))
        else:
            out.append(ch)
    return "".join(out)


def _c_caesar() -> Challenge:
    flag = f"{FLAG_PREFIX}{{caesar_shift_by_thirteen}}"
    ct = _rot(flag, 13)
    def setup(d):
        _w(os.path.join(d, "cipher.txt"), ct + "\n")
    return Challenge(
        "cls-caesar", "crypto/classical-caesar",
        "cipher.txt is a Caesar-shifted flag. Find the shift and recover the flag.",
        setup, flag, "rot13 / try all 25 shifts")


def _c_atbash() -> Challenge:
    flag = f"{FLAG_PREFIX}{{atbash_mirror_cipher}}"
    def atbash(s):
        out = []
        for ch in s:
            if ch.isupper():
                out.append(chr(90 - (ord(ch) - 65)))
            elif ch.islower():
                out.append(chr(122 - (ord(ch) - 97)))
            else:
                out.append(ch)
        return "".join(out)
    ct = atbash(flag)
    def setup(d):
        _w(os.path.join(d, "cipher.txt"), ct + "\n")
    return Challenge(
        "cls-atbash", "crypto/classical-atbash",
        "cipher.txt is enciphered with a monoalphabetic substitution that maps "
        "a<->z, b<->y, ... (Atbash). Recover the flag.",
        setup, flag, "atbash involution")


# ----------------------------------------------------------------- crypto/xor
def _xor_bytes(data: bytes, key: int) -> bytes:
    return bytes(b ^ key for b in data)


def _c_xor_single() -> Challenge:
    flag = f"{FLAG_PREFIX}{{x0r_single_byte_42}}"
    key = 0x42
    def setup(d):
        _w(os.path.join(d, "cipher.bin"), _xor_bytes(flag.encode(), key), mode="wb")
    return Challenge(
        "xor-single", "crypto/xor-single",
        "cipher.bin is the flag XORed with a single unknown byte. Recover the flag "
        "(the plaintext starts with 'picoCTF{').",
        setup, flag, "known-plaintext: key = cipher[0] ^ ord('p')")


def _c_xor_repeat() -> Challenge:
    flag = f"{FLAG_PREFIX}{{xor_repeating_key_cafe}}"
    key = b"\xca\xfe"
    def setup(d):
        data = flag.encode()
        ct = bytes(data[i] ^ key[i % len(key)] for i in range(len(data)))
        _w(os.path.join(d, "cipher.bin"), ct, mode="wb")
    return Challenge(
        "xor-repeat", "crypto/xor-repeat",
        "cipher.bin is the flag XORed with a 2-byte repeating key. The plaintext "
        "starts with 'picoCTF{'. Recover the key and the flag.",
        setup, flag, "key bytes = ct[0]^'p', ct[1]^'i'")


# ----------------------------------------------------------------- forensics
def _c_strings() -> Challenge:
    flag = f"{FLAG_PREFIX}{{str1ngs_finds_pr1ntable}}"
    def setup(d):
        blob = (os.urandom(200) + b"\x00garbage\x00" + flag.encode()
                + b"\x00" + os.urandom(200))
        _w(os.path.join(d, "dump.bin"), blob, mode="wb")
    return Challenge(
        "for-strings", "forensics/recover-strings",
        "dump.bin is a binary blob with the flag hidden as printable text inside. "
        "Extract it.",
        setup, flag, "strings dump.bin | grep picoCTF")


def _c_grep() -> Challenge:
    flag = f"{FLAG_PREFIX}{{grep_the_one_real_line}}"
    def setup(d):
        lines = [f"log entry {i}: nothing to see here" for i in range(500)]
        lines.insert(321, f"DEBUG token={flag}")
        _w(os.path.join(d, "server.log"), "\n".join(lines) + "\n")
    return Challenge(
        "for-grep", "forensics/recover-grep",
        "server.log has 500+ lines; exactly one contains the flag. Find it.",
        setup, flag, "grep picoCTF server.log")


def _c_zip_comment() -> Challenge:
    flag = f"{FLAG_PREFIX}{{hidden_in_plain_bytes}}"
    def setup(d):
        # flag split across two files; concatenation reveals it
        half = len(flag) // 2
        _w(os.path.join(d, "part_a.txt"), flag[:half])
        _w(os.path.join(d, "part_b.txt"), flag[half:])
    return Challenge(
        "for-split", "forensics/recover-split",
        "The flag was split across part_a.txt and part_b.txt. Reassemble it.",
        setup, flag, "cat part_a.txt part_b.txt")


# ----------------------------------------------------------------- reversing
def _c_rev_logic() -> Challenge:
    flag = f"{FLAG_PREFIX}{{r3ad_the_s0urce_luke}}"
    def setup(d):
        # a check script whose constant, de-rot13'd, is the flag
        enc = codecs.encode(flag, "rot13")
        src = (
            "#!/usr/bin/env python3\n"
            "import sys, codecs\n"
            f"SECRET = {enc!r}\n"
            "def check(guess):\n"
            "    return guess == codecs.decode(SECRET, 'rot13')\n"
            "if __name__ == '__main__':\n"
            "    g = sys.argv[1] if len(sys.argv) > 1 else ''\n"
            "    print('CORRECT' if check(g) else 'WRONG')\n"
        )
        _w(os.path.join(d, "check.py"), src)
    return Challenge(
        "rev-logic", "reversing/constant-rot13",
        "check.py validates a guess against an obfuscated constant. Read the "
        "source, reverse the transform, and recover the flag it accepts.",
        setup, flag, "rot13 the SECRET literal in check.py")


def _c_rev_arith() -> Challenge:
    flag = f"{FLAG_PREFIX}{{add_one_subtract_one}}"
    def setup(d):
        enc = bytes((b + 1) & 0xFF for b in flag.encode())
        _w(os.path.join(d, "secret.bin"), enc, mode="wb")
        src = (
            "#!/usr/bin/env python3\n"
            "# secret.bin holds each flag byte with +1 (mod 256) applied.\n"
            "data = open('secret.bin','rb').read()\n"
            "# flag = bytes((b-1) & 0xff for b in data)\n"
        )
        _w(os.path.join(d, "hint.py"), src)
    return Challenge(
        "rev-arith", "reversing/constant-arith",
        "secret.bin holds the flag with a simple per-byte arithmetic transform "
        "(see hint.py). Invert it to recover the flag.",
        setup, flag, "bytes((b-1)&0xff for b in data)")


# ----------------------------------------------------------------- registry / splits

def all_challenges() -> list[Challenge]:
    return [
        _c_base64(), _c_hex(), _c_base32(),
        _c_caesar(), _c_atbash(),
        _c_xor_single(), _c_xor_repeat(),
        _c_strings(), _c_grep(), _c_zip_comment(),
        _c_rev_logic(), _c_rev_arith(),
    ]


def by_id() -> dict[str, Challenge]:
    return {c.id: c for c in all_challenges()}


# Transfer protocol splits (brief: train on surface X, eval on sibling surface Y
# under the same parent mechanism). For each parent with >=2 surfaces we hold the
# first out as the in-distribution train/test surface and a sibling as transfer.
# These are the DATA ARTIFACT for H2 on the local suite; logged to results/.
TRANSFER_SPLITS = {
    # parent          : {"train": [...ids], "transfer": [...ids]}
    "crypto/encoding": {"train": ["enc-base64", "enc-hex"], "transfer": ["enc-base32"]},
    "crypto/classical": {"train": ["cls-caesar"], "transfer": ["cls-atbash"]},
    "crypto/xor":      {"train": ["xor-single"], "transfer": ["xor-repeat"]},
    "forensics/recover": {"train": ["for-strings", "for-grep"], "transfer": ["for-split"]},
    "reversing/constant": {"train": ["rev-logic"], "transfer": ["rev-arith"]},
}


if __name__ == "__main__":
    import tempfile
    cs = all_challenges()
    print(f"{len(cs)} challenges across parents:",
          sorted({c.parent for c in cs}))
    # self-check: every flag is well-formed and every setup runs
    with tempfile.TemporaryDirectory() as d:
        for c in cs:
            cd = os.path.join(d, c.id)
            os.makedirs(cd)
            c.setup(cd)
            assert c.flag.startswith(FLAG_PREFIX + "{") and c.flag.endswith("}"), c.id
            assert os.listdir(cd), f"{c.id} wrote no files"
    print("all setups OK; flags well-formed")
