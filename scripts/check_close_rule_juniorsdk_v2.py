#!/usr/bin/env python3
"""Local self-check for the JuniorCloud LLC close rule (JuniorAGI_SDK).

Verifies: trit rate 0.05, close = sha256(prev || pack || text)[:16],
guard flags false, loopback-only bind. Ports named: Home, OSai, FieldCore,
Flagstaff, JuniorLLM. No network, no model pull, no GGUF.
"""
import hashlib
import sys

RATE = 0.05
HOST = "127.0.0.1"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def close(prev: bytes, pack: bytes, text: bytes) -> str:
    return hashlib.sha256(prev + pack + text).hexdigest()[:16]


def layer(n: int) -> str:
    if n < 512 * 512:
        return "T0"
    if n < 1024 * 1024:
        return "T1"
    return "T2"


def main() -> int:
    a = close(b"", b"pack", b"text")
    b = close(b"", b"pack", b"text")
    c = close(a.encode(), b"pack", b"text")
    checks = {
        "rate": RATE == 0.05,
        "close_len": len(a) == 16,
        "close_deterministic": a == b,
        "close_chains": a != c,
        "host_loopback": HOST == "127.0.0.1",
        "flags_false": not any(FLAGS.values()),
        "layers": (layer(1), layer(512 * 512), layer(1024 * 1024)) == ("T0", "T1", "T2"),
    }
    for k, v in checks.items():
        print(f"{'ok' if v else 'FAIL'} {k}")
    print("ports:", ", ".join(PORTS))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
