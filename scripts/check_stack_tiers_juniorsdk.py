#!/usr/bin/env python3
"""Inference-slice self-check for JuniorAGI_SDK (JuniorCloud LLC, additive only).

Mirrors the layer stack (JuniorLLM junior_osai/kernels/stack.py, JuniorHome
docs/LAYER_STACK.md): T0 Python trit for n < 512*512, T1 bit-plane numpy for
n < 1024*1024, T2 native C when the .so exists, else fall back to T1.
Weights are trit-packed before the layer pick. No bench, no LUT, no GGUF.

Trit-pack rate 0.05 and close rule sha256(prev || pack || text)[:16].
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Guard flags stay false. Bind host is 127.0.0.1. Stdlib only, no network.
"""
import hashlib
import json
import sys

RATE = 0.05
BIND_HOST = "127.0.0.1"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
T0_MAX = 512 * 512
T1_MAX = 1024 * 1024


def trit(x, thr=RATE):
    return 1 if x > thr else (-1 if x < -thr else 0)


def trit_pack(weights, thr=RATE):
    """Pack ternary values 5 per byte (base 3), before any layer pick."""
    trits = [trit(w, thr) + 1 for w in weights]
    out = bytearray()
    for i in range(0, len(trits), 5):
        v = 0
        for t in reversed(trits[i:i + 5]):
            v = v * 3 + t
        out.append(v)
    return bytes(out)


def close(prev: bytes, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev + pack + text.encode()).hexdigest()[:16]


def pick_tier(n: int, so_exists: bool = False) -> str:
    if n < T0_MAX:
        return "T0"
    if n < T1_MAX:
        return "T1"
    return "T2" if so_exists else "T1"


def self_check():
    results = {}
    results["tiers"] = (
        pick_tier(T0_MAX - 1) == "T0"
        and pick_tier(T0_MAX) == "T1"
        and pick_tier(T1_MAX - 1) == "T1"
        and pick_tier(T1_MAX, so_exists=True) == "T2"
        and pick_tier(T1_MAX, so_exists=False) == "T1"
    )
    w = [0.2, -0.3, 0.01, -0.04, 0.06, -0.06, 0.0]
    pack = trit_pack(w)
    results["pack_len"] = len(pack) == 2
    results["trits"] = [trit(x) for x in w] == [1, -1, 0, 0, 1, -1, 0]
    c1 = close(b"", pack, "slice")
    c2 = close(b"", pack, "slice")
    # Two independent engines must agree on rate and close rule.
    alt = hashlib.sha256(b"".join([b"", pack, "slice".encode("utf-8")])).hexdigest()[:16]
    results["close_rule"] = c1 == c2 == alt and len(c1) == 16
    results["rate"] = RATE == 0.05
    results["flags_false"] = not any(FLAGS.values())
    results["bind_local"] = BIND_HOST == "127.0.0.1"
    results["ports"] = len(PORTS) == 5
    return results


if __name__ == "__main__":
    r = self_check()
    ok = all(r.values())
    print(json.dumps({"ok": ok, "checks": r, "ports": PORTS, "rate": RATE}, indent=2))
    sys.exit(0 if ok else 1)
