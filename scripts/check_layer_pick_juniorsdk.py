#!/usr/bin/env python3
"""Layer-pick self-check for JuniorAGI_SDK (Inference Core lane, JuniorCloud LLC).

Standalone, stdlib-only. Mirrors the layer stack:
  T0 Python trit   for n < 512*512
  T1 bit-plane     for n < 1024*1024
  T2 native C      when the .so exists, else falls back to T1
Weights are trit-packed before the pick. Rate 0.05, close rule
sha256(prev || pack || text)[:16]. No bench, no LUT, no GGUF, no network.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM (read-only constants).
"""
import hashlib
import sys

BIND_HOST = "127.0.0.1"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
RATE = 0.05
T0_MAX = 512 * 512
T1_MAX = 1024 * 1024
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def trit_pack(weights, rate=RATE):
    """Map floats to {-1, 0, 1} with a dead zone of `rate`, then pack 5 trits/byte."""
    trits = [0 if abs(w) <= rate else (1 if w > 0 else -1) for w in weights]
    out = bytearray()
    for i in range(0, len(trits), 5):
        v = 0
        for t in reversed(trits[i:i + 5]):
            v = v * 3 + (t + 1)
        out.append(v)
    return bytes(out), trits


def pick_layer(n, so_exists=False):
    if n < T0_MAX:
        return "T0"
    if n < T1_MAX:
        return "T1"
    return "T2" if so_exists else "T1"


def close_rule(prev, pack, text):
    return hashlib.sha256(prev + pack + text.encode()).hexdigest()[:16]


def main():
    fails = []
    if BIND_HOST != "127.0.0.1":
        fails.append("bind")
    if any(FLAGS.values()):
        fails.append("flags")
    cases = [
        (0, False, "T0"), (T0_MAX - 1, False, "T0"), (T0_MAX, False, "T1"),
        (T1_MAX - 1, True, "T1"), (T1_MAX, False, "T1"), (T1_MAX, True, "T2"),
    ]
    for n, so, want in cases:
        got = pick_layer(n, so)
        if got != want:
            fails.append("pick n=%d so=%s got=%s want=%s" % (n, so, got, want))
    pack, trits = trit_pack([0.2, -0.04, 0.05, -0.3, 0.0, 0.51])
    if trits != [1, 0, 0, -1, 0, 1]:
        fails.append("trits %r" % trits)
    a = close_rule(b"\x00" * 16, pack, "juniorsdk")
    b = close_rule(b"\x00" * 16, pack, "juniorsdk")
    c = close_rule(b"\x01" * 16, pack, "juniorsdk")
    if a != b or a == c or len(a) != 16:
        fails.append("close_rule")
    # Engines agree: Python close rule vs. bytes-level recomputation.
    if a != hashlib.sha256(b"\x00" * 16 + pack + b"juniorsdk").hexdigest()[:16]:
        fails.append("engine_agree")
    print("ports:", ", ".join(PORTS))
    print("rate:", RATE, "close:", a)
    if fails:
        print("FAIL:", "; ".join(fails))
        return 1
    print("OK: layer pick, trit pack, close rule, bind, flags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
