#!/usr/bin/env python3
"""Inference-slice self-check for JuniorAGI_SDK (JuniorCloud LLC, Inference Core lane).

Stdlib only. No network, no model pull, no GGUF, no bench, no LUT.
Mirrors the layer stack (T0 trit < 512*512, T1 bit-plane < 1024*1024,
T2 native C when the .so exists, else T1) and the shared close rule
sha256(prev || pack || text)[:16] at trit-pack rate 0.05.
Not a quant file; touches no ports (Home, OSai, FieldCore, Flagstaff, JuniorLLM).
"""
import hashlib
import os
import sys

RATE = 0.05
BIND = "127.0.0.1"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
T0_MAX = 512 * 512
T1_MAX = 1024 * 1024


def trit_pack(weights):
    """Quantize to {-1, 0, 1} with a dead zone of RATE, then pack 5 trits per byte."""
    trits = [0 if abs(w) <= RATE else (1 if w > 0 else -1) for w in weights]
    out = bytearray()
    for i in range(0, len(trits), 5):
        v = 0
        for t in reversed(trits[i:i + 5]):
            v = v * 3 + (t + 1)
        out.append(v)
    return bytes(out), trits


def pick_layer(n, so_path=None):
    if n < T0_MAX:
        return "T0"
    if n < T1_MAX:
        return "T1"
    if so_path and os.path.exists(so_path):
        return "T2"
    return "T1"


def close(prev, pack, text):
    return hashlib.sha256(prev + pack + text.encode("utf-8")).hexdigest()[:16]


def main():
    failures = []
    pack, trits = trit_pack([0.9, -0.7, 0.01, -0.04, 0.2, 0.0, -0.3])
    if trits != [1, -1, 0, 0, 1, 0, -1]:
        failures.append("trit_pack")
    if pick_layer(10) != "T0" or pick_layer(T0_MAX) != "T1":
        failures.append("layer T0/T1")
    if pick_layer(T1_MAX, "/nonexistent/kernel.so") != "T1":
        failures.append("layer T2 fallback")
    a = close(b"\x00" * 16, pack, "slice")
    b = close(b"\x00" * 16, pack, "slice")
    if a != b or len(a) != 16:
        failures.append("close rule")
    if any(FLAGS.values()):
        failures.append("flags")
    if BIND != "127.0.0.1":
        failures.append("bind")
    if failures:
        print("FAIL: " + ", ".join(failures))
        return 1
    print("OK rate=%.2f close=%s layers=T0/T1/T2->T1 bind=%s flags=all-false" % (RATE, a, BIND))
    return 0


if __name__ == "__main__":
    sys.exit(main())
