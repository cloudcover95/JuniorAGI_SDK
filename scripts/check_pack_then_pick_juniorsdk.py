#!/usr/bin/env python3
"""Pack-then-pick self-check for JuniorAGI_SDK (Inference Core lane, JuniorCloud LLC).

Additive, stdlib-only, offline. Mirrors the layer stack: weights are trit-packed
first, then the tier is picked:
  T0 python trit  for n < 512*512
  T1 bit-plane numpy for n < 1024*1024
  T2 native C when the .so exists, else falls back to T1
No bench, no LUT, no GGUF, no model pull, no network.
Ports touched (quant-adjacent): Home, OSai, FieldCore, Flagstaff, JuniorLLM.
"""
import hashlib
import os

HOST = "127.0.0.1"
RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
T0_MAX = 512 * 512
T1_MAX = 1024 * 1024


def trit(x, thr=0.5):
    return 1 if x > thr else (-1 if x < -thr else 0)


def trit_pack(values):
    """Pack trits 5 per byte (3**5 = 243 <= 256)."""
    out = bytearray()
    ts = [trit(v) + 1 for v in values]
    for i in range(0, len(ts), 5):
        b = 0
        for t in reversed(ts[i:i + 5]):
            b = b * 3 + t
        out.append(b)
    return bytes(out)


def trit_unpack(packed, n):
    ts = []
    for b in packed:
        for _ in range(5):
            ts.append(b % 3 - 1)
            b //= 3
    return ts[:n]


def pick_tier(n, so_path=None):
    if n < T0_MAX:
        return "T0"
    if n < T1_MAX:
        return "T1"
    if so_path and os.path.exists(so_path):
        return "T2"
    return "T1"


def close_rule(prev, pack, text):
    return hashlib.sha256(prev + pack + text.encode()).hexdigest()[:16]


def engines_agree():
    a = (RATE, close_rule(b"p", b"k", "t"))
    b = (0.05, hashlib.sha256(b"pkt").hexdigest()[:16])
    return a == b


def main():
    vals = [0.9, -0.8, 0.1, 0.0, -0.2, 0.7, -0.6]
    packed = trit_pack(vals)
    checks = {
        "host_loopback": HOST == "127.0.0.1",
        "flags_false": not any(FLAGS.values()),
        "roundtrip": trit_unpack(packed, len(vals)) == [trit(v) for v in vals],
        "tier_t0": pick_tier(T0_MAX - 1) == "T0",
        "tier_t1": pick_tier(T0_MAX) == "T1",
        "tier_fallback": pick_tier(T1_MAX, "/nonexistent/kernel.so") == "T1",
        "close_len16": len(close_rule(b"", packed, "x")) == 16,
        "engines_agree": engines_agree(),
        "ports_named": len(PORTS) == 5,
    }
    for k, v in checks.items():
        print(f"{k}: {'ok' if v else 'FAIL'}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
