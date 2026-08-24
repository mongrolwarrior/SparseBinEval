#!/usr/bin/env python3
"""Concatenate two .sbcsr files into one (same feature dimension required).

  python3 concat_sbcsr.py a.sbcsr b.sbcsr out.sbcsr
"""
import struct
import sys
from array import array

_MAGIC = b"SBCSR1\x00\x00"


def read(path):
    with open(path, "rb") as f:
        assert f.read(8) == _MAGIC, f"not a .sbcsr file: {path}"
        n, V, nnz, _ = struct.unpack("<IIII", f.read(16))
        rp = array("I"); rp.frombytes(f.read(4 * (n + 1)))
        ci = array("H"); ci.frombytes(f.read(2 * nnz))
    return n, V, rp, ci


def main():
    if len(sys.argv) != 4:
        sys.exit(f"usage: {sys.argv[0]} a.sbcsr b.sbcsr out.sbcsr")
    n1, V, rp1, ci1 = read(sys.argv[1])
    n2, V2, rp2, ci2 = read(sys.argv[2])
    assert V == V2, f"feature dimension mismatch: {V} vs {V2}"

    offset = len(ci1)
    rp = array("I", rp1)
    for i in range(1, len(rp2)):
        rp.append(rp2[i] + offset)
    ci = array("H", ci1)
    ci.extend(ci2)

    with open(sys.argv[3], "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<IIII", n1 + n2, V, len(ci), 0))
        f.write(rp.tobytes())
        f.write(ci.tobytes())
    print(f"{sys.argv[3]}: {n1+n2:,} samples, {V:,} features, {len(ci):,} nnz")


if __name__ == "__main__":
    main()
