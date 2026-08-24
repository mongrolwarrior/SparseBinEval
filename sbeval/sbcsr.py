"""Write SparseBinSOM's .sbcsr corpus format, and generate small random corpora.

.sbcsr layout (little-endian; see SparseBinSOM src/core/dataset.cpp):
    24-byte header: magic "SBCSR1\0\0" (8 bytes), then uint32 n_samples, n_features,
                    n_nonzeros, reserved(=0).
    uint32 row_ptr[n_samples + 1]   (CSR offsets; row_ptr[n_samples] == n_nonzeros)
    uint16 col_idx[n_nonzeros]      (present feature ids per sample)
"""

import random
import struct
import sys
from array import array

_MAGIC = b"SBCSR1\x00\x00"  # 8 bytes


def write_sbcsr(path, n_features, rows):
    """Write `rows` (a list of lists of feature ids) to `path` as .sbcsr.

    Returns (n_samples, n_nonzeros).
    """
    row_ptr = array("I", [0])  # uint32 CSR offsets
    col_idx = array("H")       # uint16 feature ids
    for r in rows:
        col_idx.extend(r)
        row_ptr.append(len(col_idx))
    if sys.byteorder != "little":   # the format is little-endian; byteswap on big-endian hosts
        row_ptr.byteswap()
        col_idx.byteswap()

    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<IIII", len(rows), n_features, len(col_idx), 0))
        f.write(row_ptr.tobytes())
        f.write(col_idx.tobytes())
    return len(rows), len(col_idx)


def read_stats(path):
    """Read a .sbcsr header and per-row lengths; return shape + nnz mean/std.

    Used when a sweep runs on a prebuilt corpus (e.g. real MEDLINE) instead of a
    generated one, so the memory model and σ defaults see the true corpus shape.
    Returns {n_samples, n_features, n_nonzeros, nnz_mean, nnz_std}."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != _MAGIC:
            raise RuntimeError(f"not a .sbcsr file: {path}")
        n_samples, n_features, n_nonzeros, _ = struct.unpack("<IIII", f.read(16))
        row_ptr = array("I")
        row_ptr.frombytes(f.read(4 * (n_samples + 1)))
    if sys.byteorder != "little":
        row_ptr.byteswap()
    # Per-row nnz (number of present features) → mean and population std-dev.
    s = sq = 0
    for i in range(n_samples):
        L = row_ptr[i + 1] - row_ptr[i]
        s += L
        sq += L * L
    mean = s / n_samples if n_samples else 0.0
    var = (sq / n_samples - mean * mean) if n_samples else 0.0
    return {"n_samples": n_samples, "n_features": n_features, "n_nonzeros": n_nonzeros,
            "nnz_mean": round(mean, 4), "nnz_std": round(var ** 0.5, 4)}


def iter_rows(path):
    """Yield each sample's present feature ids (an array('H')) from a .sbcsr, streaming."""
    with open(path, "rb") as f:
        if f.read(8) != _MAGIC:
            raise RuntimeError(f"not a .sbcsr file: {path}")
        n_samples, n_features, n_nonzeros, _ = struct.unpack("<IIII", f.read(16))
        row_ptr = array("I")
        row_ptr.frombytes(f.read(4 * (n_samples + 1)))
        if sys.byteorder != "little":
            row_ptr.byteswap()
        for i in range(n_samples):
            length = row_ptr[i + 1] - row_ptr[i]
            cols = array("H")
            cols.frombytes(f.read(2 * length))
            if sys.byteorder != "little":
                cols.byteswap()
            yield cols


def generate_random(path, n_samples, n_features, nnz_mean, seed=0, nnz_std=0.0):
    """Generate a random sparse-binary corpus. Each sample gets a number of distinct features
    (drawn uniformly from [0, n_features)) equal to `nnz_mean`, or — if nnz_std > 0 — to
    round(Normal(nnz_mean, nnz_std)) clamped to [1, n_features]. Returns (n_samples, n_nonzeros)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        if nnz_std > 0.0:
            k = max(1, min(n_features, round(rng.gauss(nnz_mean, nnz_std))))
        else:
            k = max(1, min(int(round(nnz_mean)), n_features))
        rows.append(sorted(rng.sample(range(n_features), k)))
    return write_sbcsr(path, n_features, rows)
