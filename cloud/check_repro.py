#!/usr/bin/env python3
"""Compare a sweep's held-out QE against a reference, per (edge, impl), within a tolerance.

Reproducibility here is WITHIN-TOLERANCE, not bit-exact: the factorized update uses float
atomicAdd (order-dependent rounding) and different GPU architectures schedule differently, so
held-out QE matches to ~2-3 significant figures and the converged epoch count can wobble by a
few. We therefore check the held-out QE of the GPU implementations (the deterministic-quality
anchor) to a relative tolerance; somoclu is excluded (its row is an extrapolated estimate).

Exit 0 if every compared row is within tolerance, else 1.
"""
import argparse
import csv


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["edge"], r["impl"])] = r
    return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("reference")
    ap.add_argument("--tol", type=float, default=0.02, help="relative tolerance on held-out QE")
    ap.add_argument("--metric", default="qe_held")
    a = ap.parse_args()

    cand, ref = load(a.candidate), load(a.reference)
    worst, fails, checked = 0.0, [], 0
    print(f"{'edge':>5} {'impl':>12} {'ref':>9} {'cand':>9} {'rel%':>7}  verdict")
    for key in sorted(ref, key=lambda k: (int(k[0]), k[1])):
        edge, impl = key
        if impl == "somoclu":                       # extrapolated estimate, not a repro target
            continue
        rv = fnum(ref[key].get(a.metric))
        cv = fnum(cand.get(key, {}).get(a.metric))
        if rv is None:
            continue
        if cv is None:
            print(f"{edge:>5} {impl:>12} {rv:>9.4f} {'MISSING':>9} {'-':>7}  FAIL (absent)")
            fails.append(key); continue
        rel = abs(cv - rv) / rv if rv else 0.0
        worst = max(worst, rel)
        checked += 1
        ok = rel <= a.tol
        if not ok:
            fails.append(key)
        print(f"{edge:>5} {impl:>12} {rv:>9.4f} {cv:>9.4f} {100*rel:>6.2f}%  {'ok' if ok else 'FAIL'}")

    print(f"\nchecked {checked} rows; worst deviation {100*worst:.2f}% (tol {100*a.tol:.0f}%)")
    if fails:
        print(f"REPRODUCIBILITY FAIL: {len(fails)} row(s) outside tolerance: {fails}")
        raise SystemExit(1)
    print("REPRODUCIBILITY PASS")


if __name__ == "__main__":
    main()
