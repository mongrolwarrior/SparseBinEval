#!/usr/bin/env python3
"""Step 5: hierarchy-recovery scoring of a SparseSOM clustering against a reference taxonomy.

Cophenetic correlation (sombench's headline "B1" metric): over sampled article pairs, correlate
the SOM's MAP distance (grid distance between the two articles' BMUs) with the reference TREE
distance (levels at which they differ — `reference_paths.tree_dist`). A positive correlation means
the SOM layout encodes the reference hierarchy. Reported per level and aggregate, exactly as
sombench/experiments/medline_som_viz/_perlevel.py — here generalized to ANY taxonomy path array
via the adapter, so MeSH / OpenAlex / FoR all run through the same code.

Also does the taxonomy-vs-taxonomy baseline: cophenetic(tree_dist(A), tree_dist(B)) — how much two
reference trees agree with each other over the corpus (the ceiling any SOM-vs-tree score is read
against).

  score_recovery.py --bmu bmu.u32 --edge 350 \
      --ref openalex:data/reference_taxonomies/openalex_path.bin:4 \
      --ref mesh:data/reference_taxonomies/mesh_path.bin:8 --pairs 10000000
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_paths import load_path_array, to_levels   # noqa: E402


def pearson(x, y):
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def map_dist(bmu, edge, a, b):
    """Euclidean grid distance between the BMUs of article pairs (a, b). bmu = neuron index.

    Cast each coordinate to float BEFORE subtracting. bmu is uint32, so `(ra - rb)` in unsigned
    arithmetic underflows to ~4e9 for every pair with rb>ra, which (after the float cast) swamps
    the real distances and collapses the cophenetic correlation to ~0. Validated: with this fix the
    harness reproduces sombench's B1 (+0.1757 @50², +0.1640 @350²); the buggy version read +0.006."""
    ra = (bmu[a] // edge).astype(np.float32); ca = (bmu[a] % edge).astype(np.float32)
    rb = (bmu[b] // edge).astype(np.float32); cb = (bmu[b] % edge).astype(np.float32)
    return np.hypot(ra - rb, ca - cb)


def sample_pairs(n, p, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, p); b = rng.integers(0, n, p)
    keep = a != b
    return a[keep], b[keep]


def cophenetic_vs_map(bmu, edge, L, a, b):
    """Per-level + aggregate cophenetic Pearson(map_dist, tree-distance), pairs where both labelled."""
    top = (L[0][a] >= 0) & (L[0][b] >= 0)
    a, b = a[top], b[top]
    gd = map_dist(bmu, edge, a, b)
    out, td = {}, np.zeros(len(a), np.int32)
    for l, lv in enumerate(L, 1):
        va, vb = lv[a], lv[b]
        m = (va >= 0) & (vb >= 0)
        out[f"L{l}"] = pearson(gd[m], (va[m] != vb[m]).astype(np.float32))
        td += (m & (va != vb)).astype(np.int32)
    out["agg"] = pearson(gd, td.astype(np.float32))
    out["pairs"] = int(len(a))
    return out


def cophenetic_tree_vs_tree(LA, LB, a, b):
    """Baseline: cophenetic(tree_dist(A), tree_dist(B)) over pairs labelled in BOTH trees."""
    top = (LA[0][a] >= 0) & (LA[0][b] >= 0) & (LB[0][a] >= 0) & (LB[0][b] >= 0)
    a, b = a[top], b[top]

    def td(L):
        d = np.zeros(len(a), np.int32)
        for lv in L:
            va, vb = lv[a], lv[b]
            ok = (va >= 0) & (vb >= 0)
            d += (ok & (va != vb)).astype(np.int32)
        return d.astype(np.float32)
    return {"cophenetic": pearson(td(LA), td(LB)), "pairs": int(len(a))}


def _load_refs(specs):
    refs = {}
    for s in specs:
        name, path, depth = s.split(":")
        refs[name] = to_levels(load_path_array(path, int(depth)))
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bmu", help="per-article BMU dump (uint32[n])")
    ap.add_argument("--edge", type=int, help="map edge M (grid is edge x edge)")
    ap.add_argument("--ref", action="append", default=[], help="name:path.bin:depth (repeatable)")
    ap.add_argument("--pairs", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    refs = _load_refs(a.ref)
    n = len(next(iter(refs.values()))[0]) if refs else None

    if a.bmu and a.edge:
        bmu = np.fromfile(a.bmu, dtype=np.uint32)
        n = len(bmu)
        A, B = sample_pairs(n, a.pairs, a.seed)
        print(f"SOM-vs-tree recovery (cophenetic Pearson), {a.edge}² map, {len(A):,} pairs:")
        for name, L in refs.items():
            r = cophenetic_vs_map(bmu, a.edge, L, A, B)
            print(f"  {name:9s} agg={r['agg']:+.4f}  " +
                  " ".join(f"{k}={r[k]:+.3f}" for k in r if k.startswith('L')) +
                  f"  (pairs {r['pairs']:,})")

    if len(refs) >= 2:
        A, B = sample_pairs(n, a.pairs, a.seed)
        names = list(refs)
        print("taxonomy-vs-taxonomy baseline (cophenetic Pearson between trees):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = cophenetic_tree_vs_tree(refs[names[i]], refs[names[j]], A, B)
                print(f"  {names[i]} vs {names[j]}: {r['cophenetic']:+.4f}  (pairs {r['pairs']:,})")


# ── self-test: a layout that respects a synthetic tree should score high; random ≈ 0 ──
if __name__ == "__main__" and "--selftest" in sys.argv:
    rng = np.random.default_rng(0)
    edge = 32                                          # 4 levels × 2×2 splits = leaf 2×2 cells
    # A genuinely tree-respecting 2-D layout: recursively place each 4-ary level into a quadrant
    # (domain→quadrant of the map, field→sub-quadrant, …), so tree-distance maps cleanly onto map-
    # distance scales — what a good SOM approximates. (Row-major does NOT, hence the ceiling above.)
    paths, bmus = [], []
    for d in range(4):
        for f in range(4):
            for s in range(4):
                for tq in range(4):
                    r0 = c0 = 0; sz = edge
                    for lv in (d, f, s, tq):
                        sz //= 2; r0 += (lv // 2) * sz; c0 += (lv % 2) * sz
                    for j in range(4):                 # 4 articles in this leaf's 2×2 region
                        paths.append([d, d * 4 + f, (d * 4 + f) * 4 + s, len(paths)])
                        bmus.append((r0 + j // 2) * edge + (c0 + j % 2))
    arr = np.array(paths, np.int16); L = to_levels(arr); N = len(paths)
    bmu_good = np.array(bmus, np.uint32)               # quadrant-nested → tree-close ⇒ grid-close
    bmu_rand = rng.permutation(N).astype(np.uint32)    # destroys the correspondence
    A, B = sample_pairs(N, 300000, 1)
    g = cophenetic_vs_map(bmu_good, edge, L, A, B)["agg"]
    r = cophenetic_vs_map(bmu_rand, edge, L, A, B)["agg"]
    print(f"selftest cophenetic: tree-respecting layout = {g:+.4f} ; random layout = {r:+.4f}")
    # flat-grid geometric ceiling keeps these modest (sombench: real SOM ≈ +0.18); the test only
    # asserts the harness REWARDS hierarchy and scores ~0 on random.
    assert g > 0.02 and abs(r) < 0.01 and g > 10 * abs(r), \
        "harness must reward a tree-respecting layout and score ~0 on random"
    print("OK — rewards hierarchy, ~0 on random (sombench B1 semantics; real SOM ≈ +0.18)")
elif __name__ == "__main__":
    main()
