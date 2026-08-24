"""Adapter (Step 0b): bridge the int16[n*depth] reference path arrays (.bin) to the EXISTING
sombench recovery kernels, which consume `L` = a list of per-level label arrays.

The key observation (see docs/reference_taxonomies_handover_correction.md): sombench's reference
tree distance `tree_dist(L, a, b)` takes `L = [level_0_labels, level_1_labels, ...]` (each an
array over articles) and counts the levels at which two articles differ, skipping `-1`. The
handover's `int16[n*depth]` path array is exactly that — **column `l` IS the level-`l` label
array**, with `-1` padding == the skip. So feeding a new taxonomy to cophenetic / AMI is a pure
column-slice; no new metric code.

  from reference_paths import load_path_array, to_levels, tree_dist
  L = to_levels(load_path_array("data/reference_taxonomies/openalex_path.bin", 4))
  # ... existing cophenetic Pearson(map_dist, tree_dist(L, a, b)) over sampled pairs ...
"""
import numpy as np


def load_path_array(path_bin, depth):
    """Load int16[n*depth] (article-major) → (n, depth) int16, -1-padded."""
    a = np.fromfile(path_bin, dtype=np.int16)
    if a.size % depth:
        raise ValueError(f"{path_bin}: size {a.size} not divisible by depth {depth}")
    return a.reshape(-1, depth)


def to_levels(path_arr):
    """(n, depth) path array → L = [col_0, col_1, …] (the `L` the existing kernels consume)."""
    return [np.ascontiguousarray(path_arr[:, l]) for l in range(path_arr.shape[1])]


def tree_dist(L, a, b):
    """Reference tree distance over sampled pairs (a, b): the number of levels at which the two
    articles carry a DIFFERENT label, counting only levels where BOTH are labelled. Identical to
    sombench/experiments/medline_som_viz/scale_sweep.py. depth − tree_dist = longest common prefix
    = the LCA depth. Pairs touching an unlabelled (`-1`) article contribute 0 at that level."""
    td = np.zeros(len(a), np.int32)
    for lv in L:
        va, vb = lv[a], lv[b]
        ok = (va >= 0) & (vb >= 0)
        td += (ok & (va != vb)).astype(np.int32)
    return td


def labelled(L):
    """Boolean mask of articles with a top-level label — pairs must skip unlabelled articles,
    exactly as MeSH-unlabelled points are skipped on the MeSH reference."""
    return L[0] >= 0


# tiny self-test / demo: feed a synthetic strict-tree path array and check the LCA semantics
if __name__ == "__main__":
    import os, tempfile
    # 3 articles, depth 4. a0 and a1 share domain+field+subfield (differ only at topic);
    # a2 shares only the domain with a0; a3 is unlabelled (all -1).
    arr = np.array([[0, 1, 2, 10],
                    [0, 1, 2, 11],
                    [0, 5, 9, 30],
                    [-1, -1, -1, -1]], dtype=np.int16)
    f = os.path.join(tempfile.gettempdir(), "_demo_path.bin"); arr.tofile(f)
    L = to_levels(load_path_array(f, 4))
    A = np.array([0, 0, 0]); B = np.array([1, 2, 3])
    td = tree_dist(L, A, B)
    lca = 4 - td
    print("pairs (a0,a1) (a0,a2) (a0,a3):")
    print("  tree_dist:", td.tolist(), " (expect [1, 3, 0])")
    print("  LCA depth:", lca.tolist(), " (a0~a1 share 3 levels; a0~a2 share 1; a0~a3 unlabelled→0)")
    print("  labelled mask:", labelled(L).tolist(), " (expect [True, True, True, False])")
    assert td.tolist() == [1, 3, 0] and labelled(L).tolist() == [True, True, True, False]
    print("OK — adapter feeds the existing tree_dist/cophenetic/AMI kernels unchanged.")
