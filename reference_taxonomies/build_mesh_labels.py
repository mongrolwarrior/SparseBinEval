#!/usr/bin/env python3
"""Build the MeSH reference path array (mesh_path.bin, int16[n*depth]) for corpus.sbcsr's rows —
the original SOM-vs-MeSH check and the taxonomy-vs-taxonomy baseline (MeSH vs OpenAlex).

Mirrors sombench/experiments/exp3_mesh_hierarchy/mesh_hierarchy.py, adapted to our corpus: an
article's level-L label is the MAJORITY-VOTE level-L MeSH-tree ancestor across its descriptors
(level-L ancestor = first L dotted components of a descriptor's tree number, e.g. L=2 of
"C04.557.386" → "C04.557"). corpus.sbcsr's col_idx ARE the compact vocab ids (vocab.json[idx]=UI),
so the join is direct. Each level is dense-coded independently to int16; -1 = no level-L ancestor.

  build_mesh_labels.py [--depth 8]   ->  data/reference_taxonomies/mesh_path.bin (+ mesh_tree.json)
"""
import argparse
import json
import multiprocessing as mp
import os
import struct
import sys
import xml.etree.ElementTree as ET
from array import array
from collections import Counter

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC = os.path.join(ROOT, "data/medline/processed/pubmed26_ge5")
XML = os.path.join(ROOT, "data/mesh/desc2026.xml")
REF = os.path.join(ROOT, "data/reference_taxonomies")
_MAGIC = b"SBCSR1\x00\x00"


def parse_desc_xml(path):
    """raw_id (int) -> list of tree-number strings (DescriptorRecords only)."""
    out = {}
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "DescriptorRecord":
            continue
        ui = elem.find("DescriptorUI")
        if ui is not None and ui.text and ui.text.startswith("D"):
            try:
                rid = int(ui.text[1:])
            except ValueError:
                elem.clear(); continue
            trees = [tn.text.strip() for tn in elem.findall(".//TreeNumber") if tn.text]
            if trees:
                out[rid] = trees
        elem.clear()
    return out


def load_corpus():
    with open(os.path.join(PROC, "corpus.sbcsr"), "rb") as f:
        assert f.read(8) == _MAGIC
        n, V, nnz, _ = struct.unpack("<IIII", f.read(16))
        row_ptr = np.frombuffer(f.read(4 * (n + 1)), dtype="<u4")
        col_idx = np.frombuffer(f.read(2 * nnz), dtype="<u2")
    return n, V, row_ptr, col_idx


# globals shared to fork()ed workers (Linux copy-on-write — no re-pickling of the big arrays)
_RP = _COL = _LEVELCODE = _DEPTH = None


def _init(rp, col, lc, depth):
    global _RP, _COL, _LEVELCODE, _DEPTH
    _RP, _COL, _LEVELCODE, _DEPTH = rp, col, lc, depth


def _worker(rng):
    start, end = rng
    out = np.full((end - start, _DEPTH), -1, np.int16)
    for i in range(start, end):
        cols = _COL[_RP[i]:_RP[i + 1]]
        codes = _LEVELCODE[cols]                       # (nnz, depth) int16 ancestor codes
        for l in range(_DEPTH):
            v = codes[:, l]
            v = v[v >= 0]
            if v.size:
                u, c = np.unique(v, return_counts=True)
                out[i - start, l] = u[c.argmax()]      # majority-vote level-l ancestor
    return start, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8)
    a = ap.parse_args()
    if not os.path.exists(XML):
        raise SystemExit(f"missing {XML} — download the MeSH descriptor XML first")

    vocab = json.load(open(os.path.join(PROC, "vocab.json")))      # [UI, ...] indexed by compact id
    raw_of = [int(ui[1:]) for ui in vocab]
    print(f"vocab {len(vocab):,} MeSH UIs; parsing {os.path.basename(XML)} ...", flush=True)
    raw_to_trees = parse_desc_xml(XML)
    print(f"  {len(raw_to_trees):,} descriptors with tree numbers", flush=True)

    # per (compact id, level) ancestor string, then dense-code each level independently to int16
    cid_trees = {cid: raw_to_trees[r] for cid, r in enumerate(raw_of) if r in raw_to_trees}
    print(f"  {len(cid_trees):,}/{len(vocab):,} vocab terms have tree numbers", flush=True)
    V = len(vocab)
    level_code = np.full((V, a.depth), -1, np.int16)
    tree_meta = {}
    for l in range(1, a.depth + 1):
        anc = {}
        for cid, trees in cid_trees.items():
            for t in trees:
                parts = t.split(".")
                if len(parts) >= l:
                    anc[cid] = ".".join(parts[:l]); break
        codes = {s: i for i, s in enumerate(sorted(set(anc.values())))}
        for cid, s in anc.items():
            level_code[cid, l - 1] = codes[s]
        tree_meta[f"L{l}"] = {"classes": len(codes), "labelled_terms": len(anc)}
        print(f"  L{l}: {len(codes)} classes over {len(anc):,} terms", flush=True)

    n, _, row_ptr, col_idx = load_corpus()
    print(f"voting per-article labels over {n:,} articles ({a.depth} levels)...", flush=True)
    chunks = [(s, min(s + 200_000, n)) for s in range(0, n, 200_000)]
    mesh_path = np.empty((n, a.depth), np.int16)
    with mp.Pool(min(16, os.cpu_count() or 4), initializer=_init,
                 initargs=(row_ptr, col_idx, level_code, a.depth)) as pool:
        done = 0
        for start, block in pool.imap_unordered(_worker, chunks):
            mesh_path[start:start + len(block)] = block
            done += len(block)
            if done % 4_000_000 < 200_000:
                print(f"  {done:,}/{n:,}", flush=True)

    os.makedirs(REF, exist_ok=True)
    mesh_path.tofile(os.path.join(REF, "mesh_path.bin"))
    cov = {f"L{l+1}": float((mesh_path[:, l] >= 0).mean()) for l in range(a.depth)}
    json.dump({"depth": a.depth, "n": n, "coverage": cov, "levels": tree_meta,
               "source": "MeSH desc2026.xml; majority-vote level-L ancestor"},
              open(os.path.join(REF, "mesh_tree.json"), "w"), indent=2)
    print(f"wrote mesh_path.bin (int16[{n}*{a.depth}]); L1 coverage {cov['L1']:.3%}, "
          f"L{a.depth} coverage {cov[f'L{a.depth}']:.3%}")


if __name__ == "__main__":
    main()
