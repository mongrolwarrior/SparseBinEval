#!/usr/bin/env python3
"""Step 4: pack the reference taxonomies into article_idx order → the drop-in path arrays.

Keyed to corpus.sbcsr's rows via metadata.sqlite(article_idx, pmid):
    openalex_path.bin   int16[n*4]   primary-topic dense path, -1-padded
    for_path.bin        int16[n*2]   primary FoR (division, group), -1-padded   (if for_labels exists)
    join_report.json    coverage + multi-label distribution + source versions

`-1` rows = articles with no classification (no PMID, or not in that source) — the recovery
kernels skip pairs touching a `-1` path (see reference_paths.labelled), exactly as for MeSH-
unlabelled points.

Memory: the label tables are joined to metadata.sqlite in SQLite (ATTACH + streamed cursor) and
scatter-written into a preallocated int16 array by article_idx, so peak RAM is the output array
(~n*depth*2 bytes), NOT the full pmid→label map. Nothing is materialised in a Python dict/list.
"""
import json
import os
import sqlite3

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REF = os.path.join(ROOT, "data/reference_taxonomies")
META = os.path.join(REF, "metadata.sqlite")

CHUNK = 500_000


def _tree_counts():
    t = sqlite3.connect(os.path.join(REF, "openalex_tree.sqlite"))
    c = dict(t.execute("SELECT key,value FROM meta"))
    t.close()
    return c.get("counts")


def main():
    if not os.path.exists(META):
        raise SystemExit(f"missing {META} — run build_pmid_map.py (Step 0) first")
    con = sqlite3.connect(META)
    con.execute("PRAGMA query_only=ON")                 # never mutate the audited map
    n = con.execute("SELECT MAX(article_idx) + 1 FROM articles").fetchone()[0]
    report = {"n": n, "sources": {}}

    # ── OpenAlex (depth 4) ──
    oal = os.path.join(REF, "openalex_labels.sqlite")
    if os.path.exists(oal):
        oa_path = np.full((n, 4), -1, np.int16)
        labelled = nlabels_total = 0
        con.execute("ATTACH DATABASE ? AS oal", (oal,))
        # INNER JOIN: only labelled articles reach Python. lab.pmid is the PK, so each row is an
        # indexed probe; the array scatter-write means row order is irrelevant (no ORDER BY / sort).
        cur = con.execute(
            "SELECT a.article_idx, l.domain_dense, l.field_dense, l.subfield_dense, "
            "       l.topic_dense, l.topics_json "
            "FROM articles a JOIN oal.lab l ON a.pmid = l.pmid")
        while True:
            chunk = cur.fetchmany(CHUNK)
            if not chunk:
                break
            idxs = np.fromiter((r[0] for r in chunk), np.int64, len(chunk))
            oa_path[idxs] = np.array([r[1:5] for r in chunk], np.int16)
            labelled += len(chunk)
            for r in chunk:
                nlabels_total += len(json.loads(r[5])) if r[5] else 1
        cur.close()
        con.execute("DETACH DATABASE oal")
        oa_path.tofile(os.path.join(REF, "openalex_path.bin"))
        cov = labelled / n
        report["sources"]["openalex"] = {
            "depth": 4, "coverage": round(cov, 6), "labelled": labelled,
            "mean_topics_per_labelled": round(nlabels_total / max(1, labelled), 3),
            "tree_counts": _tree_counts(), "path_bin": "openalex_path.bin"}
        print(f"openalex_path.bin: {labelled:,}/{n:,} labelled (coverage {cov:.4%})")
    else:
        print("(openalex_labels.sqlite not present — run extract_openalex_labels.py on the host; "
              "skipping openalex_path.bin)")

    # ── FoR (depth 2): code IS the path (2-digit division = group//100) ──
    forl = os.path.join(REF, "for_labels.sqlite")
    if os.path.exists(forl):
        fr_path = np.full((n, 2), -1, np.int16)
        labelled = 0
        con.execute("ATTACH DATABASE ? AS forl", (forl,))
        cur = con.execute(
            "SELECT a.article_idx, l.for_grp_json "
            "FROM articles a JOIN forl.lab l ON a.pmid = l.pmid")
        while True:
            chunk = cur.fetchmany(CHUNK)
            if not chunk:
                break
            idxs, paths = [], []
            for idx, gj in chunk:
                g = json.loads(gj)
                if not g:
                    continue
                grp = int(g[0])
                idxs.append(idx)
                paths.append((grp // 100, grp))          # division = 2-digit prefix
            if idxs:
                fr_path[np.array(idxs, np.int64)] = np.array(paths, np.int16)
                labelled += len(idxs)
        cur.close()
        con.execute("DETACH DATABASE forl")
        fr_path.tofile(os.path.join(REF, "for_path.bin"))
        cov = labelled / n
        report["sources"]["for"] = {"depth": 2, "coverage": round(cov, 6),
                                    "labelled": labelled, "path_bin": "for_path.bin"}
        print(f"for_path.bin: {labelled:,}/{n:,} labelled (coverage {cov:.4%})")
    else:
        print("(for_labels.sqlite not present — FoR is the optional BigQuery-gated reference; skipping)")

    con.close()
    json.dump(report, open(os.path.join(REF, "join_report.json"), "w"), indent=2)
    print(f"wrote join_report.json: {json.dumps(report)}")


if __name__ == "__main__":
    main()
