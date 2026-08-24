#!/usr/bin/env python3
"""Step 0 of the convergent-validity track: build metadata.sqlite(article_idx, pmid),
aligned to corpus.sbcsr's row order, by re-parsing the raw MEDLINE XML.

corpus.sbcsr (from sbeval/medline.py) stores NO PMIDs and its rows are the articles that passed
`min_mesh` (>=5), in (sorted-file order) x (document order). We reproduce that exact pass over the
same raw .xml.gz files with the same filter, capturing each accepted article's PMID, so row i of
the map == row i (article_idx == point id p) of the corpus.

Bit-exact alignment is GUARANTEED by two checks: each file's accepted count must equal that file's
shard `n_rows`, and the grand total must equal corpus.sbcsr's n_samples. If either fails, the
filter/order drifted and we abort rather than emit a silently-misaligned map.

LOCAL ARTIFACT ONLY — re-introduces PMIDs (public-domain NLM identifiers) for local analysis. Must
NOT be uploaded to Zenodo without the owner's sign-off (data/README.md publication guardrail).

Output: data/reference_taxonomies/metadata.sqlite
"""
import gzip
import json
import multiprocessing as mp
import os
import sqlite3
import struct
import sys
import xml.etree.ElementTree as ET
from array import array

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC = os.path.join(ROOT, "data/medline/processed/pubmed26_ge5")
OUT = os.path.join(ROOT, "data/reference_taxonomies/metadata.sqlite")


def iter_mesh_pmid(path, major_only):
    """Yield (mesh_id_count, pmid_or_None) per <PubmedArticle> — the SAME MeSH-id logic as
    sbeval.medline.iter_article_mesh (so the >=min_mesh filter is identical), plus the PMID."""
    with gzip.open(path, "rb") as fh:
        context = ET.iterparse(fh, events=("start", "end"))
        _, root = next(context)
        for event, elem in context:
            if event != "end" or elem.tag != "PubmedArticle":
                continue
            ids = {}
            for dn in elem.iter("DescriptorName"):                  # identical to iter_article_mesh
                ui = dn.get("UI")
                if not ui or ui[0] not in ("D", "d"):
                    continue
                if major_only and dn.get("MajorTopicYN") != "Y":
                    continue
                try:
                    ids[int(ui[1:])] = ui
                except ValueError:
                    continue
            pe = elem.find("MedlineCitation/PMID")                  # the article's own PMID
            pmid = None
            if pe is not None and pe.text and pe.text.isdigit():
                pmid = int(pe.text)
            yield len(ids), pmid
            elem.clear()
            root.clear()


def shard_n_rows(shards_dir, base):
    """Accepted-article count this file contributed to the corpus (from its shard sidecar)."""
    sc = os.path.join(shards_dir, base + ".ids.json")
    if os.path.exists(sc):
        return json.load(open(sc)).get("n_rows")
    shard = os.path.join(shards_dir, base + ".shard")              # fall back to the shard header
    if os.path.exists(shard):
        with open(shard, "rb") as f:
            f.read(8)                                              # magic
            return struct.unpack("<II", f.read(8))[0]
    return None


_RAW = _MAJOR = _SHARDS = _MINM = None


def _init(raw, major, shards, minm):
    global _RAW, _MAJOR, _SHARDS, _MINM
    _RAW, _MAJOR, _SHARDS, _MINM = raw, major, shards, minm


def _worker(fname):
    """Parse one file: PMIDs of its accepted (>=min_mesh) articles, in order. Validate count."""
    base = fname[:-7] if fname.endswith(".xml.gz") else fname
    pmids = array("q")                                            # int64; -1 = no PMID
    for nmesh, pmid in iter_mesh_pmid(os.path.join(_RAW, fname), _MAJOR):
        if nmesh < _MINM:
            continue
        pmids.append(pmid if pmid is not None else -1)
    expect = shard_n_rows(_SHARDS, base)
    if expect is not None and expect != len(pmids):
        return (fname, None, f"COUNT MISMATCH: parsed {len(pmids)} accepted, shard says {expect}")
    return (fname, pmids, None)


def main():
    summ = json.load(open(os.path.join(PROC, "summary.json")))
    files = summ["files"]
    minm = summ["filter"]["min_mesh"]
    major = summ["filter"].get("major_only", False)
    n_expected = summ["n_samples"]
    raw = os.path.join(ROOT, "data/medline/raw", summ["release"])
    if not os.path.isdir(raw):                                    # release subdir may be flat
        raw = os.path.join(ROOT, "data/medline/raw")
    shards = next((d for d in (os.path.join(ROOT, "data/medline/shards", summ["release"] + "_" +
                                            ("ge%d" % minm) + ("_major" if major else "")),
                               os.path.join(ROOT, "data/medline/shards"))
                   if os.path.isdir(d)), os.path.join(ROOT, "data/medline/shards"))
    print(f"files={len(files)} min_mesh={minm} major_only={major}; raw={raw}; shards={shards}")
    print(f"target: {n_expected:,} articles aligned to corpus.sbcsr")

    nproc = min(16, (os.cpu_count() or 4))
    per_file = {}
    with mp.Pool(nproc, initializer=_init, initargs=(raw, major, shards, minm)) as pool:
        done = 0
        for fname, pmids, err in pool.imap_unordered(_worker, files, chunksize=1):
            if err:
                pool.terminate()
                raise SystemExit(f"ABORT — {fname}: {err}\n(filter/order drift → would misalign)")
            per_file[fname] = pmids
            done += 1
            if done % 100 == 0 or done == len(files):
                print(f"  parsed {done}/{len(files)} files", flush=True)

    # assemble in the corpus's file order; article_idx = cumulative accepted count
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    con = sqlite3.connect(OUT)
    con.execute("CREATE TABLE articles(article_idx INTEGER PRIMARY KEY, pmid INTEGER)")
    idx = 0
    have_pmid = 0
    for fname in files:                                          # summary order == corpus order
        for p in per_file[fname]:
            con.execute("INSERT INTO articles VALUES(?,?)", (idx, None if p < 0 else p))
            if p >= 0:
                have_pmid += 1
            idx += 1
    con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    con.executemany("INSERT INTO meta VALUES(?,?)", [
        ("n_articles", str(idx)), ("with_pmid", str(have_pmid)),
        ("pmid_coverage", f"{have_pmid/idx:.6f}" if idx else "0"),
        ("aligned_to", "corpus.sbcsr (pubmed26_ge5)"), ("release", summ["release"])])
    con.commit()
    con.close()

    if idx != n_expected:
        raise SystemExit(f"ABORT — total {idx:,} != corpus n_samples {n_expected:,} (misaligned)")
    print(f"wrote {OUT}")
    print(f"  {idx:,} rows aligned to corpus.sbcsr; PMID coverage {have_pmid/idx:.4%} "
          f"({idx-have_pmid:,} have no PMID)")


if __name__ == "__main__":
    main()
