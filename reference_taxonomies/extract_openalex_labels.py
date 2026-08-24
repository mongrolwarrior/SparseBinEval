#!/usr/bin/env python3
"""Step 2: stream-filter the OpenAlex `works` snapshot against our PMID set → openalex_labels.sqlite.

RUN ON THE HOST — the works snapshot is hundreds of GB gzipped; this container can't hold it. Sync
(or mount) the partitions and point $SNAP at the `works/` dir:
    aws s3 sync s3://openalex/data/works/ "$SNAP" --no-sign-request        # on the host
    SNAP="$SNAP" python3 extract_openalex_labels.py

For each work whose PMID is in our corpus it records the primary-topic dense path plus the full
multi-label topic set (mirrors the MeSH primary/multi-label split). Validate the parse + the join
end-to-end WITHOUT the snapshot via:  python3 extract_openalex_labels.py --selftest

⚠️ Verify before a full run (handover callouts): the S3 path (docs.openalex.org/download-all-data)
and that `ids.pmid` is still the pubmed URL form.
"""
import glob
import gzip
import multiprocessing as mp
import os
import re
import sqlite3
import sys

import orjson

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REF = os.path.join(ROOT, "data/reference_taxonomies")
META = os.path.join(REF, "metadata.sqlite")
TREE = os.path.join(REF, "openalex_tree.sqlite")
OUT = os.path.join(REF, "openalex_labels.sqlite")

_PMID_RE = re.compile(r'(\d+)$')
_T_RE = re.compile(r'/T(\d+)$')


def pmid_of(work):
    u = (work.get("ids") or {}).get("pmid")          # e.g. https://pubmed.ncbi.nlm.nih.gov/12345678
    if not u:
        return None
    m = _PMID_RE.search(u)
    return int(m.group(1)) if m else None


def topic_int(turl):
    m = _T_RE.search(turl)
    return int(m.group(1)) if m else None


def load_tpath():
    tc = sqlite3.connect(TREE)
    tp = {r[0]: (r[1], r[2], r[3], r[4]) for r in tc.execute(
        "SELECT topic_oa,domain_dense,field_dense,subfield_dense,topic_dense FROM topic_path")}
    tc.close()
    return tp


def extract_one(work, pmids, TPATH):
    """→ (pmid, path4, topics_json) for a matching, classified work; else None."""
    p = pmid_of(work)
    if p is None or p not in pmids:
        return None
    pt = work.get("primary_topic")
    if not pt:
        return None                                   # unclassified work
    path = TPATH.get(topic_int(pt["id"] or ""))
    if not path:
        return None
    multi = [{"t": topic_int(t["id"]), "s": t.get("score")} for t in (work.get("topics") or [])]
    return p, path, orjson.dumps(multi).decode()


# ── snapshot streaming (host) ────────────────────────────────────────────────────
_PMIDS = _TPATH = None
PMID_BITMAP = os.path.join(REF, "pmids.roaring")     # prebuilt once; workers deserialize (instant)


def build_pmid_bitmap():
    from pyroaring import BitMap
    mc = sqlite3.connect(META)
    bm = BitMap(r[0] for r in mc.execute("SELECT pmid FROM articles WHERE pmid IS NOT NULL"))
    mc.close()
    with open(PMID_BITMAP, "wb") as f:
        f.write(bm.serialize())
    print(f"prebuilt PMID bitmap: {len(bm):,} ids ({os.path.getsize(PMID_BITMAP)/1e6:.0f} MB)")


def _init(_=None):
    global _PMIDS, _TPATH
    from pyroaring import BitMap
    with open(PMID_BITMAP, "rb") as f:
        _PMIDS = BitMap.deserialize(f.read())        # ~instant vs re-reading 30M sqlite rows
    _TPATH = load_tpath()


def _worker(fn):
    rows, seen = [], set()
    with gzip.open(fn) as fh:
        for line in fh:
            w = orjson.loads(line)
            r = extract_one(w, _PMIDS, _TPATH)
            if r and r[0] not in seen:               # rare duplicate works per PMID → keep first
                seen.add(r[0])
                rows.append(r)
    return fn, rows


def run_snapshot(snap):
    parts = sorted(glob.glob(os.path.join(snap, "**", "*.gz"), recursive=True))
    if not parts:
        raise SystemExit(f"no *.gz under {snap} — set $SNAP to the works/ dir")
    print(f"{len(parts)} partitions under {snap}")
    build_pmid_bitmap()
    if os.path.exists(OUT):
        os.remove(OUT)
    out = sqlite3.connect(OUT)
    out.executescript("""CREATE TABLE lab(pmid INTEGER PRIMARY KEY,
        domain_dense INT, field_dense INT, subfield_dense INT, topic_dense INT, topics_json TEXT);""")
    n = dup = 0
    seen_global = set()
    with mp.Pool(min(16, os.cpu_count() or 4), initializer=_init, initargs=(META,)) as pool:
        for i, (fn, rows) in enumerate(pool.imap_unordered(_worker, parts, chunksize=1), 1):
            for p, path, tj in rows:
                if p in seen_global:
                    dup += 1
                    continue
                seen_global.add(p)
                out.execute("INSERT INTO lab VALUES(?,?,?,?,?,?)", (p, *path, tj))
                n += 1
            if i % 50 == 0 or i == len(parts):
                print(f"  {i}/{len(parts)} partitions, matched {n:,}", flush=True)
    out.commit()
    out.close()
    print(f"wrote {OUT}: {n:,} PMIDs labelled, {dup} cross-partition dup works dropped")


# ── self-test: synthetic parse + a small real API join (no snapshot) ─────────────
def selftest():
    TPATH = load_tpath()
    any_topic = next(iter(TPATH))                     # a real topic id from our tree
    w = {"ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/12345678"},
         "primary_topic": {"id": f"https://openalex.org/T{any_topic}"},
         "topics": [{"id": f"https://openalex.org/T{any_topic}", "score": 0.91}]}
    r = extract_one(w, {12345678}, TPATH)
    assert r and r[0] == 12345678 and r[1] == TPATH[any_topic], "synthetic parse failed"
    print(f"synthetic: pmid={r[0]} path={r[1]} topics={r[2]}  OK")

    if not os.path.exists(META):
        print("(metadata.sqlite not built yet — skipping the live API join check)")
        return
    import requests
    mc = sqlite3.connect(META)
    # recent PMIDs (high values) are far likelier to carry an OpenAlex topic than 1960s ones
    sample = [r[0] for r in mc.execute(
        "SELECT pmid FROM articles WHERE pmid IS NOT NULL ORDER BY pmid DESC LIMIT 8")]
    mc.close()
    print(f"live API join on {len(sample)} recent PMIDs:")
    ok = 0
    for p in sample:
        try:
            w = requests.get(f"https://api.openalex.org/works/pmid:{p}",
                             params={"mailto": "mongrolwarrior@gmail.com"}, timeout=30).json()
        except Exception as e:
            print(f"  pmid {p}: API error {e}"); continue
        r = extract_one(w, {p}, TPATH)
        if r:
            ok += 1
            d, f, s, t = r[1]
            print(f"  pmid {p} -> domain={d} field={f} subfield={s} topic={t}  ({w.get('primary_topic',{}).get('display_name')})")
        else:
            print(f"  pmid {p}: no primary_topic / not in OpenAlex")
    print(f"joined {ok}/{len(sample)} — end-to-end PMID→OpenAlex→tree path works")


# ── S3 streaming (in-container; no disk landing) ─────────────────────────────────
import subprocess

MANIFEST = "s3://openalex/data/works/manifest"


def s3_partition_urls():
    raw = subprocess.run(["aws", "s3", "cp", MANIFEST, "-", "--no-sign-request"],
                         capture_output=True).stdout
    return [e["url"] for e in orjson.loads(raw)["entries"]]


def _s3_worker(url):
    """Stream one partition straight from S3 (download-and-discard), filter to our PMIDs. Retries
    transient failures so a 2k-partition job survives network blips."""
    import gzip
    for _ in range(4):
        try:
            rows = []
            proc = subprocess.Popen(["aws", "s3", "cp", url, "-", "--no-sign-request"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            with gzip.GzipFile(fileobj=proc.stdout) as fh:
                for line in fh:
                    r = extract_one(orjson.loads(line), _PMIDS, _TPATH)
                    if r:
                        rows.append(r)
            if proc.wait() == 0:
                return url, rows, None
        except Exception as e:                       # noqa: BLE001 — retry any stream error
            last = str(e)
        else:
            last = f"aws exit {proc.returncode}"
    return url, [], last


def run_s3():
    urls = s3_partition_urls()
    print(f"{len(urls)} partitions in the OpenAlex works manifest (streaming, no landing)")
    build_pmid_bitmap()                              # once; workers deserialize it in _init
    if os.path.exists(OUT):
        os.remove(OUT)
    out = sqlite3.connect(OUT)
    out.executescript("""CREATE TABLE lab(pmid INTEGER PRIMARY KEY,
        domain_dense INT, field_dense INT, subfield_dense INT, topic_dense INT, topics_json TEXT);""")
    n = failed = 0
    with mp.Pool(min(16, os.cpu_count() or 4), initializer=_init, initargs=(META,)) as pool:
        for i, (url, rows, err) in enumerate(pool.imap_unordered(_s3_worker, urls, chunksize=1), 1):
            if err:
                failed += 1
                print(f"  ⚠️ partition failed ({failed}): {url.rsplit('/',2)[-2]}/… — {err}", flush=True)
            for p, path, tj in rows:
                out.execute("INSERT OR IGNORE INTO lab VALUES(?,?,?,?,?,?)", (p, *path, tj))
                n += 1
            if i % 25 == 0 or i == len(urls):
                out.commit()
                print(f"  {i}/{len(urls)} partitions, matched {n:,} (failed {failed})", flush=True)
    real = out.execute("SELECT COUNT(*) FROM lab").fetchone()[0]
    out.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    out.executemany("INSERT INTO meta VALUES(?,?)",
                    [("labelled_pmids", str(real)), ("partitions", str(len(urls))),
                     ("partitions_failed", str(failed)), ("source", "OpenAlex works S3 snapshot")])
    out.commit()
    out.close()
    print(f"wrote {OUT}: {real:,} distinct PMIDs labelled; {failed} partitions failed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--s3" in sys.argv:
        run_s3()
    else:
        snap = os.environ.get("SNAP")
        if not snap:
            raise SystemExit("set $SNAP to a local works/ dir, pass --s3 to stream, or --selftest")
        run_snapshot(snap)
