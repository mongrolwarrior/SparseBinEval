"""Real MEDLINE/PubMed ingestion for SparseBinEval.

Turns the NCBI PubMed *baseline* (annual full snapshot of MEDLINE) into a
SparseBinSOM `.sbcsr` corpus whose features are MeSH descriptors. Pure stdlib
(urllib + gzip + xml.etree + multiprocessing) — no third-party deps.

Only the *current* baseline is served by NCBI now (the per-year archives were
withdrawn), so there is no dataset-selection knob: the source is always the
latest baseline, auto-detected from the FTP listing. You can still restrict
*which files* of it to fetch (`max_files` / `files`), since the full baseline is
~1200 files (~40 GB, ~37M articles).

Pipeline (each stage is resumable — it skips work whose output already exists):
  1. fetch   — download `pubmedNNnXXXX.xml.gz` (+ `.md5`) into raw/<release>/,
               MD5-verified against the sidecar.
  2. process — parse each file in parallel into a per-file *shard* (raw MeSH ids
               in CSR form, already min-MeSH-filtered), then compact the shards
               into one corpus.sbcsr with a dense uint16 MeSH vocabulary.

On-disk layout under <root> (default <repo>/data/medline, git-ignored):
  raw/<release>/pubmedNNnXXXX.xml.gz(.md5)
  shards/<tag>/pubmedNNnXXXX.shard(.ids.json)     # intermediate, per file
  processed/<tag>/corpus.sbcsr | vocab.json | meta.json
where <release> is e.g. "pubmed26" and <tag> adds the filter, e.g.
"pubmed26_ge5" (or "..._ge5_major").
"""

import gzip
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import Pool

from . import host, sbcsr

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"

# Rough per-file sizes for capacity estimates, before we know the real numbers.
# A baseline file is ~37 MB compressed and yields ~28k articles surviving the
# default >=5-MeSH filter, ~12 MeSH terms each.
_BYTES_PER_FILE = 37 * 2**20
_ARTICLES_PER_FILE = 28_000
_MESH_PER_ARTICLE = 12
# A baseline filename looks like pubmed26n0001.xml.gz: "26" is the release
# (2026), "0001" the file's sequence number.
_FILE_RE = re.compile(r"pubmed(\d\d)n(\d+)\.xml\.gz")

# Intermediate shard format (little-endian): magic, uint32 n_rows, n_nnz,
# then uint32 row_ptr[n_rows+1] and uint32 vals[n_nnz] (raw MeSH ids, the
# numeric part of the descriptor UI before vocabulary compaction).
_SHARD_MAGIC = b"MSHARD1\x00"


def default_root():
    """Where corpora live by default: <repo>/data/medline (sibling of sbeval/)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "medline")


def corpus_tag(min_mesh, major_only):
    """Folder tag encoding the processing filter, e.g. 'ge5' or 'ge5_major'."""
    return f"ge{int(min_mesh)}" + ("_major" if major_only else "")


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def baseline_year(release):
    """'pubmed26' -> 2026 (the two release digits are the year-2000)."""
    m = re.search(r"(\d\d)$", release or "")
    return 2000 + int(m.group(1)) if m else None


def estimate_sizes(n_files):
    """Rough resource footprint for `n_files` baseline files:
    raw download bytes, and the in-memory/on-disk size of the resulting corpus
    (CSR = uint32 row pointers + uint16 columns). Returns a dict of byte counts."""
    articles = _ARTICLES_PER_FILE * n_files
    corpus = articles * 4 + articles * _MESH_PER_ARTICLE * 2     # row_ptr + col_idx
    return {"articles": articles, "raw_bytes": _BYTES_PER_FILE * n_files,
            "corpus_bytes": corpus}


def warn_capacity(n_files, root, assume_yes=False, log=print):
    """Warn (and let the user bail on a TTY) if this container looks too small to
    download / hold the dataset. Checks free disk against the raw download + the
    corpus, and available RAM against holding the corpus in memory. Returns True
    to proceed. Never hard-blocks: the warning is advisory, as requested."""
    est = estimate_sizes(n_files)
    ram = host.available_ram_bytes()
    disk = host.free_disk_bytes(root)
    # Raw files + corpus + intermediate shards (~corpus again), with headroom.
    need_disk = int((est["raw_bytes"] + 2 * est["corpus_bytes"]) * 1.15)
    need_ram = int(est["corpus_bytes"] * 2)             # corpus + working overhead
    tight = []
    if disk and disk < need_disk:
        tight.append(f"disk: need ~{host.human(need_disk)} free, have {host.human(disk)}")
    if ram and ram < need_ram:
        tight.append(f"memory: dataset wants ~{host.human(need_ram)}, "
                     f"available {host.human(ram)}")
    if not tight:
        return True
    log("WARNING: this machine may be too small for the selected dataset "
        f"({n_files} files, ~{est['articles']:,} articles):")
    for t in tight:
        log("  - " + t)
    log("  You can continue anyway; if it runs out of memory it will fail "
        "cleanly — re-run with fewer --max-files, a higher --min-mesh, or more RAM/disk.")
    if assume_yes or not sys.stdin.isatty():
        log("  Continuing (override).")
        return True
    return input("  Continue anyway? [y/N] ").strip().lower().startswith("y")


def write_download_summary(raw_dir, n_baseline=None, verify_md5=True):
    """Write/refresh download_summary.json in the raw folder: when it was fetched,
    which baseline, how many of the baseline's files are present, and whether the
    download is complete (all files) or partial."""
    release = os.path.basename(raw_dir.rstrip("/"))
    present = sorted(f for f in os.listdir(raw_dir) if f.endswith(".xml.gz"))
    total_bytes = sum(os.path.getsize(os.path.join(raw_dir, f)) for f in present)
    summary = {
        "release": release, "baseline_year": baseline_year(release),
        "source_url": BASE_URL, "downloaded_at": _now_iso(),
        "n_files_baseline": n_baseline, "n_files_present": len(present),
        "complete": (n_baseline is not None and len(present) >= n_baseline),
        "md5_verified": bool(verify_md5), "bytes_total": total_bytes,
    }
    with open(os.path.join(raw_dir, "download_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def read_download_summary(raw_dir):
    p = os.path.join(raw_dir, "download_summary.json")
    return json.load(open(p)) if os.path.isfile(p) else None


# ─── Listing the latest baseline ───────────────────────────────────────────────

def list_baseline(timeout=60):
    """Fetch the FTP directory listing and return (release, [filenames sorted]).

    `release` is the tag like 'pubmed26'. Only the current baseline is present."""
    with urllib.request.urlopen(BASE_URL, timeout=timeout) as r:
        html = r.read().decode("utf-8", "replace")
    seen = {}                       # filename -> (release_digits, seq) for sorting
    for m in _FILE_RE.finditer(html):
        seen[m.group(0)] = (m.group(1), int(m.group(2)))
    if not seen:
        raise RuntimeError(f"no baseline files found at {BASE_URL}")
    files = sorted(seen, key=lambda f: seen[f][1])
    release = "pubmed" + seen[files[0]][0]   # all files share one release digit-pair
    return release, files


def select_files(files, max_files=None, which=None):
    """Pick a subset of baseline files. `which` is a 1-based spec like '1-5,20'
    over the sorted list; `max_files` caps the count. None/None = all of them."""
    if which:
        idx = _parse_ranges(which, len(files))
        files = [files[i - 1] for i in idx]
    if max_files is not None:
        files = files[:max_files]
    return files


def _parse_ranges(spec, n):
    """'1-5,20' -> sorted unique 1-based indices, clamped to [1, n]."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            for i in range(int(a), int(b) + 1):
                out.add(i)
        else:
            out.add(int(part))
    return sorted(i for i in out if 1 <= i <= n)


# ─── Download ──────────────────────────────────────────────────────────────────

def _md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _expected_md5(sidecar_path):
    """Parse the NCBI '.md5' sidecar: 'MD5(file)= <hex>'."""
    with open(sidecar_path) as f:
        txt = f.read()
    m = re.search(r"([0-9a-fA-F]{32})", txt)
    return m.group(1).lower() if m else None


def _download(url, dest, timeout, retries=8):
    """Download `url` to `dest` atomically (via .part), with exponential backoff.

    NCBI throttles bulk access with HTTP 503, so retry persistently with growing
    backoff (honouring Retry-After when present) rather than giving up quickly."""
    last = None
    for attempt in range(retries):
        try:
            tmp = dest + ".part"
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    blk = r.read(1 << 20)
                    if not blk:
                        break
                    f.write(blk)
            os.replace(tmp, dest)
            return
        except urllib.error.HTTPError as e:          # 503/429 throttling: back off hard
            last = e
            wait = e.headers.get("Retry-After") if e.headers else None
            time.sleep(float(wait) if wait and wait.isdigit() else min(60.0, 2.0 * 2 ** attempt))
        except Exception as e:                       # noqa: BLE001 - retry any net error
            last = e
            time.sleep(min(30.0, 2.0 * 2 ** attempt))
    raise last


def fetch_one(fname, raw_dir, verify_md5=True, timeout=120):
    """Ensure one baseline file is present and (optionally) MD5-verified.
    Returns 'have' (already good), 'fetched', or raises. Resumable."""
    gz = os.path.join(raw_dir, fname)
    md5 = gz + ".md5"
    if verify_md5:
        if not os.path.exists(md5):
            _download(BASE_URL + fname + ".md5", md5, timeout)
        want = _expected_md5(md5)
        if os.path.exists(gz) and want and _md5_of(gz) == want:
            return "have"
        _download(BASE_URL + fname, gz, timeout)
        if want and _md5_of(gz) != want:
            os.remove(gz)
            raise RuntimeError(f"MD5 mismatch for {fname}")
        return "fetched"
    if os.path.exists(gz):
        return "have"
    _download(BASE_URL + fname, gz, timeout)
    return "fetched"


def _fetch_pass(filenames, raw_dir, verify_md5, workers, timeout, log):
    """One download pass. Returns the list of files that still failed."""
    have = fetched = 0
    failed = []
    done = 0
    n = len(filenames)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, f, raw_dir, verify_md5, timeout): f for f in filenames}
        for fut in as_completed(futs):
            done += 1
            try:
                st = fut.result()
                have += st == "have"
                fetched += st == "fetched"
            except Exception as e:                   # noqa: BLE001
                failed.append(futs[fut])
                log(f"  ! {futs[fut]}: {e}")
            if done % max(1, n // 100) == 0 or done == n:
                log(f"  download {done}/{n}  (have {have}, fetched {fetched}, failed {len(failed)})")
    return failed


def fetch(filenames, raw_dir, verify_md5=True, workers=4, timeout=120, log=print,
          max_passes=6, n_baseline=None, assume_yes=False):
    """Download `filenames` into raw_dir, low-concurrency to avoid NCBI throttling.

    Re-passes over any files that failed (with backoff between passes) until none
    remain or max_passes is hit — so a partial download never silently truncates
    the corpus. Raises if files are still missing after the last pass. Writes
    download_summary.json (complete/partial) even when it bails part-way."""
    os.makedirs(raw_dir, exist_ok=True)
    if not warn_capacity(len(filenames), raw_dir, assume_yes=assume_yes, log=log):
        raise SystemExit("aborted: insufficient capacity for the selected dataset")
    pending = list(filenames)
    try:
        for p in range(max_passes):
            if not pending:
                break
            if p:
                log(f"  retry pass {p}: {len(pending)} files still missing; backing off...")
                time.sleep(min(120.0, 10.0 * p))
            pending = _fetch_pass(pending, raw_dir, verify_md5, workers, timeout, log)
    finally:
        write_download_summary(raw_dir, n_baseline=n_baseline, verify_md5=verify_md5)
    if pending:
        raise RuntimeError(f"{len(pending)} files failed to download after {max_passes} passes: "
                           f"{pending[:5]}{'...' if len(pending) > 5 else ''}")
    log(f"  all {len(filenames)} files present in {raw_dir}")
    return {"n": len(filenames)}


# ─── Parse one .xml.gz → MeSH sets ─────────────────────────────────────────────

def iter_article_mesh(path, major_only=False):
    """Stream a PubMed .xml.gz, yielding one {raw_id: ui_string} dict per article.

    A MeSH heading is `<DescriptorName UI="D000123" MajorTopicYN="Y">`; raw_id is
    the integer after the 'D'. `major_only` keeps only MajorTopicYN=Y headings.
    Memory stays flat by clearing each parsed article and the accumulating root."""
    with gzip.open(path, "rb") as fh:
        context = ET.iterparse(fh, events=("start", "end"))
        _, root = next(context)                      # the <PubmedArticleSet> root
        for event, elem in context:
            if event != "end" or elem.tag != "PubmedArticle":
                continue
            ids = {}
            for dn in elem.iter("DescriptorName"):
                ui = dn.get("UI")
                if not ui or ui[0] not in ("D", "d"):
                    continue
                if major_only and dn.get("MajorTopicYN") != "Y":
                    continue
                try:
                    ids[int(ui[1:])] = ui
                except ValueError:
                    continue
            yield ids
            elem.clear()
            root.clear()                             # drop the spent article reference


def _swap_if_big_endian(*arrays):
    if sys.byteorder != "little":
        for a in arrays:
            a.byteswap()


def _write_shard(path, rows):
    """Write CSR of raw MeSH ids (rows = list of sorted id-lists). Returns (n_rows, n_nnz)."""
    row_ptr = array("I", [0])
    vals = array("I")
    for r in rows:
        vals.extend(r)
        row_ptr.append(len(vals))
    _swap_if_big_endian(row_ptr, vals)
    with open(path, "wb") as f:
        f.write(_SHARD_MAGIC)
        f.write(struct.pack("<II", len(rows), len(vals)))
        f.write(row_ptr.tobytes())
        f.write(vals.tobytes())
    return len(rows), len(vals)


def _read_shard(path):
    """Read a shard back as (row_ptr array('I'), vals array('I'))."""
    with open(path, "rb") as f:
        magic = f.read(len(_SHARD_MAGIC))
        if magic != _SHARD_MAGIC:
            raise RuntimeError(f"bad shard magic in {path}")
        n_rows, n_nnz = struct.unpack("<II", f.read(8))
        row_ptr = array("I")
        row_ptr.frombytes(f.read(4 * (n_rows + 1)))
        vals = array("I")
        vals.frombytes(f.read(4 * n_nnz))
    if sys.byteorder != "little":
        row_ptr.byteswap()
        vals.byteswap()
    return row_ptr, vals


# A multiprocessing worker: parse one file into a shard + an id→UI sidecar.
def _parse_worker(task):
    src, shard, sidecar, major_only, min_mesh = task
    if os.path.exists(shard) and os.path.exists(sidecar):
        meta = json.load(open(sidecar))
        return shard, meta["n_rows"], meta["n_nnz"], meta["id2ui"]
    rows = []
    id2ui = {}
    for ids in iter_article_mesh(src, major_only):
        if len(ids) < min_mesh:                      # the --min-mesh filter
            continue
        rows.append(sorted(ids.keys()))
        id2ui.update(ids)
    n_rows, n_nnz = _write_shard(shard, rows)
    id2ui = {str(k): v for k, v in id2ui.items()}
    json.dump({"n_rows": n_rows, "n_nnz": n_nnz, "id2ui": id2ui}, open(sidecar, "w"))
    return shard, n_rows, n_nnz, id2ui


# Compaction worker: remap one shard's raw ids to dense vocab indices using a
# lookup table shared via the pool initializer (avoids re-pickling it per task).
_LUT = None


def _init_compact(lut):
    global _LUT
    _LUT = lut


def _compact_worker(shard):
    row_ptr, vals = _read_shard(shard)
    lut = _LUT
    col = array("H", (lut[v] for v in vals))         # raw id -> dense uint16 feature
    _swap_if_big_endian(col)
    lengths = array("I", (row_ptr[i + 1] - row_ptr[i] for i in range(len(row_ptr) - 1)))
    return col.tobytes(), lengths


# ─── Orchestration ─────────────────────────────────────────────────────────────

def process(filenames, release, root, min_mesh=5, major_only=False,
            workers=None, log=print, n_baseline=None, assume_yes=False):
    """Parse `filenames` (already in raw/<release>/) into processed/<tag>/corpus.sbcsr.

    Returns the corpus path. Resumable: existing shards are reused. Writes a
    dataset summary.json (baseline / filter / when downloaded+processed / whether
    the underlying download is complete or partial)."""
    workers = workers or min(28, (os.cpu_count() or 4))
    raw_dir = os.path.join(root, "raw", release)
    tag = release + "_" + corpus_tag(min_mesh, major_only)
    shard_dir = os.path.join(root, "shards", tag)
    out_dir = os.path.join(root, "processed", tag)
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    absent = [f for f in filenames if not os.path.exists(os.path.join(raw_dir, f))]
    if absent:
        raise RuntimeError(f"{len(absent)} requested files are not downloaded "
                           f"(e.g. {absent[:3]}); run fetch first")
    if not warn_capacity(len(filenames), out_dir, assume_yes=assume_yes, log=log):
        raise SystemExit("aborted: insufficient capacity to build the corpus")

    tasks = [(os.path.join(raw_dir, f),
              os.path.join(shard_dir, f.replace(".xml.gz", ".shard")),
              os.path.join(shard_dir, f.replace(".xml.gz", ".ids.json")),
              major_only, min_mesh) for f in filenames]

    # Stage A — parse files in parallel into shards.
    log(f"parsing {len(tasks)} files with {workers} workers (min_mesh={min_mesh}, "
        f"{'major-topic only' if major_only else 'all MeSH'}) ...")
    t0 = time.monotonic()
    results = []
    id2ui = {}
    done = 0
    with Pool(workers) as pool:
        for shard, n_rows, n_nnz, d in pool.imap_unordered(_parse_worker, tasks):
            results.append((shard, n_rows, n_nnz))
            for k, v in d.items():
                id2ui[int(k)] = v
            done += 1
            if done % max(1, len(tasks) // 100) == 0 or done == len(tasks):
                log(f"  parsed {done}/{len(tasks)} files  "
                    f"({sum(r[1] for r in results):,} articles, {time.monotonic()-t0:.0f}s)")
    results.sort(key=lambda r: r[0])                 # stable row order by filename

    total_rows = sum(r[1] for r in results)
    total_nnz = sum(r[2] for r in results)
    vocab_ids = sorted(id2ui)                        # raw ids ascending -> dense index
    if not vocab_ids:
        raise RuntimeError("no articles passed the filter; nothing to write")
    if len(vocab_ids) > 65535:
        raise RuntimeError(f"vocabulary {len(vocab_ids)} exceeds uint16 (.sbcsr) limit")
    index = {rid: i for i, rid in enumerate(vocab_ids)}
    log(f"vocabulary: {len(vocab_ids):,} MeSH descriptors; "
        f"corpus: {total_rows:,} articles, {total_nnz:,} nnz")

    # Stage B — compact shards into the final .sbcsr, remapping in parallel. The
    # big allocations (row_ptr, col bytes) live here; on a too-small container
    # this is where memory runs out, so fail cleanly with guidance.
    corpus = os.path.join(out_dir, "corpus.sbcsr")
    sumsq = 0                                            # Σ(row nnz)² for the std-dev
    try:
        lut = array("i", [-1]) * (max(vocab_ids) + 1)
        for rid, i in index.items():
            lut[rid] = i
        row_ptr = array("I", [0])
        running = 0
        shard_paths = [r[0] for r in results]
        with open(corpus, "wb") as f:
            f.write(sbcsr._MAGIC)
            f.write(struct.pack("<IIII", total_rows, len(vocab_ids), total_nnz, 0))
            hdr_end = f.tell()
            f.seek(hdr_end + 4 * (total_rows + 1))       # room for row_ptr; col_idx first
            with Pool(workers, initializer=_init_compact, initargs=(lut,)) as pool:
                for col_bytes, lengths in pool.imap(_compact_worker, shard_paths):
                    f.write(col_bytes)
                    for L in lengths:
                        running += L
                        sumsq += L * L
                        row_ptr.append(running)
            _swap_if_big_endian(row_ptr)
            f.seek(hdr_end)
            f.write(row_ptr.tobytes())
    except MemoryError:
        if os.path.exists(corpus):
            os.remove(corpus)                            # don't leave a truncated corpus
        raise SystemExit(
            "out of memory while building the corpus. Re-run with fewer --max-files, "
            "a higher --min-mesh, or on a machine with more RAM "
            f"(needed ~{host.human(estimate_sizes(len(filenames))['corpus_bytes']*2)}).")

    mean = total_nnz / total_rows
    nnz_std = round((sumsq / total_rows - mean * mean) ** 0.5, 4)
    vocab = [id2ui[rid] for rid in vocab_ids]            # feature index -> MeSH UI
    json.dump(vocab, open(os.path.join(out_dir, "vocab.json"), "w"))

    # If fetch didn't leave a download summary (e.g. an older run), synthesise one
    # now from what's on disk so provenance/partial flags are still accurate.
    dl = read_download_summary(raw_dir)
    if dl is None:
        dl = write_download_summary(raw_dir, n_baseline=n_baseline)
    n_base = n_baseline if n_baseline is not None else dl.get("n_files_baseline")
    raw_complete = dl.get("complete")
    # Partial if the raw download is incomplete, or we used fewer than all files.
    partial = (raw_complete is False) or (n_base is not None and len(filenames) < n_base)
    summary = {
        "release": release, "baseline_year": baseline_year(release), "source_url": BASE_URL,
        "filter": {"min_mesh": min_mesh, "major_only": major_only},
        "n_files_used": len(filenames), "n_files_baseline": n_base,
        "raw_complete": raw_complete, "partial": bool(partial),
        "downloaded_at": dl.get("downloaded_at"), "processed_at": _now_iso(),
        "n_samples": total_rows, "n_features": len(vocab_ids), "n_nonzeros": total_nnz,
        "nnz_mean": round(mean, 4), "nnz_std": nnz_std,
        "corpus_file": "corpus.sbcsr", "vocab_file": "vocab.json", "files": filenames,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    flag = "  [PARTIAL]" if partial else ""
    log(f"wrote {corpus} ({os.path.getsize(corpus)/2**20:.1f} MiB), vocab.json, summary.json{flag}")
    return corpus


def status(root, min_mesh=5, major_only=False):
    """Report what's present on disk for the latest release + the requested filter."""
    info = {"root": root, "releases_raw": [], "processed": []}
    raw_base = os.path.join(root, "raw")
    if os.path.isdir(raw_base):
        for rel in sorted(os.listdir(raw_base)):
            n = len([f for f in os.listdir(os.path.join(raw_base, rel)) if f.endswith(".xml.gz")])
            info["releases_raw"].append((rel, n, read_download_summary(os.path.join(raw_base, rel))))
    proc_base = os.path.join(root, "processed")
    if os.path.isdir(proc_base):
        for tag in sorted(os.listdir(proc_base)):
            mp = os.path.join(proc_base, tag, "summary.json")
            if os.path.isfile(mp):
                info["processed"].append((tag, json.load(open(mp))))
    return info


def corpus_summary(corpus_path):
    """Load the dataset summary.json sitting next to a corpus.sbcsr, or None.
    Lets sweep outputs flag the corpus's baseline / filter / partial status."""
    p = os.path.join(os.path.dirname(os.path.abspath(corpus_path)), "summary.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def ensure(root=None, min_mesh=5, major_only=False, max_files=None, files=None,
           do_download=False, assume_yes=False, workers=None, verify_md5=True, log=print):
    """Return a ready corpus.sbcsr path for the latest baseline + filter, building it
    if needed. Mirrors the requested default behaviour: check the expected place,
    and offer to download + process when it's missing.

    do_download=True downloads without asking; otherwise, on a TTY, prompt; off a
    TTY (or with --no-download), report what's missing and raise."""
    root = root or default_root()
    release, all_files = list_baseline()
    want = select_files(all_files, max_files=max_files, which=files)
    tag = release + "_" + corpus_tag(min_mesh, major_only)
    corpus = os.path.join(root, "processed", tag, "corpus.sbcsr")
    if os.path.isfile(corpus):
        log(f"corpus present: {corpus}")
        return corpus

    raw_dir = os.path.join(root, "raw", release)
    missing = [f for f in want if not os.path.exists(os.path.join(raw_dir, f))]
    log(f"latest baseline: {release} ({len(all_files)} files); selected {len(want)}; "
        f"{len(missing)} not yet downloaded; processed corpus '{tag}' absent.")

    if missing and not do_download:
        if assume_yes or (sys.stdin.isatty() and
                          input(f"Download {len(missing)} files and process? [y/N] ")
                          .strip().lower().startswith("y")):
            do_download = True
        else:
            raise SystemExit("corpus missing; re-run with download enabled (--download/--yes)")

    if missing:
        fetch(want, raw_dir, verify_md5=verify_md5, workers=(workers or 4), log=log,
              n_baseline=len(all_files), assume_yes=assume_yes)
    return process(want, release, root, min_mesh=min_mesh, major_only=major_only,
                   workers=workers, log=log, n_baseline=len(all_files), assume_yes=assume_yes)
