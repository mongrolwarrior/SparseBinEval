"""Comparison harness: run somoclu's sparse-CPU SOM on the same MEDLINE corpus and map
sizes as SparseBinSOM, and score both with the common cosine evaluator (tools/metrics).

Goal: show whether SparseBinSOM (GPU sparse) gives similar/superior quality at improved
efficiency vs somoclu (multicore-CPU sparse — somoclu's GPU kernel is dense-only, and dense
is infeasible at ~30k MeSH features). Both are batch SOMs with a dense K×V codebook, so the
fair knobs are: same corpus, same map sizes, same per-size epoch budget (taken from the
sbsom sweep), a matched radius schedule (σ₀ = frac·edge → σ_min, exponential cooling), and
ONE metric evaluator (cosine QE + topographic error) applied to both codebooks.

Pure stdlib here; the heavy lifting is in the somoclu binary and the C++ metrics tool.
"""

import csv
import json
import os
import subprocess
import time

from . import sbcsr

DEFAULT_SOMOCLU = "/workspaces/somoclu/somoclu_cpu"
DEFAULT_METRICS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "tools", "metrics")


def to_libsvm(sbcsr_path, out_path, log=print):
    """Convert a .sbcsr corpus to somoclu's libsvm sparse text (zero-indexed 'col:1' per row,
    binary presence). Returns the number of rows written. Skips if up-to-date."""
    if os.path.exists(out_path) and os.path.getmtime(out_path) >= os.path.getmtime(sbcsr_path):
        log(f"  libsvm cache up-to-date: {out_path}")
        with open(out_path, "rb") as f:                 # count rows cheaply
            return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))
    n = 0
    t0 = time.monotonic()
    with open(out_path, "w") as out:
        buf = []
        for cols in sbcsr.iter_rows(sbcsr_path):
            buf.append(" ".join(f"{c}:1" for c in cols))
            n += 1
            if len(buf) >= 50000:
                out.write("\n".join(buf) + "\n"); buf.clear()
        if buf:
            out.write("\n".join(buf) + "\n")
    log(f"  wrote {out_path} ({n:,} rows, {os.path.getsize(out_path)/2**20:.0f} MiB, "
        f"{time.monotonic()-t0:.0f}s)")
    return n


def radius_for(edge, sigma_frac, sigma_min):
    """somoclu start/end radius mirroring sbsom's σ schedule: σ₀ = max(round(frac·edge), σ_min)."""
    r0 = max(round(sigma_frac * edge), sigma_min)
    return float(r0), float(sigma_min)


def _score(metrics_bin, corpus_sbcsr, codebook, fmt, rows, cols, sample_eval, seed):
    """Run the common cosine evaluator on a codebook (.wts or .somw); return parsed metrics."""
    cmd = [metrics_bin, "--corpus", corpus_sbcsr, "--codebook", codebook, "--format", fmt,
           "--rows", str(rows), "--cols", str(cols), "--seed", str(seed)]
    if sample_eval:
        cmd += ["--sample", str(sample_eval)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": (p.stderr.strip().splitlines() or ["metrics failed"])[-1]}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": "unparseable metrics output"}


def run_one(somoclu_bin, libsvm, edge, epochs, out_dir, sigma_frac, sigma_min,
            metrics_bin, corpus_sbcsr, sample_eval, threads, timeout, keep_weights, log):
    """Train one somoclu map, time it, score it with the common evaluator. Returns a dict row."""
    r0, rN = radius_for(edge, sigma_frac, sigma_min)
    prefix = os.path.join(out_dir, f"somoclu_{edge}")
    cmd = [somoclu_bin, "-k", "2", "-x", str(edge), "-y", str(edge), "-e", str(epochs),
           "-r", str(r0), "-R", str(rN), "-t", "exponential", "-m", "planar", libsvm, prefix]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads)) if threads else os.environ
    row = {"edge": edge, "neurons": edge * edge, "epochs": epochs,
           "radius0": r0, "radiusN": rN, "status": "ok", "wall_seconds": 0.0,
           "qe_cosine": None, "topographic_error": None, "notes": ""}
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        row.update(status="timeout", wall_seconds=time.monotonic() - t0, notes="timeout")
        return row
    row["wall_seconds"] = round(time.monotonic() - t0, 2)
    blob = (p.stdout + p.stderr).lower()
    if p.returncode != 0:
        row.update(status="oom" if ("bad_alloc" in blob or "out of memory" in blob) else "error",
                   notes=(p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1][:200])
        return row
    wts = prefix + ".wts"
    if os.path.exists(wts):
        m = _score(metrics_bin, corpus_sbcsr, wts, "wts", edge, edge, sample_eval, seed=0)
        row["qe_cosine"] = m.get("qe_cosine")
        row["topographic_error"] = m.get("topographic_error")
        if m.get("error"):
            row["notes"] = "metrics: " + m["error"]
    if not keep_weights:                                # somoclu's dense .wts is huge at scale
        for ext in (".wts", ".umx", ".bm"):
            try: os.remove(prefix + ext)
            except OSError: pass
    return row


def sizes_from_sweep(summary_path):
    """Read (edge, epochs) and sbsom's own metrics from a SparseBinSOM sweep summary.json.
    Only successful ('ok') runs are returned, in ascending edge order."""
    s = json.load(open(summary_path))
    out = []
    for r in s.get("runs", []):
        if r.get("status") == "ok":
            out.append({"edge": r["edge"], "epochs": r.get("epochs") or 0,
                        "sbsom_wall": r.get("wall_seconds"), "sbsom_qe": r.get("final_qe"),
                        "sbsom_te": r.get("topographic_error")})
    return sorted(out, key=lambda r: r["edge"])


def run_compare(somoclu_bin, metrics_bin, corpus_sbcsr, libsvm, sizes, out_dir,
                sigma_frac=0.5, sigma_min=0.5, sample_eval=50000, threads=None,
                timeout=None, keep_weights=False, max_edge=None, sbsom_weights_dir=None,
                log=print):
    """Run somoclu over `sizes` (list of {edge, epochs, ...}); write compare.csv + summary.json.

    If sbsom_weights_dir holds weights_<edge>.somw codebooks, they are scored with the SAME
    cosine evaluator so the sbsom side reports cosine QE/TE comparable to somoclu's (sbsom's
    own summary QE is in its internal distance, kept separately as sbsom_qe)."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for sz in sizes:
        if max_edge and sz["edge"] > max_edge:
            log(f"edge {sz['edge']}: skipped (> --max-edge {max_edge})")
            continue
        log(f"edge {sz['edge']} (x{sz['edge']}={sz['edge']**2} neurons), {sz['epochs']} epochs ...")
        r = run_one(somoclu_bin, libsvm, sz["edge"], sz["epochs"], out_dir, sigma_frac, sigma_min,
                    metrics_bin, corpus_sbcsr, sample_eval, threads, timeout, keep_weights, log)
        r["sbsom_wall"] = sz.get("sbsom_wall"); r["sbsom_qe"] = sz.get("sbsom_qe")
        r["sbsom_te"] = sz.get("sbsom_te")
        r["sbsom_qe_cosine"] = r["sbsom_te_cosine"] = None
        if sbsom_weights_dir:
            somw = os.path.join(sbsom_weights_dir, f"weights_{sz['edge']}.somw")
            if os.path.exists(somw):
                m = _score(metrics_bin, corpus_sbcsr, somw, "somw", sz["edge"], sz["edge"],
                           sample_eval, seed=0)
                r["sbsom_qe_cosine"] = m.get("qe_cosine")
                r["sbsom_te_cosine"] = m.get("topographic_error")
        rows.append(r)
        r["sbsom_speedup"] = (round(r["sbsom_wall"] / r["wall_seconds"], 2)
                              if r.get("sbsom_wall") and r["wall_seconds"] else None)
        spd = f"{r['sbsom_speedup']}x" if r["sbsom_speedup"] else "?"
        log(f"  somoclu {r['status']} {r['wall_seconds']}s QE={r['qe_cosine']} TE={r['topographic_error']}"
            f"  | sbsom {r.get('sbsom_wall')}s  (sbsom speedup {spd})")
        if r["status"] in ("oom", "error"):
            log("  stopping (somoclu capacity reached or error).")
            break

    fields = ["edge", "neurons", "epochs", "radius0", "radiusN", "status", "wall_seconds",
              "sbsom_wall", "sbsom_speedup", "qe_cosine", "topographic_error",
              "sbsom_qe_cosine", "sbsom_te_cosine", "sbsom_qe", "sbsom_te", "notes"]
    with open(os.path.join(out_dir, "compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    prov = None
    sp = os.path.join(os.path.dirname(os.path.abspath(corpus_sbcsr)), "summary.json")
    if os.path.isfile(sp):
        prov = json.load(open(sp))
    summary = {
        "comparison": "SparseBinSOM (GPU sparse) vs somoclu (CPU sparse)",
        "corpus": corpus_sbcsr, "corpus_provenance": prov,
        "metric": "cosine (QE = mean 1-cos to BMU1; TE = 1st/2nd BMU not 8-adjacent)",
        "somoclu": {"binary": somoclu_bin, "kernel": "sparse CPU (-k 2)",
                    "threads": threads or "default", "radius_frac": sigma_frac,
                    "radius_min": sigma_min, "cooling": "exponential"},
        "sample_eval": sample_eval, "runs": rows,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {out_dir}/compare.csv and summary.json")
    return rows
