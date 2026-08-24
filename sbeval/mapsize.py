"""Map-size sweep for SparseBinSOM.

Square maps only: a "size" is the edge length L (the map is L×L = L² neurons). The sweep
starts at `start`, and each step multiplies the NODE count by `factor` (default 2 — double
the nodes), re-deriving the edge as round(sqrt(nodes)). It ends at `finish` (edge), or:
  - finish given            → sweep start..finish.
  - predict_oom = True      → predict the largest edge that fits in GPU memory, use it as finish.
  - neither                 → run until the GPU actually OOMs (empirical capacity).

Each size trains a fresh SOM (small epoch count) on one shared random corpus, as an isolated
`sbsom` subprocess. Per successful size it writes the codebook (.somw) and a JSON description
(parameters + quality metrics). The whole sweep is summarised in summary.json, including the
OOM point and a memory-constraint estimate.
"""

import csv
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict

from . import gpu

# Rough CUDA context + library overhead not captured by the pre-launch memory model.
_CONTEXT_OVERHEAD = 600 * 1024 * 1024


def predicted_bytes(edge, n_features, n_samples, n_nonzeros):
    """Approximate peak GPU bytes for an sbsom run at this map size: feature-major codebook
    (K·V) + factorized-update tiles (2·K·CH, CH=min(2048,V)) + per-neuron norms + BMU arrays."""
    K = edge * edge
    ch = min(2048, n_features)
    codebook  = K * n_features * 2              # FP16 codebook (d_W_fm is __half)
    upd_tiles = 2 * K * ch * 4                 # update tiles stay FP32
    norms_m   = 4 * K * 4
    bmu       = 4 * n_samples * 4
    dataset   = (n_samples + 1) * 4 + n_nonzeros * 2
    return codebook + upd_tiles + norms_m + bmu + dataset


def _per_neuron_bytes(n_features):
    ch = min(2048, n_features)
    return n_features * 2 + (2 * ch + 4) * 4   # codebook V·2 (FP16) + update 2·CH·4 + norms/m 4·4


def predict_max_edge(n_features, n_samples, n_nonzeros, free_bytes, safety=0.9):
    """Largest edge L whose predicted footprint fits within safety·free_bytes (minus context)."""
    usable = safety * free_bytes - _CONTEXT_OVERHEAD
    const = 4 * n_samples * 4 + (n_samples + 1) * 4 + n_nonzeros * 2
    k_max = (usable - const) / _per_neuron_bytes(n_features)
    return max(0, int(math.isqrt(int(k_max)))) if k_max > 0 else 0


def _next_edge(edge, factor):
    nxt = round(math.sqrt(edge * edge * factor))
    return max(nxt, edge + 1)


def _sigma_init_for(edge, train):
    """Initial neighbourhood radius for this map size. Default: map-scaled (a fixed fraction of
    the edge, so every size starts with a comparable relative neighbourhood). A fixed absolute
    value (train['sigma_fixed']) overrides the scaling."""
    if train["sigma_fixed"] is not None:
        return float(train["sigma_fixed"])
    return float(max(round(train["sigma_frac"] * edge), train["sigma_min"]))


def _sigma_policy_str(train):
    if train["sigma_fixed"] is not None:
        return f"fixed {train['sigma_fixed']:g}"
    return f"scaled: {train['sigma_frac']:g} x edge"


def impute_wall_seconds(run_dir):
    """Fallback per-size training time when wall_seconds wasn't recorded: derive it from the
    write times of the per-size som_<edge>.json files. wall[edge_i] ≈ mtime[i] - mtime[i-1];
    the first size is measured from the corpus file (sweep start). Approximate — it includes
    inter-size overhead — but lets old/interrupted runs report timing without a rerun.
    Returns {edge: seconds}."""
    descs = []
    for fn in os.listdir(run_dir):
        m = re.fullmatch(r"som_(\d+)\.json", fn)
        if m:
            p = os.path.join(run_dir, fn)
            descs.append((int(m.group(1)), os.path.getmtime(p)))
    descs.sort(key=lambda t: t[1])               # chronological (= ascending size, normally)
    if not descs:
        return {}
    corpus = os.path.join(run_dir, "corpus.sbcsr")
    start = os.path.getmtime(corpus) if os.path.exists(corpus) else descs[0][1]
    out, prev = {}, start
    for edge, mtime in descs:
        out[edge] = round(max(0.0, mtime - prev), 2)
        prev = mtime
    return out


def _epochs_for(train, sigma_init, rate):
    """Epoch budget for one run. Fixed (train['epochs']) unless train['epochs_auto'], in which
    case it SCALES WITH THE ANNEAL LENGTH: σ contracts from σ₀ to ~σ_min once the progress
    variable s reaches ~ln(σ₀/σ_min)/rate, and s advances ≤1/epoch, so the budget grows with σ₀
    (∝ map edge) and shrinks with rate. It's an upper bound — convergence/watchdog usually stop
    earlier. epochs = margin + mult · ln(σ₀/σ_min)/rate, floored."""
    if not train["epochs_auto"]:
        return int(train["epochs"])
    ratio = max(float(sigma_init) / max(float(train["sigma_min"]), 1e-6), 1.0001)
    s_anneal = math.log(ratio) / max(float(rate), 1e-6)
    return max(8, int(math.ceil(train["epochs_margin"] + train["epochs_mult"] * s_anneal)))


def _epochs_policy_str(train):
    if train["epochs_auto"]:
        return (f"auto: {train['epochs_margin']:g} + {train['epochs_mult']:g}·ln(σ₀/σ_min)/rate "
                f"(scales with anneal length)")
    return f"fixed {int(train['epochs'])}"


def _parse_metrics(out):
    """Pull the final quality metrics from the last per-epoch line of sbsom's output."""
    epoch_lines = [l for l in out.splitlines() if l.lstrip().startswith("epoch ")]
    m = {"final_qe": None, "kl": None, "stability": None,
         "topographic_error": None, "dead_fraction": None,
         "converged": ("converged after epoch" in out)}
    if epoch_lines:
        last = epoch_lines[-1]
        def g(pat):
            mm = re.search(pat, last)
            return float(mm.group(1)) if mm else None
        m["final_qe"]          = g(r"QE=([\d.eE+-]+)")
        m["kl"]                = g(r"KL=([\d.eE+-]+)")
        m["stability"]         = g(r"stab=([\d.eE+-]+)")
        m["topographic_error"] = g(r"TE=([\d.eE+-]+)")
        dead = re.search(r"dead=([\d.eE+-]+)%", last)
        m["dead_fraction"] = float(dead.group(1)) / 100.0 if dead else None
    return m


@dataclass
class RunRow:
    edge: int = 0
    neurons: int = 0
    sigma_init: float = 0.0     # σ₀ actually used at this size (scaled or fixed)
    rate_used: float = 0.0      # σ schedule rate used on the (final) attempt
    epochs: int = 0             # epoch budget used (auto-scaled or fixed)
    n_restarts: int = 0         # times the watchdog flagged restart-slower at this size
    status: str = ""            # "ok" | "oom" | "restart_slower" | "error"
    wall_seconds: float = 0.0
    final_qe: float = float("nan")
    kl: float = float("nan")
    stability: float = float("nan")
    topographic_error: float = float("nan")
    dead_fraction: float = float("nan")
    converged: bool = False
    predicted_mib: float = 0.0
    weights_path: str = ""
    desc_path: str = ""
    notes: str = ""


def _run_one(sbsom, corpus, edge, out_dir, train, corpus_meta, save_weights, timeout, rate):
    """Run sbsom for one map size as an isolated subprocess at the given σ schedule rate; on
    success write the codebook + a JSON description (parameters + quality metrics). The watchdog
    flag (sbsom exit code 3) maps to status "restart_slower". Returns a RunRow."""
    K = edge * edge
    sigma_init = _sigma_init_for(edge, train)        # map-scaled by default (σ₀ = frac·edge)
    epochs = _epochs_for(train, sigma_init, rate)    # auto-scaled with anneal length, or fixed
    weights = os.path.join(out_dir, f"weights_{edge}.somw") if save_weights else ""
    cmd = [sbsom, corpus, "--rows", str(edge), "--cols", str(edge),
           "--epochs", str(epochs),
           "--sigma-init", str(sigma_init), "--sigma-min", str(train["sigma_min"]),
           "--sigma-sched", train["sched"], "--sigma-rate", str(rate),
           "--sigma-gref", str(train["gref"]), "--sigma-window", str(train["window"]),
           "--sigma-accel", str(train["accel"]), "--wd-path-frac", str(train["wd_path_frac"]),
           "--te-converge-max", str(train["te_converge_max"])]
    if weights:
        cmd += ["--save-weights", weights]

    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return RunRow(edge=edge, neurons=K, sigma_init=sigma_init, rate_used=rate, epochs=epochs,
                      status="error", wall_seconds=time.monotonic() - t0, notes="timeout")
    secs = time.monotonic() - t0

    blob = (out + "\n" + err).lower()
    if rc == 0:                          status = "ok"
    elif rc == 3:                        status = "restart_slower"   # watchdog flag
    elif "out of memory" in blob:        status = "oom"
    else:                                status = "error"
    met = _parse_metrics(out)
    note = "" if status not in ("error",) else (err.strip().splitlines()[-1] if err.strip() else f"exit {rc}")

    desc_path = ""
    if status == "ok":
        desc_path = os.path.join(out_dir, f"som_{edge}.json")
        desc = {
            "parameters": {
                "map_rows": edge, "map_cols": edge, "neurons": K,
                "n_features": corpus_meta["n_features"], "n_samples": corpus_meta["n_samples"],
                "nnz_mean": corpus_meta["nnz_mean"], "nnz_std": corpus_meta["nnz_std"],
                "n_nonzeros": corpus_meta["n_nonzeros"], "corpus_seed": corpus_meta["seed"],
                "epochs": epochs, "epochs_policy": _epochs_policy_str(train),
                "sigma_init": sigma_init,
                "sigma_init_policy": _sigma_policy_str(train), "sigma_min": train["sigma_min"],
                "sigma_schedule": train["sched"], "sigma_rate": rate, "sigma_g_ref": train["gref"],
                "sigma_window": train["window"], "sigma_accel": train["accel"],
                "wd_path_frac": train["wd_path_frac"], "te_converge_max": train["te_converge_max"],
                "box_passes": 3, "stop_criterion": "KaskiLagus",
            },
            "metrics": {
                "final_qe": met["final_qe"], "kaski_lagus": met["kl"],
                "stability": met["stability"], "topographic_error": met["topographic_error"],
                "dead_fraction": met["dead_fraction"], "converged": met["converged"],
                "wall_seconds": round(secs, 3),
            },
            "codebook": os.path.basename(weights) if weights else None,
        }
        with open(desc_path, "w") as fh:
            json.dump(desc, fh, indent=2)

    return RunRow(edge=edge, neurons=K, sigma_init=sigma_init, rate_used=rate, epochs=epochs,
                  status=status, wall_seconds=secs, final_qe=met["final_qe"], kl=met["kl"],
                  stability=met["stability"], topographic_error=met["topographic_error"],
                  dead_fraction=met["dead_fraction"], converged=bool(met["converged"]),
                  weights_path=weights, desc_path=desc_path, notes=note)


def run_sweep(sbsom, corpus, corpus_meta, train, start, finish, factor, predict_oom,
              out_dir, save_weights=True, safety=0.9, timeout=None, max_steps=64,
              max_restarts=2, restart_slowdown=0.5, rate_scale_n=False, dry_run=False):
    """Run the map-size sweep; write sweep.csv + summary.json. Returns the list of RunRow."""
    os.makedirs(out_dir, exist_ok=True)
    V, S, nnz = corpus_meta["n_features"], corpus_meta["n_samples"], corpus_meta["n_nonzeros"]
    free, total = gpu.free_total_bytes()

    if predict_oom:
        predicted = predict_max_edge(V, S, nnz, free, safety)
        finish = predicted if finish is None else min(finish, predicted)
        print(f"PredictOOM: GPU free {free // 2**20} / {total // 2**20} MiB "
              f"(safety {safety}) -> max edge {predicted}; finishing at {finish}")
    mode = "fixed" if (finish is not None and not predict_oom) else \
           ("predicted" if predict_oom else "empirical (double until OOM)")
    print(f"sweep: start={start} finish={finish} factor={factor} mode={mode} "
          f"corpus={S}x{V} (nnz={nnz}) epochs={_epochs_policy_str(train)}")

    if dry_run:
        edge, steps = start, 0
        while (finish is None or edge <= finish) and steps < max_steps:
            print(f"  edge={edge:6d}  neurons={edge*edge:>12,d}  "
                  f"predicted={predicted_bytes(edge, V, S, nnz)/2**20:8.1f} MiB")
            edge = _next_edge(edge, factor); steps += 1
            if finish is None:
                break
        return []

    rows, edge, steps = [], start, 0
    csv_path = os.path.join(out_dir, "sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f for f in asdict(RunRow()).keys()])
        writer.writeheader()
        while steps < max_steps:
            if finish is not None and edge > finish:
                break
            # Base σ rate for this size. With --rate-scale-n the half-life grows with the map
            # diameter (rate ∝ start/edge), so larger maps get a slower, longer-annealing schedule.
            rate = train["rate"] * (start / edge) if rate_scale_n else train["rate"]
            row = _run_one(sbsom, corpus, edge, out_dir, train, corpus_meta, save_weights, timeout, rate)
            # Watchdog flagged topology unresolved → re-run this size with a slower rate.
            n_restarts = 0
            while row.status == "restart_slower" and n_restarts < max_restarts:
                rate *= restart_slowdown; n_restarts += 1
                print(f"    restart {n_restarts}: topology unresolved -> sigma rate {rate:.3g}")
                row = _run_one(sbsom, corpus, edge, out_dir, train, corpus_meta, save_weights, timeout, rate)
            row.n_restarts = n_restarts
            row.predicted_mib = predicted_bytes(edge, V, S, nnz) / 2**20
            rows.append(row)
            writer.writerow(asdict(row)); fh.flush()
            qe = row.final_qe if row.final_qe is not None else float("nan")
            print(f"  edge={edge:6d}  neurons={edge*edge:>12,d}  σ0={row.sigma_init:6.1f}  "
                  f"rate={row.rate_used:.3g}  {row.status:13s}  {row.wall_seconds:7.2f}s  "
                  f"QE={qe:.4f}  pred={row.predicted_mib:.0f} MiB"
                  + (f"  restarts={n_restarts}" if n_restarts else "")
                  + (f"  [{row.notes}]" if row.notes else ""))
            # OOM / hard error stops the sweep (capacity limit); restart_slower is a quality flag,
            # not a memory limit, so continue to the next size after recording it.
            if row.status in ("oom", "error"):
                break
            edge = _next_edge(edge, factor); steps += 1

    summary = _build_summary(rows, corpus_meta, train, start, factor, predict_oom,
                             free, total, safety, os.path.basename(out_dir.rstrip("/")))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    cap = summary["capacity"]
    print(f"done: {len(rows)} runs; largest successful edge = {cap['largest_successful_edge']} "
          f"({cap['largest_successful_neurons']:,} neurons); OOM at edge = {cap['oom_edge']}")
    print(f"  memory estimate: ~{summary['memory_estimate']['estimated_max_neurons']:,} neurons "
          f"(edge ~{summary['memory_estimate']['estimated_max_edge']}) fit in "
          f"{free // 2**20} MiB free; results -> {out_dir}")
    return rows


def _build_summary(rows, corpus_meta, train, start, factor, predict_oom,
                   free, total, safety, initiated):
    V, S, nnz = corpus_meta["n_features"], corpus_meta["n_samples"], corpus_meta["n_nonzeros"]
    ok = [r for r in rows if r.status == "ok"]
    oom = next((r for r in rows if r.status == "oom"), None)
    largest_ok = max((r.edge for r in ok), default=None)
    per_neuron = _per_neuron_bytes(V)
    est_max_edge = predict_max_edge(V, S, nnz, free, safety)

    return {
        "sweep": {
            "initiated": initiated,
            "start_edge": start, "factor": factor,
            "mode": "predicted" if predict_oom else "empirical (double until OOM)",
            "corpus": {"n_samples": S, "n_features": V, "nnz_mean": corpus_meta["nnz_mean"],
                       "nnz_std": corpus_meta["nnz_std"], "n_nonzeros": nnz,
                       "seed": corpus_meta["seed"],
                       "provenance": corpus_meta.get("provenance")},
            "training": {"epochs": _epochs_policy_str(train), "sigma_init_policy": _sigma_policy_str(train),
                         "sigma_min": train["sigma_min"], "sigma_schedule": train["sched"],
                         "sigma_rate": train["rate"], "sigma_g_ref": train["gref"],
                         "sigma_window": train["window"], "sigma_accel": train["accel"],
                         "wd_path_frac": train["wd_path_frac"], "te_converge_max": train["te_converge_max"],
                         "box_passes": 3, "stop_criterion": "KaskiLagus"},
            "gpu": {"free_mib": free // 2**20, "total_mib": total // 2**20},
        },
        "runs": [{"edge": r.edge, "neurons": r.neurons, "sigma_init": r.sigma_init,
                  "rate_used": r.rate_used, "epochs": r.epochs,
                  "n_restarts": r.n_restarts, "status": r.status,
                  "wall_seconds": round(r.wall_seconds, 3),
                  "final_qe": r.final_qe, "kl": r.kl, "stability": r.stability,
                  "topographic_error": r.topographic_error, "dead_fraction": r.dead_fraction,
                  "converged": r.converged, "predicted_mib": round(r.predicted_mib, 1),
                  "notes": r.notes} for r in rows],
        "watchdog": {
            "restart_flagged_edges": [r.edge for r in rows if r.n_restarts > 0],
            "unresolved_after_restarts": [r.edge for r in rows if r.status == "restart_slower"],
        },
        "capacity": {
            "largest_successful_edge": largest_ok,
            "largest_successful_neurons": (largest_ok * largest_ok) if largest_ok else 0,
            "predicted_mib_at_largest_ok": round(predicted_bytes(largest_ok, V, S, nnz) / 2**20, 1)
                                            if largest_ok else None,
            "oom_edge": oom.edge if oom else None,
            "oom_neurons": (oom.edge * oom.edge) if oom else None,
            "predicted_mib_at_oom": round(predicted_bytes(oom.edge, V, S, nnz) / 2**20, 1)
                                     if oom else None,
        },
        "memory_estimate": {
            "free_vram_mib": free // 2**20, "total_vram_mib": total // 2**20,
            "safety_fraction": safety, "context_overhead_mib": _CONTEXT_OVERHEAD // 2**20,
            "per_neuron_bytes": per_neuron,
            "estimated_max_neurons": int((safety * free - _CONTEXT_OVERHEAD) / per_neuron),
            "estimated_max_edge": est_max_edge,
            "note": ("True capacity is bracketed between the largest successful run "
                     "and the OOM run; per_neuron_bytes ~= (V + 2*min(2048,V) + 4)*4."),
        },
        "timing": {
            "total_train_seconds": round(sum(r.wall_seconds for r in rows if r.status == "ok"), 1),
            "per_size": [{"edge": r.edge, "epochs": r.epochs,
                          "wall_seconds": round(r.wall_seconds, 2),
                          "seconds_per_epoch": round(r.wall_seconds / r.epochs, 3) if r.epochs else None}
                         for r in rows if r.status == "ok"],
            "note": ("wall_seconds = sbsom subprocess wall (corpus load + training + metrics + "
                     "weight save); the per-size training cost for the efficiency comparison."),
        },
    }
