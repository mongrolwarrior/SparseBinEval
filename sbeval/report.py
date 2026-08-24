"""Render a sweep's summary.json into a human-readable report (Markdown or HTML).

Reads only summary.json (which already carries the sweep params, per-size runs, capacity, the
memory estimate, and the watchdog flags), so it's a pure post-processing step over a finished
sweep directory.
"""

import html as _html


def _f(x, nd=4):
    """Format a float/None/NaN for display."""
    if x is None:
        return "–"
    try:
        xf = float(x)
        if xf != xf:        # NaN
            return "–"
        return f"{xf:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _g(x):
    """Compact float (trailing-zero-trimmed)."""
    return "–" if x is None else f"{float(x):g}"


# Per-run table columns: (header, key, formatter)
_COLUMNS = [
    ("edge", "edge", lambda v: f"{v}"),
    ("neurons", "neurons", lambda v: f"{v:,}"),
    ("σ₀", "sigma_init", _g),
    ("rate", "rate_used", _g),
    ("epochs", "epochs", lambda v: f"{v or 0}"),
    ("status", "status", str),
    ("wall s", "wall_seconds", lambda v: _f(v, 2)),
    ("s/epoch", "sec_per_epoch", lambda v: _f(v, 2)),
    ("QE", "final_qe", lambda v: _f(v, 4)),
    ("KL", "kl", lambda v: _f(v, 4)),
    ("stab", "stability", lambda v: _f(v, 3)),
    ("TE", "topographic_error", lambda v: _f(v, 3)),
    ("dead%", "dead_fraction", lambda v: "–" if v is None else f"{100*float(v):.1f}"),
    ("conv", "converged", lambda v: "yes" if v else "no"),
    ("restarts", "n_restarts", lambda v: f"{v or 0}"),
    ("pred MiB", "predicted_mib", lambda v: _f(v, 0)),
]


def _fmt_secs(x):
    """Seconds as a compact h/m/s string, e.g. 1786 -> '29m46s'."""
    x = int(round(x))
    if x < 60:
        return f"{x}s"
    if x < 3600:
        return f"{x // 60}m{x % 60:02d}s"
    return f"{x // 3600}h{(x % 3600) // 60:02d}m"


def _augment_timing(s):
    """Add a derived seconds-per-epoch to each run, and return (total_secs, slowest_run)
    over the runs that actually trained. Pure post-processing of existing wall_seconds."""
    total, slowest = 0.0, None
    for r in s.get("runs", []):
        wall, ep = r.get("wall_seconds"), r.get("epochs")
        r["sec_per_epoch"] = (wall / ep) if (wall and ep) else None
        if r.get("status") == "ok" and wall:
            total += wall
            if slowest is None or wall > slowest.get("wall_seconds", 0):
                slowest = r
    return total, slowest


def _timing_line(s):
    """One-line training-time rollup for the header."""
    total, slowest = _augment_timing(s)
    n = sum(1 for r in s.get("runs", []) if r.get("status") == "ok" and r.get("wall_seconds"))
    if not n:
        return "–"
    imp = " (imputed from file mtimes)" if s.get("_timing_imputed") else ""
    slow = (f"; slowest edge {slowest['edge']} at {_fmt_secs(slowest['wall_seconds'])} "
            f"({_f(slowest.get('sec_per_epoch'), 1)} s/epoch)") if slowest else ""
    return f"{_fmt_secs(total)} total over {n} sizes{slow}{imp}"


def _sweep_lines(s):
    """Common header facts as (label, value) pairs."""
    sw, c, t, g = s["sweep"], s["sweep"]["corpus"], s["sweep"]["training"], s["sweep"]["gpu"]
    cap, mem = s["capacity"], s["memory_estimate"]
    prov = c.get("provenance")
    lines = [
        ("Initiated", sw.get("initiated", "?")),
        ("Mode", sw.get("mode", "?")),
        ("Start edge / factor", f"{sw['start_edge']} / ×{sw['factor']} (node count)"),
        ("Corpus", f"{c['n_samples']:,} samples × {c['n_features']:,} features "
                   f"(~{c['nnz_mean']}±{c['nnz_std']} per sample, {c['n_nonzeros']:,} nnz, seed {c['seed']})"),
    ]
    if prov:
        flt = prov.get("filter", {})
        flt_s = f"≥{flt.get('min_mesh')} MeSH" + (", major-topic only" if flt.get("major_only") else "")
        files = (f"{prov.get('n_files_used')}/{prov.get('n_files_baseline')} files"
                 if prov.get("n_files_baseline") else f"{prov.get('n_files_used')} files")
        partial = " — ⚠ PARTIAL DOWNLOAD" if prov.get("partial") else " (complete)"
        lines.append(("Dataset", f"MEDLINE {prov.get('release')} baseline, {flt_s}; {files}{partial}; "
                                 f"downloaded {prov.get('downloaded_at','?')}"))
    return lines + [
        ("Training", f"epochs {t['epochs']}, σ {t['sigma_init_policy']} → σ_min {t['sigma_min']} "
                     f"({t.get('sigma_schedule','?')}, rate {t.get('sigma_rate','?')}, "
                     f"g_ref {t.get('sigma_g_ref','?')}, window {t.get('sigma_window','?')}), "
                     f"accel ×{t.get('sigma_accel','?')}, wd_path_frac {t.get('wd_path_frac','?')}, "
                     f"TE-gate {t.get('te_converge_max', t.get('bmu_beta','?'))}, {t['stop_criterion']}"),
        ("GPU", f"{g['free_mib']:,} / {g['total_mib']:,} MiB free"),
        ("Training time", _timing_line(s)),
        ("Capacity", f"largest ok edge {cap['largest_successful_edge']} "
                     f"({cap['largest_successful_neurons']:,} neurons, ~{_f(cap['predicted_mib_at_largest_ok'],0)} MiB); "
                     f"OOM at edge {cap['oom_edge']}"
                     + (f" ({cap['oom_neurons']:,} neurons, ~{_f(cap['predicted_mib_at_oom'],0)} MiB)"
                        if cap['oom_edge'] else "")),
        ("Memory estimate", f"~{mem['estimated_max_neurons']:,} neurons (edge ~{mem['estimated_max_edge']}) "
                            f"fit in {mem['free_vram_mib']:,} MiB; per-neuron ≈ {mem['per_neuron_bytes']:,} B"),
        ("Watchdog", f"restart-flagged: {s['watchdog']['restart_flagged_edges'] or 'none'}; "
                     f"unresolved after restarts: {s['watchdog']['unresolved_after_restarts'] or 'none'}"),
    ]


def render_markdown(s):
    _augment_timing(s)                                   # add sec_per_epoch to each run first
    out = [f"# SparseBinSOM map-size sweep — {s['sweep'].get('initiated','?')}", ""]
    for label, val in _sweep_lines(s):
        out.append(f"- **{label}:** {val}")
    out += ["", "## Per-size runs", ""]
    out.append("| " + " | ".join(h for h, _, _ in _COLUMNS) + " |")
    out.append("|" + "|".join("---" for _ in _COLUMNS) + "|")
    for r in s["runs"]:
        out.append("| " + " | ".join(fmt(r.get(key)) for _, key, fmt in _COLUMNS) + " |")
    return "\n".join(out) + "\n"


def render_html(s):
    _augment_timing(s)                                   # add sec_per_epoch to each run first
    rows = "\n".join(
        "<tr>" + "".join(f"<td>{_html.escape(fmt(r.get(key)))}</td>" for _, key, fmt in _COLUMNS) + "</tr>"
        for r in s["runs"])
    head = "".join(f"<th>{_html.escape(h)}</th>" for h, _, _ in _COLUMNS)
    facts = "\n".join(f"<li><b>{_html.escape(l)}:</b> {_html.escape(str(v))}</li>" for l, v in _sweep_lines(s))
    return f"""<!doctype html><meta charset="utf-8">
<title>SparseBinSOM sweep — {_html.escape(s['sweep'].get('initiated','?'))}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:70rem}}
 table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
 th{{background:#f3f3f3}} td:first-child,th:first-child{{text-align:left}}
 ul{{line-height:1.6}}
</style>
<h1>SparseBinSOM map-size sweep — {_html.escape(s['sweep'].get('initiated','?'))}</h1>
<ul>{facts}</ul>
<h2>Per-size runs</h2>
<table><thead><tr>{head}</tr></thead>
<tbody>
{rows}
</tbody></table>
"""
