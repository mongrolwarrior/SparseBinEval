# SparseBinEval

> **▶ Reproduce the full sweep on a cloud GPU:** see **[REPRODUCE.md](REPRODUCE.md)** — one Docker
> image builds all five implementations; the dataset downloads from Zenodo; one `docker run` does
> the whole all-implementations, held-out-QE map-size sweep.

A small evaluation / benchmark harness for **[SparseSOM](https://github.com/mongrolwarrior/SparseSOM)**
(formerly SparseBinSOM) and the baselines it's compared against. It drives the `sparsesom` CLI (and
`standardsparsesom` / `somoclu`) over parameter sweeps and records the results, adding features only
as they're needed. The canonical sweep is `compare_sweep.py` (all five implementations, held-out QE,
`--until oom1|optimal|size:N|mem:GB`); the older single-implementation map-size sweep
(`run_mapsize_sweep.py`) is documented below.

Python harness, no third-party dependencies (stdlib + `nvidia-smi`). Each sweep point runs as
an isolated `sbsom` subprocess, so an out-of-memory or crash at one size can't take down the
harness — essential for capacity testing.

## Map-size sweep (the first sweep)

Square maps only: a **size** is the edge length `L` (the map is `L×L = L²` neurons). The sweep
starts at `--start`, and each step multiplies the **node count** by `--factor` (default `2.0` —
double the neurons), re-deriving the edge as `round(sqrt(nodes))`. It ends at:

- **`--finish L`** given → sweep `start..finish`.
- **`--predict-oom`** → predict the largest edge that fits in GPU memory and use it as the finish
  (so it sweeps up to capacity without actually OOMing).
- **neither** → run until the GPU actually OOMs — i.e. **empirically** find the capacity on the
  current hardware.

Each size trains a fresh SOM on one shared random sparse-binary corpus, with full persistence of
the trained codebook (`.somw`) unless `--no-weights`.

The neighbourhood width σ uses SparseBinSOM's endpoint, KL-driven schedule (the schedule runs
inside `sbsom`; the harness only *sets its parameters*). σ₀ is **map-scaled** (`--sigma-frac`,
default 0.5 → σ₀ = N/2), held constant in normalised terms across the sweep so stop-epoch
differences reflect size, not an inconsistent neighbourhood policy. If a size's watchdog flags
`RestartSlower` (topology unresolved at small σ), the harness re-runs it with a slower σ rate
(`--max-restarts`, `--restart-slowdown`); `--rate-scale-n` makes the base rate scale with map
size (longer half-life for larger maps). `g_ref` (`--gref`) is exposed as a primary sweepable knob.

```bash
# Fixed range, doubling neurons each step:
./run_mapsize_sweep.py --start 16 --finish 256

# Predict the GPU capacity limit and sweep up to it:
./run_mapsize_sweep.py --start 64 --predict-oom

# Empirically find capacity (run until OOM):
./run_mapsize_sweep.py --start 256

# Just show the planned sizes and predicted memory footprints:
./run_mapsize_sweep.py --start 16 --finish 1024 --dry-run
```

By default every size trains for a fixed `--epochs`. But the σ schedule's anneal length grows with
the map (σ₀ = N/2 ∝ edge, and σ only reaches σ_min once the KL-driven progress variable `s` has
advanced ~`ln(σ₀/σ_min)/rate`, with `s` rising ≤1 per epoch), so a fixed budget under-anneals large
maps. `--epochs-auto` gives each size an epoch budget that **scales with its anneal length** —
`epochs(size) = margin + mult·ln(σ₀/σ_min)/rate` (`--epochs-margin`, `--epochs-mult`; floored at 8)
— so small maps stay cheap and large maps actually finish annealing. It's an upper bound;
convergence (Kaski–Lagus stop) or the watchdog usually stop a run earlier. The per-size budget is
recorded in `som_<edge>.json`, `sweep.csv`, and the report's `epochs` column.

Key options: `--epochs`/`--epochs-auto`/`--epochs-mult`/`--epochs-margin`; `--samples`/`--features`/`--nnz`/`--nnz-std` (the random corpus shape,
fixed across the sweep; `--nnz-std>0` draws per-sample feature counts ~ Normal); the σ-schedule
knobs `--sigma-frac`/`--sigma-init`/`--sigma-min`/`--sigma-sched`/`--sigma-rate`/`--gref`/
`--sigma-window`/`--sigma-accel`/`--wd-path-frac`; restart handling `--max-restarts`/
`--restart-slowdown`/`--rate-scale-n`; `--seed`, `--safety`, `--timeout`, `--out`/`--out-base`,
`--sbsom` (default `/workspaces/SparseBinarySOM/build/sbsom`).

## Output

Each sweep writes a dated subfolder `runs/dd.mm.yy-hhmm/` (or `--out`):
- `corpus.sbcsr` — the shared random corpus,
- `weights_<edge>.somw` — the trained codebook per size (unless `--no-weights`),
- `som_<edge>.json` — that SOM's parameters + quality metrics,
- `sweep.csv` — one row per size (edge, neurons, σ₀, rate, restarts, status, wall, QE/KL/…),
- `summary.json` — sweep params + every run + capacity (OOM point) + memory estimate + watchdog flags.

## Reading the results

`report.py` renders a sweep's `summary.json` into a readable report:

```bash
./report.py runs/19.06.26-0012           # -> runs/19.06.26-0012/report.md (and prints it)
./report.py runs/19.06.26-0012 --html    # -> report.html
./report.py runs/.../summary.json --stdout
```

It produces a header (corpus / training params / GPU / capacity / memory estimate / watchdog) and
a per-size table (status, σ₀, QE, KL, stability, topographic error, dead %, converged, restarts).

## Real MEDLINE data

> **Released corpus.** The MEDLINE/MeSH corpus used in the Phase 1 paper is distributed
> ready-made on Zenodo (concept DOI [10.5281/zenodo.20770707](https://doi.org/10.5281/zenodo.20770707),
> CC0 1.0) — an anonymised sparse-binary `.sbcsr` with no PMIDs, titles or abstracts. The
> extraction script described below (`medline.py`) is a local data-preparation tool and is
> **not part of this repository**; the notes stay here to document how the corpus was built.
>
> The underlying data are **courtesy of the U.S. National Library of Medicine**. The corpus is a
> fixed snapshot of the 2026 PubMed baseline and **does not reflect the most current or accurate
> data available from NLM**. NLM does not endorse or recommend this project or any product or
> service derived from it.

`medline.py` turns the NCBI PubMed **baseline** (the annual MEDLINE snapshot) into a `.sbcsr`
corpus whose features are MeSH descriptors — pure stdlib (`urllib` + `gzip` + `xml.etree` +
`multiprocessing`). NCBI now serves only the *current* baseline (the per-year archives were
withdrawn), so there is no dataset selector: the source is always the latest baseline
(`https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/`), auto-detected from the listing. You can still
restrict *which files* of it to use, and filter articles by MeSH-term count.

The default behaviour (no subcommand) is to **check whether the raw and processed data is in the
expected place and offer to download and process it if not**:

```bash
./medline.py                       # check; offer to download + process (prompts on a TTY)
./medline.py status                # report what's on disk
./medline.py fetch  --max-files 5  # download the first 5 baseline files (+ MD5 verify)
./medline.py process --min-mesh 5  # parse downloaded files -> corpus.sbcsr
./medline.py ensure --yes          # download (all ~1300 files, ~40 GB) + process, no prompt
./medline.py process --min-mesh 5 --major-mesh   # major-topic descriptors only
```

Filtering: `--min-mesh N` keeps only articles with ≥ N MeSH terms (default 5); `--major-mesh`
keeps only `MajorTopicYN=Y` descriptors. Selection: `--max-files N` or `--files 1-5,20`. Each stage
is **resumable** — downloads skip already-verified files, and parsing reuses per-file shards.

Data lives under `data/medline/` (git-ignored): `raw/<release>/*.xml.gz`, intermediate
`shards/<tag>/`, and `processed/<tag>/{corpus.sbcsr, vocab.json, summary.json}` where `<tag>` is
e.g. `pubmed26_ge5`. `vocab.json` maps each feature index back to its MeSH UI.

**Capacity check.** The package is meant to be cloned into containers of unknown size, and the
baseline is large (~48 GiB raw, ~1 GiB corpus, 37 M articles). Before downloading or building, it
estimates the disk + RAM needed and, if the machine looks too small, prints a warning — you can
continue anyway (it's advisory). If a build then runs out of memory it fails cleanly with guidance
(use fewer `--max-files`, a higher `--min-mesh`, or more RAM) rather than leaving a truncated corpus.

**Summaries.** `raw/<release>/download_summary.json` records when it was downloaded, which baseline,
how many of the baseline's files are present, and whether the download is **complete or partial**.
`processed/<tag>/summary.json` records the baseline, the filter condition, when it was downloaded and
processed, the corpus shape, and a `partial` flag. Sweep outputs read this summary: the report header
shows a **Dataset** line and a prominent ⚠ warning when the corpus came from a partial download.

Run a sweep on it with `--corpus` (a prebuilt `.sbcsr`) or `--medline` (auto-ensure then sweep):

```bash
./run_mapsize_sweep.py --start 32 --corpus data/medline/processed/pubmed26_ge5/corpus.sbcsr --epochs-auto
./run_mapsize_sweep.py --start 32 --medline --min-mesh 5 --epochs-auto --yes
```

The memory model and σ defaults read the true corpus shape (`n_samples`, `n_features`, `nnz`) from
the `.sbcsr` header, so capacity prediction works the same as for generated corpora.

## Comparison: SparseBinSOM vs somoclu

`compare_somoclu.py` runs [somoclu](https://github.com/peterwittek/somoclu)'s sparse-CPU SOM on
the same MEDLINE corpus, map sizes, and per-size epoch budget as a SparseBinSOM sweep, and scores
**both** codebooks with one cosine evaluator (`tools/metrics`, C++/OpenMP) — so quality is judged
identically. somoclu's GPU kernel is dense-only (infeasible at ~28k MeSH features), so its sparse
CPU kernel is the fair counterpart to sbsom's GPU sparse kernel. See **[docs/COMPARISON.md](docs/COMPARISON.md)**
for the full design, fairness rules, and caveats.

```bash
tools/build_somoclu.sh                                   # build CPU somoclu (no autotools needed)
g++ -O3 -fopenmp tools/metrics.cpp -o tools/metrics      # build the common cosine evaluator
./compare_somoclu.py --corpus data/medline/processed/pubmed26_ge5/corpus.sbcsr \
    --sbsom-sweep runs/<ts> --threads 28 --max-edge 128  # mirror the sweep's sizes/epochs
```

Output `runs_compare/<ts>/{compare.csv,summary.json}`: per size — somoclu wall/QE/TE, sbsom wall,
sbsom cosine QE/TE (scored from its `.somw`), and the sbsom-vs-somoclu speedup. The sbsom side needs
the compared sizes trained **with weights** (a capacity sweep uses `--no-weights`).

## Memory model (for `--predict-oom`)

Approximate peak GPU bytes ≈ codebook `K·V` + factorized-update tiles `2·K·min(2048,V)` +
per-neuron norms + per-sample BMU arrays, with `K = L²`, `V = features`. The predictor inverts
this against `safety · free_VRAM` (minus a fixed CUDA-context allowance) to get the largest edge.
It's deliberately approximate; the empirical mode (run-until-OOM) is the ground truth.

## Layout

```
sbeval/sbcsr.py    write/read .sbcsr; generate a random sparse-binary corpus
sbeval/medline.py  download + parse the PubMed baseline into a MeSH .sbcsr corpus
sbeval/gpu.py      query free/total VRAM via nvidia-smi
sbeval/mapsize.py  memory model, OOM prediction, the sweep driver
sbeval/report.py   render a sweep summary.json as Markdown / HTML
run_mapsize_sweep.py   CLI: run a sweep
report.py              CLI: render a sweep report
medline.py             CLI: fetch/process the MEDLINE baseline corpus
```

Future sweeps (other parameters) slot in alongside `mapsize.py`.
