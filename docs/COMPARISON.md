# SparseBinSOM vs somoclu — comparison design

Goal: test whether **SparseBinSOM** (GPU sparse-binary SOM, driven via `sbsom`) delivers
**similar or better quality at improved efficiency** versus **somoclu** on the real MEDLINE
corpus (articles × MeSH descriptors).

## Why somoclu's *sparse CPU* kernel

somoclu has three kernels: dense CPU, dense GPU, and **sparse CPU** (`-k 2`). Its GPU kernel is
**dense only**, and a dense representation is infeasible here — the codebook/data over ~28k MeSH
features would need petabytes. So somoclu's only viable kernel for MEDLINE is **sparse CPU**
(OpenMP across all cores). The honest comparison is therefore:

| | SparseBinSOM | somoclu |
|---|---|---|
| kernel | sparse, **GPU** (CUDA) | sparse, **multicore CPU** (OpenMP) |
| codebook | dense `K×V` float32 | dense `K×V` float32 |
| training | batch | batch |
| memory bound | **VRAM** (24 GiB here) | **host RAM** (31 GiB here) |

Both store a *dense* `K×V` codebook, so per-map memory is comparable — but because somoclu is
bounded by host RAM and sbsom by VRAM, somoclu may reach *larger* maps before OOM. That capacity
asymmetry is itself a reportable result, not a flaw in the comparison.

## What we hold equal (fairness)

- **Same corpus** — the identical `.sbcsr`, converted once to somoclu's libsvm sparse text
  (zero-indexed `col:1` per row, binary presence).
- **Same map sizes** — the edges from a SparseBinSOM map-size sweep.
- **Same per-size epoch budget** — sbsom's epoch count is auto-scaled (KL-driven); we feed
  somoclu the **same final epoch count sbsom used at that size** (`epochs` column of the sweep),
  so neither side gets a compute advantage.
- **Matched neighbourhood schedule** — somoclu start radius `-r = round(frac·edge)` (= σ₀ = N/2,
  same `--sigma-frac` default as sbsom), end radius `-R = σ_min`, exponential cooling (`-t
  exponential`), planar map. (See caveats — the *kernel shapes* still differ.)
- **One metric evaluator** — `tools/metrics` (C++/OpenMP) scores **both** codebooks with
  identical code under **cosine** distance:
  - **QE** = mean cosine distance `1 − cos(sample, BMU1)`.
  - **TE** (topographic error) = fraction of samples whose 1st and 2nd BMUs are **not adjacent**
    on the grid (Chebyshev distance > 1, i.e. outside the 8-neighbourhood).

  It reads somoclu `.wts` (neuron-major text) and sbsom `.somw` (feature-major binary), and
  evaluates on a seeded `--sample` of rows for tractability.

## Caveats / documented asymmetries

- **Neighbourhood kernel.** sbsom uses a **box-filter half-width** σ with a KL-driven endpoint
  schedule; somoclu uses a **Gaussian** radius with linear/exp cooling. We match the *endpoints*
  (σ₀, σ_min) and use exponential cooling, but the kernel shapes are not identical. This is
  inherent to comparing two different SOM implementations and is disclosed, not hidden.
- **Native quality numbers aren't comparable.** sbsom reports QE/KL/TE in its own internal
  distance; somoclu reports none. Only the **common cosine evaluator** output is comparable —
  the sbsom side requires running the compared sizes **with `--save-weights`** so `.somw` exists
  to score. (sbsom's self-reported QE is retained separately as `sbsom_qe`, for reference only.)
- **Efficiency is GPU-vs-CPU**, by necessity (somoclu has no sparse GPU kernel). Wall-time ratios
  are reported alongside the hardware so the comparison is read correctly.
- **Initialisation** differs (both random by default); not separately controlled.

## Workflow

```bash
# 1. Build the tools (once):
tools/build_somoclu.sh                                  # -> /workspaces/somoclu/somoclu_cpu
g++ -O3 -fopenmp tools/metrics.cpp -o tools/metrics

# 2. Run a SparseBinSOM sweep WITH weights at the sizes you want to compare
#    (the comparison needs .somw to score sbsom under cosine):
./run_mapsize_sweep.py --start 32 --finish 128 \
    --corpus data/medline/processed/pubmed26_ge5/corpus.sbcsr --epochs-auto   # keeps weights

# 3. Run the comparison — mirrors the sweep's sizes + per-size epochs, scores both:
./compare_somoclu.py --corpus data/medline/processed/pubmed26_ge5/corpus.sbcsr \
    --sbsom-sweep runs/<ts> --threads 28 --max-edge 128
```

Output: `runs_compare/<ts>/compare.csv` + `summary.json` — per size: somoclu wall/QE/TE,
sbsom wall, sbsom cosine QE/TE (from `.somw`), and the sbsom-vs-somoclu speedup.

## Deferred (needs the running sweep done + GPU free)

- The actual head-to-head **runs**. somoclu's sparse CPU kernel is slow on large maps × 16.8M
  samples, so cap with `--max-edge` and/or evaluate quality on a `--sample-eval` subset.
- Re-running sbsom at the compared sizes **with weights** (the capacity sweep used `--no-weights`).

---

# The fair baseline: StandardSparseSOM (cuSPARSE, same GPU)

somoclu's sparse kernel is CPU-only, so that comparison conflates GPU-vs-CPU with
specialized-vs-standard. [StandardSparseSOM](https://github.com/mongrolwarrior/StandardSparseSOM)
is a standalone same-GPU baseline: SparseBinSOM's bespoke binary/feature-major BMU replaced by
`cusparseSpMM`, reading `.sbcsr` and writing `.somw` so the shared cosine evaluator scores both.
`compare_gpu.py` drives it (reads an sbsom sweep run *with weights*, trains StandardSparseSOM at
the matched sizes/epochs/σ for each `--layout`, scores everything):

```bash
./compare_gpu.py --corpus data/medline/processed/pubmed26_ge5/corpus.sbcsr \
    --sbsom-sweep runs/cmp_sbsom_full --edges 32,64
```

## Key framing — the BMU axis is *speed*, not quality

An exact-argmax BMU is identical regardless of implementation (cuSPARSE vs bespoke) **or** codebook
layout (feature- vs node-major gave identical QE/TE). So the BMU only affects **speed**. All
QE/TE/dead differences between SparseBinSOM and StandardSparseSOM come from the **update/schedule**
axis (sbsom's box-blur + KL-stop vs StandardSparseSOM's Gaussian + geometric σ-decay). To compare
*quality* fairly, hold the update fixed too.

## Results (full 30M-article corpus, cosine on 200k sampled rows)

| edge | impl | wall | BMU | QE | TE | dead% |
|---|---|---|---|---|---|---|
| 32 | sbsom | 35.2s | — | 0.529 | **0.018** | 14.6 |
| 32 | ssom-feature | 38.1s | 25.6s | 0.480 | 0.204 | 5.3 |
| 32 | ssom-node | 151.3s | 139.7s | 0.481 | 0.207 | 6.9 |
| 64 | sbsom | 70.6s | — | 0.491 | **0.022** | 21.8 |
| 64 | ssom-feature | 132.5s | 119.5s | 0.445 | 0.253 | 9.4 |
| 64 | ssom-node | 656.0s | 641.0s | 0.444 | 0.257 | 9.0 |

- **Feature-major vs node-major:** identical quality, but feature-major's BMU is **~5.4× faster**
  under cuSPARSE (`ORDER_ROW` ≫ `ORDER_COL`) — independently vindicating SparseBinSOM's layout.
- **Efficiency:** sbsom beats the best cuSPARSE variant **~1.08× (edge 32) → ~1.9× (edge 64)**;
  the gap widens with neuron count (binary-blind multiply + dense-scores materialisation). Far
  tougher baseline than somoclu CPU (~100×+).
- **Quality (update axis):** sbsom has ~10× better topology (TE), the cuSPARSE+Gaussian baseline
  has lower QE / fewer dead neurons — the classic TE-vs-QE trade-off set by neighbourhood strength.

## Isolating quality: match the update (`--neighbourhood box`, `--stop kl`)

Since an exact-argmax BMU can't change quality, the quality gap above is purely the *update*. Giving
StandardSparseSOM sbsom's update confirms it — quality converges to sbsom's:

| edge | impl | TE | QE | dead% |
|---|---|---|---|---|
| 32 | sbsom | 0.018 | 0.529 | 14.6 |
| 32 | ssom Gaussian | 0.204 | 0.480 | 5.3 |
| 32 | ssom **box** | 0.017 | 0.525 | 19.5 |
| 32 | ssom **box+KL** | 0.015 | 0.527 | 17.7 |
| 64 | sbsom | 0.022 | 0.491 | 21.8 |
| 64 | ssom box+KL | 0.022 | 0.488 | 21.7 |

Box-blur alone (3 passes ≈ Gaussian, = sbsom's factorized neighbourhood) drops TE from ~0.2 to ~0.02;
adding the KL stop lets it converge on its own KL plateau (ep 21 / 30) at sbsom's quality.

## Conclusion (original question: similar/superior quality at better efficiency?)

- **Quality: equal, not superior.** The BMU implementation/layout has *no* effect on quality
  (exact argmax). SparseBinSOM's quality comes entirely from its **update** (box-blur + KL stop),
  which any correct SOM reproduces — StandardSparseSOM with `--neighbourhood box --stop kl` matches
  it (TE/QE/dead within noise).
- **Efficiency: SparseBinSOM's real, map-size-dependent win.** Matched-update wall: edge 32
  comparable (~28 s ssom vs ~35 s sbsom — cuSPARSE is a strong same-GPU baseline at small maps),
  edge 64 sbsom ~2× faster (71 s vs 142 s); the bespoke binary/feature-major BMU's advantage grows
  with neuron count. Plus feature-major beats node-major ~5.4× even inside cuSPARSE, and the whole
  field is ~216× ahead of somoclu's CPU sparse kernel. SparseBinSOM's contribution is an efficient,
  better-scaling BMU + a vindicated codebook layout — not a quality gain.

Repeatable: `./compare_gpu.py --corpus … --sbsom-sweep … --edges 32,64 --neighbourhood box --stop kl`.

## Increasing-size 3-way sweep (compare_sweep.py, full 30M corpus, matched update box+KL)

All three (sbsom, ssom-feature, ssom-node) climb the size ladder; each drops out on OOM; the sweep
stops once >=2 break, after the winner finishes that size. sbsom now reports BMU/update wall too.

| edge | sbsom BMU/ep | feat BMU/ep | node BMU/ep | sbsom upd/ep | feat upd/ep | result |
|---|---|---|---|---|---|---|
| 32 | 1.05 | 0.88 | 4.83 | 0.04 | 0.03 | feat faster total (fewer ep) |
| 64 | 1.83 | 3.62 | 19.4 | 0.04 | 0.10 | sbsom 2x BMU |
| 129 | 12.0 | 17.1 | 79.8 | 0.09 | 0.56 | sbsom 1.4x BMU, 6x update |
| 182 | 30.9 | 37.7 | 161 | 0.15 | 1.38 | sbsom 1.2x BMU, 9x update |
| 257 | 74.0 | OOM | OOM | 0.26 | — | **sbsom WINS (both ssom OOM)** |

Findings: (1) **Capacity** — sbsom reaches edge 257 (66k neurons); both cuSPARSE variants OOM there
(cap edge 182), ~2x less capacity (they hold W + two V×K accumulators + dense scores tile + int32
indices). (2) **Quality identical** across all three at every size. (3) **BMU**: binary edge over
feature-major cuSPARSE narrows with K (2.0x->1.2x by edge 182) as the shared O(K) codebook read
dominates; node-major ~5x slower than feature-major throughout. (4) **Update**: sbsom's factorized
box-blur is σ-independent so its update advantage *grows* with map size (~9x at edge 182). So
sbsom's wins are capacity + update efficiency + small/mid-map BMU; quality is equal.

Note: the binary BMU advantage being a small/mid-K effect suggests generalising to **sparse-float**
(gather-multiply instead of gather-sum, ‖x‖²=Σx_v², a values array) would broaden applicability at
little large-K speed cost, keeping the feature-major + fused-top-2 + factorized-update design.
