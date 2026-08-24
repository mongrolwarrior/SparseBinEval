# Finding the optimal SOM size (the elbow) on a high-memory cloud GPU

We showed the optimal map size for the 29.9 M-article MEDLINE corpus sits **beyond** a 24 GB GPU:
held-out QE was still falling at edge 411 (192 articles/neuron) with **no flattening** (~3.7 %
improvement *per node-doubling*, constant). To locate the elbow we need a bigger GPU — and the
binding constraint is **device memory, not time** (a full sweep is ~4 h; see below).

## GPU: H200 141 GB (Hopper)

| GPU | mem | max edge | neurons | art/neuron | full sweep |
|---|---|---|---|---|---|
| RTX 4090 | 24 GB | ~411 | 169 k | 192 | (done — still falling) |
| H100 | 80 GB | ~743 | 552 k | 54 | ~3.5 h |
| **H200** | **141 GB** | **~992** | **984 k** | **30** | **~4.4 h** |
| B200 | 192 GB | ~1160 | 1.35 M | 22 | ~3.6 h |

**H200** is the cost-effective pick: Hopper (so thread-block-clusters/DSMEM are available), and at
30 articles/neuron it has the best single-GPU shot at reaching the elbow. The existing multi-arch
build already targets sm_90, so the current `sparsesom` binary runs on it as-is.

Only the **best large-map implementation is needed: `sbsom-bin`** — the binary gather-sum path. The
corpus is binary, and `--bin` has the largest capacity (no values array → biggest maps) and is at
BMU parity with float at large K. No need for the other four implementations here.

## The efficient search pattern

Each training run is expensive and **cost grows with size** (time ∝ K = edge²), so the pattern
matters. In order of leverage:

1. **Start high.** We already characterised the steep region (≤411) on 24 GB. Don't re-train it —
   begin at **2× the 24 GB OOM-1 in nodes ⇒ edge 581** (`round(411·√2)`). The expensive budget goes
   where the answer is: the unexplored high range.

2. **Probe the cap *first*.** Train the two extremes — **581 and the memory cap (992)** — before
   anything in between. The most expensive run (992) is also the most informative:
   - if held-QE is **still steeply falling** at 992 (Δ/doubling ≈ the same ~3–4 %), the elbow is
     **beyond this GPU** → stop after **2 runs** (~5 h). Verdict: multi-GPU (the distributed-SOM
     roadmap) — itself a publishable statement about the corpus's intrinsic dimensionality.
   - if held-QE has **flattened** by 992, the knee is **bracketed in [581, 992]** → step 3.

   This beats a blind doubling ascent (581→822→992), which trains the middle point even in the
   common "still falling" case.

3. **Bisect to pin the knee.** If bracketed, add the geometric midpoint **822**, then bisect in
   log-neuron space toward where Δheld-QE/doubling crosses the "flat" threshold. 1–2 more runs.

4. **Online diminishing-returns stop** (the ascent alternative): `--until optimal` trains the
   doubling ladder and stops the moment a doubling buys < `--knee-eps` — so you never pay for runs
   *past* the knee.

5. **(Optional) cheap downsampled scout.** A run on, say, 10 % of articles is ~10× cheaper and
   reveals the curve *shape* + a rough knee; since the optimal size scales ~with N, scale that up to
   target the full-data probes. Good when you're unsure the knee is even near 992.

Held-out QE is sampled (cheap) — **training dominates**, so minimising the number and size of
training runs is the whole game. Probe-the-cap + start-high does that.

### Commands

```bash
# on the H200, via the cloud image (sbsom-bin only, held-out QE, explicit probe edges):
compare_sweep.py --corpus data/.../corpus.train.sbcsr --test-corpus data/.../corpus.test.sbcsr \
    --options sbsom-bin --until mem:130 --edges 581,992          # probe the cap first (2 runs)
# if flattening seen, refine:
compare_sweep.py ... --options sbsom-bin --until mem:130 --edges 822
# ascent-with-auto-stop alternative:
compare_sweep.py ... --options sbsom-bin --start 581 --until optimal

# splice with the existing ≤411 points and locate the elbow:
./find_elbow.py runs_compare/*/sweep.csv --impl sbsom-bin --n-samples 29903261
```
`find_elbow.py` prints the Δ-per-doubling table, runs Kneedle, and either reports the elbow edge or
says "still falling → beyond this GPU."

## Compute optimisation — thread block clusters + DSMEM (sm_90), phase 2

The BMU at large K is **HBM-bandwidth-bound** — it streams the feature-major codebook `W[V×K]`. The
cluster opportunity: a **thread-block cluster** processing different sample-tiles but the *same*
neuron slice loads each `W[v,·]` column once into one block's shared memory and shares it to the
others via **DSMEM**, cutting redundant HBM reads. The factorized box-blur could likewise hold
larger node-major tiles across a cluster's distributed shared memory.

Implementation: sm_90 kernel variants launched with `cudaLaunchKernelEx` + a cluster dim, using the
cooperative-groups cluster API (`cluster.map_shared_rank`). **Untestable without Hopper** (this dev
box is sm_89), so it's phase-2: get the baseline H200 sweep running first, profile the BMU, then
prototype the cluster-BMU only if HBM-read redundancy is the measured bottleneck. The surer,
larger win is simply H200's 4.8 TB/s HBM (~5× the 4090) + the bigger memory.
```
