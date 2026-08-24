# Cloud-GPU reproducibility

Run the held-out size sweep on a fresh cloud GPU and check it reproduces the reference results.
One multi-arch image builds all three implementations + the harness, so it runs on any modern
NVIDIA card (T4 → H100). Memory differences are absorbed by the `--until mem:GB` ceiling.

## What reproduces (and what doesn't)

The **held-out QE of the GPU implementations** is the reproducibility anchor — it should match
the reference to ~2–3 significant figures across machines. It is *not* bit-exact: the factorized
update uses float `atomicAdd` (order-dependent rounding) and GPU architectures schedule
differently, so QE matches within tolerance (default 2%) and the converged epoch count can wobble
by a few. `somoclu` is an extrapolated CPU estimate and is excluded from the numeric check.

## Steps

1. **Check out the three repos** side by side (the build stages source from them — no GitHub
   auth needed inside the image):

   ```
   SparseBinarySOM/  StandardSparseSOM/  SparseBinEval/
   ```

2. **Build the image** (multi-arch; takes a while):

   ```
   SparseBinEval/cloud/build.sh                      # -> sparsesom-repro:latest
   # known card? trim the arch list for a faster build:
   DOCKER_BUILD_ARGS="--build-arg CUDA_ARCHES=80-real;80-virtual" SparseBinEval/cloud/build.sh
   ```

3. **Transfer the corpus split** to a `data/` dir on the box (the exact files the reference was
   built from, so results are numerically comparable):

   ```
   data/corpus.train.sbcsr      # ~770 MB
   data/corpus.test.sbcsr       # ~16 MB
   # corpus.train.libsvm is regenerated on the box if absent (somoclu input)
   ```

4. **Run** (mounts the corpus; the bundled reference is checked automatically):

   ```
   docker run --gpus all -v $PWD/data:/data -v $PWD/out:/out sparsesom-repro
   ```

   Ends with `REPRODUCIBILITY PASS` / `FAIL` and a per-row held-QE table.

## Knobs (env vars on `docker run -e ...`)

| var | default | meaning |
|---|---|---|
| `UNTIL` | `mem:6` | ceiling mode. `mem:6` matches the reference (comparable). `auto` = `mem:<90% of this GPU>` for a full-capacity run. Also `oom1`, `size:N`. |
| `OPTIONS` | all five | drop `somoclu` for a fast plumbing-only check |
| `TOL` | `0.02` | held-QE relative tolerance |
| `CORPUS_DIR` | `/data` | where the split is mounted |
| `MODE` | `repro` | `elbow` for the elbow sweep (sbsom-bin only, full ladder) |
| `TRAIN_SUBSAMPLE` | `1.0` | fraction of training set for sweep (0.25 = ~4× speedup, <0.2% QE diff) |
| `INIT_PCA` | `1` | PCA codebook init (1 = on, 0 = random). Reduces epoch count ~2× |

## Notes

- The image pins CUDA 12.8 + gcc-12 and builds SASS for sm_70/75/80/86/89/90 plus a PTX
  fallback, so the same image JITs forward onto newer GPUs.
- For a different corpus, regenerate the split with `split_corpus.py` and rebuild the reference
  with one local sweep before comparing.
- PCA init + 25% subsample together give ~5× speedup with <0.2% QE difference vs full-data
  random init. The sweep trains on the subsample; final eval at the elbow trains on the full set.
