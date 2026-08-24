# Convergent-Validity & Clustering-Comparison Study — Consolidated

**Status:** living document, last updated 2026-06-22. **Supersedes** for operational use the pair
`reference_taxonomies_handover.md` (original) + `reference_taxonomies_handover_correction.md`
(re-anchoring) — both retained as historical methodology. This doc folds in what was actually
**built, run, and validated** this session.

**Question (from the project plan):** is the SparseSOM a good way to organise the MEDLINE/MeSH
knowledge structure, and how does it compare to other clustering/visualisation methods — judged by
a *vector* of criteria, and validated against taxonomies the SOM never trained on?

---

## 1. Anchors (verified, not assumed)

| Thing | Reality |
|---|---|
| SOM under test | **SparseSOM** (feature-major codebook; `SparseBinarySOM/build/sparsesom`), planar **square** lattice, edge-clamped neighbourhood. BMU = the clustering. |
| Corpus | `data/medline/processed/pubmed26_ge5/corpus.sbcsr` — 29,903,261 articles ≥5 MeSH, V=30,766. |
| PMID key | `data/reference_taxonomies/metadata.sqlite` `articles(article_idx,pmid)` — **alignment audited** (counts, contiguity, per-file shard guard, independent content checks at head/mid/tail). `article_idx` = `.sbcsr` row = BMU point id. Local/git-ignored (Zenodo guardrail). |
| MeSH reference | `mesh_path.bin` int16[n*8], majority-vote level-ℓ tree ancestor, dense-coded, −1-padded. |
| OpenAlex reference | `openalex_tree.sqlite` (4/26/252/4516, dense ids **verified globally-unique + strict single-parent** → exact LCA), `openalex_path.bin` int16[n*4] — built, **96.5% coverage** (28.85M arts, 2.74 topics/art). |
| Metrics | **Cophenetic correlation** (layout) + **AMI** (partition). The original's "AMRI" = AMI here; **Dendrogram Purity is not implemented** (deferred). |
| Scoring | `reference_taxonomies/score_recovery.py` + the `reference_paths.py` column-slice adapter (path array → per-level `L`, `tree_dist` byte-identical to sombench `scale_sweep.py`). |

**Pipeline scripts:** `build_pmid_map.py` (Step 0 alignment), `build_mesh_labels.py`,
`build_openalex_tree.py`, `extract_openalex_labels.py` (S3 stream-filter), `assemble_reference_paths.py`
(streaming SQL join — memory-fixed), `score_recovery.py`.

> **Critical correctness fix (2026-06-22):** `map_dist` computed grid-coordinate differences in
> unsigned `uint32`, underflowing to ~4e9 for half the pairs and collapsing cophenetic to ~0. Fixed
> by casting to float before subtracting. **Regression-anchored** against sombench B1: the harness now
> reproduces +0.1757 @50² / +0.1640 @350² on the known-good MedSOM dumps. Any new layout metric must
> re-validate against that anchor.

---

## 2. Validated results

### 2.1 MeSH cophenetic — the SOM *does* encode the hierarchy
SparseSOM agg cophenetic **+0.18–0.22** (45²/64²/91², σ_min 0.5 and 1.0; 10M pairs), matching/exceeding
MedSOM-Naive (+0.176@50²). σ_min and map size are **not** meaningful levers (flat across the topological
regime — the geometric ceiling). k-means/random layout cophenetic ≈ 0.

### 2.2 Partition recovery (AMI) — close, but a size-dependent crossover
Full-corpus per-level AMI (45²/64²/91², 5-seed CIs): SOM−k-means gaps are small (~±0.01) but **not a flat
tie** — k-means wins coarse levels (L1–L4), the SOM wins deep levels (L7–L8), and the **crossover deepens
with map size** (flips at L5→L6→L7 for 45→64→91; at 91² k-means wins L1–L6, SOM only L7–L8). Absolute AMI
decreases modestly with K *even at full corpus* (over-partitioning of the fixed reference classes — not only
the 30k-sample artifact). Extends HANDOVER's "k-means ties SOM on AMI" from L5 to L8. **Training-seed CIs (5 SOM × 5 k-means initialisations @64², fixed eval sample) confirm the crossover is
real, not a single-seed artifact:** k-means significantly wins L1–L5 (gaps −0.006 to −0.016, all CIs
exclude 0), the SOM significantly wins L7–L8 (+0.004/+0.010), and **L6 is a genuine tie (the crossover
point)**. The training-seed CIs (±0.002–0.003) are ~25× the evaluation-resample CIs (the SOM is more
init-sensitive than k-means), but 7/8 gaps still survive. (Training-seed CIs at 45²/91² remain to do.)
**AMI also cures the coarse-level differ-rate saturation** that makes per-level binary cophenetic
uninformative at L1–L3 — so AMI is the right coarse-level metric.

### 2.3 Method panel (AMI, common 30k sample, K=4096) — the metric discriminates
`SOM (0.144) ≳ k-means (0.137) ≳ PCA+km (0.130) ≳ Ward (0.114) > UMAP+km (0.107) ≫ single-linkage
(−0.03) > random (0.000)`, consistent at every level. The **weak anchors work**: random floors at 0,
single-linkage goes *negative* (chaining) — proving the ~0.1–0.15 values are real signal, not saturation.
PCA≈Ward tie; UMAP-for-clustering is a notch lower (2-D bottleneck).

### 2.4 Layout cophenetic (5-seed CIs) — the dissociation
On *layout* fidelity the order **flips**: `PCA (+0.285) > UMAP (+0.248) > SOM (+0.183±0.002)`, every level,
non-overlapping CIs. The SOM's quantised grid caps layout cophenetic (same-node distance = 0). **So the
SOM is not the best 2-D layout of the hierarchy — continuous embeddings encode it better.** Its value is
the *bundle*: partition quality on par with the best **plus** a discrete, addressable, stable, browsable
atlas at full corpus scale that UMAP/PCA-for-clustering don't provide.

### 2.5 Size sweep
Method rankings are **stable across 45²/64²/91²**. Absolute AMI *decreases* with K on a fixed sample — a
K/n over-clustering artifact, **not** "bigger maps worse"; use full-corpus AMI for absolute size trends.
The single-linkage discrimination washes out at K=8281 (forced over-clustering prevents chaining) → report
the discrimination story at K ≤ 4096.

### 2.6 Convergent validity vs OpenAlex — the external-criterion result (DONE 2026-06-22)
OpenAlex Topics attached to **96.5%** of the corpus (28,847,783 articles; 2.74 topics/art; strict tree →
*true-LCA* cophenetic, stronger than the MeSH majority-vote proxy). The **MeSH↔OpenAlex agreement ceiling**
(taxonomy-vs-taxonomy cophenetic — the max any method could plausibly reach) is **+0.111**.

| map | OpenAlex cophenetic (agg) | % of ceiling | MeSH cophenetic (agg) |
|---|---|---|---|
| SOM 45² | +0.089 | 80% | +0.213 |
| SOM 64² | +0.062 | 56% | +0.181 |
| SOM 91² | +0.098 | 88% | +0.220 |
| k-means 64² (no layout) | +0.032 | 29% | +0.073 |

**The SOM's *layout* recovers the independent, citation-derived, MeSH-orthogonal OpenAlex taxonomy** at
56–88% of the achievable ceiling — clean convergent validity (it trained only on MeSH vectors). The no-layout
k-means floor (~29%) confirms the recovery is the *topological layout*, not just clustering. Per-level
OpenAlex cophenetic **decays domain→topic** (91²: L1 .098 → L2 .082 → L3 .039 → L4 .011) — the geometric
ceiling reproduced on a strict-tree, true-LCA *external* criterion. ANZSRC FoR remains deferred (BigQuery
grant); OpenAlex alone satisfies the convergent-validity argument.

---

## 3. Convergent-validity methodology (OpenAlex / FoR)

Full detail in the archived original (`reference_taxonomies_handover.md`) and its re-anchoring
(`..._correction.md`). The argument: the SOM trains on MeSH-incidence vectors, so OpenAlex Topics
(citation-derived, MeSH-orthogonal) and ANZSRC FoR are **external** criteria — recovery against them is
clean convergent validity, not circular. Both are strict single-parent trees → exact longest-common-prefix
LCA. Output keyed to `corpus.sbcsr` row order via `metadata.sqlite`; `-1` rows skipped by the kernels.
**Bonus:** OpenAlex gives *true-LCA* cophenetic (vs the MeSH majority-vote proxy), which also closes
HANDOVER publication-gap #5 on an external criterion.

---

## 4. Deferred to future research

These are real questions intentionally **out of scope** for the current study, to be characterised later:

1. **Lattice topology — hexagonal vs rectangular, toroidal vs planar.** All current results use a
   **square, planar, edge-clamped** lattice. The advantages/disadvantages are *not yet characterised on
   this corpus*: (a) **toroidal** wrap removes boundary/edge effects and dead border nodes (no corners),
   and would change `map_dist` to a wrapped metric — plausibly lifting cophenetic for periphery articles;
   it stays flat (genus-1, zero curvature) so it does **not** escape the geometric ceiling, and a *toroidal
   hyperbolic* SOM is impossible (Gauss–Bonnet, needs genus ≥ 2). (b) **Hexagonal** neighbourhoods give
   uniform 6-neighbour adjacency (vs rect 4/8) and more isotropic neighbourhood functions, which may change
   TE and fine-level recovery. Open question: do hex/toroidal materially change cophenetic / AMI / coverage,
   or (as HANDOVER's synthetic work hinted for hex) are they second-order vs the geometric ceiling? Needs a
   controlled hex×{planar,toroidal} sweep with the validated harness. (See HANDOVER §4.3 for the
   curvature/Gauss–Bonnet context.)
2. **ANZSRC FoR (third reference)** — requires the Dimensions BigQuery academic grant; OpenAlex alone
   satisfies the track, FoR is a bonus.
3. **Dendrogram Purity** — net-new kernel, not implemented; cophenetic + AMI carry the argument.
4. **Depth-8 MeSH labelling soundness** — majority-vote per level thins to fewer distinct codes at L7/L8
   (non-monotonic differ-rate); whether a true tree-number LCA reference changes deep-level reads is open.
5. **Tuned UMAP** — current UMAP uses fixed `n_neighbors=30`; HANDOVER found `n_neighbors` must scale with
   N, so UMAP's layout/partition numbers are likely a floor.
6. **Training-seed CIs** — current CIs capture sample/embedding variability; map-to-map (init-seed)
   variability for the small SOM−k-means gaps is a separate, more expensive check.
7. **More comparators** — LDA (topic-argmax) and HDBSCAN to round out the AMI panel.
8. **Curvature / hyperbolic embeddings** — the geoopt Poincaré line (HANDOVER §4.3, run notes
   `medline_som_poincare_geoopt`, `medline_hyperbolic_som`) lifts the ceiling for *embeddings* but not for a
   lattice SOM; revisiting under the validated harness is deferred.

---

## 5. Provenance
- Scripts: `SparseBinEval/reference_taxonomies/*.py`.
- Historical spec (retired): `docs/Retired/reference_taxonomies_handover.md` + `..._correction.md`.
- Scientific baseline & geometric-ceiling/curvature context: `sombench/HANDOVER.md` (2026-06-17).
- Validated-results detail + operational gotchas: memories `convergent-validity-results`, `som-scoring-gotchas`.
- Harness regression anchor: `sombench/results/medline_som_{50,350}/dump/bmu_year.bin` (+0.1757/+0.1640).
