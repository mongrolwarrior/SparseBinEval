# CORRECTION ADDENDUM — Reference-Taxonomies Handover (re-anchored to SparseSOM)

**Read this alongside the original "Independent Reference Taxonomies (OpenAlex Topics + ANZSRC
FoR)" handover.** That handover's *methodology, output contract, metric wiring, and all three
⚠️ verify-before-trust callouts are correct and unchanged.* What's stale is the environment and
the data anchors — the original author wrote it against the `MedSOM` / `ubuntu-gpu` era and was
unaware the project has since become **SparseSOM** + **SparseBinEval**, running inside a Docker
container *on* `ubuntu-gpu`. This addendum re-anchors it and records what was verified in the
workspace on 2026-06-21.

---

## Verified state of the workspace (facts, not assumptions)

| Thing the handover assumes | Verified reality |
|---|---|
| `ubuntu-gpu`, "reached from ubuntu-gpu" | You are **inside a Docker container hosted on ubuntu-gpu**. Egress still works: **`api.openalex.org` is reachable from the container** (confirmed). |
| OpenAlex/`pyroaring`/`orjson`/BigQuery libs available | **NOT installed** — `pyroaring`, `orjson`, `google-cloud-bigquery` all fail to import; no `aws` CLI. `pip install pyroaring orjson requests google-cloud-bigquery awscli` first. |
| `articles.bin` (PMID per point id) "exists" | The **producer exists and is built**: `MedSOM/medline_extract/` (C++) streams `(pmid, year)` to `articles.bin` and writes MeSH labels (`main_mesh_labels`, `writers.hpp`, `xml_parser.hpp` with a `skipped_no_pmid` counter). But **no `articles.bin` file is on disk** → must be (re)generated, and see the pipeline-alignment problem below. |
| `metadata.sqlite(article_idx, pmid)` "exists" | **Does not exist** anywhere in the workspace. Must be built (§A). |
| `mesh_path.bin`, depth 8, DP/coph/AMRI kernels "existing" | **Mostly real, one gap.** MeSH tree machinery exists (`sombench/experiments/exp3_mesh_hierarchy/mesh_hierarchy.py`). **Cophenetic correlation is implemented and battle-tested** — the headline "B1" metric: Pearson(map-dist, MeSH-tree-dist) over sampled pairs + per-level (`experiments/medline_som_viz/{scale_sweep,tree_node_som,_perlevel}.py`; real results SOM +0.176–0.187 vs k-means ≈0). **NMI / AMI is implemented** (multi-level, `exp3`/`scale_sweep`). **Dendrogram Purity is NOT** — absent from the code *and* from `HANDOVER.md`'s own plan → net-new. The handover's **"AMRI" has no match**; the implemented metric is **AMI**. No `mesh_path.bin` *file* exists — tree distance is computed on the fly by `tree_dist(L, a, b)` from a **list of per-level label arrays `L`**, onto which the `int16[n*depth]` path array maps by column (see §B). |
| The SOM being scored | **SparseSOM** over `corpus.sbcsr` (the `.scsr`/`SCSR1` MeSH-incidence corpus); the clustering = BMU/node assignments. |

---

## §A — The PMID alignment problem (new Step 0; this is the real blocker)

There are **two different MEDLINE pipelines** in the workspace, and they don't share row order:

1. **`MedSOM/medline_extract/`** (C++) — extracts `<PMID>` and writes `articles.bin` of `(pmid, year)`
   aligned to its own MeSH output. This is where PMIDs live.
2. **`SparseBinEval/sbeval/medline.py`** (Python) — produced the **current `corpus.sbcsr`** (and the
   Zenodo dataset). It extracts **only `<DescriptorName UI>`, stores no PMID**, applies `--min-mesh 5`,
   and orders rows by its own shard pass. `data/README.md` states the published corpus is
   deliberately anonymized (no PMIDs — rows can't be traced to articles).

So `medline_extract`'s `articles.bin` is **not** aligned to `corpus.sbcsr`'s rows — you cannot just
join them. To get a PMID↔`corpus.sbcsr`-row map (`article_idx` = `.sbcsr` row index = point id `p`),
pick one:

- **(a) Regenerate the corpus through `medline_extract`** so `articles.bin` + the MeSH CSR share order,
  then (re)train the SparseSOM map on that corpus. Cleanest alignment; costs a retrain.
- **(b) Add PMID capture to `medline.py`** in the *same pass / same file order / same `--min-mesh`*
  that wrote `corpus.sbcsr`, emitting `(article_idx, pmid)`. No retrain, but the order/filter
  invariant must match exactly or every downstream join silently misaligns.

Either way, write `metadata.sqlite(article_idx INTEGER PRIMARY KEY, pmid INTEGER)`. **Keep it
local / git-ignored** (under `data/`, already ignored). This re-introduces PMIDs for the local
convergent-validity analysis only, and replaces the original's ⚠️ "assumed `metadata.sqlite`
schema" caveat.

> **⚠️ Publication guardrail.** `metadata.sqlite` (and any title/abstract data this track touches)
> **must NOT be uploaded to Zenodo or otherwise published without explicit review and confirmation
> by the repository owner.** The Zenodo bundle's commitment is *no per-article title/abstract text
> and no PMIDs* — see `data/README.md`. These artifacts stay local by design.

## §B — Rename / path corrections (apply throughout)

| Handover says | Use instead |
|---|---|
| `~/dev/projects/<medsom>/data/reference_taxonomies/` | **`SparseBinEval/data/reference_taxonomies/`** (`data/` is git-ignored — correct home for PMID-bearing local artifacts) |
| `articles.bin` as the article anchor | **`corpus.sbcsr`** (`.scsr`/SCSR1); row index = `article_idx` = point id `p`. `articles.bin` (from `medline_extract`) is the PMID *source*, reconciled per §A |
| "MeSH reference `mesh_path.bin` (existing)" | Build it from `sombench/experiments/exp3_mesh_hierarchy/mesh_hierarchy.py`'s tree machinery (MeSH UI → tree-number → dense path). Note `vocab.json` holds MeSH **UIs (D-codes), not tree numbers**, so you need the MeSH descriptor→tree-number table (the `mesh_hierarchy.py` desc-XML parser already does this), and MeSH is **poly-hierarchical** (a UI → several tree numbers) → the MeSH reference keeps the multi-path handling the strict OpenAlex/FoR trees avoid |
| "DP / cophenetic / AMRI kernels already consume `(path_array, depth)`" | **Cophenetic + NMI/AMI already exist and are battle-tested**; they consume **`L` = a list of per-level label arrays** via `tree_dist(L, a, b)` (counts the levels at which two articles differ, skipping `-1`). That is *exactly* your `int16[n*depth]` path array **sliced by column** (`-1` padding = the skip), so the OpenAlex/FoR `*_path.bin` feed the existing cophenetic/AMI code through a trivial column-slice adapter — **no new metric code**. The handover's **"AMRI" = AMI** here. Only **Dendrogram Purity** is genuinely missing (not in the repo or the prior plan) — treat it as optional/net-new; cophenetic is the headline metric regardless |

## §C — Output contract: unchanged

The `int16[n*depth]` path arrays, the `*_labels.sqlite`, and `join_report.json` are
taxonomy-agnostic and correct as written. Only the anchors change: key everything to
`corpus.sbcsr`'s row order via the new `metadata.sqlite`, and write under
`SparseBinEval/data/reference_taxonomies/`.

## §D — Container setup (verified)

- **Deps (not installed):** `pip install pyroaring orjson requests google-cloud-bigquery awscli`.
- **Network:** OpenAlex API egress from the container is **confirmed working** → `build_openalex_tree.py`
  runs in-container as-is.
- **Disk is the constraint:** the OpenAlex `works` S3 snapshot is **hundreds of GB gzipped** and this
  container won't hold it. Since you're a *container on* `ubuntu-gpu`, run the `aws s3 sync` +
  stream-filter **on the host** and read the result through a mount, or stream-filter on the fly —
  don't land it inside the container.
- **BigQuery:** still needs the Dimensions academic grant + `gcloud` auth in the container; FoR stays
  the optional third reference.

## §E — Unchanged and still correct (don't re-litigate)

The convergent-validity argument; OpenAlex's independence-from-MeSH (and the Scopus-ASJC caveat); the
three ⚠️ items (S3 path, `ids.pmid` URL format, Dimensions BQ schema); the multi-label vs `primary`
projection; the taxonomy-vs-taxonomy baseline; strict-tree LCA = longest-common-prefix. The
"trained on MeSH vectors → OpenAlex/FoR are external criteria" logic holds for **SparseSOM** verbatim,
and the MeSH-seeded routing's oracle/fair-entrant distinction applies unchanged.

## §F — Revised build order

0. **(new)** Reconcile pipelines (§A) and build `metadata.sqlite(article_idx, pmid)` in `corpus.sbcsr`
   row order; keep local/git-ignored.
0b. **(adapter, not a rebuild)** Write a thin column-slice adapter so `*_path.bin` → `L` (path-array
   column `l` = the level-`l` array) and feed the **existing** `tree_dist` + cophenetic + AMI kernels.
   Materialize `mesh_path.bin` the same way if you want one uniform interface. Dendrogram Purity is the
   only net-new kernel — optional.
1. `build_openalex_tree.py` → `openalex_tree.sqlite` (API egress confirmed; verify counts 4/26/254/~4516).
2. Sync (on host) + `extract_openalex_labels.py` → `openalex_labels.sqlite`.
3. (If granted) `extract_for.sql` + loader → `for_labels.sqlite`; build `for_tree.sqlite` from ANZSRC 2020.
4. `assemble_reference_paths.py` → `openalex_path.bin`, `for_path.bin`, `join_report.json` (keyed to `corpus.sbcsr`).
5. Run the existing cophenetic + AMI kernels with the OpenAlex `L` (4 levels) and FoR `L` (2 levels);
   add the taxonomy-vs-taxonomy baseline. (Optionally add Dendrogram Purity.)

---

### Provenance pointers (where the ancestor pieces live)
- MeSH tree + per-article level labels: `sombench/experiments/exp3_mesh_hierarchy/{mesh_hierarchy.py, analyse_exp3.py, run_exp3.py, infer_bmu.py}`
- Recovery kernels — cophenetic + AMI/NMI + the reusable `tree_dist(L,a,b)`: `sombench/experiments/medline_som_viz/{scale_sweep.py, tree_node_som.py, _perlevel.py, _svd_perlevel.py}` (Dendrogram Purity is **not** here — net-new)
- PMID/year + MeSH-label extractor (built): `MedSOM/medline_extract/` (`src/main_mesh_labels.cpp`, `include/{writers,xml_parser,mesh_parser}.hpp`)
- First-handover design (MeSH reference, DP/coph/AMRI plan): `sombench/HANDOVER.md`, `MedSOM/medline_extract/docs/SOM_HANDOFF.md`
- Current corpus + ingestion: `SparseBinEval/sbeval/medline.py`, `data/medline/processed/pubmed26_ge5/{corpus.sbcsr, vocab.json, summary.json}`, `data/README.md`
