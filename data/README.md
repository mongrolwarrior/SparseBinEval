# MEDLINE MeSH-incidence corpus (PubMed 2026 baseline) — dataset record

This is the dataset bundle hosted on Zenodo for reproducing the SparseSOM size-sweep. It is a
**MeSH-descriptor incidence matrix** derived from the PubMed annual baseline: each row is one
article, each column is one MeSH descriptor, and an entry marks that the article was indexed with
that descriptor.

## What it contains — and deliberately does NOT

Contains **only** NLM-produced, public-domain controlled-vocabulary content:

| file | contents |
|---|---|
| `corpus.sbcsr` | full incidence matrix (CSR: `uint32 row_ptr`, `uint16 col_idx`) |
| `corpus.train.sbcsr` / `corpus.test.sbcsr` | the 98% / 2% split used for held-out QE (seed 0) |
| `vocab.json` | the column→MeSH map: a list of MeSH descriptor **UI codes** (e.g. `"D000123"`) |
| `summary.json` | provenance: article/feature counts, the `min_mesh≥5` filter, source filenames, timestamps |

It contains **no abstracts, no titles, no author/journal data, and no PMIDs** — rows are anonymous
and cannot be traced back to individual articles. Only the MeSH descriptor *codes* are stored, not
the descriptor strings. The pipeline (`sbeval/medline.py`) reads exactly one field from each
`<PubmedArticle>`: the `UI` attribute of `<DescriptorName>`. Nothing else is parsed or retained.

> ## ⚠️ Publication guardrail — do not bypass without sign-off
> Keeping the Zenodo bundle free of per-article title/abstract text **and** PMIDs is a deliberate
> commitment. Downstream work may build **local** artifacts that re-introduce PMIDs (e.g. to join
> external taxonomies — see `docs/reference_taxonomies_handover_correction.md`) or that touch
> article text. **Any per-article title/abstract data, and any PMID-bearing artifact, must NOT be
> uploaded to Zenodo (or otherwise published) without explicit review and confirmation by the
> repository owner.** The published dataset stays anonymized by design; such artifacts are
> local/git-ignored only.

## License & attribution

- **Derived structures (these files): CC0 1.0** — public-domain dedication. The underlying MeSH
  indexing is produced by the U.S. National Library of Medicine and is not subject to copyright.
- **Courtesy of the U.S. National Library of Medicine.** This product uses publicly available data
  from NLM but is not endorsed or certified by NLM.
- **Static snapshot — not current data.** This is a frozen snapshot of the **PubMed 2026 annual
  baseline**; it does not reflect the most current or accurate data available from NLM. For live
  data, ingest directly from <https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/>.

## Provenance / regenerating from source

Produced by `sbeval/medline.py` from the 1334-file PubMed 2026 baseline with the `--min-mesh 5`
filter (drop articles with fewer than 5 MeSH descriptors). To rebuild from scratch instead of
downloading this bundle — pulling the raw XML yourself under NLM's terms — run the pipeline's
`download` + `process` stages. See `summary.json` for the exact file list and parameters.

## Format

`.sbcsr`: 24-byte header `magic "SBCSR1\0\0", uint32 n_samples, n_features, n_nonzeros, reserved`,
then `uint32 row_ptr[n_samples+1]` and `uint16 col_idx[n_nonzeros]` (little-endian). A column id
`c` maps to MeSH UI `vocab.json[c]`.
