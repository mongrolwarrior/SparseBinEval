> **ARCHIVED ORIGINAL (recovered 2026-06-22).** This is the base handover that
> `reference_taxonomies_handover_correction.md` re-anchors. It was written for the
> `MedSOM`/`ubuntu-gpu` era; the environment and data anchors are **stale** (now SparseSOM +
> SparseBinEval, `corpus.sbcsr` not `articles.bin`, metric is **AMI** not "AMRI", Dendrogram
> Purity not implemented, PMID-alignment Step 0 required). Read it **with** the correction
> addendum. For the *validated outcomes* of the implemented pipeline (incl. the `map_dist`
> uint32 fix and MeSH cophenetic +0.18–0.22), see memory `convergent-validity-results`.

---

# Handover: Independent Reference Taxonomies (OpenAlex Topics + ANZSRC FoR) for the Convergent-Validity Track

**Target:** an implementing agent in VS Code on `ubuntu-gpu`.
**Goal:** attach two *MeSH-independent* hierarchical taxonomies to the MedSOM corpus, keyed on PMID, in exactly the format the existing hierarchy-recovery kernels (Dendrogram Purity, Cophenetic Correlation, AMRI) already consume — so the convergent-validity track can score any clustering against a **second** (and third) reference tree with no change to the metric code.

This runs in your environment, not in a sandbox: the OpenAlex S3 snapshot, the OpenAlex API, and Google BigQuery are all reached from `ubuntu-gpu`. The Python libraries used (`pyroaring`, `orjson`, `requests`, `google-cloud-bigquery`) are all on PyPI.

> **VERIFY-before-trust callouts** are marked ⚠️ throughout. They flag the three things most likely to have drifted (S3 snapshot path, OpenAlex `ids.pmid` format, Dimensions BigQuery schema) and the one thing I'm assuming about your side (`metadata.sqlite` schema). Confirm each against current docs / your DB before a full run.

---

## 0. How this plugs into the existing metrics

The first handover defined the MeSH reference as a per-article **integer path** (`int16 mesh_path[n*8]`, dense-coded, `-1`-padded) with reference tree-distance = *longest common prefix* of two paths (because MeSH tree-numbers are dotted prefixes). The metric kernels are parameterised by `(path_array, depth)`.

This spec produces the identical artifact for two more taxonomies, so the only change to the recovery code is swapping the array and the depth constant:

| Reference | Path depth | Array file | LCA rule |
|---|---|---|---|
| MeSH (existing) | 8 | `mesh_path.bin` | longest common prefix |
| OpenAlex Topics | 4 (domain→field→subfield→topic) | `openalex_path.bin` | longest common prefix |
| ANZSRC FoR | 2 (division→group) | `for_path.bin` | longest common prefix |

Both new taxonomies are **strict trees** — in OpenAlex each topic has exactly one subfield, one field, one domain — which is *cleaner* than MeSH (where a descriptor can carry several tree-numbers). So the longest-common-prefix LCA is exact, with no poly-hierarchy special-casing on the reference side.

---

## 1. Output contract (build to this)

All files land in `~/dev/projects/<medsom>/data/reference_taxonomies/` and are keyed to the **same `article_idx`** as `articles.bin` (row index = point id `p`).

```
reference_taxonomies/
  openalex_tree.sqlite     # static 4-level tree: nodes + topic→path map + dense-index maps
  openalex_labels.sqlite   # per-PMID: primary topic + full multi-label topic set
  openalex_path.bin        # int16[n*4], article_idx order, primary-topic path, -1 padded
  for_tree.sqlite          # static 2-level tree: divisions + groups
  for_labels.sqlite        # per-PMID: FoR codes (multi-label)
  for_path.bin             # int16[n*2], article_idx order, primary FoR path, -1 padded
  join_report.json         # coverage, dedupe counts, multi-label distribution, source versions
```

`primary` path = the single-label projection used by the drop-in metric kernels; the full multi-label sets live in the `*_labels.sqlite` for the set-based robustness checks (same policy as the MeSH single-label projection in the first handover).

---

## 2. Reference tree #1 — OpenAlex Topics

### 2.1 Structure
Four levels: **4 domains → 26 fields → 254 subfields → ~4,516 topics**. Topics are derived by citation-community clustering + LLM labelling + a deep-learning assignment model (CWTS Leiden lineage); domain/field/subfield align to Scopus ASJC but are applied **per work**, not per journal. Each work carries a `primary_topic` plus a scored `topics[]` array. Independent of MeSH (different signal: citation community vs manual indexing).

> Interpretation note for the methods section: because OpenAlex's upper three levels *are* Scopus ASJC, OpenAlex and Scopus are not independent of each other at domain/field/subfield — but both are independent of MeSH, which is all the convergent-validity argument needs. The genuinely citation-derived, MeSH-orthogonal resolution is the topic level.

### 2.2 Build the static tree from the OpenAlex API → `openalex_tree.sqlite`

The authoritative hierarchy comes from the `/domains`, `/fields`, `/subfields`, `/topics` endpoints (the topic object embeds its subfield/field/domain). Build once; it changes only when OpenAlex revises the taxonomy (pin the date).

```python
# build_openalex_tree.py  — run on ubuntu-gpu (reaches api.openalex.org)
import requests, sqlite3, re, time

MAILTO = "andrew@<your-domain>"   # polite pool; faster, more reliable
BASE = "https://api.openalex.org"

def _intid(url):                  # ".../domains/1" -> 1 ; ".../T11636" -> 11636
    m = re.search(r'/([A-Za-z]?)(\d+)$', url)
    return int(m.group(2))

def page(entity):
    cur, out = "*", []
    while cur:
        r = requests.get(f"{BASE}/{entity}",
                         params={"per-page": 200, "cursor": cur, "mailto": MAILTO},
                         timeout=60).json()
        out += r["results"]
        cur = r["meta"].get("next_cursor")
        time.sleep(0.1)
    return out

domains  = page("domains")     # 4
fields   = page("fields")      # 26
subfields= page("subfields")   # 254
topics   = page("topics")      # ~4516, each has .domain/.field/.subfield {id,display_name}

con = sqlite3.connect("openalex_tree.sqlite"); c = con.cursor()
c.executescript("""
CREATE TABLE nodes(level TEXT, oa_int INTEGER, dense INTEGER, name TEXT,
                   PRIMARY KEY(level, oa_int));
CREATE TABLE topic_path(topic_oa INTEGER PRIMARY KEY,
                        domain_dense INTEGER, field_dense INTEGER,
                        subfield_dense INTEGER, topic_dense INTEGER);
""")

# dense indices per level (contiguous int16, stable sort by oa_int)
def dense_map(items):
    ids = sorted(_intid(x["id"]) for x in items)
    return {oid:i for i,oid in enumerate(ids)}
dm = {lvl:dense_map(its) for lvl,its in
      [("domain",domains),("field",fields),("subfield",subfields),("topic",topics)]}

for lvl,its in [("domain",domains),("field",fields),("subfield",subfields),("topic",topics)]:
    for x in its:
        oi=_intid(x["id"])
        c.execute("INSERT INTO nodes VALUES(?,?,?,?)",(lvl,oi,dm[lvl][oi],x["display_name"]))

for t in topics:
    to=_intid(t["id"])
    c.execute("INSERT INTO topic_path VALUES(?,?,?,?,?)",(
        to,
        dm["domain"][_intid(t["domain"]["id"])],
        dm["field"][_intid(t["field"]["id"])],
        dm["subfield"][_intid(t["subfield"]["id"])],
        dm["topic"][to]))
con.commit(); con.close()
```

The 4-tuple `(domain_dense, field_dense, subfield_dense, topic_dense)` is the article's reference path; common-prefix length → LCA depth, exactly as MeSH tree-numbers. All four levels fit `int16` after dense remapping (≤4516 < 32767).

### 2.3 Per-article labels from the bulk snapshot → `openalex_labels.sqlite`

The full OpenAlex `works` snapshot is large (hundreds of GB gzipped). **Stream-filter it against your PMID set** rather than landing it whole.

⚠️ **Verify the current snapshot path** at `docs.openalex.org/download-all-data` before syncing. As of writing it is the public, unsigned S3 bucket:
```bash
# only the works partitions; no AWS account needed
aws s3 sync "s3://openalex/data/works/" "$SNAP/works/" --no-sign-request
```

⚠️ **Verify `ids.pmid` format** — OpenAlex stores it as a URL (e.g. `https://pubmed.ncbi.nlm.nih.gov/12345678`); extract the trailing integer.

```python
# extract_openalex_labels.py
import glob, gzip, sqlite3, re, orjson
from pyroaring import BitMap            # compact 30M-int membership test

# 1) PMID set from your corpus
mc = sqlite3.connect("…/metadata.sqlite")
pmids = BitMap(r[0] for r in mc.execute("SELECT pmid FROM articles WHERE pmid IS NOT NULL"))
mc.close()

# 2) topic -> dense path lookup
tc = sqlite3.connect("openalex_tree.sqlite")
TPATH = {r[0]:(r[1],r[2],r[3],r[4]) for r in tc.execute("SELECT * FROM topic_path")}
tc.close()

PMID_RE = re.compile(r'(\d+)$')
def pmid_of(work):
    u = (work.get("ids") or {}).get("pmid")
    if not u: return None
    m = PMID_RE.search(u); return int(m.group(1)) if m else None
def topic_int(turl): return int(re.search(r'/T(\d+)$', turl).group(1))

out = sqlite3.connect("openalex_labels.sqlite"); oc = out.cursor()
oc.executescript("""
CREATE TABLE lab(pmid INTEGER PRIMARY KEY,
  domain_dense INT, field_dense INT, subfield_dense INT, topic_dense INT,
  topics_json TEXT);          -- full multi-label set + scores
""")
seen=set(); dup=0
for fn in glob.glob(f"{SNAP}/works/**/*.gz", recursive=True):
    with gzip.open(fn) as fh:
        for line in fh:
            w = orjson.loads(line)
            p = pmid_of(w)
            if p is None or p not in pmids: continue
            if p in seen: dup+=1; continue       # rare duplicate works per PMID
            pt = w.get("primary_topic")
            if not pt: continue                   # some works unclassified
            path = TPATH.get(topic_int(pt["id"]))
            if not path: continue
            seen.add(p)
            multi=[{"t":topic_int(t["id"]),"s":t.get("score")} for t in (w.get("topics") or [])]
            oc.execute("INSERT INTO lab VALUES(?,?,?,?,?,?)",
                       (p,*path, orjson.dumps(multi).decode()))
out.commit(); out.close()
print("matched", len(seen), "dup_works", dup)
```

Notes: parallelise across `*.gz` parts with `multiprocessing` (one worker per part, merge at the end); `pyroaring` keeps the 30M-PMID membership test at ~tens of MB; `orjson` ≫ stdlib `json` for this volume. Multi-label policy mirrors MeSH: `primary_topic` → the drop-in path; full `topics[]` retained for set-based checks.

---

## 3. Reference tree #2 — ANZSRC Fields of Research (Dimensions via BigQuery)

### 3.1 Structure & access
ANZSRC 2020 FoR, applied **at the document level** by Dimensions' text classifier (not journal-level), multi-label. Dimensions implements the **2-digit Divisions (22)** and **4-digit Groups (~171)** — two usable levels. Directly relevant to you as the Australian/NZ standard.

⚠️ **Access caveat:** the bulk FoR labels are reached through **Dimensions on Google BigQuery**, which is free for academic/non-commercial use **but requires an access grant** (apply via Digital Science). If you can't get it, OpenAlex alone fully satisfies the convergent-validity track; treat FoR as a bonus third reference.

### 3.2 Extraction
Upload your PMID list to a BigQuery temp table and join server-side; export only the small result.

⚠️ **Verify dataset and field names** against the current Dimensions BQ schema — the nested `category_for` structure and the `pmid` field name have changed across releases. Representative query:

```sql
-- extract_for.sql  (adjust to the Dimensions BQ schema you were granted)
WITH mine AS (SELECT CAST(pmid AS STRING) AS pmid FROM `myproj.medsom.pmids`)
SELECT
  p.pmid,
  ARRAY(SELECT code FROM UNNEST(p.category_for.first_level.codes)  AS code) AS for_div,
  ARRAY(SELECT code FROM UNNEST(p.category_for.second_level.codes) AS code) AS for_grp
FROM `dimensions-ai.data_analytics.publications` AS p
JOIN mine USING (pmid)
WHERE p.pmid IS NOT NULL;
```

Load PMIDs and pull results with `google-cloud-bigquery`; write `for_labels.sqlite(pmid, for_div_json, for_grp_json)`. Build `for_tree.sqlite` from the ANZSRC 2020 code list (divisions are 2-digit, groups 4-digit with the division as their 2-digit prefix — so the **code itself is the path**, no remap needed; 4-digit ≤ 9999 < int16 max). Primary FoR = the highest-ranked group Dimensions assigns (or the single group if one); store the rest as the multi-label set.

---

## 4. Join to the pipeline → `*_path.bin`

Pack both taxonomies into `article_idx` order so they sit beside `articles.bin`.

⚠️ **Assumed schema:** `metadata.sqlite` has `articles(article_idx INTEGER PRIMARY KEY, pmid INTEGER, …)`. Adjust the SELECT to your actual table/column names.

```python
# assemble_reference_paths.py
import sqlite3, numpy as np, orjson, json

meta = sqlite3.connect("…/metadata.sqlite")
rows = list(meta.execute("SELECT article_idx, pmid FROM articles ORDER BY article_idx"))
n = rows[-1][0] + 1

oa = {r[0]:(r[1],r[2],r[3],r[4]) for r in
      sqlite3.connect("openalex_labels.sqlite").execute(
      "SELECT pmid,domain_dense,field_dense,subfield_dense,topic_dense FROM lab")}
fr = {}
for pmid, grp_json in sqlite3.connect("for_labels.sqlite").execute("SELECT pmid,for_grp_json FROM lab"):
    g = json.loads(grp_json)
    if g:
        grp = int(g[0]); fr[pmid] = (grp//100, grp)   # division = 2-digit prefix

oa_path = np.full((n,4), -1, np.int16)
fr_path = np.full((n,2), -1, np.int16)
miss_oa = miss_fr = 0
for idx, pmid in rows:
    if pmid in oa: oa_path[idx] = oa[pmid]
    else: miss_oa += 1
    if pmid in fr: fr_path[idx] = fr[pmid]
    else: miss_fr += 1

oa_path.tofile("openalex_path.bin")
fr_path.tofile("for_path.bin")
json.dump({"n":n,
           "openalex_coverage":1-miss_oa/n,
           "for_coverage":1-miss_fr/n},
          open("join_report.json","w"), indent=2)
```

`-1` rows are articles with no classification (genuinely unclassified, or absent from that source) — the metric kernels must skip pairs touching a `-1` path, identical to how they handle MeSH-unlabelled points.

---

## 5. Wiring into the recovery metrics (what changes: almost nothing)

In the DP / cophenetic / AMRI code, the reference side is `(path_array, depth)`. Run each metric three times:

- MeSH: `mesh_path.bin`, depth 8 (existing)
- OpenAlex: `openalex_path.bin`, depth 4
- FoR: `for_path.bin`, depth 2

The cophenetic kernel's `mesh_tree_distance()` (longest-common-prefix on the int path) is reused verbatim — just point it at the new array and depth. For AMRI's resolution sweep, the level truncations are: OpenAlex {domain, field, subfield, topic}; FoR {division, group}.

**The new analyses this second tree unlocks:**

1. **Convergent-validity scores.** Compute each unsupervised method's recovery (DP/coph/AMRI) against OpenAlex and FoR, *not just MeSH*. Because the SOM was trained on MeSH vectors, OpenAlex/FoR are fully external criteria → recovery against them is clean convergent validity, not circular. Your top-down MeSH-seeded routing, which is an oracle on the MeSH row, becomes a **fair entrant** on the OpenAlex/FoR rows (it never saw those taxonomies).
2. **Taxonomy-vs-taxonomy baseline.** Score the MeSH tree against the OpenAlex tree (and FoR) directly over the corpus — i.e. how much do the curated, citation-derived, and ANZSRC organisations agree with *each other*? This is the essential context line: it tells you the ceiling of agreement any method could plausibly reach, so a SOM-vs-OpenAlex score is read against "how well even MeSH itself agrees with OpenAlex," not against a perfect-1.0 fantasy.

---

## 6. QA / reproducibility

- **Coverage** (`join_report.json`): expect high OpenAlex match (it ingests all of PubMed) but <100% — some works are unclassified for want of metadata. Report it; stratify downstream metrics by covered-only.
- **Join sanity cross-tab:** on a sample, articles with MeSH `Neoplasms` should concentrate in OpenAlex domain *Health Sciences* / an Oncology subfield, and FoR division *32 Biomedical and Clinical Sciences*. A quick contingency check catches a broken PMID join immediately (it's a correctness check, not the analysis).
- **Multi-label distribution:** record #topics/article and #FoR-groups/article; mirrors the MeSH multi-label handling and tells you how lossy the single-label primary projection is.
- **Version pinning:** record the OpenAlex snapshot date, the OpenAlex taxonomy build date, and `ANZSRC FoR 2020` in `join_report.json` for the methods section.
- **Dedupe:** log duplicate works-per-PMID (kept = first/highest-score); should be rare.

---

## 7. Honest caveats

- All three sources classify from **title/abstract/metadata + (for OpenAlex) citations**, none from full text — the same input regime as your corpus and as MeSH indexing, so no full-text advantage confound.
- OpenAlex domain/field/subfield **are** Scopus ASJC (independent of MeSH; not independent of Scopus) — state this so the convergent-validity claim is scoped to "independent of MeSH."
- FoR requires the Dimensions BigQuery grant; if unavailable, proceed with OpenAlex as the sole independent reference — the track still stands.
- The two `⚠️` infra details (S3 path, BQ schema) and the assumed `metadata.sqlite` schema are the only places this spec can go stale; verify them first and the rest follows.

---

## 8. Build order

1. `build_openalex_tree.py` → `openalex_tree.sqlite` (verify counts: 4 / 26 / 254 / ~4516).
2. Sync + `extract_openalex_labels.py` → `openalex_labels.sqlite` (check match rate).
3. (If granted) `extract_for.sql` + loader → `for_labels.sqlite`; build `for_tree.sqlite` from the ANZSRC code list.
4. `assemble_reference_paths.py` → `openalex_path.bin`, `for_path.bin`, `join_report.json`.
5. Re-run DP / cophenetic / AMRI with `(openalex_path.bin, 4)` and `(for_path.bin, 2)`; add the taxonomy-vs-taxonomy baseline.

---

## 9. References
- OpenAlex Topics: four-level domain→field→subfield→topic taxonomy (~4,516 topics), citation-clustering + LLM labelling + DL classifier, CWTS Leiden lineage; docs at docs.openalex.org/api-entities/topics; bulk data docs.openalex.org/download-all-data (CC0).
- OpenAlex topic-classification code/model: github.com/ourresearch/openalex-topic-classification.
- ANZSRC 2020 Fields of Research; Dimensions article-level FoR classifier (Quantitative Science Studies 4(1):127, 2023, "Recategorising research: Mapping from FoR 2008 to FoR 2020 in Dimensions"); Dimensions on Google BigQuery (`dimensions-ai.data_analytics.publications`).
