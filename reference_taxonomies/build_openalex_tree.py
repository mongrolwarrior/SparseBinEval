#!/usr/bin/env python3
"""Step 1 of the convergent-validity track: build the static OpenAlex Topics tree.

Four levels — 4 domains → 26 fields → 254 subfields → ~4516 topics — pulled from the OpenAlex
API (the topic object embeds its subfield/field/domain). Each level is dense-remapped to a
contiguous int16 index (stable sort by OpenAlex id), so the 4-tuple
(domain_dense, field_dense, subfield_dense, topic_dense) is an article's reference PATH and
common-prefix length is the LCA depth — exactly like MeSH tree-numbers. Build once; pin the date.

Output: data/reference_taxonomies/openalex_tree.sqlite  (tables: nodes, topic_path)
See docs/reference_taxonomies_handover_correction.md.
"""
import os
import re
import sqlite3
import time

import requests

MAILTO = os.environ.get("OPENALEX_MAILTO", "mongrolwarrior@gmail.com")   # polite pool
BASE = "https://api.openalex.org"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "data", "reference_taxonomies", "openalex_tree.sqlite")


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


def dense_map(items):              # contiguous int16 index per level, stable by OpenAlex id
    ids = sorted(_intid(x["id"]) for x in items)
    return {oid: i for i, oid in enumerate(ids)}


def main():
    print("paging OpenAlex taxonomy (domains/fields/subfields/topics)...")
    domains = page("domains")
    fields = page("fields")
    subfields = page("subfields")
    topics = page("topics")
    print(f"  domains={len(domains)} fields={len(fields)} subfields={len(subfields)} topics={len(topics)}")
    # domains/fields are the stable ASJC top levels; subfields/topics drift as OpenAlex revises
    # the taxonomy (handover said 254/~4516; observed 252/4516 on 2026-06-21). Record actuals.
    if not (len(domains) == 4 and len(fields) == 26):
        raise SystemExit("unexpected top-level counts (domains!=4 or fields!=26) — investigate before trusting")
    if not (240 <= len(subfields) <= 260 and 4000 <= len(topics) <= 5200):
        print(f"  ⚠️ subfield/topic counts well outside expected range — verify the taxonomy")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    con = sqlite3.connect(OUT)
    c = con.cursor()
    c.executescript("""
    CREATE TABLE nodes(level TEXT, oa_int INTEGER, dense INTEGER, name TEXT,
                       PRIMARY KEY(level, oa_int));
    CREATE TABLE topic_path(topic_oa INTEGER PRIMARY KEY,
                            domain_dense INTEGER, field_dense INTEGER,
                            subfield_dense INTEGER, topic_dense INTEGER);
    CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
    """)

    levels = [("domain", domains), ("field", fields), ("subfield", subfields), ("topic", topics)]
    dm = {lvl: dense_map(its) for lvl, its in levels}
    for lvl, its in levels:
        for x in its:
            oi = _intid(x["id"])
            c.execute("INSERT INTO nodes VALUES(?,?,?,?)", (lvl, oi, dm[lvl][oi], x["display_name"]))

    for t in topics:
        to = _intid(t["id"])
        c.execute("INSERT INTO topic_path VALUES(?,?,?,?,?)", (
            to,
            dm["domain"][_intid(t["domain"]["id"])],
            dm["field"][_intid(t["field"]["id"])],
            dm["subfield"][_intid(t["subfield"]["id"])],
            dm["topic"][to]))

    c.execute("INSERT INTO meta VALUES('source','OpenAlex API /topics')")
    c.execute("INSERT INTO meta VALUES('depth','4')")
    c.execute("INSERT INTO meta VALUES('counts',?)",
              (f"{len(domains)}/{len(fields)}/{len(subfields)}/{len(topics)}",))
    con.commit()
    con.close()
    print(f"wrote {OUT}  (depth 4; LCA = longest common prefix of the dense 4-tuple)")


if __name__ == "__main__":
    main()
