#!/usr/bin/env python3
"""Quick test: authenticate with the Dimensions API and inspect category_for_2020 structure.

Usage:  DIMENSIONS_API_KEY=your_key python3 test_dimensions_api.py
"""
import json
import os
import sys
import urllib.request

API_KEY = os.environ.get("DIMENSIONS_API_KEY", "")
AUTH_URL = "https://app.dimensions.ai/api/auth.json"
QUERY_URL = "https://app.dimensions.ai/api/dsl/v2"


def auth(key):
    body = json.dumps({"key": key}).encode()
    req = urllib.request.Request(AUTH_URL, data=body,
                                headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())["token"]


def query(token, dsl):
    req = urllib.request.Request(QUERY_URL, data=dsl.encode(),
                                headers={"Authorization": f"JWT {token}",
                                         "Content-Type": "text/plain"})
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def main():
    if not API_KEY:
        print("Set DIMENSIONS_API_KEY env var", file=sys.stderr)
        sys.exit(1)

    print("Authenticating...", end=" ", flush=True)
    token = auth(API_KEY)
    print("OK")

    # 1) Fetch a few publications with PMIDs and FoR 2020 codes
    dsl = (
        'search publications where pmid is not empty '
        'return publications[pmid + category_for_2020] limit 5'
    )
    print(f"\nQuery: {dsl}")
    result = query(token, dsl)

    print(f"Total count: {result.get('_stats', {}).get('total_count', '?')}")
    pubs = result.get("publications", [])
    print(f"Returned: {len(pubs)}\n")

    for pub in pubs:
        print(json.dumps(pub, indent=2))
        print()

    # 2) Also try category_for (the alias) on 1 record
    dsl2 = (
        'search publications where pmid is not empty '
        'return publications[pmid + category_for] limit 1'
    )
    print(f"Alias query: {dsl2}")
    result2 = query(token, dsl2)
    for pub in result2.get("publications", []):
        print(json.dumps(pub, indent=2))
        print()

    # 3) Check what a specific well-known PMID looks like
    dsl3 = (
        'search publications where pmid = "1" '
        'return publications[pmid + doi + category_for_2020] limit 1'
    )
    print(f"Single PMID query: {dsl3}")
    result3 = query(token, dsl3)
    for pub in result3.get("publications", []):
        print(json.dumps(pub, indent=2))
        print()


if __name__ == "__main__":
    main()
