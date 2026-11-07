"""Laptop CPU demo: semantic search against a local LiteRT-LM embedding server.

Start the server first (or just run ./laptop_demo.sh, which does both):

    litert-lm serve --host 127.0.0.1 --port 9379

Then:

    python laptop_demo.py                                  # interactive
    python laptop_demo.py --query "my container keeps crashing"

Standard library only: the server speaks the OpenAI /v1/embeddings API.
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

from sentences import SENTENCES

# "<model-id>,<backend>". The model ID is what `litert-lm import` printed / `litert-lm list` shows.
DEFAULT_MODEL = os.environ.get("LITERT_MODEL", "embeddinggemma-2-text-270m.litertlm,cpu")
DEFAULT_URL = os.environ.get("LITERT_URL", "http://127.0.0.1:9379/v1/embeddings")

# Prefixes from the LiteRT-LM embedding guide.
QUERY_PREFIX = "task: search query | text: "
DOC_PREFIX = "task: search result | text: "


def embed(texts, url, model):
    """POST a list of strings to /v1/embeddings and return one vector per string."""
    body = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)["data"]
    return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def search(query, docs, doc_vecs, url, model, k):
    t0 = time.perf_counter()
    [qvec] = embed([QUERY_PREFIX + query], url, model)
    ms = (time.perf_counter() - t0) * 1000
    ranked = sorted(zip(docs, doc_vecs), key=lambda dv: cosine(qvec, dv[1]), reverse=True)
    print(f'\nQuery: "{query}"   (embedded in {ms:.0f} ms on CPU)')
    for (group, text), vec in ranked[:k]:
        print(f"  {cosine(qvec, vec):.3f}  [{group:<7}] {text}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", help="run one query and exit (no interactive prompt)")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-k", type=int, default=5, help="results to show (default 5)")
    args = ap.parse_args()

    docs = SENTENCES
    print(f"Embedding {len(docs)} documents with {args.model} ...")
    t0 = time.perf_counter()
    try:
        doc_vecs = embed([DOC_PREFIX + text for _, text in docs], args.url, args.model)
    except urllib.error.HTTPError as e:
        sys.exit(f"Server error {e.code}: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"Can't reach {args.url} ({e.reason}). Is `litert-lm serve` running?")
    ms = (time.perf_counter() - t0) * 1000
    print(f"Indexed {len(docs)} documents in {ms:.0f} ms ({ms / len(docs):.0f} ms each), {len(doc_vecs[0])} dims")

    if args.query:
        search(args.query, docs, doc_vecs, args.url, args.model, args.k)
        return

    print("\nType a query (English, Malayalam, ...). Empty line or Ctrl+D to quit.")
    while True:
        try:
            query = input("\nsearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        search(query, docs, doc_vecs, args.url, args.model, args.k)


if __name__ == "__main__":
    main()
