"""On-device intent matching with EmbeddingGemma 2 (LiteRT-LM, laptop CPU, offline).

    python app.py                 # serve the demo UI on http://127.0.0.1:8090
    python app.py --eval          # top-1 / top-3 accuracy + latency on Banking77 test messages
    python app.py --download      # fetch Banking77 into data/ (run.sh does this for you)

Banking77: 77 customer-support intents. Each intent gets a prototype vector built from its
readable name ("card arrival") and a handful of real training messages. A user message is
embedded once (~80 ms on CPU) and compared with the 77 prototypes by dot product.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE = os.path.join(DATA, "cache")
MODEL = os.environ.get(
    "INTENT_MODEL",
    os.path.join(HERE, "..", "..", "demo", "models", "embeddinggemma-2-text-270m.litertlm"),
)
DATA_URL = "https://huggingface.co/datasets/mteb/banking77/resolve/main/{split}.jsonl"

# Symmetric task prefix: both the intent side and the user message use it.
PREFIX = "task: classification | text: "

EXAMPLES_PER_INTENT = 10  # training messages embedded per intent
NAME_WEIGHT = 0.5         # prototype = normalize(mean(examples) + NAME_WEIGHT * name_vec)
SEED = 0

# Confidence rule (chosen from `python app.py --eval`, see README):
# confident only if top score >= MIN_SCORE and (top - second) >= MIN_GAP.
MIN_SCORE = 0.82
MIN_GAP = 0.03

UPPER = {"atm": "ATM", "pin": "PIN", "visa": "Visa", "mastercard": "Mastercard",
         "apple": "Apple", "google": "Google", "fiat": "fiat"}


def readable(label):
    """card_arrival -> 'card arrival'; atm_support -> 'ATM support'."""
    words = label.strip("?").lower().split("_")
    return " ".join(UPPER.get(w, w) for w in words)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- data

def data_path(split):
    return os.path.join(DATA, f"banking77_{split}.jsonl")


def download():
    os.makedirs(DATA, exist_ok=True)
    for split in ("train", "test"):
        dst = data_path(split)
        if os.path.exists(dst):
            continue
        url = DATA_URL.format(split=split)
        log(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as r:
            body = r.read()
        with open(dst + ".part", "wb") as f:
            f.write(body)
        os.replace(dst + ".part", dst)
        log(f"  -> {dst} ({len(body) // 1024} KB)")


def load_split(split):
    path = data_path(split)
    if not os.path.exists(path):
        sys.exit(f"Missing {path}. Run: python app.py --download (needs internet once)")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- model

class Embedder:
    """One LiteRT-LM EmbeddingEngine, shared across threads behind a lock."""

    def __init__(self, path=MODEL):
        if not os.path.exists(path):
            sys.exit(f"Model not found: {path}\nRun demo/laptop_demo.sh once to download it.")
        from litert_lm import Backend, EmbeddingEngine
        t0 = time.perf_counter()
        self._ctx = EmbeddingEngine(path, backend=Backend.CPU())
        self.engine = self._ctx.__enter__()
        self.load_s = time.perf_counter() - t0
        self.lock = threading.Lock()

    def embed(self, text):
        with self.lock:
            vec = np.asarray(self.engine.compute_embedding(PREFIX + text).embedding, dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def close(self):
        self._ctx.__exit__(None, None, None)


# ---------------------------------------------------------------- index

def pick_examples(train, k=EXAMPLES_PER_INTENT, seed=SEED):
    by_label = OrderedDict()
    for row in sorted(train, key=lambda r: r["label"]):
        by_label.setdefault(row["label_text"], []).append(row["text"])
    rng = random.Random(seed)
    return OrderedDict((lab, rng.sample(texts, k)) for lab, texts in by_label.items())


def index_path():
    return os.path.join(CACHE, f"index_k{EXAMPLES_PER_INTENT}_s{SEED}.npz")


def build_or_load_index(embedder=None):
    """Returns dict(labels, names, examples, name_vecs[77,768], ex_vecs[77,k,768])."""
    path = index_path()
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return dict(labels=meta["labels"], names=meta["names"], examples=meta["examples"],
                    name_vecs=z["name_vecs"], ex_vecs=z["ex_vecs"], build_s=meta["build_s"])
    embedder = embedder or Embedder()
    chosen = pick_examples(load_split("train"))
    labels = list(chosen)
    names = [readable(lab) for lab in labels]
    n_total = len(labels) * (1 + EXAMPLES_PER_INTENT)
    log(f"Building intent index: {len(labels)} intents x (name + {EXAMPLES_PER_INTENT} examples)"
        f" = {n_total} embeddings ...")
    t0 = time.perf_counter()
    name_vecs, ex_vecs = [], []
    for i, lab in enumerate(labels):
        name_vecs.append(embedder.embed(names[i]))
        ex_vecs.append([embedder.embed(t) for t in chosen[lab]])
        if (i + 1) % 11 == 0:
            log(f"  {i + 1}/{len(labels)} intents  ({time.perf_counter() - t0:.0f} s)")
    build_s = time.perf_counter() - t0
    log(f"First build took {build_s:.1f} s ({build_s / n_total * 1000:.0f} ms per embedding). Cached to {path}")
    os.makedirs(CACHE, exist_ok=True)
    meta = dict(labels=labels, names=names, examples=[chosen[l] for l in labels], build_s=build_s,
                prefix=PREFIX, model=os.path.basename(MODEL))
    np.savez(path, name_vecs=np.stack(name_vecs), ex_vecs=np.asarray(ex_vecs, dtype=np.float32),
             meta=np.asarray(json.dumps(meta)))
    return dict(labels=labels, names=names, examples=meta["examples"],
                name_vecs=np.stack(name_vecs), ex_vecs=np.asarray(ex_vecs, dtype=np.float32), build_s=build_s)


def normalize(m):
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12)


def prototypes(index, k=EXAMPLES_PER_INTENT, name_weight=NAME_WEIGHT):
    """k=0 -> name only. Otherwise normalized mean of k example vectors, plus name_weight * name."""
    if k == 0:
        return normalize(index["name_vecs"])
    mean = normalize(index["ex_vecs"][:, :k].mean(axis=1))
    return normalize(mean + name_weight * index["name_vecs"])


def decide(scores_sorted):
    top, second = float(scores_sorted[0]), float(scores_sorted[1])
    if top < MIN_SCORE:
        return "unsure", f"top score {top:.2f} < {MIN_SCORE:.2f}"
    if top - second < MIN_GAP:
        return "unsure", f"gap to #2 is {top - second:.3f} < {MIN_GAP:.2f}"
    return "confident", f"top {top:.2f}, gap {top - second:.3f}"


# ---------------------------------------------------------------- eval

def run_eval(n, fresh):
    index = build_or_load_index()
    test = load_split("test")
    rng = random.Random(SEED)
    sample = rng.sample(test, min(n, len(test)))
    label_ix = {lab: i for i, lab in enumerate(index["labels"])}
    y = np.array([label_ix[r["label_text"]] for r in sample])

    cache = os.path.join(CACHE, f"eval_n{len(sample)}_s{SEED}.npz")
    if os.path.exists(cache) and not fresh:
        z = np.load(cache)
        q, lat = z["q"], z["lat"]
        log(f"Using cached test embeddings from {cache} (latencies are from that measured run; --fresh to redo)")
    else:
        emb = Embedder()
        log(f"Model loaded in {emb.load_s:.2f} s. Embedding {len(sample)} Banking77 test messages ...")
        emb.embed("warm up")
        q, lat = [], []
        for r in sample:
            t0 = time.perf_counter()
            q.append(emb.embed(r["text"]))
            lat.append((time.perf_counter() - t0) * 1000)
        q, lat = np.stack(q), np.array(lat)
        emb.close()
        os.makedirs(CACHE, exist_ok=True)
        np.savez(cache, q=q, lat=lat)

    log(f"\nBanking77 test, {len(sample)} random messages (seed {SEED}), 77 intents, chance = 1.3%")
    log(f"Query embedding latency on CPU: median {np.median(lat):.0f} ms, p90 {np.percentile(lat, 90):.0f} ms")
    log(f"Index first build: {index['build_s']:.1f} s\n")

    variants = [("intent name only", 0, 0.0), ("name + 1 example", 1, NAME_WEIGHT),
                ("name + 3 examples", 3, NAME_WEIGHT), ("10 examples, no name", 10, 0.0),
                (f"name + {EXAMPLES_PER_INTENT} examples (app default)", EXAMPLES_PER_INTENT, NAME_WEIGHT)]
    log(f"{'prototype':<38}{'top-1':>8}{'top-3':>8}")
    for title, k, w in variants:
        s = q @ prototypes(index, k, w).T
        order = np.argsort(-s, axis=1)
        top1 = (order[:, 0] == y).mean()
        top3 = (order[:, :3] == y[:, None]).any(axis=1).mean()
        log(f"{title:<38}{top1 * 100:>7.1f}%{top3 * 100:>7.1f}%")

    # Threshold table for the default prototypes.
    s = q @ prototypes(index).T
    srt = -np.sort(-s, axis=1)
    correct = np.argmax(s, axis=1) == y
    top, gap = srt[:, 0], srt[:, 0] - srt[:, 1]
    log(f"\nTop score: in-scope test messages p5 {np.percentile(top, 5):.3f}, p10 {np.percentile(top, 10):.3f}")
    log(f"Top-score stats: correct median {np.median(top[correct]):.3f}, wrong median {np.median(top[~correct]):.3f}")
    log(f"Gap stats:       correct median {np.median(gap[correct]):.3f}, wrong median {np.median(gap[~correct]):.3f}")
    log(f"\n{'min score':>10}{'min gap':>9}{'confident':>11}{'acc when confident':>20}{'acc when unsure':>17}")
    for ms in (0.0, 0.80, 0.82, 0.85):
        for mg in (0.0, 0.01, 0.02, 0.03):
            conf = (top >= ms) & (gap >= mg)
            if not conf.any():
                continue
            un = f"{correct[~conf].mean() * 100:.1f}%" if (~conf).any() else "-"
            mark = "  <- app" if (ms, mg) == (MIN_SCORE, MIN_GAP) else ""
            log(f"{ms:>10.2f}{mg:>9.2f}{conf.mean() * 100:>10.0f}%{correct[conf].mean() * 100:>19.1f}%{un:>17}{mark}")


# ---------------------------------------------------------------- server

class Matcher:
    def __init__(self):
        self.emb = Embedder()
        self.index = build_or_load_index(self.emb)
        self.protos = prototypes(self.index)
        self.emb.embed("warm up")
        self.cache = OrderedDict()

    def match(self, text, k=5):
        text = text.strip()[:500]
        t0 = time.perf_counter()
        hit = text in self.cache
        if hit:
            vec = self.cache[text]
            self.cache.move_to_end(text)
        else:
            vec = self.emb.embed(text)
            self.cache[text] = vec
            if len(self.cache) > 256:
                self.cache.popitem(last=False)
        embed_ms = (time.perf_counter() - t0) * 1000
        scores = self.protos @ vec
        order = np.argsort(-scores)
        status, reason = decide(scores[order])
        total_ms = (time.perf_counter() - t0) * 1000
        top = int(order[0])
        return dict(
            query=text, status=status, reason=reason, cached=hit,
            embed_ms=round(embed_ms, 1), total_ms=round(total_ms, 1),
            results=[dict(label=self.index["labels"][i], name=self.index["names"][i],
                          score=round(float(scores[i]), 4)) for i in order[:k]],
            examples=self.index["examples"][top][:2],
        )

    def info(self):
        return dict(model=os.path.basename(MODEL), intents=len(self.protos), dims=int(self.protos.shape[1]),
                    examples_per_intent=EXAMPLES_PER_INTENT, name_weight=NAME_WEIGHT,
                    min_score=MIN_SCORE, min_gap=MIN_GAP, load_s=round(self.emb.load_s, 2),
                    build_s=round(self.index["build_s"], 1), prefix=PREFIX)


def make_handler(matcher, page):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    with open(page, "rb") as f:  # re-read so UI edits show on reload
                        self.send(200, f.read(), "text/html; charset=utf-8")
                elif url.path == "/api/match":
                    q = qs.get("q", [""])[0]
                    if not q.strip():
                        self.send(400, b'{"error":"empty query"}', "application/json")
                        return
                    out = json.dumps(matcher.match(q), ensure_ascii=False).encode("utf-8")
                    self.send(200, out, "application/json; charset=utf-8")
                elif url.path == "/api/info":
                    self.send(200, json.dumps(matcher.info()).encode(), "application/json")
                else:
                    self.send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser aborted a stale request

    return Handler


def serve(host, port):
    page = os.path.join(HERE, "index.html")
    matcher = Matcher()
    log(f"Model loaded in {matcher.emb.load_s:.2f} s; {len(matcher.protos)} intent prototypes ready.")
    srv = ThreadingHTTPServer((host, port), make_handler(matcher, page))
    srv.daemon_threads = True
    log(f"\n  Intent matcher running at  http://{host}:{port}\n  (Ctrl+C to stop)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        matcher.emb.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="store_true", help="evaluate on Banking77 test messages")
    ap.add_argument("-n", type=int, default=500, help="test messages for --eval (default 500)")
    ap.add_argument("--fresh", action="store_true", help="--eval: re-embed test messages even if cached")
    ap.add_argument("--download", action="store_true", help="download Banking77 and exit")
    ap.add_argument("--build", action="store_true", help="build the intent cache and exit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    if args.download:
        download()
    elif args.build:
        build_or_load_index()
    elif args.eval:
        run_eval(args.n, args.fresh)
    else:
        serve(args.host, args.port)


if __name__ == "__main__":
    main()
