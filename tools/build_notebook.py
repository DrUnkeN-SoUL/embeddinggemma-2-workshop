"""Builds workshop.ipynb (the Colab notebook for the three labs). Edit cells here, then run:
    python tools/build_notebook.py
"""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# EmbeddingGemma 2 workshop · GDG Cochin

Hands-on labs on real public datasets. Run the cells top to bottom.

**Before you start:** *File → Save a copy in Drive*, then *Runtime → Change runtime type → T4 GPU* (CPU also works, just slower).

The model is Apache 2.0 and not gated, so no Hugging Face token is needed.

| Lab | Dataset |
|---|---|
| 1 | Natural Questions (Google search questions + Wikipedia answers), OPUS-100 English–Malayalam, AG News |
| 2 | Natural Questions |
| 3 | RAG Mini Wikipedia, Banking77 (customer-support intents), Flickr8k (captioned photos) |
""")
md("## Setup")
code("""
# Colab already has datasets, scikit-learn and matplotlib; upgrading only sentence-transformers avoids dependency warnings.
!pip install -q -U sentence-transformers
""")
code("""
import time, random, numpy as np, torch, matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 on GPUs that support it natively (Ampere and newer); float32 otherwise (e.g. T4, CPU).
# Never float16: it can silently produce NaN or degraded embeddings with this model.
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

MODEL_ID = "google/embeddinggemma-2"
model = SentenceTransformer(
    MODEL_ID, device=device,
    model_kwargs={"torch_dtype": dtype},
    config_kwargs={"vision_config": None, "audio_config": None},  # text-only: 270M parameters
)
print("Device:", device, "| dtype:", dtype)
dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
print("Embedding size:", dim_fn())

def enc(texts, **kw):
    \"\"\"Encode to normalized numpy vectors (so dot product = cosine similarity).\"\"\"
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64, **kw)

def recall_at(q, d, gold, ks=(1, 10)):
    \"\"\"Fraction of queries whose correct document is in the top-k results.\"\"\"
    order = np.argsort(-(q @ d.T), axis=1)
    return {k: float(np.mean([g in row[:k] for g, row in zip(gold, order)])) for k in ks}
""")

# ---------------- Lab 1 ----------------
md("""
# Lab 1 · Your first embeddings

### 1. Embed five sentences
Each sentence becomes one row of 768 numbers.
""")
code("""
sentences = [
    "Appam with vegetable stew for breakfast",
    "Puttu and kadala curry",
    "Restart the crashed Kubernetes pod",
    "My container keeps crashing",
    "Heavy monsoon rain expected in Ernakulam",
]
emb = enc(sentences, prompt_name="SentenceSimilarity")
print(emb.shape)               # (5, 768)
print(emb[0][:8])              # the first 8 of appam's 768 numbers
print(np.linalg.norm(emb[0]))  # length is 1.0 (normalized)
""")
md("### 2. Similarity matrix\nEvery sentence compared with every other. The diagonal is each sentence matching itself.")
code("""
sims = emb @ emb.T
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(sims, cmap="Blues", vmin=0, vmax=1)
short = [s[:28] + ("…" if len(s) > 28 else "") for s in sentences]
ax.set_xticks(range(len(short)), short, rotation=40, ha="right"); ax.set_yticks(range(len(short)), short)
for i in range(len(sims)):
    for j in range(len(sims)):
        ax.text(j, i, f"{sims[i, j]:.2f}", ha="center", va="center", color="white" if sims[i, j] > .6 else "black", fontsize=9)
plt.colorbar(im); plt.title("Cosine similarity"); plt.tight_layout(); plt.show()
""")
md("""
### 3. Do task prefixes matter? Measure it on real data
**Natural Questions**: real Google search queries, each paired with the Wikipedia paragraph that answers it. We hide each answer among 1,000 others and check whether search finds it.
""")
code("""
nq = load_dataset("sentence-transformers/natural-questions", split="train[:1000]")
queries, passages = nq["query"], nq["answer"]
gold = list(range(len(queries)))   # query i is answered by passage i
print(queries[0], "→", passages[0][:150], "…")

with_p = recall_at(enc(queries, prompt_name="SearchQuery"), enc(passages, prompt_name="Document"), gold)
without = recall_at(enc(queries, prompt=""), enc(passages, prompt=""), gold)
print(f"With prefixes:    top-1 {with_p[1]:.1%}   top-10 {with_p[10]:.1%}")
print(f"Without prefixes: top-1 {without[1]:.1%}   top-10 {without[10]:.1%}")
""")
md("""
### 4. Malayalam ↔ English
**OPUS-100** English–Malayalam translation pairs. For each Malayalam sentence, can the model pick its English translation out of 300 candidates?
""")
code("""
pairs = load_dataset("Helsinki-NLP/opus-100", "en-ml", split="test")["translation"]
pairs = [p for p in pairs if 40 <= len(p["en"]) <= 200 and len(p["ml"]) >= 20][:300]
en = enc([p["en"] for p in pairs], prompt_name="SentenceSimilarity")
ml = enc([p["ml"] for p in pairs], prompt_name="SentenceSimilarity")
best = (ml @ en.T).argmax(axis=1)
print(f"Malayalam → English, top-1 out of {len(pairs)}: {np.mean(best == np.arange(len(pairs))):.1%}\\n")
for i in random.Random(0).sample(range(len(pairs)), 5):
    print("ML:", pairs[i]["ml"]); print("→ ", pairs[best[i]]["en"], "✓" if best[i] == i else "✗"); print()
""")
md("""
### 5. Plot a map of real news
**AG News** headlines from four sections. The model never sees the section labels; we colour the dots afterwards.
""")
code("""
from sklearn.decomposition import PCA
news = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=0)
names = ["World", "Sports", "Business", "Sci/Tech"]
picked = {k: [] for k in range(4)}
for row in news:
    if len(picked[row["label"]]) < 100: picked[row["label"]].append(row["text"])
    if all(len(v) == 100 for v in picked.values()): break
texts = [t for k in range(4) for t in picked[k]]
labels = np.repeat(np.arange(4), 100)
vecs = enc(texts, prompt_name="Clustering")
xy = PCA(n_components=2).fit_transform(vecs)
plt.figure(figsize=(9, 6))
for k in range(4):
    plt.scatter(xy[labels == k, 0], xy[labels == k, 1], s=14, alpha=.7, label=names[k])
plt.legend(); plt.title("400 AG News headlines, 768-D → 2-D (PCA)"); plt.axis("off"); plt.show()

i = 5
print("Headline:", texts[i][:120])
for j in np.argsort(-(vecs @ vecs[i]))[1:4]:
    print(f"  {names[labels[j]]:<9} {texts[j][:110]}")
""")

# ---------------- Lab 2 ----------------
md("""
# Lab 2 · Edge trade-offs, measured

### 1. Shrink the vectors (Matryoshka)
Same Natural Questions test, keeping only the first N numbers of every vector.
""")
code("""
q_full = enc(queries, prompt_name="SearchQuery")
d_full = enc(passages, prompt_name="Document")
print(" dims   top-1   top-10   index size for 1M passages")
for dim in [768, 512, 256, 128]:
    q = q_full[:, :dim] / np.linalg.norm(q_full[:, :dim], axis=1, keepdims=True)  # truncate + re-normalize
    d = d_full[:, :dim] / np.linalg.norm(d_full[:, :dim], axis=1, keepdims=True)
    r = recall_at(q, d, gold)
    print(f"{dim:>5}   {r[1]:.1%}   {r[10]:.1%}    {1_000_000 * dim * 4 / 1e9:.2f} GB")
""")
md("### 2. Storage maths for your app")
code("""
def storage(items, dims, bytes_per_number=4):
    return f"{items:,} items × {dims}-D = {items * dims * bytes_per_number / 1e6:,.1f} MB"

for dims in [768, 256, 128]:
    print(storage(10_000, dims), "  (a phone photo library)")
print(storage(1_000_000, 256), "  (a company document index)")
""")
md("### 3. Time it: GPU vs CPU, on real Wikipedia paragraphs")
code("""
batch = passages[:256]

def per_second(m, n):
    m.encode(batch[:4], prompt_name="Document")                     # warm-up
    t = time.perf_counter(); m.encode(batch[:n], batch_size=16, prompt_name="Document")
    return n / (time.perf_counter() - t)

print(f"{device.upper()}: {per_second(model, 256):.0f} passages/sec")
if device == "cuda":
    cpu_model = SentenceTransformer(MODEL_ID, device="cpu", model_kwargs={"torch_dtype": torch.float32},
                                    config_kwargs={"vision_config": None, "audio_config": None})
    print(f"CPU: {per_second(cpu_model, 32):.1f} passages/sec   (Colab's CPU is a small shared VM; the quantized edge build below is much faster)")
    del cpu_model
""")
md("### 4. Weigh the encoders\nThe text core is small. Vision adds about 170M parameters.")
code("""
count = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
print(f"Text only:      {count(model):.0f}M parameters")
vision_model = SentenceTransformer(MODEL_ID, device=device, model_kwargs={"torch_dtype": dtype},
                                   config_kwargs={"audio_config": None})
print(f"Text + vision:  {count(vision_model):.0f}M parameters")
# Keep vision_model loaded: Lab 3's photo search uses it.
""")
md("""
### Stretch: the quantized edge build with LiteRT-LM
Runs the quantized text-only `.litertlm` file (165 MB) that phones use, on Colab's CPU. Follows Google's LiteRT-LM embedding guide (https://developers.google.com/edge/litert-lm/embedding_models); if the API has changed, this cell prints the error and the rest of the notebook still runs.
""")
code("""
try:
    !pip install -q litert-lm
    from huggingface_hub import hf_hub_download
    from litert_lm import Backend, EmbeddingEngine
    path = hf_hub_download("litert-community/embeddinggemma-2-text-270m-litert-lm", "embeddinggemma-2-text-270m.litertlm")
    with EmbeddingEngine(path, backend=Backend.CPU()) as engine:
        engine.compute_embedding("task: search query | text: warm-up")
        t = time.perf_counter()
        out = engine.compute_embedding("task: search query | text: Top places to visit in Kochi").embedding
        print(f"{len(out)} numbers in {(time.perf_counter() - t) * 1000:.0f} ms on CPU (quantized edge build)")
except Exception as e:
    print("LiteRT-LM stretch skipped:", repr(e)[:300])
""")

# ---------------- Lab 3 ----------------
md("""
# Lab 3 · Build a mini offline search app

### 1. Index a Wikipedia collection
**RAG Mini Wikipedia**: about 3,200 Wikipedia passages plus questions about them. We store 256-D vectors: 3× smaller, nearly the same quality.
""")
code("""
corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")["passage"]
qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
DIM = 256
t = time.perf_counter()
index = enc(corpus, prompt_name="Document", truncate_dim=DIM)
print(f"Indexed {len(corpus):,} passages in {time.perf_counter() - t:.1f}s → {index.nbytes / 1e6:.1f} MB")
""")
md("### 2. Search it, and check it against the dataset's answers")
code("""
def search(query, k=3):
    q = enc([query], prompt_name="SearchQuery", truncate_dim=DIM)[0]
    top = np.argsort(-(index @ q))[:k]
    return [(float(index[i] @ q), corpus[i]) for i in top]

for row in qa.select(range(3)):
    print("Q:", row["question"], "| expected:", row["answer"])
    for score, passage in search(row["question"], k=2):
        print(f"   {score:.3f}  {passage[:140]}…")
    print()

# How often does the expected answer appear in the top 3 passages? (skip yes/no answers)
rows = [r for r in qa if r["answer"].lower() not in ("yes", "no") and len(r["answer"]) > 2][:300]
hits = [any(r["answer"].lower() in p.lower() for _, p in search(r["question"])) for r in rows]
print(f"Exact answer text appears in the top-3 passages for {np.mean(hits):.1%} of {len(rows)} questions")
print("(A strict check: answers phrased differently from the passage count as misses, so the true hit rate is higher.)")
""")
md("""
### 3. Zero-shot router on Banking77
Real customer-support messages, 77 intents. **No training**: we embed each intent's name and send every message to the closest one. Random guessing would get 1.3%.

*Banking77 is a public benchmark the model may have seen during training, so expect somewhat lower accuracy on your own tickets. Always test on your own data.*
""")
code("""
bank = load_dataset("mteb/banking77", split="test").shuffle(seed=0).select(range(1000))
intents = sorted(set(bank["label_text"]))
intent_emb = enc([i.replace("_", " ") for i in intents], prompt_name="Classification")
msg_emb = enc(bank["text"], prompt_name="Classification")
ranked = np.argsort(-(msg_emb @ intent_emb.T), axis=1)
truth = np.array([intents.index(l) for l in bank["label_text"]])
print(f"Top-1 accuracy: {np.mean(ranked[:, 0] == truth):.1%}   Top-3: {np.mean([t in r[:3] for t, r in zip(truth, ranked)]):.1%}\\n")
for text, label, r in list(zip(bank["text"], bank["label_text"], ranked))[:6]:
    print(f"{intents[r[0]]:<32} (true: {label})  ←  {text}")

def route(message):
    return intents[int(np.argmax(enc([message], prompt_name="Classification") @ intent_emb.T))]
print("\\nYour turn:", route("I was charged twice for the same coffee"))
""")
md("""
### 4. Photo search with text
**Flickr8k**: photos with human-written captions. We embed 200 photos, then search them with each photo's caption and check how often the right photo comes first.
""")
code("""
rows = list(load_dataset("jxie/flickr8k", split="test", streaming=True).take(200))  # stream just 200 photos
images = [r["image"].convert("RGB") for r in rows]
captions = [r["caption_0"] for r in rows]

try:
    img_emb = vision_model.encode(images, normalize_embeddings=True, convert_to_numpy=True, batch_size=16)
except Exception as e:
    # Some sentence-transformers versions expect dict inputs for multimodal models.
    print("PIL input failed, retrying with dict inputs:", repr(e)[:120])
    img_emb = vision_model.encode([{"image": im} for im in images], normalize_embeddings=True, convert_to_numpy=True, batch_size=16)
cap_emb = vision_model.encode(captions, prompt_name="SearchQuery", normalize_embeddings=True, convert_to_numpy=True)
r = recall_at(cap_emb, img_emb, list(range(len(images))), ks=(1, 5))
print(f"Caption → photo, out of {len(images)}: top-1 {r[1]:.1%}   top-5 {r[5]:.1%}")
""")
code("""
def photo_search(query, k=4):
    q = vision_model.encode([query], prompt_name="SearchQuery", normalize_embeddings=True, convert_to_numpy=True)[0]
    best = np.argsort(-(img_emb @ q))[:k]
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    for ax, i in zip(axes, best):
        ax.imshow(images[i]); ax.set_title(f"{img_emb[i] @ q:.2f}"); ax.axis("off")
    plt.suptitle(query); plt.show()

photo_search("a dog running on the beach")
photo_search("children playing in the water")
""")
md("""
## Where to go next
- Model card: https://huggingface.co/google/embeddinggemma-2
- Run it on a phone: Google AI Edge Gallery (Play Store / App Store)
- On-device guide: https://developers.googleblog.com/google-ai-edge-with-embeddinggemma-2/
- LiteRT builds: https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm
- Datasets: sentence-transformers/natural-questions · Helsinki-NLP/opus-100 · fancyzhx/ag_news · rag-datasets/rag-mini-wikipedia · mteb/banking77 · jxie/flickr8k
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).resolve().parent.parent / "workshop.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")
