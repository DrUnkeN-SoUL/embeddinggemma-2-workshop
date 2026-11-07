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

Hands-on labs for the session. Run the cells top to bottom.

**Before you start:** *File → Save a copy in Drive*, then *Runtime → Change runtime type → T4 GPU* (CPU also works, just slower).

The model is Apache 2.0 and not gated, so no Hugging Face token is needed.
""")
md("## Setup")
code("""
!pip install -q -U sentence-transformers scikit-learn matplotlib
""")
code("""
import time, numpy as np, torch, matplotlib.pyplot as plt
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
print("Embedding size:", model.get_sentence_embedding_dimension())
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
emb = model.encode(sentences, prompt_name="SentenceSimilarity", normalize_embeddings=True)
print(emb.shape)          # (5, 768)
print(emb[0][:8])         # the first 8 of appam's 768 numbers
print(np.linalg.norm(emb[0]))  # length is 1.0 (normalized)
""")
md("""
### 2. Similarity matrix
Every sentence compared with every other. 1.0 on the diagonal is each sentence matching itself.
""")
code("""
sims = model.similarity(emb, emb).cpu().numpy()
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
**Look for:** the two food sentences and the two container sentences score high with each other, even though *My container keeps crashing* and *Restart the crashed Kubernetes pod* share almost no words.

### 3. With vs without task prefixes
Search uses **different** prefixes for the query and the documents. `prompt=""` turns the prefix off.
""")
code("""
docs = [
    "The Kochi Metro runs from Aluva to Thrippunithura.",
    "Kerala's backwaters are a network of lagoons and lakes.",
    "Use kubectl rollout restart to restart a deployment.",
    "Palada payasam is a sweet rice-flake pudding served at sadya.",
    "Monsoon season in Kerala usually starts in early June.",
    "Python dictionaries raise KeyError for missing keys.",
]
query = "how do I bring my app back after it crashed on k8s?"

def rank(q_emb, d_emb):
    scores = (d_emb @ q_emb.T).ravel()
    return sorted(zip(scores, docs), reverse=True)

with_p = rank(model.encode([query], prompt_name="SearchQuery", normalize_embeddings=True),
              model.encode(docs, prompt_name="Document", normalize_embeddings=True))
without = rank(model.encode([query], prompt="", normalize_embeddings=True),
               model.encode(docs, prompt="", normalize_embeddings=True))
print("WITH prefixes");    [print(f"  {s:.3f}  {d}") for s, d in with_p[:3]]
print("WITHOUT prefixes"); [print(f"  {s:.3f}  {d}") for s, d in without[:3]]
""")
md("""
### 4. Malayalam ↔ English
Try your own sentences in Malayalam, Hindi, Tamil… and see whether they find their English twin.
""")
code("""
english = ["It is raining heavily", "Where is the railway station?", "I love fish curry", "The meeting starts at ten"]
other = ["മഴ ശക്തമായി പെയ്യുന്നു",            # it is raining heavily
         "എനിക്ക് മീൻ കറി ഇഷ്ടമാണ്",            # I love fish curry
         "റെയിൽവേ സ്റ്റേഷൻ എവിടെയാണ്?"]          # where is the railway station?
e = model.encode(english, prompt_name="SentenceSimilarity", normalize_embeddings=True)
o = model.encode(other, prompt_name="SentenceSimilarity", normalize_embeddings=True)
for text, row in zip(other, model.similarity(o, e).cpu().numpy()):
    best = row.argmax()
    print(f"{text}  →  {english[best]}  ({row[best]:.2f})")
""")
md("""
### 5. Plot your own map
PCA squashes 768 dimensions down to 2 so we can draw them. Add your own sentences to the list.
""")
code("""
from sklearn.decomposition import PCA
groups = {
    "food":    ["Appam with vegetable stew", "Puttu and kadala curry", "Malabar chicken biryani", "A strong filter coffee"],
    "places":  ["Houseboat on the Alleppey backwaters", "Tea gardens of Munnar", "Chinese fishing nets at Fort Kochi", "Sunset at Varkala cliff"],
    "tech":    ["Restart the crashed Kubernetes pod", "Python script throws a KeyError", "Deploying a container to Cloud Run", "GPU ran out of memory"],
    "weather": ["Heavy monsoon rain in Ernakulam", "Orange alert for coastal districts", "മഴ ശക്തമായി പെയ്യുന്നു", "Humid and cloudy afternoon"],
}
texts = [t for ts in groups.values() for t in ts]
labels = [g for g, ts in groups.items() for _ in ts]
xy = PCA(n_components=2).fit_transform(model.encode(texts, prompt_name="Clustering", normalize_embeddings=True))
plt.figure(figsize=(9, 6))
for g in groups:
    idx = [i for i, l in enumerate(labels) if l == g]
    plt.scatter(xy[idx, 0], xy[idx, 1], s=60, label=g)
for (x, y), t in zip(xy, texts):
    plt.annotate(t, (x, y), fontsize=8, xytext=(5, 3), textcoords="offset points")
plt.legend(); plt.title("Your sentences, 768-D → 2-D"); plt.axis("off"); plt.show()
""")
md("*(Colab's default font may not render Malayalam in plots. The embedding itself is fine.)*")

# ---------------- Lab 2 ----------------
md("""
# Lab 2 · Edge trade-offs, measured

### 1. Shrink the vectors (Matryoshka)
Keep only the first N numbers. Does search still find the right answer?
""")
code("""
qa = [  # (question, the document that should come first)
    ("When does the monsoon arrive in Kerala?", 4),
    ("How do I restart a Kubernetes deployment?", 2),
    ("What sweet is served at a sadya?", 3),
    ("Which metro line goes to Aluva?", 0),
    ("Why does my Python dict throw an error?", 5),
    ("What are the backwaters?", 1),
]
for dim in [768, 512, 256, 128]:
    d = model.encode(docs, prompt_name="Document", normalize_embeddings=True, truncate_dim=dim)
    q = model.encode([x for x, _ in qa], prompt_name="SearchQuery", normalize_embeddings=True, truncate_dim=dim)
    top1 = (q @ d.T).argmax(axis=1)
    acc = np.mean([t == want for t, (_, want) in zip(top1, qa)])
    print(f"{dim:>4}-D  top-1 correct: {acc:.0%}   bytes per vector (float32): {dim * 4}")
""")
md("### 2. Storage maths for your app")
code("""
def storage(items, dims, bytes_per_number=4):
    mb = items * dims * bytes_per_number / 1e6
    return f"{items:,} items × {dims}-D = {mb:,.1f} MB"

for dims in [768, 256, 128]:
    print(storage(10_000, dims), "  (a phone photo library)")
print(storage(1_000_000, 256), "  (a company document index)")
""")
md("### 3. Time it: GPU vs CPU")
code("""
batch = [f"Sample sentence number {i} about Kerala, code and the monsoon." for i in range(256)]

def per_second(m):
    m.encode(batch[:8])                      # warm-up
    t = time.perf_counter(); m.encode(batch, batch_size=32); dt = time.perf_counter() - t
    return len(batch) / dt

print(f"{device.upper()}: {per_second(model):.0f} sentences/sec")
if device == "cuda":
    cpu_model = SentenceTransformer(MODEL_ID, device="cpu", model_kwargs={"torch_dtype": torch.float32},
                                    config_kwargs={"vision_config": None, "audio_config": None})
    print(f"CPU: {per_second(cpu_model):.0f} sentences/sec")
    del cpu_model
""")
md("""
### 4. Weigh the encoders
The text core is small. Vision adds about 170M parameters.
""")
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
This runs the same `.litertlm` file that phones use, on Colab's CPU. The API follows Google's LiteRT-LM embedding guide; if it changes, check https://developers.google.com/edge/litert-lm/embedding_models.
""")
code("""
!pip install -q litert-lm
from huggingface_hub import hf_hub_download
from litert_lm import Backend, EmbeddingEngine

path = hf_hub_download("litert-community/embeddinggemma-2-740m-litert-lm", "embeddinggemma-2-740m.litertlm")
with EmbeddingEngine(path, backend=Backend.CPU()) as engine:
    t = time.perf_counter()
    out = engine.compute_embedding("task: search query | text: Top places to visit in Kochi").embedding
    print(f"{len(out)} numbers in {(time.perf_counter() - t) * 1000:.0f} ms on CPU (first call includes warm-up)")
""")

# ---------------- Lab 3 ----------------
md("""
# Lab 3 · Build a mini offline search app

### 1. Index a notes collection
Paste in your own notes if you like. We store 256-D vectors: 3× smaller, nearly the same quality.
""")
code("""
notes = [
    "GDG Cochin meetup is on the second Saturday of every month.",
    "Remember to renew the domain before the end of March.",
    "Grandma's recipe: soak the rice overnight for soft appams.",
    "The staging cluster runs on GKE in asia-south1.",
    "Book train tickets to Thiruvananthapuram for the conference.",
    "To roll back a bad deploy, use the previous image tag.",
    "Buy coconut oil, curry leaves and mustard seeds.",
    "The office Wi-Fi password changes every quarter.",
    "Kochi Metro cards can be recharged at any station.",
    "ഓണത്തിന് സദ്യ ഒരുക്കാൻ വാഴയില വാങ്ങണം",   # buy banana leaves to prepare the Onam sadya
    "Car service is due at 40,000 km.",
    "Ask the team to review the on-call runbook.",
]
DIM = 256
index = model.encode(notes, prompt_name="Document", normalize_embeddings=True, truncate_dim=DIM)
print(index.shape)
""")
md("### 2. Search it")
code("""
def search(query, k=3):
    q = model.encode([query], prompt_name="SearchQuery", normalize_embeddings=True, truncate_dim=DIM)
    scores = (index @ q.T).ravel()
    for i in scores.argsort()[::-1][:k]:
        print(f"  {scores[i]:.3f}  {notes[i]}")

for q in ["what do I need for the festival lunch?", "production is broken, how do I undo it?", "groceries", "മെട്രോ കാർഡ്"]:
    print(q); search(q); print()
""")
md("""
### 3. Zero-shot router
No training data: each team is described in a sentence, and every message goes to the closest description. Add a team by adding a line.
""")
code("""
teams = {
    "Payments":  "billing, refunds, invoices, payment failed, charged twice",
    "App bugs":  "the app crashes, freezes, shows an error or won't open",
    "Delivery":  "order delayed, courier, tracking, package not received",
    "Account":   "login, password reset, change email, delete my account",
}
team_names = list(teams)
team_emb = model.encode(list(teams.values()), prompt_name="Classification", normalize_embeddings=True)

def route(message):
    m = model.encode([message], prompt_name="Classification", normalize_embeddings=True)
    scores = (team_emb @ m.T).ravel()
    return team_names[scores.argmax()], scores.max()

for msg in ["I was charged two times for one order", "the app closes as soon as I open it",
            "my parcel still hasn't come", "പാസ്‌വേഡ് മറന്നുപോയി",   # I forgot my password
            "where is my refund?"]:
    team, s = route(msg)
    print(f"{team:<10} {s:.2f}  {msg}")
""")
md("""
### 4. Stretch: search photos with text
Runs on a few sample photos by default. Set `USE_MY_PHOTOS = True` to upload 5–10 of your own instead. Uses the text + vision model from Lab 2.
""")
code("""
USE_MY_PHOTOS = False

import io, urllib.request
from PIL import Image
if USE_MY_PHOTOS:
    from google.colab import files
    uploaded = files.upload()
    images = [Image.open(io.BytesIO(data)).convert("RGB") for data in uploaded.values()]
else:  # sample photos from the COCO dataset
    urls = [f"http://images.cocodataset.org/val2017/{n:012d}.jpg" for n in (39769, 139, 285, 632, 724, 785, 872)]
    images = [Image.open(io.BytesIO(urllib.request.urlopen(u).read())).convert("RGB") for u in urls]

img_emb = vision_model.encode(images, normalize_embeddings=True)
print(img_emb.shape)
""")
code("""
def photo_search(query, k=3):
    q = vision_model.encode([query], prompt_name="SearchQuery", normalize_embeddings=True)
    scores = (img_emb @ q.T).ravel()
    best = scores.argsort()[::-1][:k]
    fig, axes = plt.subplots(1, len(best), figsize=(4 * len(best), 4))
    for ax, i in zip(np.atleast_1d(axes), best):
        ax.imshow(images[i]); ax.set_title(f"{scores[i]:.2f}"); ax.axis("off")
    plt.suptitle(query); plt.show()

photo_search("cats sleeping on a sofa")
photo_search("someone skiing down a slope")
""")
md("""
## Where to go next
- Model card: https://huggingface.co/google/embeddinggemma-2
- Run it on a phone: Google AI Edge Gallery (Play Store / App Store)
- On-device guide: https://developers.googleblog.com/google-ai-edge-with-embeddinggemma-2/
- LiteRT builds: https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).resolve().parent.parent / "workshop.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")
