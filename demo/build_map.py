"""Embed the sample sentences and queries with EmbeddingGemma 2 and write demo/map-data.js.

The deck's map slide reads map-data.js. Until you run this it shows an illustrative layout;
afterwards it shows real 2-D coordinates (PCA of the 768-D vectors) and real cosine scores.

    pip install -U sentence-transformers
    cd demo && python build_map.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from sentences import QUERIES, SENTENCES

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
model = SentenceTransformer(
    "google/embeddinggemma-2",
    model_kwargs={"torch_dtype": dtype},
    config_kwargs={"vision_config": None, "audio_config": None},  # text-only: 270M params
)

def embed(texts):
    # Symmetric task: every item gets the same prefix, so queries and sentences share one geometry.
    return model.encode(texts, prompt_name="SentenceSimilarity", normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

docs = embed([t for _, t in SENTENCES])
qs = embed([t for _, t in QUERIES])

def top3(vec, exclude=None):
    sims = docs @ vec
    if exclude is not None:
        sims[exclude] = -1
    return [[int(j), round(float(sims[j]), 3)] for j in np.argsort(-sims)[:3]]

# PCA fitted on the sentences only, then queries projected into the same 2-D plane.
mean = docs.mean(axis=0)
_, _, vt = np.linalg.svd(docs - mean, full_matrices=False)
proj = lambda m: (m - mean) @ vt[:2].T
d2, q2 = proj(docs), proj(qs)
scale = np.abs(np.vstack([d2, q2])).max()
d2, q2 = d2 / scale, q2 / scale

data = {
    "source": "embeddinggemma-2",
    "points": [{"g": g, "t": t, "p": [round(float(v), 4) for v in d2[i]], "n": top3(docs[i], i)} for i, (g, t) in enumerate(SENTENCES)],
    "queries": [{"g": g, "t": t, "p": [round(float(v), 4) for v in q2[i]], "n": top3(qs[i])} for i, (g, t) in enumerate(QUERIES)],
}
out = Path(__file__).with_name("map-data.js")
out.write_text("window.MAP_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
print(f"Wrote {out}: {len(data['points'])} sentences, {len(data['queries'])} queries")
