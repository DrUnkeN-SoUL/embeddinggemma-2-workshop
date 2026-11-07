"""Tiny local semantic search with EmbeddingGemma 2 — the live-demo script.

    python demo/search.py "where should I eat breakfast in Kochi?"
    python demo/search.py --dim 256 "my container keeps crashing"
"""
import argparse

import torch
from sentence_transformers import SentenceTransformer

from sentences import SENTENCES

parser = argparse.ArgumentParser()
parser.add_argument("query")
parser.add_argument("--dim", type=int, default=768, choices=[768, 512, 256, 128])
parser.add_argument("-k", type=int, default=5)
args = parser.parse_args()

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
model = SentenceTransformer(
    "google/embeddinggemma-2",
    model_kwargs={"torch_dtype": dtype},
    config_kwargs={"vision_config": None, "audio_config": None},
    truncate_dim=args.dim,
)

docs = [t for _, t in SENTENCES]
doc_emb = model.encode(docs, prompt_name="Document", normalize_embeddings=True)
query_emb = model.encode(args.query, prompt_name="SearchQuery", normalize_embeddings=True)

scores = model.similarity(query_emb, doc_emb)[0]
top = scores.argsort(descending=True)[: args.k]
print(f"\nQuery: {args.query}   ({args.dim}-D)\n")
for i in top.tolist():
    print(f"  {scores[i]:.3f}  {docs[i]}")
