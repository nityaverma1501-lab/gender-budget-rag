"""
Build a hybrid retrieval index over index/chunks.jsonl:
  - dense embeddings from a multilingual sentence-transformer, so that a
    Hindi-language query can match the English half of a bilingual
    document (the Delhi PDFs render Hindi in a legacy non-Unicode font,
    so lexical/BM25 matching on the Hindi query text alone would fail).
  - BM25 for exact keyword / number / scheme-name matches.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "index" / "chunks.jsonl"
EMB_PATH = ROOT / "index" / "embeddings.npy"

EMBED_MODEL = "intfloat/multilingual-e5-base"


def load_chunks():
    return [json.loads(l) for l in open(CHUNKS_PATH, encoding="utf-8")]


def build():
    chunks = load_chunks()
    model = SentenceTransformer(EMBED_MODEL)
    texts = [f"passage: {c['text']}" for c in chunks]
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32
    )
    np.save(EMB_PATH, embeddings.astype(np.float32))
    print(f"Saved {embeddings.shape} embeddings to {EMB_PATH}")


if __name__ == "__main__":
    build()
