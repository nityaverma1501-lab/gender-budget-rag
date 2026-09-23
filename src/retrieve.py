"""Hybrid (dense + BM25) retrieval over the indexed chunks."""
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from index import EMBED_MODEL, EMB_PATH, load_chunks

ROOT = Path(__file__).resolve().parent.parent

_TOKEN_RE = re.compile(r"[a-zA-Z0-9%.]+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def _is_devanagari_query(text, threshold=0.3):
    """BM25's tokenizer only extracts [a-zA-Z0-9%.], so a Hindi-script query
    reduces to just its stray digits (e.g. "2024", "25") -- meaningless
    noise that can still out-rank the dense embedding score on the actual
    right page (observed: pulled a wrong page above the correct one purely
    on a coincidental digit match). Detect a majority-Devanagari query so
    the caller can lean on dense similarity alone instead."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    devanagari = sum(1 for c in letters if _DEVANAGARI_RE.match(c))
    return devanagari / len(letters) >= threshold


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.embeddings = np.load(EMB_PATH)
        self.model = SentenceTransformer(EMBED_MODEL)
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])

    def search(self, query, top_k=8, dense_weight=0.6, max_per_file=3):
        if _is_devanagari_query(query):
            # BM25 can't meaningfully score a Hindi-script query (see
            # _is_devanagari_query) -- rely on dense cross-lingual
            # similarity alone rather than let stray digit matches skew
            # the ranking.
            dense_weight = 1.0
        q_emb = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        dense_scores = self.embeddings @ q_emb  # cosine sim, already normalized

        bm25_scores = np.array(self.bm25.get_scores(_tokenize(query)))
        if bm25_scores.max() > 0:
            bm25_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_norm = bm25_scores

        # dense scores are roughly in [0, 1] for this model on relevant text
        dense_norm = (dense_scores - dense_scores.min()) / (
            dense_scores.max() - dense_scores.min() + 1e-9
        )

        combined = dense_weight * dense_norm + (1 - dense_weight) * bm25_norm

        # Long prose documents (e.g. the 57-page Odisha report) tend to flood
        # a plain top-k with several similarly-worded pages, crowding out a
        # single terse-but-correct row from a table-like source (e.g. the
        # Union Budget CSV). Cap results per source file so the ranked list
        # stays diverse across documents.
        ranked_idx = np.argsort(-combined)
        results = []
        per_file_count = {}
        for i in ranked_idx:
            if len(results) >= top_k:
                break
            c = self.chunks[i]
            f = c["file"]
            if per_file_count.get(f, 0) >= max_per_file:
                continue
            per_file_count[f] = per_file_count.get(f, 0) + 1
            entry = dict(c)
            entry["score"] = float(combined[i])
            entry["dense_score"] = float(dense_scores[i])
            results.append(entry)
        return results

    def rerank_by_passage(self, query, chunks, window=3):
        """Reorder chunks by their best-matching short passage instead of
        whole-page similarity. A page is embedded as one vector, so a single
        answer-bearing sentence gets diluted by the ~40 unrelated lines
        around it -- observed: the page stating "Australia was the first
        country ... in 1984" ranked below a neighbouring page that never
        mentions it. Scoring overlapping 3-line windows lets that one
        sentence carry the page."""
        if len(chunks) < 2:
            return chunks
        windows, owner = [], []
        for ci, c in enumerate(chunks):
            lines = [l.strip() for l in c["text"].split("\n") if l.strip()]
            for i in range(max(1, len(lines) - window + 1)):
                windows.append("passage: " + " ".join(lines[i : i + window]))
                owner.append(ci)
        q_emb = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        w_emb = self.model.encode(windows, normalize_embeddings=True, batch_size=64)
        sims = w_emb @ q_emb
        best = np.full(len(chunks), -1.0)
        for s, ci in zip(sims, owner):
            best[ci] = max(best[ci], s)
        order = np.argsort(-best, kind="stable")
        return [dict(chunks[i], passage_score=float(best[i])) for i in order]
