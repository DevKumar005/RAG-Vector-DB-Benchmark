import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def tokenize(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in ENGLISH_STOP_WORDS]

class HybridRetriever:
    def __init__(self, chunks, doc_emb: np.ndarray, ef_search: int = 64):
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.doc_emb = doc_emb.astype("float32")
        dim = self.doc_emb.shape[1]

        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

        self.flat = faiss.IndexFlatIP(dim)
        self.flat.add(self.doc_emb)

        self.hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.hnsw.hnsw.efConstruction = 200
        self.hnsw.hnsw.efSearch = ef_search
        self.hnsw.add(self.doc_emb)

    def dense_search(self, q_vec: np.ndarray, k: int = 20, use_hnsw: bool = False):
        index = self.hnsw if use_hnsw else self.flat
        _, ix = index.search(q_vec.reshape(1, -1).astype("float32"), k)
        return [self.chunk_ids[i] for i in ix[0]]

    def bm25_search(self, query_text: str, k: int = 20):
        scores = self.bm25.get_scores(tokenize(query_text))
        return [self.chunk_ids[i] for i in np.argsort(scores)[::-1][:k]]

    @staticmethod
    def reciprocal_rank_fusion(ranked_lists, k: int = 60, top: int = 20, weights=None):
        weights = weights or [1.0] * len(ranked_lists)
        scores = {}
        for w, lst in zip(weights, ranked_lists):
            for rank, cid in enumerate(lst, 1):
                scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
        return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])[:top]]

    def hybrid_search(self, query_text: str, q_vec: np.ndarray, top: int = 20):
        return self.reciprocal_rank_fusion(
            [self.dense_search(q_vec, k=20), self.bm25_search(query_text, k=20)],
            top=top,
        )
