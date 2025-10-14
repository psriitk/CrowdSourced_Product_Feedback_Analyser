"""
Hybrid retriever: Dense (FAISS) + Sparse (TF-IDF / Hashing) + KG expansion.
Usage:
    from cfpipe.retriever import HybridRetriever
    r = HybridRetriever(...)
    results = r.search("battery life on headphones", top_k=10)
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from scipy import sparse

from sentence_transformers import SentenceTransformer

from .utils_logging import get_logger

LOGGER = get_logger("cfpipe.retriever")

class HybridRetriever:
    def __init__(self,
                 id_map_path: Path,
                 emb_model_name: str,
                 faiss_index_path: Path,
                 tfidf_vectorizer_path: Path,
                 tfidf_matrix_path: Path,
                 kg_pickle: Path,
                 weights: Dict[str, float] = None,
                 device: str = "cpu"):
        import faiss  # local import
        self.faiss = faiss
        self.id_map = {int(d["index"]): d for d in json.loads(Path(id_map_path).read_text())}
        self.model = SentenceTransformer(emb_model_name, device=device)
        self.index = self.faiss.read_index(str(faiss_index_path))

        # Load sparse backend
        with open(tfidf_vectorizer_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and obj.get("backend") == "hashing":
            # Hashing backend
            from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
            self.h_cfg = obj
            self.hv = HashingVectorizer(n_features=obj["n_features"], alternate_sign=False, norm=obj["norm"], dtype=np.float32)
            self.tf = TfidfTransformer()
            self.Xs = sparse.load_npz(tfidf_matrix_path)
            self.backend = "hashing"
        else:
            self.vec = obj  # TfidfVectorizer
            self.Xs = sparse.load_npz(tfidf_matrix_path)
            self.backend = "tfidf"

        # Load KG
        import networkx as nx
        self.G = nx.read_gpickle(kg_pickle)

        self.w_dense = weights.get("dense", 0.5) if weights else 0.5
        self.w_sparse = weights.get("sparse", 0.35) if weights else 0.35
        self.w_kg = weights.get("kg", 0.15) if weights else 0.15

    def _dense_search(self, q: str, k: int = 20) -> List[Tuple[int, float]]:
        qv = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(qv.astype(np.float32), k)
        return list(zip(I[0].tolist(), D[0].tolist()))  # (idx, score)

    def _sparse_search(self, q: str, k: int = 40) -> List[Tuple[int, float]]:
        if self.backend == "tfidf":
            qv = self.vec.transform([q])
        else:
            qv = self.tf.transform(self.hv.transform([q]))
        sims = qv @ self.Xs.T  # [1, N]
        sims = sims.toarray().ravel()
        top_idx = np.argpartition(-sims, kth=min(k, sims.size-1))[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(int(i), float(sims[i])) for i in top_idx]

    def _kg_expand(self, q: str, k: int = 40) -> List[Tuple[int, float]]:
        # Heuristic: find aspects that appear in the query; pull chunks connected to these aspects
        q_low = q.lower()
        candidate_chunks = {}
        for node in self.G.nodes:
            if node.startswith("aspect:"):
                aspect = self.G.nodes[node].get("aspect", "")
                if aspect in q_low:
                    # get neighbors (chunks)
                    for cnode in self.G.predecessors(node):  # chunk -> aspect edges direction
                        if cnode.startswith("chunk:"):
                            # map chunk_id to dense index by reversing id_map
                            chunk_id = cnode.replace("chunk:", "")
                            # search in id_map for matching chunk_id (linear scan but acceptable for demo)
                            for idx, meta in self.id_map.items():
                                if meta["chunk_id"] == chunk_id:
                                    # weight negative sentiment higher if question contains 'problem' etc.
                                    weight = 1.0
                                    candidate_chunks[idx] = candidate_chunks.get(idx, 0.0) + weight
        # Return top-k by weight
        items = sorted(candidate_chunks.items(), key=lambda x: -x[1])[:k]
        return [(idx, float(w)) for idx, w in items]

    def search(self, q: str, top_k: int = 10) -> List[Dict[str, Any]]:
        d = self._dense_search(q, k=max(top_k*2, 20))
        s = self._sparse_search(q, k=max(top_k*2, 40))
        kgs = self._kg_expand(q, k=max(top_k*2, 40))

        # Combine
        scores = {}
        for i, sc in d:
            scores[i] = scores.get(i, 0.0) + self.w_dense * float(sc)
        for i, sc in s:
            scores[i] = scores.get(i, 0.0) + self.w_sparse * float(sc)
        for i, sc in kgs:
            scores[i] = scores.get(i, 0.0) + self.w_kg * float(sc)

        top = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, sc in top:
            meta = self.id_map[idx]
            results.append({"index": idx, "score": float(sc), **meta})
        return results
