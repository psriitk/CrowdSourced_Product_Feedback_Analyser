"""
Create a sparse retrieval backend.
Backends:
- tfidf: sklearn TfidfVectorizer (good quality, needs fit; OK for subsets)
- hashing: HashingVectorizer + stored config (scales to very large corpora, no fit)

Writes: tfidf_vectorizer.pkl (if tfidf), tfidf_matrix.npz
"""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer

from .config import ROOT, load_config, get_paths, ensure_parents
from .utils_logging import get_logger

LOGGER = get_logger("cfpipe.tfidf")

def run(input_csv: Path, out_vectorizer: Path, out_matrix: Path, backend: str = "tfidf",
        max_features: int = 200000, n_features_hash: int = 2**20):
    ensure_parents(out_vectorizer, out_matrix)
    df = pd.read_csv(input_csv, usecols=["chunk_text"])
    texts = df["chunk_text"].astype(str).tolist()

    if backend == "tfidf":
        LOGGER.info("Fitting TfidfVectorizer (may use memory for large corpora; tune max_features/min_df) ...")
        vec = TfidfVectorizer(max_features=max_features, min_df=3, dtype=np.float32)
        X = vec.fit_transform(texts)
        with open(out_vectorizer, "wb") as f:
            pickle.dump(vec, f)
        sparse.save_npz(out_matrix, X.astype(np.float32))
        LOGGER.info(f"✅ Wrote TF-IDF vectorizer and matrix: {out_vectorizer}, {out_matrix} (shape={X.shape})")
    elif backend == "hashing":
        LOGGER.info("Using HashingVectorizer + TfidfTransformer (no-fit, scalable) ...")
        hv = HashingVectorizer(n_features=n_features_hash, alternate_sign=False, norm="l2", dtype=np.float32)
        X_counts = hv.transform(texts)
        tfidf = TfidfTransformer()
        X = tfidf.fit_transform(X_counts)
        # Save config (since HashingVectorizer is stateless)
        cfg = {"backend": "hashing", "n_features": n_features_hash, "alternate_sign": False, "norm": "l2"}
        with open(out_vectorizer, "wb") as f:
            pickle.dump(cfg, f)
        sparse.save_npz(out_matrix, X.astype(np.float32))
        LOGGER.info(f"✅ Wrote Hashing TF-IDF matrix + config: {out_vectorizer}, {out_matrix} (shape={X.shape})")
    else:
        raise ValueError("backend must be 'tfidf' or 'hashing'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(ROOT / "data" / "chunked_reviews.csv"))
    ap.add_argument("--out-vectorizer", type=str, default=str(ROOT / "embeddings" / "tfidf_vectorizer.pkl"))
    ap.add_argument("--out-matrix", type=str, default=str(ROOT / "embeddings" / "tfidf_matrix.npz"))
    ap.add_argument("--backend", type=str, default="tfidf", choices=["tfidf", "hashing"])
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--n-features-hash", type=int, default=1048576)
    args = ap.parse_args()
    run(Path(args.input), Path(args.out_vectorizer), Path(args.out_matrix),
        backend=args.backend, max_features=args.max_features, n_features_hash=args.n_features_hash)

if __name__ == "__main__":
    main()
