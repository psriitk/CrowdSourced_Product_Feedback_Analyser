"""
Build FAISS index from memmapped embeddings.
- CPU HNSW (no training) for laptop
- GPU IVF-PQ / IVF-Flat export for one-time GPU run
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from .config import ROOT, ensure_parents
from .utils_logging import get_logger

LOGGER = get_logger("cfpipe.build_faiss")

def load_memmap(path: Path, dim: int) -> np.memmap:
    return np.memmap(path, dtype=np.float32, mode="r", shape=(-1, dim))

def build_hnsw(x: np.ndarray, M: int = 32, efConstruction: int = 200):
    import faiss
    index = faiss.IndexHNSWFlat(x.shape[1], M)
    index.hnsw.efConstruction = efConstruction
    index.add(x)
    return index

def build_ivf_flat_gpu(x: np.ndarray, nlist: int = 4096):
    import faiss
    res = faiss.StandardGpuResources()
    d = x.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    cpu_index.train(x)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.add(x)
    return faiss.index_gpu_to_cpu(gpu_index)

def build_ivf_pq_gpu(x: np.ndarray, nlist: int = 4096, m: int = 16, nbits: int = 8):
    import faiss
    res = faiss.StandardGpuResources()
    d = x.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    cpu_index.train(x)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.add(x)
    return faiss.index_gpu_to_cpu(gpu_index)

def run(emb_path: Path, dim: int, out_path: Path, index_type: str = "hnsw", gpu: bool = False):
    ensure_parents(out_path)
    X = load_memmap(emb_path, dim)
    LOGGER.info(f"Loaded embeddings {emb_path} with shape {X.shape}")
    # Normalize already done; if not, do here for IP
    x = np.array(X, copy=False)

    if index_type == "hnsw":
        index = build_hnsw(x, M=32, efConstruction=200)
    else:
        if not gpu:
            raise RuntimeError("IVF indexes require --gpu for efficient training/build")
        if index_type == "ivf_flat":
            index = build_ivf_flat_gpu(x, nlist=4096)
        elif index_type == "ivf_pq":
            index = build_ivf_pq_gpu(x, nlist=4096, m=16, nbits=8)
        else:
            raise ValueError("index_type must be one of: hnsw, ivf_flat, ivf_pq")

    import faiss
    faiss.write_index(index, str(out_path))
    LOGGER.info(f"✅ Wrote FAISS index: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=str, default=str(ROOT / "embeddings" / "roberta_embeddings.memmap"))
    ap.add_argument("--dim", type=int, required=True, help="Embedding dimension (768 for distilroberta, 1024 for roberta-large)")
    ap.add_argument("--out", type=str, default=str(ROOT / "indexes" / "index.faiss"))
    ap.add_argument("--index-type", type=str, default="hnsw", choices=["hnsw", "ivf_flat", "ivf_pq"])
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    run(Path(args.emb), args.dim, Path(args.out), index_type=args.index_type, gpu=args.gpu)

if __name__ == "__main__":
    main()
