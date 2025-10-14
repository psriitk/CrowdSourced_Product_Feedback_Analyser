"""
Compute sentence-level embeddings for chunks using sentence-transformers (RoBERTa variants).
Writes:
- id_to_text_map.json   (index -> {chunk_id, product_id})
- roberta_embeddings.memmap (float32 memmap array shape [N, D])
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

from .config import ROOT, load_config, get_paths, ensure_parents
from .utils_logging import get_logger

LOGGER = get_logger("cfpipe.embed")

def run(input_csv: Path, id_map_path: Path, emb_memmap_path: Path,
        model_name: str = "sentence-transformers/all-distilroberta-v1",
        batch_size: int = 64, device: str = "cpu"):
    ensure_parents(id_map_path, emb_memmap_path)

    LOGGER.info(f"Loading model {model_name} on device={device} ...")
    model = SentenceTransformer(model_name, device=device)
    # Determine embedding dim
    dummy = model.encode(["hello world"], convert_to_numpy=True, normalize_embeddings=True)
    dim = dummy.shape[1]
    LOGGER.info(f"Embedding dim = {dim}")

    df = pd.read_csv(input_csv)
    texts = df["chunk_text"].tolist()
    chunk_ids = df["chunk_id"].tolist()
    product_ids = df["product_id"].tolist()
    N = len(texts)

    # Prepare memmap file
    dtype = np.float32
    emb_mmap = np.memmap(emb_memmap_path, dtype=dtype, mode="w+", shape=(N, dim))

    id_map = []
    idx = 0
    for i in tqdm(range(0, N, batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size)
        bsz = embs.shape[0]
        emb_mmap[idx:idx+bsz] = embs.astype(dtype, copy=False)
        # Map indices -> ids
        for k in range(bsz):
            id_map.append({"index": idx+k, "chunk_id": chunk_ids[i+k], "product_id": product_ids[i+k]})
        idx += bsz

    emb_mmap.flush()
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2)

    LOGGER.info(f"✅ Wrote embeddings to {emb_memmap_path} and id map to {id_map_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(ROOT / "data" / "chunked_reviews.csv"))
    ap.add_argument("--id-map", type=str, default=str(ROOT / "embeddings" / "id_to_text_map.json"))
    ap.add_argument("--emb", type=str, default=str(ROOT / "embeddings" / "roberta_embeddings.memmap"))
    ap.add_argument("--model", type=str, default="sentence-transformers/all-distilroberta-v1")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    run(Path(args.input), Path(args.id_map), Path(args.emb), model_name=args.model,
        batch_size=args.batch_size, device=args.device)

if __name__ == "__main__":
    main()
