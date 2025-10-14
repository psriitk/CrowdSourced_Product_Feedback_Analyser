"""
Chunk cleaned reviews into ~N-token segments with overlap.
Uses RoBERTa tokenizer to respect token limits.
Outputs: chunked_reviews.csv (one row per chunk)
"""
from __future__ import annotations
import argparse, math, csv
from pathlib import Path
from typing import List, Dict
import pandas as pd
from transformers import AutoTokenizer

from .config import load_config, get_paths, ensure_parents, ROOT
from .utils_logging import get_logger

LOGGER = get_logger("cfpipe.chunking")

def roberta_chunks(text: str, tokenizer, chunk_tokens: int, overlap: int) -> List[str]:
    # Tokenize and slice by token ids (excluding special tokens)
    toks = tokenizer.encode(text, add_special_tokens=False)
    if not toks:
        return []
    stride = max(1, chunk_tokens - overlap)
    chunks = []
    for i in range(0, len(toks), stride):
        window = toks[i:i+chunk_tokens]
        if not window:
            break
        chunk = tokenizer.decode(window, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        chunks.append(chunk)
        if i + chunk_tokens >= len(toks):
            break
    return chunks

def run(input_csv: Path, out_csv: Path, chunk_tokens: int = 384, overlap: int = 64, tokenizer_name: str = "roberta-base"):
    ensure_parents(out_csv)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    rows = []
    for i, row in enumerate(pd.read_csv(input_csv).itertuples(index=False), start=1):
        text = str(row.text)
        chs = roberta_chunks(text, tok, chunk_tokens, overlap)
        for j, c in enumerate(chs):
            rows.append({
                "chunk_id": f"{row.review_id}_{j}",
                "review_id": row.review_id,
                "product_id": row.product_id,
                "rating": row.rating,
                "unix_time": row.unix_time,
                "chunk_text": c
            })
        if i % 5000 == 0:
            LOGGER.info(f"Chunked {i} reviews -> {len(rows)} chunks so far")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    LOGGER.info(f"✅ Wrote {out_csv} with {len(rows)} chunks")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(ROOT / "data" / "cleaned_reviews.csv"))
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "chunked_reviews.csv"))
    ap.add_argument("--chunk-tokens", type=int, default=384)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--tokenizer", type=str, default="roberta-base")
    args = ap.parse_args()

    cfg = load_config()
    paths = get_paths(cfg)
    run(Path(args.input or paths.cleaned_csv), Path(args.out or paths.chunked_csv),
        chunk_tokens=args.chunk_tokens, overlap=args.overlap, tokenizer_name=args.tokenizer)

if __name__ == "__main__":
    main()
