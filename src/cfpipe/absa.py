"""
ABSA (Aspect-Based Sentiment Analysis) pipeline.

- Aspect detection: zero-shot with **XLM-RoBERTa** (`joeddav/xlm-roberta-large-xnli`)
- Sentiment per (chunk, aspect): **RoBERTa** sentiment (`siebert/sentiment-roberta-large-english`)
- Outputs: aspect_sentiment.json (list of records)

You can later swap in a fine-tuned RoBERTa model by implementing `predict_aspects_finetuned(...)`
and `predict_sentiment_finetuned(...)`.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline

from .config import load_config, get_paths, ROOT, ensure_parents
from .utils_logging import get_logger

LOGGER = get_logger("cfpipe.absa")

def run(input_csv: Path, out_json: Path, aspects: List[str], batch_size: int = 8, device: str = "cpu"):
    ensure_parents(out_json)
    LOGGER.info(f"Loading pipelines on device={device} ...")
    # Zero-shot aspect detection
    aspect_model = "joeddav/xlm-roberta-large-xnli"
    nli = pipeline("zero-shot-classification", model=aspect_model, device=0 if device.startswith("cuda") else -1)
    # Sentiment model (RoBERTa)
    sent_model = "siebert/sentiment-roberta-large-english"
    sent = pipeline("sentiment-analysis", model=sent_model, device=0 if device.startswith("cuda") else -1)

    df = pd.read_csv(input_csv)
    records: List[Dict[str, Any]] = []

    texts = df["chunk_text"].tolist()
    chunk_ids = df["chunk_id"].tolist()
    product_ids = df["product_id"].tolist()

    # Batched processing
    for i in tqdm(range(0, len(texts), batch_size), desc="ABSA"):
        batch_texts = texts[i:i+batch_size]
        batch_chunk_ids = chunk_ids[i:i+batch_size]
        batch_product_ids = product_ids[i:i+batch_size]

        # Aspect detection with multi-label zero-shot
        z = nli(batch_texts, candidate_labels=aspects, multi_label=True)
        # z is either dict (single) or list (batch). Normalize to list.
        if isinstance(z, dict):
            zs = [z]
        else:
            zs = z  # type: ignore

        for j, zres in enumerate(zs):
            # Keep aspects with score >= 0.4 (tune as needed)
            pairs = [(lab, sc) for lab, sc in zip(zres["labels"], zres["scores"]) if sc >= 0.4]
            chosen_aspects = [a for a, _ in pairs]

            # Sentiment for this chunk once
            s = sent(batch_texts[j])[0]
            sentiment = s["label"].lower()  # 'positive' or 'negative'
            score = float(s["score"])

            rec = {
                "chunk_id": batch_chunk_ids[j],
                "product_id": batch_product_ids[j],
                "aspects": chosen_aspects,
                "sentiment": sentiment,
                "sentiment_score": score
            }
            records.append(rec)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    LOGGER.info(f"✅ Wrote {out_json} ({len(records)} items)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(ROOT / "data" / "chunked_reviews.csv"))
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "aspect_sentiment.json"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    cfg = load_config()
    aspects = cfg["defaults"]["absa_aspects"]
    run(Path(args.input), Path(args.out), aspects=aspects, batch_size=args.batch_size, device=args.device)

if __name__ == "__main__":
    main()
