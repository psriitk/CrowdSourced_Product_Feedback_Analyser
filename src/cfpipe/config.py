"""
Config loader for cfpipe.
Reads YAML, exposes helpers to create directories and resolve paths.
"""
from __future__ import annotations
import os, yaml
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root

@dataclass
class Paths:
    raw_path: Path
    cleaned_csv: Path
    removed_csv: Path
    language_stats: Path
    chunked_csv: Path
    aspect_seed_terms: Path
    aspect_sentiment: Path
    id_to_text_map: Path
    dense_memmap: Path
    tfidf_vectorizer: Path
    tfidf_matrix: Path
    faiss_index: Path
    kg_pickle: Path
    chatbot_cfg: Path

def load_config(path: str | os.PathLike = ROOT / "config" / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_paths(cfg: dict) -> Paths:
    data = cfg["data"]
    emb = cfg["embeddings"]
    idx = cfg["indexes"]
    graph = cfg["graph"]
    chatbot = cfg["chatbot"]
    return Paths(
        raw_path=ROOT / data["raw_path"],
        cleaned_csv=ROOT / data["cleaned_csv"],
        removed_csv=ROOT / data["removed_csv"],
        language_stats=ROOT / data["language_stats"],
        chunked_csv=ROOT / data["chunked_csv"],
        aspect_seed_terms=ROOT / data["aspect_seed_terms"],
        aspect_sentiment=ROOT / data["aspect_sentiment"],
        id_to_text_map=ROOT / emb["id_to_text_map"],
        dense_memmap=ROOT / emb["dense_memmap"],
        tfidf_vectorizer=ROOT / emb["tfidf_vectorizer"],
        tfidf_matrix=ROOT / emb["tfidf_matrix"],
        faiss_index=ROOT / idx["faiss_index"],
        kg_pickle=ROOT / graph["pickle_path"],
        chatbot_cfg=ROOT / chatbot["config_yaml"],
    )

def ensure_parents(*paths: Path) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
