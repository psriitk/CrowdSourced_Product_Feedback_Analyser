"""
I/O utilities for streaming Amazon review datasets (JSON lines, possibly gzipped).
Includes:
- Reservoir sampling for large files
- Streaming JSON-lines reader
- CSV appenders
"""
from __future__ import annotations
import csv, gzip, io, json, os, random, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any

def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def stream_jsonl(path: Path) -> Iterator[dict]:
    """Stream JSON lines from a .json or .json.gz file safely."""
    with _open_text(path) as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed line
                continue

def reservoir_sample_jsonl(path: Path, k: int, seed: int = 42) -> List[dict]:
    """Reservoir sample k items from a very large jsonl file (single pass)."""
    rnd = random.Random(seed)
    sample: List[dict] = []
    for i, rec in enumerate(stream_jsonl(path), start=1):
        if i <= k:
            sample.append(rec)
        else:
            j = rnd.randint(1, i)
            if j <= k:
                sample[j-1] = rec
    return sample

def stable_id(*parts: str) -> str:
    """Create a stable review id by hashing source fields (avoid PII)."""
    h = hashlib.md5(("||".join(parts)).encode("utf-8")).hexdigest()
    return h

def write_csv(path: Path, rows: List[Dict[str, Any]], header: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if header is None:
        header = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
