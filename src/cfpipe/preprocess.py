"""
Enhanced preprocessing for Amazon review datasets (e.g., Electronics_5.json.gz).

Features:
- Streaming JSONL/.gz read (memory-safe)
- PII masking in text: [URL], [EMAIL], [PHONE]
- Verified -> bool; vote -> cleaned text + vote_int (numeric)
- Time parsing: unix_time (int) + review_time_iso (UTC ISO)
- Language detection (keep English), short/nonsense filters
- Style flattening: style_color / style_size / style_other
- Derived features: text_len, word_count
- Hash-based de-dup: (reviewerID|asin|unix_time|cleaned_text) for small RAM usage
- Chunked CSV appends + removal audit + language stats

Outputs:
- data/cleaned_reviews.csv
- data/removed_reviews.csv
- data/language_stats.json
"""
from __future__ import annotations

import argparse
import html
import io
import gzip
import json
import math
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterator, Dict, Any, Tuple

import pandas as pd
from langdetect import detect, DetectorFactory

from .config import load_config, get_paths, ensure_parents, ROOT
from .io_utils import write_csv  # we’ll stream manually; write_csv only used for small lists if needed
from .utils_logging import get_logger

DetectorFactory.seed = 42
LOGGER = get_logger("cfpipe.preprocess")

# ---------- Streaming reader ----------
def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def stream_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

# ---------- Text / PII cleaning ----------
WS_RE    = re.compile(r"\s+")
CTRL_RE  = re.compile(r"[\x00-\x1f\x7f]")
URL_RE   = re.compile(r"(https?://\S+)", re.I)
EMAIL_RE = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1,2}\d{4}\b")

def strip_ctrl(txt: str) -> str:
    return CTRL_RE.sub(" ", txt)

def normalize_ws(txt: str) -> str:
    return WS_RE.sub(" ", txt.strip())

def mask_pii(txt: str) -> str:
    txt = URL_RE.sub("[URL]", txt)
    txt = EMAIL_RE.sub("[EMAIL]", txt)
    txt = PHONE_RE.sub("[PHONE]", txt)
    return txt

def clean_text(txt: str) -> str:
    txt = str(txt or "")
    txt = html.unescape(txt)
    txt = strip_ctrl(txt)
    txt = normalize_ws(txt)
    txt = mask_pii(txt)
    return txt

# ---------- Coercers ----------
def to_int_votes(v) -> int:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0
    try:
        return int(str(v).replace(",", "").strip())
    except Exception:
        # extract first integer if embedded in text
        m = re.search(r"\d+", str(v))
        return int(m.group(0)) if m else 0

def to_bool_verified(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true", "t", "1", "yes", "y")

def parse_review_time(review_time, unix_time) -> Tuple[int | None, str | None]:
    # Prefer unix
    if unix_time is not None and not (isinstance(unix_time, float) and math.isnan(unix_time)):
        try:
            ut = int(unix_time)
            iso = pd.to_datetime(ut, unit="s", utc=True).isoformat()
            return ut, iso
        except Exception:
            pass
    # Fallback to parsing review_time
    if review_time:
        dt = pd.to_datetime(review_time, errors="coerce", utc=True)
        if not pd.isna(dt):
            return int(dt.timestamp()), dt.isoformat()
    return None, None

def detect_lang_safe(text: str) -> str:
    try:
        return detect(text) if len(text) >= 20 else "en"
    except Exception:
        return "unknown"

def is_nonsensical(text: str) -> bool:
    t = text.replace(" ", "")
    if len(t) >= 30:
        uniq_ratio = len(set(t)) / max(1, len(t))
        if uniq_ratio < 0.1:
            return True
    return False

def parse_style(style_field) -> Tuple[str, str, str, str]:
    """
    Returns (style_raw, style_color, style_size, style_other) as strings (no Nones).
    Accepts dict or loosely formatted string "key:val; key:val".
    """
    import ast
    raw = ""
    d: Dict[str, Any] = {}
    color = ""
    size = ""
    other = ""

    if style_field is None or (isinstance(style_field, float) and math.isnan(style_field)):
        return raw, color, size, other

    if isinstance(style_field, dict):
        raw = json.dumps(style_field, ensure_ascii=False)
        d = style_field
    else:
        s = str(style_field).strip()
        raw = s
        # Try parse literal dict
        try:
            v = ast.literal_eval(s)
            if isinstance(v, dict):
                d = v
        except Exception:
            # Fallback parse "k:v; k:v"
            parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    d[k.strip()] = str(v).strip()

    for k, v in d.items():
        kl = k.lower()
        if not color and ("color" in kl or "colour" in kl):
            color = str(v)
        if not size and ("size" in kl or "capacity" in kl or "storage" in kl):
            size = str(v)

    if d and not (color or size):
        other = "; ".join(f"{k}: {v}" for k, v in d.items())

    return raw or "", color or "", size or "", other or ""

# ---------- Main runner ----------
def run(input_path: Path,
        out_csv: Path,
        removed_csv: Path,
        language_stats: Path,
        max_rows: int = 0,
        sample: int = 0,
        infer_cols: bool = True,
        write_chunk_size: int = 50_000) -> None:
    """
    :param max_rows: hard cap of raw rows to read (0 = no cap)
    :param sample: if >0, reservoir-sample this many rows (randomized single-pass)
                   (Note: for simplicity here we just hard-cap via max_rows if sample>0.
                    If you need true reservoir sampling, call cfpipe.io_utils.reservoir_sample_jsonl externally.)
    :param write_chunk_size: flush cleaned rows to disk every N rows
    """
    ensure_parents(out_csv, removed_csv, language_stats)

    # Open outputs with headers once
    cleaned_cols = [
        "review_id", "product_id", "user_id",
        "rating", "verified", "vote", "vote_int",
        "unix_time", "review_time", "review_time_iso",
        "summary", "text", "lang",
        "text_len", "word_count",
        "style_raw", "style_color", "style_size", "style_other"
    ]
    pd.DataFrame(columns=cleaned_cols).to_csv(out_csv, index=False)
    pd.DataFrame(columns=["reason", "reviewerID", "asin"]).to_csv(removed_csv, index=False)

    dedupe_seen: set[bytes] = set()
    lang_ctr = Counter()
    buf: list[Dict[str, Any]] = []
    removed_buf: list[Dict[str, Any]] = []

    n_in = 0
    n_kept = 0

    # Determine iteration policy
    iterator = stream_jsonl(input_path)
    # If sample > 0 and you want true randomized sample, prefer reservoir sampling in a separate helper
    hard_cap = sample if sample > 0 else (max_rows if max_rows > 0 else 0)

    for rec in iterator:
        n_in += 1
        if hard_cap and n_in > hard_cap:
            break

        # Extract raw fields (image dropped implicitly)
        overall = rec.get("overall", None)
        vote_raw = rec.get("vote", None)
        verified_raw = rec.get("verified", None)
        reviewTime = rec.get("reviewTime", None)
        reviewerID = rec.get("reviewerID", None)
        asin = rec.get("asin", None)
        style = rec.get("style", None)
        summary_raw = rec.get("summary", "")
        reviewText_raw = rec.get("reviewText", rec.get("review", ""))
        unixReviewTime = rec.get("unixReviewTime", None)

        # Required keys
        if not reviewerID or not asin:
            removed_buf.append({"reason": "missing_keys", "reviewerID": reviewerID, "asin": asin})
            continue

        # Clean text + summary (with PII masking)
        txt = clean_text(reviewText_raw)
        summ = clean_text(summary_raw)

        # Short / nonsense filter
        if len(txt) < 20 or is_nonsensical(txt):
            removed_buf.append({"reason": "short_or_nonsense", "reviewerID": reviewerID, "asin": asin})
            continue

        # Language
        lang = detect_lang_safe(txt)
        lang_ctr[lang] += 1
        if lang != "en":
            removed_buf.append({"reason": f"lang_{lang}", "reviewerID": reviewerID, "asin": asin})
            continue

        # Rating
        try:
            rating = float(overall)
        except Exception:
            rating = None
        if rating is None or not (1.0 <= rating <= 5.0):
            removed_buf.append({"reason": "bad_rating", "reviewerID": reviewerID, "asin": asin})
            continue

        # Time
        ut, dt_iso = parse_review_time(reviewTime, unixReviewTime)
        if ut is None:
            removed_buf.append({"reason": "bad_time", "reviewerID": reviewerID, "asin": asin})
            continue

        # Hash-based de-dup (small memory)
        k = f"{reviewerID}|{asin}|{int(ut)}|{txt}"
        dedupe_key = hashlib.md5(k.encode("utf-8")).digest()
        if dedupe_key in dedupe_seen:
            removed_buf.append({"reason": "duplicate", "reviewerID": reviewerID, "asin": asin})
            continue
        dedupe_seen.add(dedupe_key)

        # Votes / verified
        vote_int = to_int_votes(vote_raw)
        verified_bool = to_bool_verified(verified_raw)

        # Style
        style_raw, style_color, style_size, style_other = parse_style(style)

        # Stable review_id
        review_id = hashlib.md5(f"{reviewerID}|{asin}|{int(ut)}|{txt[:64]}".encode("utf-8")).hexdigest()

        # Derived
        words = txt.split()
        row = {
            "review_id": review_id,
            "product_id": str(asin),
            "user_id": str(reviewerID),
            "rating": float(rating),
            "verified": bool(verified_bool),
            "vote": str(vote_raw).replace(",", "").strip() if vote_raw is not None else "",
            "vote_int": int(vote_int),
            "unix_time": int(ut),
            "review_time": reviewTime if reviewTime is not None else "",
            "review_time_iso": dt_iso if dt_iso is not None else "",
            "summary": summ or "",
            "text": txt or "",
            "lang": "en",
            "text_len": int(len(txt)),
            "word_count": int(len(words)),
            "style_raw": style_raw,
            "style_color": style_color,
            "style_size": style_size,
            "style_other": style_other,
        }
        buf.append(row)
        n_kept += 1

        # Flush
        if len(buf) >= write_chunk_size:
            pd.DataFrame(buf, columns=cleaned_cols).to_csv(out_csv, mode="a", header=False, index=False)
            buf.clear()

        # Progress
        if n_in % 10_000 == 0:
            LOGGER.info(f"Processed {n_in:,} raw lines | kept so far: {n_kept:,} | removed buffer: {len(removed_buf):,}")

    # Final flush
    if buf:
        pd.DataFrame(buf, columns=cleaned_cols).to_csv(out_csv, mode="a", header=False, index=False)
    if removed_buf:
        pd.DataFrame(removed_buf)[["reason", "reviewerID", "asin"]].to_csv(removed_csv, mode="a", header=False, index=False)

    # Language stats
    with open(language_stats, "w", encoding="utf-8") as f:
        json.dump({"counts": dict(lang_ctr)}, f, indent=2)

    LOGGER.info(f"✅ Done. Raw seen={n_in:,} | Kept={n_kept:,} | Removed={len(removed_buf):,}")
    LOGGER.info(f"Wrote: {out_csv}, {removed_csv}, {language_stats}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(ROOT / "data" / "Electronics_5.json.gz"))
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "cleaned_reviews.csv"))
    ap.add_argument("--removed", type=str, default=str(ROOT / "data" / "removed_reviews.csv"))
    ap.add_argument("--langstats", type=str, default=str(ROOT / "data" / "language_stats.json"))
    ap.add_argument("--max-rows", type=int, default=0, help="Hard cap of raw lines to read (0=no cap)")
    ap.add_argument("--sample", type=int, default=0, help="If >0, sample this many rows (use reservoir externally for true random)")
    ap.add_argument("--write-chunk-size", type=int, default=50_000, help="Flush to CSV every N kept rows")
    ap.add_argument("--infer", action="store_true", help="Use paths from config/config.yaml")
    args = ap.parse_args()

    if args.infer:
        cfg = load_config()
        paths = get_paths(cfg)
        run(paths.raw_path, paths.cleaned_csv, paths.removed_csv, paths.language_stats,
            max_rows=args.max_rows, sample=args.sample, write_chunk_size=args.write_chunk_size)
    else:
        run(Path(args.input), Path(args.out), Path(args.removed), Path(args.langstats),
            max_rows=args.max_rows, sample=args.sample, write_chunk_size=args.write_chunk_size)

if __name__ == "__main__":
    main()
