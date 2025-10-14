"""
Build a simple Knowledge Graph from chunk metadata and ABSA outputs.
Nodes: Product, Chunk, Aspect
Edges:
- Product --hasChunk--> Chunk
- Chunk --mentionsAspect--> Aspect [attr: sentiment, score]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import networkx as nx

from .config import ROOT, ensure_parents
from .utils_logging import get_logger

LOGGER = get_logger("cfpipe.kg")

def run(chunks_csv: Path, absa_json: Path, out_pickle: Path):
    ensure_parents(out_pickle)
    df = pd.read_csv(chunks_csv, usecols=["chunk_id", "product_id"])
    with open(absa_json, "r", encoding="utf-8") as f:
        absa = json.load(f)

    G = nx.MultiDiGraph()
    # Add product and chunk nodes
    for row in df.itertuples(index=False):
        G.add_node(f"product:{row.product_id}", kind="product", product_id=row.product_id)
        G.add_node(f"chunk:{row.chunk_id}", kind="chunk", chunk_id=row.chunk_id)
        G.add_edge(f"product:{row.product_id}", f"chunk:{row.chunk_id}", relation="hasChunk")

    # Add aspects edges
    for rec in absa:
        cnode = f"chunk:{rec['chunk_id']}"
        for a in rec["aspects"]:
            anode = f"aspect:{a}"
            G.add_node(anode, kind="aspect", aspect=a)
            G.add_edge(cnode, anode, relation="mentionsAspect",
                       sentiment=rec.get("sentiment"), score=float(rec.get("sentiment_score", 0.0)))

    nx.write_gpickle(G, out_pickle)
    LOGGER.info(f"✅ Wrote Knowledge Graph: {out_pickle} with {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default=str(ROOT / "data" / "chunked_reviews.csv"))
    ap.add_argument("--absa", type=str, default=str(ROOT / "data" / "aspect_sentiment.json"))
    ap.add_argument("--out", type=str, default=str(ROOT / "graph" / "knowledge_graph.pkl"))
    args = ap.parse_args()
    run(Path(args.chunks), Path(args.absa), Path(args.out))

if __name__ == "__main__":
    main()
