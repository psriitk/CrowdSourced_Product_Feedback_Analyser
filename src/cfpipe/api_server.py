"""
FastAPI server exposing search and chat endpoints.
- /search?q=... : hybrid retrieval
- /chat : uses LangChain to summarize retrieved snippets into an answer
"""
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, Query
from pydantic import BaseModel

from .config import load_config, get_paths, ROOT
from .retriever import HybridRetriever
from .utils_logging import get_logger

LOGGER = get_logger("cfpipe.api")

app = FastAPI(title="Crowdsourced Feedback Analyzer API")

class ChatRequest(BaseModel):
    query: str
    top_k: int = 10

# Lazy globals
_retriever = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        cfg = load_config()
        paths = get_paths(cfg)
        weights = {"dense": 0.5, "sparse": 0.35, "kg": 0.15}
        _retriever = HybridRetriever(
            id_map_path=paths.id_to_text_map,
            emb_model_name=cfg["defaults"]["embedding_model_laptop"],
            faiss_index_path=paths.faiss_index,
            tfidf_vectorizer_path=paths.tfidf_vectorizer,
            tfidf_matrix_path=paths.tfidf_matrix,
            kg_pickle=paths.kg_pickle,
            weights=weights,
            device="cpu"
        )
    return _retriever

@app.get("/search")
def search(q: str = Query(..., min_length=2), top_k: int = 10):
    r = get_retriever()
    results = r.search(q, top_k=top_k)
    return {"query": q, "results": results}

@app.post("/chat")
def chat(req: ChatRequest):
    r = get_retriever()
    res = r.search(req.query, top_k=req.top_k)

    # Pull chunk texts for answer assembly
    id2text = {}
    import json
    with open(ROOT / "embeddings" / "id_to_text_map.json", "r", encoding="utf-8") as f:
        id_map = json.load(f)
    chunk_ids = [d["chunk_id"] for d in id_map]
    # Load chunked CSV to fetch chunk_text
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "chunked_reviews.csv", usecols=["chunk_id", "chunk_text"])
    chunk_text_map = dict(zip(df["chunk_id"], df["chunk_text"]))

    # Summarize with LangChain: prefer local HF summarizer to avoid API keys
    try:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline
        hf_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        llm = HuggingFacePipeline(pipeline=hf_summarizer)
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        prompt = PromptTemplate(
            input_variables=["question", "snippets"],
            template=(
                "You are an assistant summarizing user reviews.\n"
                "Question: {question}\n\n"
                "Relevant snippets:\n{snippets}\n\n"
                "Write a concise, neutral answer with bullet points and mention the main aspects and their sentiment."
            ),
        )
        snippets = "\n---\n".join([chunk_text_map.get(r["chunk_id"], "")[:400] for r in res])
        chain = LLMChain(llm=llm, prompt=prompt)
        answer = chain.run({"question": req.query, "snippets": snippets})
    except Exception as e:
        answer = "Summary unavailable (LLM not configured). Here are the top snippets:\n" + \
                 "\n\n".join([chunk_text_map.get(r["chunk_id"], "")[:280] for r in res])

    return {"query": req.query, "answer": answer, "results": res}
