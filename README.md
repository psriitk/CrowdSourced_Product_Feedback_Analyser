# Crowdsourced Product Feedback Analyzer (RoBERTa + TF-IDF + KG + LangChain + Dashboard)

End-to-end, **deployable** pipeline for massive review datasets (e.g., Amazon **Electronics_5.json.gz**).
Supports:
- **Advanced preprocessing & chunking**
- **ABSA** (Aspect-Based Sentiment Analysis) via **RoBERTa** (zero-shot or fine-tuned)
- **Embeddings** with **RoBERTa (sentence-transformers)**
- **Sparse TF-IDF** (or Hashing for very large data)
- **Vector DB (FAISS)** + **Knowledge Graph (NetworkX/Neo4j)**
- **Adaptive Hybrid Retrieval** (dense + sparse + KG)
- **LangChain** chatbot interface (optional local HF summarizer or OpenAI)
- **Streamlit dashboard** with basic monitoring
- **One-time GPU run** to precompute embeddings/ABSA + export FAISS index, then serve cheaply on CPU

This scaffold contains:
- A **small, CPU-friendly notebook** for laptop proof-of-concept on a subset.
- **Generalized .py scripts** for streaming/large-scale processing and GPU deployment.
- **Dockerfiles** for CPU and GPU.
- Install scripts, Makefile, and configuration files.

---

## Quick start (Laptop / CPU demo)

1. Place your dataset at: `data/Electronics_5.json.gz`
2. Create a Python 3.10+ venv and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   # (Optional, for spaCy if you want it)
   # python -m spacy download en_core_web_sm
   ```
3. Open the notebook and run the demo (uses a small sample so it fits in 8GB RAM):
   ```bash
   jupyter lab  # or jupyter notebook
   # open notebooks/01_small_pipeline_demo.ipynb
   ```

**Outputs** will be written under: `data/`, `embeddings/`, `indexes/`, `graph/`, `analytics/`.

---

## One-time GPU run (H100/A100) — full data

Use the included **conda** env and scripts:

```bash
# Create GPU env (CUDA 12.x)
conda env create -f environment-gpu.yml
conda activate cfpipe-gpu

# Run the big-data pipeline in streaming mode
# 1) Preprocess + chunk
python -m cfpipe.preprocess --input data/Electronics_5.json.gz --out data/cleaned_reviews.csv --max-rows 0 --infer
python -m cfpipe.chunking --input data/cleaned_reviews.csv --out data/chunked_reviews.csv --chunk-tokens 384 --overlap 64

# 2) ABSA (zero-shot or fine-tuned); batched on GPU
python -m cfpipe.absa --input data/chunked_reviews.csv --out data/aspect_sentiment.json --batch-size 64 --device cuda

# 3) Embeddings + TF-IDF (streaming); embeddings on GPU
python -m cfpipe.embed --input data/chunked_reviews.csv --id-map embeddings/id_to_text_map.json --emb emb/roberta_embeddings.memmap --model sentence-transformers/all-roberta-large-v1 --batch-size 512 --device cuda
python -m cfpipe.tfidf --input data/chunked_reviews.csv --out-vectorizer embeddings/tfidf_vectorizer.pkl --out-matrix embeddings/tfidf_matrix.npz --backend hashing

# 4) FAISS index (GPU build/export), KG
python -m cfpipe.build_faiss --emb emb/roberta_embeddings.memmap --dim 1024 --out indexes/index.faiss --gpu --index-type ivf_pq
python -m cfpipe.kg --chunks data/chunked_reviews.csv --absa data/aspect_sentiment.json --out graph/knowledge_graph.pkl
```

Then **deploy** a low-cost CPU API + Streamlit dashboard using the precomputed artifacts:

```bash
# API (FastAPI)
uvicorn cfpipe.api_server:app --host 0.0.0.0 --port 8000

# Dashboard (Streamlit)
streamlit run src/cfpipe/dashboard.py
```

---

## Project layout

```
crowd_feedback_pipeline/
├── config/
│   └── config.yaml
├── data/                         # place Electronics_5.json.gz here
├── embeddings/                   # id_to_text_map.json, TF-IDF, embeddings
├── indexes/                      # FAISS index
├── graph/                        # knowledge_graph.pkl
├── analytics/                    # aggregated parquet/csv
├── src/cfpipe/                   # all python modules (importable as cfpipe)
├── notebooks/                    # small, CPU-friendly demo
├── docker/                       # Dockerfiles for CPU & GPU
├── scripts/                      # helper scripts
├── requirements.txt              # CPU/laptop deps
├── environment-gpu.yml           # GPU env
└── README.md
```

---

## Notes

- **TF-IDF backend**: for very large data, the script supports `--backend hashing` (no fit, low memory) and stores config for reproducibility.
- **FAISS index types**:
  - Laptop / CPU: HNSW (no training, good quality/latency balance).
  - GPU (one-time): IVF-PQ or IVF-Flat (train on a sample, very fast retrieval).
- **ABSA**: default is **zero-shot** with **XLM-RoBERTa** (`joeddav/xlm-roberta-large-xnli`) for aspect detection + **RoBERTa sentiment** (`siebert/sentiment-roberta-large-english`). You can later fine-tune your own model and swap it in (`models/domain_adapted_roberta/`).

Good luck, and happy shipping!
