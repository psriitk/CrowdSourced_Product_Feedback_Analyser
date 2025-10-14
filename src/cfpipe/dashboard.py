"""
Streamlit dashboard for interactive analysis.
- Reads aspect_sentiment.json and chunked_reviews.csv
- Charts: sentiment pie, top negative aspects bar, aspect trend over time
"""
import json
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from .config import ROOT

st.set_page_config(page_title="Crowdsourced Feedback Dashboard", layout="wide")
st.title("📊 Crowdsourced Product Feedback Dashboard")

# Load data
aspect_path = ROOT / "data" / "aspect_sentiment.json"
chunked_path = ROOT / "data" / "chunked_reviews.csv"

if not aspect_path.exists() or not chunked_path.exists():
    st.warning("Run the pipeline first to generate aspect_sentiment.json and chunked_reviews.csv")
    st.stop()

with open(aspect_path, "r", encoding="utf-8") as f:
    absa = json.load(f)
df_absa = pd.DataFrame(absa)
df_chunks = pd.read_csv(chunked_path)

# Merge for time-based charts
df = df_absa.merge(df_chunks[["chunk_id", "unix_time"]], on="chunk_id", how="left")
df["ts"] = pd.to_datetime(df["unix_time"], unit="s", errors="coerce")
df["month"] = df["ts"].dt.to_period("M").astype(str)

# Sidebar filters
aspects = sorted(set(a for xs in df["aspects"] for a in xs))
chosen_aspects = st.sidebar.multiselect("Filter by aspect(s)", aspects, default=[])

if chosen_aspects:
    df = df[df["aspects"].apply(lambda xs: any(a in xs for a in chosen_aspects))]

# 1) Sentiment pie
st.subheader("Sentiment distribution")
sent_counts = df["sentiment"].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(sent_counts.values, labels=sent_counts.index, autopct="%1.1f%%")
st.pyplot(fig1)

# 2) Top negative aspects
st.subheader("Top negative aspects")
neg = df[df["sentiment"] == "negative"]
aspect_freq = {}
for xs in neg["aspects"]:
    for a in xs: aspect_freq[a] = aspect_freq.get(a, 0) + 1
top_neg = sorted(aspect_freq.items(), key=lambda x: -x[1])[:15]
fig2, ax2 = plt.subplots()
ax2.bar([a for a, _ in top_neg], [c for _, c in top_neg])
ax2.set_xticklabels([a for a, _ in top_neg], rotation=45, ha="right")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# 3) Aspect sentiment over time (avg sentiment score: positive=+1, negative=-1)
st.subheader("Aspect sentiment over time")
def to_signed(s):
    return 1.0 if s == "positive" else -1.0
df["signed"] = df["sentiment"].apply(to_signed) * df["sentiment_score"].astype(float)
trend = df.groupby("month")["signed"].mean().reset_index()
fig3, ax3 = plt.subplots()
ax3.plot(trend["month"], trend["signed"])
ax3.set_xticklabels(trend["month"], rotation=45, ha="right")
ax3.set_ylabel("Avg sentiment score")
st.pyplot(fig3)

st.success("✅ Dashboard ready. Use the sidebar to filter by aspects.")
