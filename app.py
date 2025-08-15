import os, json, time, yaml, math
from pathlib import Path
import streamlit as st
import pandas as pd

from src.utils.text_clean import normalize_text
from src.processors.trend_detector import detect_trends
from src.processors.moderation import analyze_sentiment, passes_moderation
from src.processors.embedding import embed_texts
from src.storage.vectorstore import VectorStore
from src.processors.rag import generate_answer

# --- Config
cfg_path = Path("config.yaml")
cfg = {}
if cfg_path.exists():
    cfg = yaml.safe_load(open(cfg_path, "r"))
else:
    cfg = yaml.safe_load(open("config.example.yaml","r"))

OPENAI_API_KEY = (cfg.get("openai",{}) or {}).get("api_key") or os.getenv("OPENAI_API_KEY","")

# --- State
if "posts" not in st.session_state:
    # Try to load sample data
    sample = json.load(open("data/sample_posts.json","r"))
    st.session_state.posts = sample

st.set_page_config(page_title="Social Media RAG + Trends", layout="wide")

st.title(" Social Media RAG with Trend Analysis")

colA, colB = st.columns([2,1])
with colA:
    st.markdown("Ingest posts from multiple platforms, detect trends, and ask context-aware questions.                 Works offline with sample data; add API keys to go live.")

with colB:
    refresh = st.button(" Refresh (demo)")

# --- Ingestion (demo uses sample; with keys you could wire connectors here)
with st.expander("Ingestion Settings", expanded=False):
    search_terms = st.text_input("Search terms (comma-separated)", ", ".join(cfg["ingestion"]["search_terms"]))
    st.write("Platforms:", cfg["ingestion"]["platforms"])
    st.write("Window (minutes):", cfg["ingestion"]["window_minutes"])
    st.write("Max posts:", cfg["ingestion"]["max_posts"])

# --- Moderation + Vectorization
df = pd.DataFrame(st.session_state.posts)
if not df.empty:
    df["clean_text"] = df["text"].map(normalize_text)
    df["ok"] = df["clean_text"].map(passes_moderation)
    df["sentiment"] = df["clean_text"].map(lambda t: analyze_sentiment(t)["sentiment"])
    kept = df[df["ok"]].copy()
else:
    kept = df

st.subheader(" Latest Posts (moderated)")
st.dataframe(kept[["platform","author","created_at","text","sentiment","url"]], use_container_width=True, height=260)

# --- Vector store
vs = VectorStore(path=".chroma")
if not kept.empty:
    ids = kept["id"].tolist()
    docs = kept["clean_text"].tolist()
    metas = kept[["platform","author","created_at","url"]].to_dict("records")
    embs = embed_texts(docs)
    vs.upsert(ids, docs, metas, embs)

# --- Trend detection
st.subheader(" Trending Terms")
trends = detect_trends(
    kept.to_dict("records"),
    ngram_min=cfg["trend"]["ngram_min"],
    ngram_max=cfg["trend"]["ngram_max"],
    min_count=cfg["trend"]["min_count"],
    z_thresh=cfg["trend"]["zscore_threshold"],
    window_minutes=cfg["ingestion"]["window_minutes"],
)
tDF = pd.DataFrame(trends)
if tDF.empty:
    st.info("No trends detected yet. Add more sample posts or connect APIs.")
else:
    st.dataframe(tDF, use_container_width=True, height=220)

# --- Ask the RAG
st.subheader(" Ask about a trend")
trend_term = st.selectbox("Pick a term", ["(type your own)"] + tDF["term"].tolist() if not tDF.empty else ["(type your own)"])
user_q = st.text_input("Your question", f"What is the cultural context of '{trend_term}'?" if trend_term != "(type your own)" else "")
topk = st.slider("Top-K documents", 3, 15, 5)

if st.button("Answer"):
    # Retrieve
    res = vs.query(user_q or trend_term, n=topk, embedding_fn=embed_texts)
    docs = res["documents"][0] if res and res.get("documents") else []
    metas = res["metadatas"][0] if res and res.get("metadatas") else []
    retrieved = [{"text":d, **m} for d, m in zip(docs, metas)]
    # Generate
    answer = generate_answer(user_q or trend_term, retrieved, os.getenv("OPENAI_API_KEY", OPENAI_API_KEY))
    st.markdown("#### Answer")
    st.write(answer)
    if retrieved:
        st.markdown("##### Sources")
        for i, r in enumerate(retrieved):
            st.write(f"[{i+1}] {r['url']} — {r['text'][:120]}…")

st.caption("Tip: Connect real APIs by adding keys to config.yaml and wiring connectors in app.py.")
