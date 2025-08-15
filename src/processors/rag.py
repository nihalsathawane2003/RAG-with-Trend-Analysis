from typing import List, Dict, Optional
import os
import textwrap
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

SYSTEM = "You are a culturally aware analyst. Explain memes, trends, and movements succinctly with citations to provided snippets. Avoid speculation beyond context."

def generate_answer(query: str, retrieved: List[Dict], openai_api_key: Optional[str] = None) -> str:
    # If OpenAI available, use it; else produce extractive answer.
    context = "\n\n".join([f"[{i+1}] {r['text']} (source: {r.get('url','')})" for i, r in enumerate(retrieved)])
    if openai_api_key and OpenAI is not None:
        client = OpenAI(api_key=openai_api_key)
        msg = [
            {"role":"system","content":SYSTEM},
            {"role":"user","content": f"Question: {query}\n\nContext:\n{context}\n\nAnswer with references like [1], [2]."}
        ]
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0.2, max_tokens=350)
        return resp.choices[0].message.content.strip()
    # Fallback: extractive
    if not retrieved:
        return "No context available."
    snippet = "\n\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(retrieved)])
    return textwrap.shorten(f"Based on the posts: {snippet}", width=800, placeholder="…")
