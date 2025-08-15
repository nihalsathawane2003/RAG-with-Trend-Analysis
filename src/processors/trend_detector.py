from collections import Counter, defaultdict
from datetime import datetime, timedelta
import math
import re
from typing import List, Dict, Tuple
from ..utils.text_clean import normalize_text, ngrams

WORD_RE = re.compile(r"[\w#@']+")

def tokenize(text: str):
    return [t.lower() for t in WORD_RE.findall(text)]

def _window(posts: List[Dict], minutes: int) -> List[Dict]:
    # Keep posts within the last N minutes
    now = max(datetime.fromisoformat(p["created_at"].replace("Z","+00:00")) for p in posts)
    cutoff = now - timedelta(minutes=minutes)
    return [p for p in posts if datetime.fromisoformat(p["created_at"].replace("Z","+00:00")) >= cutoff]

def detect_trends(posts: List[Dict], ngram_min=1, ngram_max=2, min_count=3, z_thresh=2.0, window_minutes=120):
    if not posts:
        return []
    recent = _window(posts, window_minutes)
    # Count in-window
    in_counts = Counter()
    for p in recent:
        text = normalize_text(p["text"])
        toks = tokenize(text)
        for ng in ngrams(toks, ngram_min, ngram_max):
            in_counts[ng] += 1
    # Historical baseline (all-time)
    all_counts = Counter()
    for p in posts:
        text = normalize_text(p["text"])
        toks = tokenize(text)
        for ng in ngrams(toks, ngram_min, ngram_max):
            all_counts[ng] += 1

    # Naive z-score using sqrt(N) variance approx.
    trends = []
    for term, c_in in in_counts.items():
        c_all = max(all_counts[term], 1)
        expected = c_all * (len(recent)/max(len(posts),1))
        std = math.sqrt(max(expected, 1.0))
        z = (c_in - expected)/std if std > 0 else 0.0
        if c_in >= min_count and z >= z_thresh:
            trends.append({"term": term, "count": c_in, "zscore": z})

    trends.sort(key=lambda x: (x["zscore"], x["count"]), reverse=True)
    return trends[:25]
