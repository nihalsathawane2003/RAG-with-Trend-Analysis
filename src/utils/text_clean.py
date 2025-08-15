import re
from typing import List

URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")

def normalize_text(t: str) -> str:
    t = URL_RE.sub("", t)
    t = t.replace("\n", " ").strip()
    t = WS_RE.sub(" ", t)
    return t

def ngrams(tokens: List[str], n_min: int, n_max: int):
    for n in range(n_min, n_max+1):
        for i in range(len(tokens)-n+1):
            yield " ".join(tokens[i:i+n])
