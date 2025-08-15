from typing import List, Dict
import time

def measure_latency(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0

def retrieval_precision_at_k(retrieved: List[str], relevant: List[str], k:int=5) -> float:
    topk = set(retrieved[:k])
    rel = set(relevant)
    if not topk:
        return 0.0
    return len(topk & rel) / min(k, len(retrieved))

def dummy_ragas_answer_relevance(answer: str) -> float:
    # Placeholder: higher if answer is longer than 50 chars and contains at least one reference [n]
    score = 0.0
    if len(answer) > 50:
        score += 0.5
    if "[" in answer and "]" in answer:
        score += 0.5
    return score
