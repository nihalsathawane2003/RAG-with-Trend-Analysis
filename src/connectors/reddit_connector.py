from typing import List, Dict
import os
import praw

def fetch_reddit(client_id: str, client_secret: str, user_agent: str, query: str, limit: int = 50) -> List[Dict]:
    if not client_id or not client_secret:
        return []
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent or "social-rag-demo")
    out = []
    for sub in reddit.subreddit("all").search(query, limit=limit, sort="new"):
        out.append({
            "platform":"reddit",
            "id": sub.id,
            "created_at": (sub.created_utc),
            "author": str(sub.author) if sub.author else "unknown",
            "text": sub.title + (" " + sub.selftext if getattr(sub, "selftext", None) else ""),
            "url": f"https://reddit.com{sub.permalink}"
        })
    # Convert epoch to ISO
    from datetime import datetime, timezone
    for p in out:
        if isinstance(p["created_at"], (int, float)):
            p["created_at"] = datetime.fromtimestamp(p["created_at"], tz=timezone.utc).isoformat().replace("+00:00","Z")
    return out
