from typing import List, Dict
import os
import requests

def fetch_tweets(bearer_token: str, query: str, max_results: int = 50) -> List[Dict]:
    if not bearer_token:
        return []
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {"query": query, "max_results": min(max_results,100), "tweet.fields":"created_at,author_id"}
    headers = {"Authorization": f"Bearer {bearer_token}"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    out = []
    for t in data.get("data", []):
        out.append({
            "platform":"twitter",
            "id": t["id"],
            "created_at": t["created_at"],
            "author": t["author_id"],
            "text": t.get("text",""),
            "url": f"https://x.com/i/web/status/{t['id']}"
        })
    return out
