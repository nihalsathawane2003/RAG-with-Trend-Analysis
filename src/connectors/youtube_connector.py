from typing import List, Dict
import requests

def fetch_youtube(api_key: str, query: str, max_results: int = 25) -> List[Dict]:
    if not api_key:
        return []
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {"part":"snippet", "q": query, "type":"video", "maxResults": min(max_results,50), "key": api_key, "order":"date"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    out = []
    for item in data.get("items", []):
        snip = item["snippet"]
        out.append({
            "platform":"youtube",
            "id": item["id"]["videoId"],
            "created_at": snip["publishedAt"],
            "author": snip.get("channelTitle",""),
            "text": snip.get("title",""),
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        })
    return out
