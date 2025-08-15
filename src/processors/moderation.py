from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> Dict:
    scores = analyzer.polarity_scores(text)
    label = "positive" if scores["compound"] >= 0.05 else "negative" if scores["compound"] <= -0.05 else "neutral"
    return {"sentiment": label, **scores}

def passes_moderation(text: str) -> bool:
    # Simple placeholder: extend with toxicity/hate models if needed.
    # For demo, reject if contains extreme slurs or threats (basic heuristic).
    lowered = text.lower()
    banned = ["kill yourself", "hate crime"]  # extend list or use a classifier
    return not any(b in lowered for b in banned)
