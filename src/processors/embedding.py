from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        # Force CPU device to avoid CUDA issues on Streamlit Cloud
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _model

def embed_texts(texts):
    model = get_model()
    return model.encode(texts, show_progress_bar=False)
