from typing import List
from fastembed import TextEmbedding
from .config import EMBED_MODEL_NAME, CACHE_DIR

embedding_model = None

def get_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = TextEmbedding(model_name=EMBED_MODEL_NAME, cache_dir=CACHE_DIR, threads=1)
    return embedding_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    return [v.tolist() for v in model.embed(texts)]

def embed_query(text: str) -> List[float]:
    model = get_model()
    return list(model.embed([text]))[0].tolist()
