from typing import List
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Allow overriding via env; default to the provided Space URL
HF_SPACE_API_URL = os.getenv("HF_SPACE_API_URL", "https://rahul258789-embedding-service.hf.space/embed")
HF_SPACE_TIMEOUT = int(os.getenv("HF_SPACE_TIMEOUT", "30"))


def _call_space(text: str) -> List[float]:
    payload = {"text": text}
    try:
        resp = requests.post(HF_SPACE_API_URL, json=payload, timeout=HF_SPACE_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # Support both {"embedding": [...] } and {"data": {...}} shapes
        if isinstance(data, dict) and "embedding" in data:
            return data["embedding"]
        # Some spaces return {"data": {"embedding": [...]}}
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict) and "embedding" in data["data"]:
            return data["data"]["embedding"]
        # Some endpoints may return a list directly
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, float)):
            return data[0]
        raise ValueError(f"Unexpected embedding response shape: {data}")
    except Exception as e:
        raise RuntimeError(f"Failed to get embedding from HF Space: {e}")


def embed_texts(texts: List[str]) -> List[List[float]]:
    embeddings = []
    for t in texts:
        emb = _call_space(t)
        embeddings.append(list(emb))
    return embeddings


def embed_query(text: str) -> List[float]:
    emb = _call_space(text)
    return list(emb)
