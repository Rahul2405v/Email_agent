import requests
from langchain.embeddings.base import Embeddings

class CustomAPIEmbedding(Embeddings):
    def embed_documents(self, texts):
        result = []
        for t in texts:
            emb = self._embed(t)
            result.append(emb)
        return result

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        response = requests.post(
            "https://embedding-model-dtv9.vercel.app/embed",
            json={"text": text},
            timeout=20
        )
        data = response.json()
        return data["embedding"]