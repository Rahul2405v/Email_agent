from fastapi import APIRouter
from pydantic import BaseModel
from rag.indexer import build_index
from rag.service import rag_answer
from rag.embedding import embed_query, embed_texts
from rag.extract_idx import find as find_ids
router = APIRouter(prefix="/rag", tags=["RAG"])

class AskBody(BaseModel):
    prompt: str
    k: int = 4

@router.get("/init")
def init():
    return {"indexed": build_index()}

@router.post("/embed")
def embed(body: AskBody):
    return {"vector": embed_query(body.prompt)}

@router.post("/ask")
def ask(body: AskBody):
    reply, docs = rag_answer(body.prompt, body.k)
    extract_ids = find_ids(reply)
    return {"answer": reply, "chunks": docs, "extracted_ids": extract_ids}