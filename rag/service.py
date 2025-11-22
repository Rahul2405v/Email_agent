from .embedding import embed_query
from .vector_search import vector_search
from .groq_llm import answer_question

def rag_answer(prompt, k):
    q = embed_query(prompt + " what action / task / issue?")
    docs = vector_search(q, k)
    reply = answer_question(prompt, docs)
    return reply, docs
