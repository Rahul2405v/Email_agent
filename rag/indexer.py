from .embedding import embed_texts
from .chunking import flatten_email, chunk_text, load_emails
from .db import collection

def upsert(email_id, chunks, vectors):
    for chunk, vec in zip(chunks, vectors):
        collection.replace_one(
            {"_id": chunk["id"]},
            {"_id": chunk["id"], "email_id": email_id, "chunk": chunk["chunk"], "embedding": vec},
            upsert=True,
        )

def build_index(path="mock_emails.json"):
    emails = load_emails(path)
    for email in emails:
        text = flatten_email(email)
        chunks = chunk_text(text)
        vectors = embed_texts([c["chunk"] for c in chunks])
        upsert(email["id"], chunks, vectors)
    return len(emails)
