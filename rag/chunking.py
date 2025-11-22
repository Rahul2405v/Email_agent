import json, os, uuid
from rag.db_client import get_emails as db_get_emails
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_emails(path="mock_emails.json"):
    # Prefer DB-backed emails; fall back to file if DB empty
    try:
        emails = db_get_emails()
        if emails:
            return emails
    except Exception:
        pass

    if not os.path.exists(path):
        return []
    return json.load(open(path, "r", encoding="utf-8"))

def flatten_email(email):
    return (
        f"Email ID: {email.get('id')}\n"
        f"Subject: {email.get('subject')}\n"
        f"From: {email.get('sender_email')}\n"
        f"Timestamp: {email.get('timestamp')}\n\n"
        f"{email.get('body_text')}"
    )

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return [{"id": str(uuid.uuid4()), "chunk": c} for c in splitter.split_text(text)]