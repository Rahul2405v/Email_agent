from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv
import uvicorn

# AGENT system
from agents.agent_helper import ask_agent
from agents.parllel_runner import process_all_emails
from agents.reply_draft import generate_reply_draft
from models.GenerateReplyRequest import GenerateReplyRequest

# RAG system
from rag.indexer import build_index
from rag.service import rag_answer
from rag.embedding import embed_query
from rag.extract_idx import find as find_ids

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Files
PROMPT_FILE = "prompts.json"
EMAIL_FILE = "mock_emails.json"


# UTILS  -----------------------------------------------------

def load_prompts():
    return json.load(open(PROMPT_FILE, "r")) if os.path.exists(PROMPT_FILE) else {}

def save_prompts(data):
    json.dump(data, open(PROMPT_FILE, "w"), indent=4)

def load_all_emails():
    return json.load(open(EMAIL_FILE, "r")) if os.path.exists(EMAIL_FILE) else []


# MODELS -----------------------------------------------------

class Query(BaseModel):
    prompt: str

class AskBody(BaseModel):
    prompt: str
    k: int = 4


# PROMPT ROUTES ----------------------------------------------

@app.get("/prompts")
def get_prompts():
    return load_prompts()

@app.post("/prompts")
def update_prompts(data: dict):
    save_prompts(data)
    process_all_emails()
    return {"status": "success"}


# EMAIL ROUTES -----------------------------------------------

@app.get("/emails")
def get_all_emails():
    return load_all_emails()

@app.post("/process-all-emails")
def process_all():
    return process_all_emails()

@app.post("/process-email")
def process_email(payload: dict):
    email_id = payload.get("id")
    prmopt = payload.get("prmopt")

    if not email_id or not prmopt:
        raise HTTPException(status_code=400, detail="id and prmopt are required")

    emails = load_all_emails()
    matched_email = next((e for e in emails if e.get("id") == email_id), None)
    if not matched_email:
        raise HTTPException(status_code=404, detail="Email not found")

    subject = matched_email.get("subject", "")
    body = matched_email.get("body_text", "")
    timestamp = matched_email.get("timestamp", "")

    body = body + f"\n\nTimestamp: {timestamp}"

    final_prompt = f"User instruction: {prmopt}. Strictly follow this instruction."
    result = ask_agent(subject, body, final_prompt, timestamp, email_id)
    return result


@app.post("/generate-reply")
def generate_reply(request: GenerateReplyRequest):
    id = request.id
    prompt = request.prompt
    emails = load_all_emails()

    matched_email = next((e for e in emails if e.get("id") == id), None)
    if not matched_email:
        raise HTTPException(status_code=404, detail="Email not found")

    subject = matched_email.get("subject", "")
    body = matched_email.get("body_text", "")
    timestamp = matched_email.get("timestamp", "")
    sender = matched_email.get("sender_name", "Unknown Sender")

    body = body + f"\n\nTimestamp: {timestamp}\nSender: {sender}"

    reply_prompt = prompt if prompt else "reply_prompt"
    result = generate_reply_draft(subject=subject, body=body, prompt=reply_prompt)

    try:
        return json.loads(result)
    except:
        return {"error": "Model returned invalid JSON"}


# RAG ROUTES --------------------------------------------------

@app.get("/init")
def rag_init():
    return {"indexed": build_index()}

@app.post("/embed")
def rag_embed(body: AskBody):
    return {"vector": embed_query(body.prompt)}

@app.post("/ask")
def rag_ask(body: AskBody):
    reply, docs = rag_answer(body.prompt, body.k)
    extract_ids = find_ids(reply)
    return {"answer": reply, "chunks": docs, "extracted_ids": extract_ids}


# START -------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
