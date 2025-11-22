from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from agents.agent_helper import ask_agent
from agents.parllel_runner import process_all_emails
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.agents import initialize_agent
from agents.reply_draft import generate_reply_draft
from models.GenerateReplyRequest import GenerateReplyRequest
from rag.rag_routes import router as rag_router

import json
import os
import uvicorn
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()
app.include_router(rag_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


PROMPT_FILE = "prompts.json"
EMAIL_FILE = "mock_emails.json"
class Query(BaseModel):
    prompt: str

def load_prompts():
    if not os.path.exists(PROMPT_FILE):
        return {}
    with open(PROMPT_FILE, "r") as f:
        return json.load(f)


def save_prompts(data):
    with open(PROMPT_FILE, "w") as f:
        json.dump(data, f, indent=4)


def load_all_emails():
    if not os.path.exists(EMAIL_FILE):
        return []

    with open(EMAIL_FILE, "r") as f:
        return json.load(f)

@app.get("/prompts")
def get_prompts():
    return load_prompts()


@app.post("/prompts")
def update_prompts(data: dict):
    save_prompts(data)
    process_all()
    return {"status": "success"}

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

    body = body + "\n\nTimestamp: " + timestamp

    allDetails = {
        "subject": subject,
        "timestamp": timestamp,
        "body_text": body,
        "prompt": prmopt,
        "id" : email_id
    }
    allDetails['prompt'] = f"User instruction: {prmopt}. Strictly follow this instruction."
    result = ask_agent(**allDetails)
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
    body = (
        body
        + f"\n\nTimestamp: {timestamp}"
        + f"\nSender: {sender}"
    )
    if prompt == "":
        reply_prompt = "reply_prompt"
    else :
        reply_prompt = ({prompt})
    result = generate_reply_draft(
        subject=subject,
        body=body,
        prompt=reply_prompt
    )
    try:
        return json.loads(result)
    except:
        print("Model returned invalid JSON")
        return ""


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
