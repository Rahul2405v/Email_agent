import json
from langchain_core.runnables import RunnableParallel, RunnableSequence
from agents.categorization_agent import categorization_prompt_template
from agents.action_agent import action_item_prompt_template
from langchain_groq import ChatGroq
import time

import os
from dotenv import load_dotenv
from rag.db_client import get_prompts as db_get_prompts, get_emails as db_get_emails, update_email as db_update_email
load_dotenv()


llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
)


parallel_runner = RunnableParallel(
    category = categorization_prompt_template | llm,
    actions = action_item_prompt_template | llm
)
def run_parallel_processing(subject: str, body: str, prompts: dict):
    input_data = {
        "subject": subject,
        "body": body,
        "categorization_prompt": prompts["categorization"],
        "action_item_prompt": prompts["action_item"]
    }

    full_sequence = RunnableSequence(
        lambda x: input_data,
        parallel_runner
    )

    result = full_sequence.invoke(input_data)

    category = result["category"].content.strip()
    raw_actions = result["actions"].content.strip()

    try:
        action_items = json.loads(raw_actions)
    except:
        action_items = []

    return category, action_items

def process_all_emails():
    prompts = db_get_prompts()
    try:
        emails = db_get_emails()
    except Exception:
        return {"success": False, "message": "Cannot read emails from DB"}

    updated_emails = []
    for email in emails:
        time.sleep(2)
        email_id = email.get("id")
        subject = email.get("subject", "")
        body = email.get("body_text", "")
        time_stamp = email.get("timestamp", "")
        body = body + "\n\nTimestamp: " + time_stamp
        category, action_items = run_parallel_processing(subject, body, prompts)
        email["category"] = category
        email["actions"] = action_items

        # persist per-email updates to DB
        if email_id:
            try:
                db_update_email(email_id, {"category": category, "actions": action_items})
            except Exception:
                pass

        updated_emails.append(email)

    return {
        "success": True,
        "total_processed": len(updated_emails),
        "data": updated_emails
    }

