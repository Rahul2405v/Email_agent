import json
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag.db_client import get_prompts as db_get_prompts

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
)

load_dotenv()

reply_draft_prompt_template = PromptTemplate.from_template("""
You are an AI system that drafts a professional and context-aware email reply.
Below is the original email received:

Subject: {subject}

Body:
{body}

Draft a reply email following these instructions:
{prompt}

OUTPUT FORMAT (VERY IMPORTANT):
Return only valid JSON in the following structure:

{{
  "subject": "<reply email subject>",
  "body": "<reply email body>"
}}

Do not include explanations, comments, or text outside the JSON.
""")


def generate_reply_draft(subject: str, body: str, prompt: str):
    # refresh prompts from DB
    prompts = db_get_prompts()
    if prompt == "reply_prompt":
        r_p = prompts.get("auto_reply")
    else:
        r_p = prompt

    formatted_prompt = reply_draft_prompt_template.format(
        subject=subject,
        body=body,
        prompt=r_p
    )

    res = llm.invoke(formatted_prompt)
    return res.content.strip()
