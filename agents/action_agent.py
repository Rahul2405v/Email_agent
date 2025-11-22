import json
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
)

load_dotenv()

def load_prompts():
    with open("prompts.json", "r") as f:
        return json.load(f)
prompts = load_prompts()


action_item_prompt_template = PromptTemplate.from_template("""
{action_item_prompt}
 in case of deadline mention as mm/dd/yyyy and time as HH:MM AM/PM
Email:
Subject: {subject}
Body: {body}
""")

def run_action_extraction(subject: str, body: str, action_prompt: str):
    prompt = action_item_prompt_template.format(
        action_item_prompt=action_prompt,
        subject=subject,
        body=body
    )

    result = llm.invoke(prompt)
    text = result.content.strip()

    try:
        return json.loads(text)
    except:
        return []
