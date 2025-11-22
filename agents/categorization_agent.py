import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

def load_prompts():
    with open("prompts.json", "r") as f:
        return json.load(f)
prompts = load_prompts()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
)

categorization_prompt_template = PromptTemplate.from_template("""
{categorization_prompt}

Email:
Subject: {subject}
Body: {body}

Return ONLY the category.
""")


def run_categorization(subject: str, body: str, categorization_prompt: str):
    prompt = categorization_prompt_template.format(
        categorization_prompt=categorization_prompt,
        subject=subject,
        body=body
    )

    result = llm.invoke(prompt)
    return result.content.strip()