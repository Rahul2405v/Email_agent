import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
)

MESSAGE_HISTORY = []
LAST_ID = None

def ask_agent(subject: str, body_text: str, prompt: str, timestamp: str, id: str):
    global MESSAGE_HISTORY, LAST_ID

    if LAST_ID != id:
        MESSAGE_HISTORY = [
            SystemMessage(
                content="You are an AI assistant that processes emails strictly based on the user instructions. "
                        "Never assume anything outside the email."
            )
        ]
        LAST_ID = id

    MESSAGE_HISTORY.append(
        HumanMessage(
            content=f"""
INSTRUCTIONS: {prompt}

EMAIL DETAILS:
Subject: {subject}
Body: {body_text}
Timestamp: {timestamp}

Follow instructions EXACTLY.
 Do NOT invent or guess missing information.if it is out of context say user to stict to email content only.
"""
        )
    )

    response = llm.invoke(MESSAGE_HISTORY)

    MESSAGE_HISTORY.append(
        AIMessage(content=response.content)
    )

    return response.content.strip()
