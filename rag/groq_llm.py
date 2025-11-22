from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

def answer_question(prompt: str, docs):
    """
    Generates an answer using only the provided docs.
    If answer is not found, the model must respond "I don't know".
    Also at the end, include list of email IDs used in: [msg_id1, msg_id2]
    """
    context = "\n\n".join([d["chunk"] for d in docs])

    final_prompt = f"""
You are an AI that answers strictly from the given context.

CONTEXT:
{context}

QUESTION:
{prompt}

RESPONSE RULES:
- If the answer is not present in the context, respond only with: "I don't know"
- Do not hallucinate or assume anything not in the context
- At the end of your response, add a new final line containing ONLY the list of email IDs used in the response in format: [msg_id1, msg_id2]

ANSWER:
"""

    result = llm.invoke(final_prompt)
    return result.content
