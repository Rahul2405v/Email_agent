from pydantic import BaseModel

class GenerateReplyRequest(BaseModel):
    id: str
    prompt: str
