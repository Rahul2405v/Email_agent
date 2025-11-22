from pydantic import BaseModel

class EmailPayload(BaseModel):
    id: str
    subject: str
    body: str