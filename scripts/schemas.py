from pydantic import BaseModel

class Essay(BaseModel):
    question: str
    answer: str
    reference: str