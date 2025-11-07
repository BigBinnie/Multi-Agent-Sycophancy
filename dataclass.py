from pydantic import BaseModel
from typing import Optional, Dict

class Question(BaseModel):
    content: str


class Answer(BaseModel):
    content: str


class SolverRequest(BaseModel):
    content: str
    question: str


class IntermediateSolverResponse(BaseModel):
    content: str
    question: str
    answer: str
    round: int


class FinalSolverResponse(BaseModel):
    answer: str
    answers_by_round: Optional[Dict[int, str]] = None
    confidence_by_round: Optional[Dict[int, float]] = None
    sender_id: Optional[str] = None
    system_prompt: Optional[list] = None
    dialogue_history: Optional[list] = None


class ErrorResponse(BaseModel):
    """Message type for error responses"""
    content: str
    error_code: Optional[str] = None
    details: Optional[Dict] = None
