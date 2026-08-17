from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    file: str
    location: str
    excerpt: str


class TopicCandidate(BaseModel):
    name: str
    description: str
    importance: int = Field(ge=1, le=5)
    source_ids: list[str]


class TopicMap(BaseModel):
    topics: list[TopicCandidate]


class GeneratedQuestion(BaseModel):
    question: str
    expected_answer: str
    answerable: bool
    difficulty: str
    rationale: str
    source_ids: list[str]


class QuestionBatch(BaseModel):
    questions: list[GeneratedQuestion]


class SilverQuestion(BaseModel):
    id: str
    topic: str
    question: str
    expected_answer: str
    answerable: bool = True
    difficulty: str = "medium"
    rationale: str = ""
    sources: list[SourceRef] = Field(default_factory=list)
    review_status: str = "pending"
    reviewer_notes: str = ""


class ChatbotResult(BaseModel):
    question_id: str
    answer: str
    retrieved_context: str = ""
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class JudgeScores(BaseModel):
    correctness: int = Field(ge=1, le=4)
    completeness: int = Field(ge=1, le=4)
    relevance: int = Field(ge=1, le=4)
    groundedness: int = Field(ge=1, le=4)
    response_is_abstention: bool
    explanation: str
    missing_or_wrong: str = ""


class Outcome(str, Enum):
    CORRECT_ANSWER = "correct_answer"
    PARTIAL_ANSWER = "partial_answer"
    INCORRECT_ANSWER = "incorrect_answer"
    INCORRECT_ABSTENTION = "incorrect_abstention"
    CORRECT_ABSTENTION = "correct_abstention"
    SHOULD_HAVE_ABSTAINED = "should_have_abstained"
    CHATBOT_ERROR = "chatbot_error"
    JUDGE_ERROR = "judge_error"


class EvaluationRecord(BaseModel):
    question: SilverQuestion
    result: ChatbotResult
    outcome: Outcome
    scores: JudgeScores | None = None
    judge_error: str = ""

