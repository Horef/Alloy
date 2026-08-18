from __future__ import annotations

from enum import Enum
from typing import Any, Literal

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
    answer_scope: Literal["exact", "too_little", "too_much"]
    incorrect_type: Literal["not_applicable", "unrelated", "hallucination"]
    retrieval_relevance: int = Field(ge=0, le=4)
    retrieval_correctness: int = Field(ge=0, le=4)
    retrieval_completeness: int = Field(ge=0, le=4)
    response_is_abstention: bool
    explanation: str
    missing_or_wrong: str
    retrieval_explanation: str


class Outcome(str, Enum):
    CORRECT_ANSWER = "correct_answer"
    PARTIAL_TOO_LITTLE = "partial_too_little"
    PARTIAL_TOO_MUCH = "partial_too_much"
    UNRELATED_ANSWER = "unrelated_answer"
    MISLEADING_HALLUCINATION = "misleading_hallucination"
    INCORRECT_ABSTENTION = "incorrect_abstention"
    CORRECT_ABSTENTION = "correct_abstention"
    SHOULD_HAVE_ABSTAINED = "should_have_abstained"
    CHATBOT_ERROR = "chatbot_error"
    JUDGE_ERROR = "judge_error"


class TopicAssignment(BaseModel):
    question_id: str
    topic: str


class TopicAssignments(BaseModel):
    assignments: list[TopicAssignment]


class EvaluationRecord(BaseModel):
    question: SilverQuestion
    result: ChatbotResult
    outcome: Outcome
    scores: JudgeScores | None = None
    judge_error: str = ""
