from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SourceRef(BaseModel):
    source_id: str = ""
    file: str
    location: str
    excerpt: str


class EvidenceQuote(BaseModel):
    source_id: str
    quote: str = Field(min_length=3)


class QuestionType(str, Enum):
    BASIC_KNOWLEDGE = "basic_knowledge"
    TOPIC_INTEGRATION = "topic_integration"
    DOCUMENT_WIDE = "document_wide"
    CROSS_DOCUMENT = "cross_document"
    PERSONAL_BASIC = "personal_basic"
    PERSONAL_INTEGRATION = "personal_integration"
    UNANSWERABLE = "unanswerable"


class QuestionForm(str, Enum):
    CANONICAL = "canonical"
    NATURAL_USER = "natural_user"
    AMBIGUOUS = "ambiguous"


class ExpectedBehavior(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


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
    question_type: QuestionType = QuestionType.BASIC_KNOWLEDGE
    reference_claims: list[str]
    supporting_quotes: list[EvidenceQuote] = Field(default_factory=list)


class QuestionBatch(BaseModel):
    questions: list[GeneratedQuestion]


class GeneratedVariation(BaseModel):
    source_question_id: str
    question: str
    question_form: Literal["natural_user", "ambiguous"]
    required_clarification: str = ""
    rationale: str


class VariationBatch(BaseModel):
    variations: list[GeneratedVariation]


class SilverQuestion(BaseModel):
    id: str
    topic: str
    question: str
    expected_answer: str
    answerable: bool = True
    difficulty: str = "medium"
    rationale: str = ""
    question_type: QuestionType = QuestionType.BASIC_KNOWLEDGE
    question_form: QuestionForm = QuestionForm.CANONICAL
    expected_behavior: ExpectedBehavior = ExpectedBehavior.ANSWER
    parent_question_id: str = ""
    reference_claims: list[str] = Field(default_factory=list)
    supporting_quotes: list[EvidenceQuote] = Field(default_factory=list)
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


class ClaimAssessment(BaseModel):
    claim_id: str
    addressed: bool
    correct: bool
    explanation: str = ""

    @model_validator(mode="after")
    def correct_claim_must_be_addressed(self) -> "ClaimAssessment":
        if self.correct and not self.addressed:
            raise ValueError("a correct claim must also be addressed")
        return self


class JudgeScores(BaseModel):
    claim_assessments: list[ClaimAssessment]
    required_points_total: int = Field(ge=0)
    answer_points_addressed: int = Field(ge=0)
    answer_points_correct: int = Field(ge=0)
    answer_false_claims: int = Field(ge=0)
    answer_unsupported_claims: int = Field(ge=0)
    answer_extraneous_claims: int = Field(ge=0)
    retrieval_points_found: int = Field(ge=0)
    retrieved_chunks_total: int = Field(ge=0)
    retrieved_chunks_relevant: int = Field(ge=0)
    retrieved_chunks_contradictory: int = Field(ge=0)
    answer_scope: Literal["exact", "too_little", "too_much"]
    incorrect_type: Literal["not_applicable", "unrelated", "hallucination"]
    response_is_abstention: bool
    response_is_clarification: bool = False
    explanation: str
    missing_or_wrong: str
    retrieval_explanation: str

    @model_validator(mode="after")
    def validate_counts(self) -> "JudgeScores":
        if self.answer_points_correct > self.answer_points_addressed:
            raise ValueError("answer_points_correct cannot exceed answer_points_addressed")
        if self.answer_points_addressed > self.required_points_total:
            raise ValueError("answer_points_addressed cannot exceed required_points_total")
        if self.retrieval_points_found > self.required_points_total:
            raise ValueError("retrieval_points_found cannot exceed required_points_total")
        if self.retrieved_chunks_relevant > self.retrieved_chunks_total:
            raise ValueError("retrieved_chunks_relevant cannot exceed retrieved_chunks_total")
        if self.retrieved_chunks_contradictory > self.retrieved_chunks_total:
            raise ValueError("retrieved_chunks_contradictory cannot exceed retrieved_chunks_total")
        return self


class Outcome(str, Enum):
    CORRECT_ANSWER = "correct_answer"
    PARTIAL_TOO_LITTLE = "partial_too_little"
    PARTIAL_TOO_MUCH = "partial_too_much"
    UNRELATED_ANSWER = "unrelated_answer"
    MISLEADING_HALLUCINATION = "misleading_hallucination"
    INCORRECT_ABSTENTION = "incorrect_abstention"
    CORRECT_ABSTENTION = "correct_abstention"
    SHOULD_HAVE_ABSTAINED = "should_have_abstained"
    CORRECT_CLARIFICATION = "correct_clarification"
    MISSING_CLARIFICATION = "missing_clarification"
    CHATBOT_ERROR = "chatbot_error"
    JUDGE_ERROR = "judge_error"


class TopicAssignment(BaseModel):
    question_id: str
    topic: str


class TopicAssignments(BaseModel):
    assignments: list[TopicAssignment]


class InsightIssue(BaseModel):
    title: str
    priority: Literal["high", "medium", "low"]
    confidence: Literal["high", "medium", "low"]
    evidence_count: int = Field(ge=1)
    affected_topics: list[str] = Field(max_length=12)
    observed_pattern: str
    likely_cause_hypothesis: str
    recommendation: str
    example_question_ids: list[str] = Field(min_length=1, max_length=8)


class EvaluationInsights(BaseModel):
    executive_summary: str
    strengths: list[str] = Field(max_length=4)
    issues: list[InsightIssue] = Field(max_length=6)
    methodology_note: str


class PromptPackage(BaseModel):
    system_prompt_hebrew: str = Field(min_length=40)
    corpus_scope_summary: list[str] = Field(min_length=1)
    assumptions_requiring_review: list[str] = Field(min_length=1)
    application_guardrails: list[str] = Field(min_length=1)
    manager_review_checklist: list[str] = Field(min_length=1)
    suggested_test_questions: list[str] = Field(min_length=1)
    revision_summary: list[str] = Field(default_factory=list)
    revision_evidence_question_ids: list[str] = Field(default_factory=list)


class EvaluationRecord(BaseModel):
    question: SilverQuestion
    result: ChatbotResult
    outcome: Outcome
    scores: JudgeScores | None = None
    judge_error: str = ""
