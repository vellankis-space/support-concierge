from __future__ import annotations

from datetime import datetime
from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Category(StrEnum):
    BILLING = "billing"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    ACCOUNT_ACCESS = "account_access"
    SECURITY = "security"
    PRIVACY_LEGAL = "privacy_legal"
    SPAM = "spam"
    FEEDBACK = "feedback"
    OTHER = "other"


class Action(StrEnum):
    AUTO_RESOLVE = "auto_resolve"
    DRAFT_FOR_REVIEW = "draft_for_review"
    ESCALATE = "escalate"


class RiskFlag(StrEnum):
    MONEY = "money"
    ACCOUNT_DELETION = "account_deletion"
    LEGAL = "legal"
    SECURITY = "security"
    PROMPT_INJECTION = "prompt_injection"
    MULTI_INTENT = "multi_intent"
    MALFORMED_INPUT = "malformed_input"
    DUPLICATE_CONTEXT = "duplicate_context"
    LOW_CONFIDENCE = "low_confidence"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QueueKind(StrEnum):
    DRAFT = "draft"
    ESCALATION = "escalation"


class TicketStatus(StrEnum):
    PROCESSING = "processing"
    CLOSED = "closed"
    DRAFT_PENDING = "draft_pending"
    ESCALATED = "escalated"


class ReviewAction(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"
    RESOLVE = "resolve"
    REASSIGN = "reassign"


class DomainModel(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(use_enum_values=False)


class Ticket(DomainModel):
    id: str
    received_at: datetime
    from_name: str
    from_email: str
    subject: str
    body: str


class NormalizedTicket(DomainModel):
    ticket_id: str
    subject: str
    body: str
    combined_text: str
    warnings: list[str] = Field(default_factory=list)
    duplicate_context: bool = False


class Entity(DomainModel):
    value: str
    kind: str


class TriageResult(DomainModel):
    category: Category
    secondary_category: Category | None = None
    product_area: str | None = None
    urgency: Urgency
    sentiment: Sentiment
    customer_identifiers: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    intent_count: int = Field(default=1, ge=1)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    prompt_injection_indicators: list[str] = Field(default_factory=list)
    input_quality: str
    requested_action: bool = False
    model_confidence: float = Field(ge=0, le=1)
    summary: str
    rationale: str


class DecisionResult(DomainModel):
    proposed_action: Action
    model_confidence: float = Field(ge=0, le=1)
    rationale: str
    evidence: list[str] = Field(default_factory=list)


class DraftResult(DomainModel):
    response_text: str
    template_id: str
    claims: list[str] = Field(default_factory=list)


class ReviewResult(DomainModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)
    rationale: str


class PolicyEvaluation(DomainModel):
    raw_risk_flags: list[RiskFlag] = Field(default_factory=list)
    final_risk_flags: list[RiskFlag] = Field(default_factory=list)
    penalties: dict[str, float] = Field(default_factory=dict)
    base_confidence: float = Field(ge=0, le=1)
    effective_confidence: float = Field(ge=0, le=1)
    final_action: Action
    override_reason: str | None = None
    rationale: str


class PipelineResult(DomainModel):
    ticket: Ticket
    normalized: NormalizedTicket
    triage: TriageResult | None = None
    decision: DecisionResult | None = None
    policy: PolicyEvaluation | None = None
    draft: DraftResult | None = None
    review: ReviewResult | None = None
    final_action: Action
    status: TicketStatus
    send_status: str | None = None
    error: str | None = None
    run_id: str | None = None


def model_dump_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    return value
