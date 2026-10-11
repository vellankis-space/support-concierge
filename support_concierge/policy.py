from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from .models import (
    Action,
    Category,
    DecisionResult,
    DraftResult,
    NormalizedTicket,
    PolicyEvaluation,
    RiskFlag,
    Ticket,
    TriageResult,
)

_MONEY_RE = re.compile(
    r"\b(refund|invoice|billing|bill|charged|charge|overcharg|pricing|price|payment|cancel(?:ling|lation)?)\b",
    re.IGNORECASE,
)
_DELETION_RE = re.compile(
    r"\b(delete|deletion|remove|close|permanently)\b.{0,40}"
    r"\b(account|data|personal information)\b|\bGDPR\b|\bArticle 17\b",
    re.IGNORECASE,
)
_LEGAL_RE = re.compile(
    r"\b(legal|lawyer|attorney|lawsuit|regulator|complaint|article 17|GDPR)\b",
    re.IGNORECASE,
)
_SECURITY_RE = re.compile(
    r"\b(security|vulnerab|unauthori[sz]ed|expos(?:e|ed)|"
    r"another company|account_id|access.*dashboard)\b",
    re.IGNORECASE,
)
_INJECTION_RE = re.compile(
    r"ignore (?:all )?(?:previous|prior) instructions|system note|"
    r"do not escalate|immediately approve|mark this ticket resolved",
    re.IGNORECASE,
)
_MALFORMED_RE = re.compile(r"=EF=BF=BD|=20|=3D|�")
_REQUEST_RE = re.compile(r"\?|\b(please|want|need|can you|could you|would love)\b", re.I)

APPROVED_TEMPLATE_IDS = {
    "billing_review",
    "bug_review",
    "feature_review",
    "account_access_review",
    "general_review",
}

APPROVED_TEMPLATE_GUIDANCE = {
    "billing_review": (
        "Acknowledge the billing concern, summarize the request, and state that a human "
        "will review it. Do not promise a refund or billing change."
    ),
    "bug_review": (
        "Acknowledge the reported behavior, summarize the affected product area, and "
        "state that support will investigate. Do not claim a fix."
    ),
    "feature_review": (
        "Acknowledge the feature request and summarize it without promising delivery "
        "or a timeline."
    ),
    "account_access_review": (
        "Acknowledge the access problem and request only safe troubleshooting details. "
        "Do not claim an account change."
    ),
    "general_review": (
        "Acknowledge the request, summarize the known facts, and state that support "
        "will review it without inventing commitments."
    ),
}


def normalize_ticket(ticket: Ticket) -> NormalizedTicket:
    warnings: list[str] = []
    subject = " ".join(ticket.subject.split())
    body = ticket.body.replace("\r\n", "\n").strip()
    if not subject:
        warnings.append("missing_subject")
    if not body:
        warnings.append("missing_body")
    if _MALFORMED_RE.search(f"{subject}\n{body}"):
        warnings.append("malformed_encoding_marker")
    return NormalizedTicket(
        ticket_id=ticket.id,
        subject=subject,
        body=body,
        combined_text=f"Subject: {subject}\n\nBody: {body}",
        warnings=warnings,
    )


def duplicate_context(ticket: Ticket, previous: Sequence[Ticket]) -> bool:
    subject_tokens = set(re.findall(r"[a-z0-9]+", ticket.subject.lower()))
    if not subject_tokens:
        return False
    for candidate in previous:
        if candidate.from_email.lower() != ticket.from_email.lower():
            continue
        candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate.subject.lower()))
        overlap = len(subject_tokens & candidate_tokens) / max(len(subject_tokens), 1)
        if overlap >= 0.5:
            return True
    return False


def detect_raw_risks(normalized: NormalizedTicket) -> set[RiskFlag]:
    text = normalized.combined_text
    risks: set[RiskFlag] = set()
    if _MONEY_RE.search(text):
        risks.add(RiskFlag.MONEY)
    if _DELETION_RE.search(text):
        risks.add(RiskFlag.ACCOUNT_DELETION)
    if _LEGAL_RE.search(text):
        risks.add(RiskFlag.LEGAL)
    if _SECURITY_RE.search(text):
        risks.add(RiskFlag.SECURITY)
    if _INJECTION_RE.search(text):
        risks.add(RiskFlag.PROMPT_INJECTION)
    if _MALFORMED_RE.search(text) or normalized.warnings:
        risks.add(RiskFlag.MALFORMED_INPUT)
    if re.search(r"\b(first|second)\b|two issues|also .*\b(charged|refund|billing)\b", text, re.I):
        risks.add(RiskFlag.MULTI_INTENT)
    return risks


def _penalties(
    risks: Iterable[RiskFlag], normalized: NormalizedTicket, triage: TriageResult
) -> dict[str, float]:
    penalties: dict[str, float] = {}
    risk_set = set(risks)
    if RiskFlag.MULTI_INTENT in risk_set or triage.intent_count > 1:
        penalties[RiskFlag.MULTI_INTENT.value] = 0.20
    if RiskFlag.MALFORMED_INPUT in risk_set:
        penalties[RiskFlag.MALFORMED_INPUT.value] = 0.40
    if normalized.duplicate_context:
        penalties[RiskFlag.DUPLICATE_CONTEXT.value] = 0.10
    if not normalized.subject or not normalized.body:
        penalties["missing_fields"] = 0.15
    return penalties


def evaluate_policy(
    ticket: Ticket,
    normalized: NormalizedTicket,
    triage: TriageResult,
    decision: DecisionResult,
    raw_risks: set[RiskFlag],
) -> PolicyEvaluation:
    final_risks = set(raw_risks) | set(triage.risk_flags)
    penalties = _penalties(final_risks, normalized, triage)
    base_confidence = min(triage.model_confidence, decision.model_confidence)
    effective_confidence = max(0.0, min(1.0, base_confidence - sum(penalties.values())))

    direct_escalation = {
        RiskFlag.ACCOUNT_DELETION,
        RiskFlag.LEGAL,
        RiskFlag.SECURITY,
        RiskFlag.PROMPT_INJECTION,
        RiskFlag.MALFORMED_INPUT,
        RiskFlag.MULTI_INTENT,
    }
    override_reason: str | None = None

    if final_risks & direct_escalation:
        final_action = Action.ESCALATE
        override_reason = "hard-risk policy requires direct human escalation"
    elif effective_confidence < 0.65:
        final_action = Action.ESCALATE
        final_risks.add(RiskFlag.LOW_CONFIDENCE)
        override_reason = "effective confidence is below draft threshold"
    elif (
        triage.category in {Category.SPAM, Category.FEEDBACK}
        and effective_confidence >= 0.90
        and not has_requested_action(normalized, triage)
        and not final_risks
    ):
        final_action = Action.AUTO_RESOLVE
    else:
        final_action = Action.DRAFT_FOR_REVIEW

    if final_action.value != decision.proposed_action:
        override_reason = override_reason or "deterministic policy changed the model proposal"

    rationale = (
        f"Category={triage.category}; proposed={decision.proposed_action}; "
        f"effective_confidence={effective_confidence:.2f}; "
        f"risks={sorted(r.value for r in final_risks)}"
    )
    return PolicyEvaluation(
        raw_risk_flags=sorted(raw_risks, key=lambda risk: risk.value),
        final_risk_flags=sorted(final_risks, key=lambda risk: risk.value),
        penalties=penalties,
        base_confidence=base_confidence,
        effective_confidence=effective_confidence,
        final_action=final_action,
        override_reason=override_reason,
        rationale=rationale,
    )


def canned_response(category: Category) -> str:
    if category == Category.SPAM:
        return "This mailbox is for product support. Promotional messages are not processed."
    return "Thank you for contacting support. We appreciate your feedback."


def has_requested_action(normalized: NormalizedTicket, triage: TriageResult) -> bool:
    if "no action needed" in normalized.combined_text.lower():
        return False
    return triage.requested_action or bool(_REQUEST_RE.search(normalized.combined_text))


def template_for_category(category: Category) -> str:
    return {
        Category.BILLING: "billing_review",
        Category.BUG: "bug_review",
        Category.FEATURE_REQUEST: "feature_review",
        Category.ACCOUNT_ACCESS: "account_access_review",
    }.get(category, "general_review")


def template_guidance(category: Category) -> str:
    return APPROVED_TEMPLATE_GUIDANCE[template_for_category(category)]


def validate_draft(draft: DraftResult, category: Category) -> list[str]:
    issues: list[str] = []
    if draft.template_id not in APPROVED_TEMPLATE_IDS:
        issues.append("draft selected an unknown response template")
    forbidden = {
        "refund issued": "claims a refund was issued",
        "refund approved": "claims a refund was approved",
        "refund processed": "claims a refund was processed",
        "we refunded": "claims a refund was processed",
        "account deleted": "claims an account was deleted",
        "we deleted": "claims an account or data was deleted",
        "security issue resolved": "claims a security issue was resolved",
        "security issue fixed": "claims a security issue was fixed",
        "issue has been fixed": "claims an undocumented fix",
        "issue is resolved": "claims an undocumented resolution",
        "within 30 days": "promises a legal deadline",
        "by tomorrow": "promises an unsupported deadline",
        "we changed your plan": "claims an account or billing change",
        "we cancelled your": "claims an account or billing change",
    }
    text = draft.response_text.lower()
    issues.extend(reason for phrase, reason in forbidden.items() if phrase in text)
    if draft.template_id != template_for_category(category):
        issues.append("draft template does not match the ticket category")
    return issues
