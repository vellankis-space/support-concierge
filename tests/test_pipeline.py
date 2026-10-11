import pytest

from support_concierge.models import (
    Action,
    DraftResult,
    QueueKind,
    ReviewAction,
    ReviewResult,
    TicketStatus,
)
from support_concierge.pipeline import run_batch
from support_concierge.provider import FailureInjectingProvider, FakeProvider
from support_concierge.storage import AuditStore

from .conftest import make_ticket


class BadDraftProvider(FakeProvider):
    def draft(self, ticket, normalized, triage, decision) -> DraftResult:
        draft = super().draft(ticket, normalized, triage, decision)
        return draft.model_copy(update={"template_id": "unknown_template"})


def test_auto_resolution_persists_canned_response_and_closes(tmp_path):
    store = AuditStore(tmp_path / "audit.db")
    try:
        ticket = make_ticket(subject="Thanks", body="Thank you, no action needed!")
        result = run_batch([ticket], store, FakeProvider())[0]

        assert result.final_action == Action.AUTO_RESOLVE
        assert result.status == TicketStatus.CLOSED
        inspected = store.ticket(ticket.id)
        assert inspected["ticket"]["final_response"]
        assert inspected["ticket"]["send_status"] == "sent_simulated"
        assert inspected["events"]
        assert inspected["events"][0]["model"] == "test-fixture"
        assert inspected["events"][0]["prompt_version"] == "support-concierge-prompts-v1"
        assert store.list_queue(QueueKind.DRAFT) == []
        assert store.list_queue(QueueKind.ESCALATION) == []
    finally:
        store.close()


def test_draft_review_can_be_edited_and_approved(tmp_path):
    store = AuditStore(tmp_path / "audit.db")
    try:
        ticket = make_ticket(subject="Export is broken", body="The export button does not work.")
        result = run_batch([ticket], store, FakeProvider())[0]

        assert result.final_action == Action.DRAFT_FOR_REVIEW
        assert len(store.list_queue(QueueKind.DRAFT)) == 1

        store.review(
            ticket.id,
            ReviewAction.EDIT,
            reviewer="alice",
            response="We are reviewing the export issue now.",
        )
        store.review(ticket.id, ReviewAction.APPROVE, reviewer="alice")
        inspected = store.ticket(ticket.id)
        assert inspected["ticket"]["status"] == TicketStatus.CLOSED.value
        assert inspected["ticket"]["final_response"] == "We are reviewing the export issue now."
        assert [row["action"] for row in inspected["review_actions"]] == ["edit", "approve"]
        assert (
            inspected["review_actions"][-1]["response"]
            == "We are reviewing the export issue now."
        )
    finally:
        store.close()


def test_rejecting_draft_updates_decision_and_opens_escalation(tmp_path):
    store = AuditStore(tmp_path / "reject.db")
    try:
        ticket = make_ticket(subject="Export is broken", body="The export button does not work.")
        run_batch([ticket], store, FakeProvider())

        store.review(ticket.id, ReviewAction.REJECT, reviewer="alice", note="Needs specialist")
        inspected = store.ticket(ticket.id)
        assert inspected["ticket"]["final_action"] == Action.ESCALATE.value
        assert inspected["decision"]["final_action"] == Action.ESCALATE.value
        assert len(store.list_queue(QueueKind.ESCALATION)) == 1
    finally:
        store.close()


def test_escalation_can_be_reassigned_and_resolved(tmp_path):
    store = AuditStore(tmp_path / "escalation.db")
    try:
        ticket = make_ticket(
            subject="Possible security issue",
            body="Changing account_id exposes another company's dashboard.",
        )
        run_batch([ticket], store, FakeProvider())

        store.review(ticket.id, ReviewAction.REASSIGN, reviewer="alice", assigned_to="security")
        store.review(
            ticket.id,
            ReviewAction.RESOLVE,
            reviewer="bob",
            note="Security team reviewed the report.",
        )
        inspected = store.ticket(ticket.id)
        assert inspected["ticket"]["status"] == TicketStatus.CLOSED.value
        assert [row["action"] for row in inspected["review_actions"]] == [
            "reassign",
            "resolve",
        ]
    finally:
        store.close()


def test_repeated_approval_cannot_send_twice(tmp_path):
    store = AuditStore(tmp_path / "repeat.db")
    try:
        ticket = make_ticket(subject="Export is broken", body="The export button does not work.")
        run_batch([ticket], store, FakeProvider())
        store.review(ticket.id, ReviewAction.APPROVE, reviewer="alice")

        with pytest.raises(ValueError, match="no pending review item"):
            store.review(ticket.id, ReviewAction.APPROVE, reviewer="alice")
    finally:
        store.close()


def test_invalid_response_is_recorded_as_failed_agent_event(tmp_path):
    store = AuditStore(tmp_path / "invalid-draft.db")
    try:
        ticket = make_ticket(subject="Export is broken", body="The export button does not work.")
        result = run_batch([ticket], store, BadDraftProvider())[0]

        assert result.final_action == Action.ESCALATE
        response_events = [
            event
            for event in store.ticket(ticket.id)["events"]
            if event["agent"] == "response"
        ]
        assert response_events[0]["status"] == "failed"
    finally:
        store.close()


def test_agent_failure_escalates_only_affected_ticket_and_continues(tmp_path):
    store = AuditStore(tmp_path / "failure.db")
    try:
        failed = make_ticket("TCK-FAIL", "Broken", "Please help with this broken feature.")
        healthy = make_ticket("TCK-OK", "Thanks", "Thank you, no action needed!")
        provider = FailureInjectingProvider(FakeProvider(), "TCK-FAIL", "triage")

        results = run_batch([failed, healthy], store, provider)

        assert results[0].error == "triage: controlled failure injection"
        assert results[0].final_action == Action.ESCALATE
        assert results[1].final_action == Action.AUTO_RESOLVE
        assert len(store.list_queue(QueueKind.ESCALATION)) == 1
        events = store.ticket("TCK-FAIL")["events"]
        assert events[0]["agent"] == "triage"
        assert events[0]["status"] == "failed"
    finally:
        store.close()


class RejectingReviewerProvider(FakeProvider):
    def review_draft(self, ticket, normalized, triage, draft) -> ReviewResult:
        return ReviewResult(
            approved=False,
            issues=["Draft contains unverified claims", "Tone is inappropriate"],
            rationale="Failed factual consistency check",
        )


def test_reviewer_rejection_overrides_to_escalation_and_clears_draft(tmp_path):
    store = AuditStore(tmp_path / "reviewer-reject.db")
    try:
        ticket = make_ticket(subject="Export is broken", body="The export button does not work.")
        result = run_batch([ticket], store, RejectingReviewerProvider())[0]

        assert result.final_action == Action.ESCALATE
        assert result.status == TicketStatus.ESCALATED
        assert result.draft is None
        assert result.review is not None
        assert not result.review.approved
        assert "Draft contains unverified claims" in result.review.issues
        assert result.policy is not None
        assert "reviewer agent rejected draft" in (result.policy.override_reason or "")
        assert len(store.list_queue(QueueKind.ESCALATION)) == 1
        assert len(store.list_queue(QueueKind.DRAFT)) == 0
    finally:
        store.close()


def test_reviewer_failure_escalates_and_records_event(tmp_path):
    store = AuditStore(tmp_path / "reviewer-fail.db")
    try:
        ticket = make_ticket(subject="Export is broken", body="The export button does not work.")
        provider = FailureInjectingProvider(FakeProvider(), ticket.id, "reviewer")
        result = run_batch([ticket], store, provider)[0]

        assert result.error == "reviewer: controlled failure injection"
        assert result.final_action == Action.ESCALATE
        assert result.status == TicketStatus.ESCALATED
        assert len(store.list_queue(QueueKind.ESCALATION)) == 1
    finally:
        store.close()
