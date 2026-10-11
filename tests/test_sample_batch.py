from pathlib import Path

from support_concierge.models import Action, QueueKind
from support_concierge.pipeline import load_tickets, run_batch
from support_concierge.provider import FakeProvider
from support_concierge.storage import AuditStore


def test_all_sample_tickets_have_safe_final_actions(tmp_path):
    root = Path(__file__).parents[1]
    tickets = load_tickets(root / "sample_tickets.json")
    store = AuditStore(tmp_path / "sample.db")
    try:
        results = run_batch(tickets, store, FakeProvider())

        assert len(results) == 18
        assert all(result.final_action in set(Action) for result in results)
        auto_resolved = {
            result.ticket.id
            for result in results
            if result.final_action == Action.AUTO_RESOLVE
        }
        assert auto_resolved == {"TCK-1008", "TCK-1016"}
        escalated = {
            result.ticket.id
            for result in results
            if result.final_action == Action.ESCALATE
        }
        assert {
            "TCK-1006",
            "TCK-1009",
            "TCK-1010",
            "TCK-1012",
            "TCK-1013",
            "TCK-1015",
            "TCK-1017",
        } <= escalated
        assert len(store.list_queue(QueueKind.ESCALATION)) >= 7
        assert all(result.error is None for result in results)
    finally:
        store.close()
