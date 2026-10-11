from datetime import datetime

from support_concierge.models import Ticket


def make_ticket(
    ticket_id: str = "TCK-TEST",
    subject: str = "Question",
    body: str = "Please help.",
) -> Ticket:
    return Ticket(
        id=ticket_id,
        received_at=datetime(2026, 8, 3, 9, 12),
        from_name="Test Customer",
        from_email="test@example.com",
        subject=subject,
        body=body,
    )
