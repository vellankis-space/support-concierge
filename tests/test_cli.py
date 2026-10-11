from typer.testing import CliRunner

from support_concierge.cli import app
from support_concierge.pipeline import run_batch
from support_concierge.provider import FakeProvider
from support_concierge.storage import AuditStore

from .conftest import make_ticket


def test_cli_lists_and_reviews_draft_queue(tmp_path):
    db = tmp_path / "audit.db"
    store = AuditStore(db)
    try:
        run_batch(
            [make_ticket(subject="Export is broken", body="The export button does not work.")],
            store,
            FakeProvider(),
        )
    finally:
        store.close()

    runner = CliRunner()
    queue_result = runner.invoke(app, ["queue", "--kind", "draft", "--db", str(db)])
    assert queue_result.exit_code == 0
    assert "TCK-TEST" in queue_result.stdout

    review_result = runner.invoke(
        app,
        [
            "review",
            "TCK-TEST",
            "approve",
            "--db",
            str(db),
            "--response",
            "We are reviewing this issue.",
        ],
    )
    assert review_result.exit_code == 0

    show_result = runner.invoke(app, ["show", "TCK-TEST", "--db", str(db)])
    assert show_result.exit_code == 0
    assert '"status": "closed"' in show_result.stdout
