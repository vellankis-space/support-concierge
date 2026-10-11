from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .models import QueueKind, ReviewAction
from .pipeline import load_tickets, run_batch
from .provider import AgentProvider, FailureInjectingProvider, FakeProvider, GroqProvider
from .storage import AuditStore

app = typer.Typer(help="Support Concierge ticket triage CLI")


@app.command()
def run(
    input_path: Annotated[Path, typer.Argument(help="JSON ticket input")],
    db: Annotated[Path, typer.Option(help="SQLite audit database")],
    output: Annotated[Path, typer.Option(help="JSON result output")],
    model: Annotated[str, typer.Option(help="Groq model")] = "openai/gpt-oss-20b",
    delay: Annotated[
        float, typer.Option(help="Inter-ticket delay in seconds to avoid rate limits")
    ] = 4.0,
    simulate_failure: Annotated[
        str | None,
        typer.Option(
            help="Inject failure as TICKET_ID:triage|decision|response|reviewer"
        ),
    ] = None,
    use_fake: Annotated[
        bool, typer.Option("--fake", help="Use deterministic offline provider")
    ] = False,
) -> None:
    """Process tickets through the triage pipeline."""
    provider: AgentProvider
    if use_fake:
        provider = FakeProvider()
        if delay == 4.0:
            delay = 0.0
    else:
        try:
            provider = GroqProvider(model=model)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    if simulate_failure:
        try:
            ticket_id, agent = simulate_failure.split(":", 1)
        except ValueError as exc:
            raise typer.BadParameter("use TICKET_ID:AGENT") from exc
        provider = FailureInjectingProvider(provider, ticket_id, agent)

    tickets = load_tickets(input_path)
    store = AuditStore(db)
    try:
        results = run_batch(tickets, store, provider, delay_seconds=delay)
        output.write_text(
            json.dumps([result.model_dump(mode="json") for result in results], indent=2),
            encoding="utf-8",
        )
        typer.echo(f"Processed {len(results)} tickets; results written to {output}")
    finally:
        store.close()


@app.command("queue")
def queue(
    kind: Annotated[QueueKind, typer.Option(help="Queue type")],
    db: Annotated[Path, typer.Option(help="SQLite audit database")],
) -> None:
    """List pending draft or escalation work."""
    store = AuditStore(db)
    try:
        rows = store.list_queue(kind)
        typer.echo(json.dumps([dict(row) for row in rows], indent=2, default=str))
    finally:
        store.close()


@app.command()
def show(
    ticket_id: Annotated[str, typer.Argument()],
    db: Annotated[Path, typer.Option(help="SQLite audit database")],
) -> None:
    """Show a ticket and its complete audit trail."""
    store = AuditStore(db)
    try:
        typer.echo(json.dumps(store.ticket(ticket_id), indent=2, default=str))
    except KeyError as exc:
        raise typer.BadParameter(f"unknown ticket: {ticket_id}") from exc
    finally:
        store.close()


@app.command()
def review(
    ticket_id: Annotated[str, typer.Argument()],
    action: Annotated[ReviewAction, typer.Argument()],
    db: Annotated[Path, typer.Option(help="SQLite audit database")],
    reviewer: Annotated[str, typer.Option(help="Reviewer identity")] = "local-reviewer",
    response: Annotated[str | None, typer.Option(help="Edited or approved response")] = None,
    note: Annotated[str | None, typer.Option(help="Human review note")] = None,
    assigned_to: Annotated[str | None, typer.Option(help="Escalation destination")] = None,
) -> None:
    """Apply an append-only human review action."""
    store = AuditStore(db)
    try:
        store.review(
            ticket_id=ticket_id,
            action=action,
            reviewer=reviewer,
            note=note,
            response=response,
            assigned_to=assigned_to,
        )
        typer.echo(f"Applied {action.value} to {ticket_id}")
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        store.close()


if __name__ == "__main__":
    app()
