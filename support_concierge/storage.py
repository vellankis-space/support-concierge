from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import (
    Action,
    PipelineResult,
    QueueKind,
    ReviewAction,
    Ticket,
    TicketStatus,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


class AuditStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );
            CREATE TABLE IF NOT EXISTS tickets (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(id),
                raw_json TEXT NOT NULL,
                normalized_json TEXT,
                status TEXT NOT NULL,
                final_action TEXT,
                final_response TEXT,
                send_status TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(id),
                ticket_id TEXT NOT NULL REFERENCES tickets(id),
                agent TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT,
                output_json TEXT,
                confidence REAL,
                latency_ms REAL,
                error TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL REFERENCES tickets(id),
                proposed_action TEXT,
                final_action TEXT NOT NULL,
                confidence_json TEXT NOT NULL,
                rationale TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL REFERENCES tickets(id),
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                response TEXT,
                assigned_to TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS review_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL REFERENCES tickets(id),
                action TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                note TEXT,
                response TEXT,
                assigned_to TEXT,
                previous_status TEXT NOT NULL,
                new_status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_review_queue_kind_status
                ON review_queue(kind, status);
            CREATE INDEX IF NOT EXISTS idx_agent_events_ticket
                ON agent_events(ticket_id, created_at);
            """
        )
        self.connection.commit()

    def start_run(self, provider: str, model: str) -> str:
        run_id = str(uuid.uuid4())
        self.connection.execute(
            "INSERT INTO runs(id, provider, model, status, started_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, provider, model, "running", _now()),
        )
        self.connection.commit()
        return run_id

    def finish_run(self, run_id: str, status: str) -> None:
        self.connection.execute(
            "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
            (status, _now(), run_id),
        )
        self.connection.commit()

    def start_ticket(self, ticket: Ticket, run_id: str, normalized: Any) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO tickets
            (id, run_id, raw_json, normalized_json, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                ticket.id,
                run_id,
                ticket.model_dump_json(),
                normalized.model_dump_json(),
                TicketStatus.PROCESSING.value,
                _now(),
            ),
        )
        self.connection.commit()

    def record_agent_event(
        self,
        run_id: str,
        ticket_id: str,
        agent: str,
        status: str,
        model: str,
        prompt_version: str,
        schema_version: str,
        input_value: Any = None,
        output_value: Any = None,
        confidence: float | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        self.connection.execute(
            """INSERT INTO agent_events
            (run_id, ticket_id, agent, model, prompt_version, schema_version,
             status, input_json, output_json, confidence, latency_ms, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                ticket_id,
                agent,
                model,
                prompt_version,
                schema_version,
                status,
                json.dumps(_json_value(input_value), default=str)
                if input_value is not None
                else None,
                json.dumps(_json_value(output_value), default=str)
                if output_value is not None
                else None,
                confidence,
                latency_ms,
                error,
                _now(),
            ),
        )
        self.connection.commit()

    def record_decision(
        self,
        result: PipelineResult,
        final_response: str | None = None,
        send_status: str | None = None,
    ) -> None:
        assert result.policy is not None
        proposed = result.decision.proposed_action if result.decision else None
        self.connection.execute(
            """INSERT INTO decisions
            (ticket_id, proposed_action, final_action, confidence_json, rationale, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                result.ticket.id,
                proposed,
                result.policy.final_action.value,
                result.policy.model_dump_json(),
                result.policy.rationale,
                _now(),
            ),
        )
        self.connection.execute(
            """UPDATE tickets SET status = ?, final_action = ?, final_response = ?,
            send_status = ?, updated_at = ?
            WHERE id = ?""",
            (
                result.status.value,
                result.final_action.value,
                final_response,
                send_status,
                _now(),
                result.ticket.id,
            ),
        )
        self.connection.commit()

    def enqueue(self, result: PipelineResult, response: str | None = None) -> None:
        assert result.policy is not None
        if result.final_action == Action.DRAFT_FOR_REVIEW:
            kind = QueueKind.DRAFT.value
            status = "pending"
        elif result.final_action == Action.ESCALATE:
            kind = QueueKind.ESCALATION.value
            status = "pending"
        else:
            return
        self.connection.execute(
            """INSERT INTO review_queue(ticket_id, kind, status, response, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (result.ticket.id, kind, status, response, _now(), _now()),
        )
        self.connection.commit()

    def record_failure(self, run_id: str, ticket: Ticket, normalized: Any, message: str) -> None:
        self.connection.execute(
            """UPDATE tickets SET status = ?, final_action = ?, updated_at = ? WHERE id = ?""",
            (TicketStatus.ESCALATED.value, Action.ESCALATE.value, _now(), ticket.id),
        )
        self.connection.execute(
            """INSERT INTO decisions
            (ticket_id, proposed_action, final_action, confidence_json, rationale, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                ticket.id,
                None,
                Action.ESCALATE.value,
                json.dumps({"effective_confidence": 0.0, "failure": True}),
                f"Agent failure caused safe escalation: {message}",
                _now(),
            ),
        )
        self.connection.execute(
            """INSERT INTO review_queue(ticket_id, kind, status, response, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (ticket.id, QueueKind.ESCALATION.value, "pending", None, _now(), _now()),
        )
        self.connection.commit()

    def list_queue(self, kind: QueueKind) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """SELECT q.*, t.raw_json, t.final_action, t.status AS ticket_status
                FROM review_queue q JOIN tickets t ON t.id = q.ticket_id
                WHERE q.kind = ? AND q.status = 'pending'
                ORDER BY q.created_at""",
                (kind.value,),
            )
        )

    def ticket(self, ticket_id: str) -> dict[str, Any]:
        ticket = self.connection.execute(
            "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
        if ticket is None:
            raise KeyError(ticket_id)
        events = list(
            self.connection.execute(
                "SELECT * FROM agent_events WHERE ticket_id = ? ORDER BY id", (ticket_id,)
            )
        )
        decisions = list(
            self.connection.execute(
                "SELECT * FROM decisions WHERE ticket_id = ? ORDER BY id", (ticket_id,)
            )
        )
        reviews = list(
            self.connection.execute(
                "SELECT * FROM review_actions WHERE ticket_id = ? ORDER BY id", (ticket_id,)
            )
        )
        return {
            "ticket": dict(ticket),
            "events": [dict(row) for row in events],
            "decision": dict(decisions[-1]) if decisions else None,
            "decision_history": [dict(row) for row in decisions],
            "review_actions": [dict(row) for row in reviews],
        }

    def review(
        self,
        ticket_id: str,
        action: ReviewAction,
        reviewer: str,
        note: str | None = None,
        response: str | None = None,
        assigned_to: str | None = None,
    ) -> None:
        queue = self.connection.execute(
            "SELECT * FROM review_queue WHERE ticket_id = ? AND status = 'pending' "
            "ORDER BY id DESC LIMIT 1",
            (ticket_id,),
        ).fetchone()
        ticket = self.connection.execute(
            "SELECT status FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
        if queue is None or ticket is None:
            raise ValueError("ticket has no pending review item")
        kind = queue["kind"]
        current_status = ticket["status"]
        if action == ReviewAction.EDIT:
            if kind != QueueKind.DRAFT.value or not response:
                raise ValueError("edit requires a draft queue item and response")
            self.connection.execute(
                "UPDATE review_queue SET response = ?, updated_at = ? WHERE id = ?",
                (response, _now(), queue["id"]),
            )
            new_status = current_status
        elif action == ReviewAction.APPROVE:
            if kind != QueueKind.DRAFT.value:
                raise ValueError("approve requires a draft queue item")
            final_response = response or queue["response"]
            if not final_response:
                raise ValueError("approve requires a response")
            response = final_response
            self.connection.execute(
                "UPDATE review_queue SET status = 'resolved', response = ?, "
                "updated_at = ? WHERE id = ?",
                (final_response, _now(), queue["id"]),
            )
            self.connection.execute(
                "UPDATE tickets SET status = 'closed', final_response = ?, "
                "send_status = 'sent_simulated', "
                "updated_at = ? WHERE id = ?",
                (final_response, _now(), ticket_id),
            )
            new_status = TicketStatus.CLOSED.value
        elif action == ReviewAction.REJECT:
            if kind != QueueKind.DRAFT.value:
                raise ValueError("reject requires a draft queue item")
            self.connection.execute(
                "UPDATE review_queue SET status = 'rejected', updated_at = ? WHERE id = ?",
                (_now(), queue["id"]),
            )
            self.connection.execute(
                "UPDATE tickets SET status = 'escalated', final_action = 'escalate', "
                "updated_at = ? WHERE id = ?",
                (_now(), ticket_id),
            )
            self.connection.execute(
                "UPDATE decisions SET final_action = 'escalate', "
                "rationale = rationale || ? WHERE id = "
                "(SELECT id FROM decisions WHERE ticket_id = ? ORDER BY id DESC LIMIT 1)",
                ("; human reviewer rejected the draft", ticket_id),
            )
            self.connection.execute(
                "INSERT INTO review_queue(ticket_id, kind, status, created_at, updated_at) "
                "VALUES (?, 'escalation', 'pending', ?, ?)",
                (ticket_id, _now(), _now()),
            )
            new_status = TicketStatus.ESCALATED.value
        elif action == ReviewAction.RESOLVE:
            if kind != QueueKind.ESCALATION.value or not note:
                raise ValueError("resolve requires an escalation and a human note")
            self.connection.execute(
                "UPDATE review_queue SET status = 'resolved', updated_at = ? WHERE id = ?",
                (_now(), queue["id"]),
            )
            self.connection.execute(
                "UPDATE tickets SET status = 'closed', updated_at = ? WHERE id = ?",
                (_now(), ticket_id),
            )
            new_status = TicketStatus.CLOSED.value
        elif action == ReviewAction.REASSIGN:
            if kind != QueueKind.ESCALATION.value or not assigned_to:
                raise ValueError("reassign requires an escalation and destination")
            self.connection.execute(
                "UPDATE review_queue SET assigned_to = ?, updated_at = ? WHERE id = ?",
                (assigned_to, _now(), queue["id"]),
            )
            new_status = current_status
        else:
            raise ValueError(f"unsupported review action: {action}")

        self.connection.execute(
            """INSERT INTO review_actions
            (ticket_id, action, reviewer, note, response, assigned_to,
             previous_status, new_status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticket_id,
                action.value,
                reviewer,
                note,
                response,
                assigned_to,
                current_status,
                new_status,
                _now(),
            ),
        )
        self.connection.commit()
