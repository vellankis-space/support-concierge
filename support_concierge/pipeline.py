from __future__ import annotations

import time
from pathlib import Path
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from .models import (
    Action,
    DraftResult,
    PipelineResult,
    PolicyEvaluation,
    ReviewResult,
    Ticket,
    TicketStatus,
    TriageResult,
)
from .policy import (
    canned_response,
    detect_raw_risks,
    duplicate_context,
    evaluate_policy,
    normalize_ticket,
    validate_draft,
)
from .provider import AgentProvider, ProviderError
from .storage import AuditStore

PROMPT_VERSION = "support-concierge-prompts-v1"
SCHEMA_VERSION = "support-concierge-schema-v1"

# Inter-ticket delay in seconds to respect hosted-model rate limits.
BATCH_DELAY_SECONDS = 4.0


class PipelineState(TypedDict, total=False):
    ticket: Ticket
    normalized: Any
    raw_risks: set[Any]
    triage: TriageResult
    decision: Any
    policy: PolicyEvaluation
    draft: DraftResult
    review: ReviewResult
    result: PipelineResult


def _record_agent(
    store: AuditStore,
    run_id: str,
    ticket: Ticket,
    agent: str,
    input_value: Any,
    operation: Any,
    model: str,
) -> Any:
    started = time.perf_counter()
    try:
        result = operation()
    except ProviderError as exc:
        store.record_agent_event(
            run_id,
            ticket.id,
            agent,
            "failed",
            model=model,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            input_value=input_value,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=exc.message,
        )
        raise
    except Exception as exc:
        store.record_agent_event(
            run_id,
            ticket.id,
            agent,
            "failed",
            model=model,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            input_value=input_value,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
        )
        raise ProviderError(agent, str(exc)) from exc

    confidence = getattr(result, "model_confidence", None)
    store.record_agent_event(
        run_id,
        ticket.id,
        agent,
        "completed",
        model=model,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input_value=input_value,
        output_value=result,
        confidence=confidence,
        latency_ms=(time.perf_counter() - started) * 1000,
    )
    return result


class TicketPipeline:
    def __init__(self, provider: AgentProvider, store: AuditStore, run_id: str) -> None:
        self.provider = provider
        self.store = store
        self.run_id = run_id

    def _graph(self) -> Any:
        builder = StateGraph(PipelineState)

        def normalize_node(state: PipelineState) -> dict[str, Any]:
            normalized = state["normalized"]
            return {"normalized": normalized, "raw_risks": detect_raw_risks(normalized)}

        def triage_node(state: PipelineState) -> dict[str, Any]:
            ticket = state["ticket"]
            normalized = state["normalized"]
            result = _record_agent(
                self.store,
                self.run_id,
                ticket,
                "triage",
                normalized,
                lambda: self.provider.triage(ticket, normalized),
                self.provider.model,
            )
            return {"triage": result}

        def decision_node(state: PipelineState) -> dict[str, Any]:
            ticket = state["ticket"]
            normalized = state["normalized"]
            triage = state["triage"]
            result = _record_agent(
                self.store,
                self.run_id,
                ticket,
                "decision",
                {"normalized": normalized, "triage": triage},
                lambda: self.provider.decide(ticket, normalized, triage),
                self.provider.model,
            )
            return {"decision": result}

        def policy_node(state: PipelineState) -> dict[str, Any]:
            result = evaluate_policy(
                state["ticket"],
                state["normalized"],
                state["triage"],
                state["decision"],
                state["raw_risks"],
            )
            return {"policy": result}

        def response_node(state: PipelineState) -> dict[str, Any]:
            ticket = state["ticket"]
            normalized = state["normalized"]
            triage = state["triage"]
            decision = state["decision"]
            def generate_draft() -> DraftResult:
                result = self.provider.draft(ticket, normalized, triage, decision)
                issues = validate_draft(result, triage.category)
                if issues:
                    raise ProviderError("response", "; ".join(issues))
                return result

            result = _record_agent(
                self.store,
                self.run_id,
                ticket,
                "response",
                {"normalized": normalized, "triage": triage, "decision": decision},
                generate_draft,
                self.provider.model,
            )
            return {"draft": result}

        def reviewer_node(state: PipelineState) -> dict[str, Any]:
            """Self-critique step: a fourth agent reviews the drafted response."""
            ticket = state["ticket"]
            normalized = state["normalized"]
            triage = state["triage"]
            draft = state["draft"]
            result = _record_agent(
                self.store,
                self.run_id,
                ticket,
                "reviewer",
                {"draft": draft, "triage": triage},
                lambda: self.provider.review_draft(ticket, normalized, triage, draft),
                self.provider.model,
            )
            return {"review": result}

        def finalize_node(state: PipelineState) -> dict[str, Any]:
            policy = state["policy"]
            draft = state.get("draft")
            review = state.get("review")

            # If the reviewer agent rejected the draft, override to escalation.
            if review and not review.approved:
                final_action = Action.ESCALATE
                final_status = TicketStatus.ESCALATED
                override_reason = (
                    f"reviewer agent rejected draft: {'; '.join(review.issues)}"
                    if review.issues
                    else "reviewer agent rejected draft"
                )
                policy = policy.model_copy(
                    update={
                        "final_action": final_action,
                        "override_reason": override_reason,
                    }
                )
                draft = None
            else:
                final_action = policy.final_action
                final_status = (
                    TicketStatus.CLOSED
                    if final_action == Action.AUTO_RESOLVE
                    else TicketStatus.DRAFT_PENDING
                    if final_action == Action.DRAFT_FOR_REVIEW
                    else TicketStatus.ESCALATED
                )

            if final_action == Action.AUTO_RESOLVE:
                draft = DraftResult(
                    response_text=canned_response(state["triage"].category),
                    template_id=f"auto_{state['triage'].category}",
                    claims=[],
                )
            result = PipelineResult(
                ticket=state["ticket"],
                normalized=state["normalized"],
                triage=state.get("triage"),
                decision=state.get("decision"),
                policy=policy,
                draft=draft,
                review=review,
                final_action=final_action,
                status=final_status,
                send_status=(
                    "sent_simulated" if final_action == Action.AUTO_RESOLVE else None
                ),
                run_id=self.run_id,
            )
            response = draft.response_text if draft else None
            self.store.record_decision(
                result,
                final_response=response,
                send_status=result.send_status,
            )
            self.store.enqueue(result, response=response)
            return {"result": result}

        def route(state: PipelineState) -> str:
            return state["policy"].final_action.value

        builder.add_node("normalize", normalize_node)
        builder.add_node("triage_agent", triage_node)
        builder.add_node("decision_agent", decision_node)
        builder.add_node("policy_gate", policy_node)
        builder.add_node("response_agent", response_node)
        builder.add_node("reviewer_agent", reviewer_node)
        builder.add_node("finalize", finalize_node)
        builder.add_edge(START, "normalize")
        builder.add_edge("normalize", "triage_agent")
        builder.add_edge("triage_agent", "decision_agent")
        builder.add_edge("decision_agent", "policy_gate")
        builder.add_conditional_edges(
            "policy_gate",
            route,
            {
                Action.AUTO_RESOLVE.value: "finalize",
                Action.DRAFT_FOR_REVIEW.value: "response_agent",
                Action.ESCALATE.value: "finalize",
            },
        )
        builder.add_edge("response_agent", "reviewer_agent")
        builder.add_edge("reviewer_agent", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()

    def run_ticket(self, ticket: Ticket, previous: list[Ticket] | None = None) -> PipelineResult:
        normalized = normalize_ticket(ticket)
        normalized.duplicate_context = duplicate_context(ticket, previous or [])
        self.store.start_ticket(ticket, self.run_id, normalized)
        try:
            state = self._graph().invoke(
                {"ticket": ticket, "normalized": normalized},
            )
            return cast(PipelineResult, state["result"])
        except ProviderError as exc:
            self.store.record_failure(self.run_id, ticket, normalized, exc.message)
            return PipelineResult(
                ticket=ticket,
                normalized=normalized,
                final_action=Action.ESCALATE,
                status=TicketStatus.ESCALATED,
                error=f"{exc.agent}: {exc.message}",
                run_id=self.run_id,
            )


def run_batch(
    tickets: list[Ticket],
    store: AuditStore,
    provider: AgentProvider,
    delay_seconds: float = 0.0,
) -> list[PipelineResult]:
    """Process a list of tickets through the pipeline.

    A configurable inter-ticket delay (default 0 s in code, 4 s in CLI) avoids rate-limit
    errors when using hosted model APIs with token-per-minute quotas.
    """
    run_id = store.start_run(provider.name, provider.model)
    pipeline = TicketPipeline(provider, store, run_id)
    results: list[PipelineResult] = []
    previous: list[Ticket] = []
    try:
        for idx, ticket in enumerate(tickets):
            if idx > 0 and delay_seconds > 0:
                time.sleep(delay_seconds)
            result = pipeline.run_ticket(ticket, previous=previous)
            results.append(result)
            previous.append(ticket)
        failed = any(result.error for result in results)
        store.finish_run(run_id, "completed_with_failures" if failed else "completed")
    except Exception:
        store.finish_run(run_id, "failed")
        raise
    return results


def load_tickets(path: str | Path) -> list[Ticket]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Ticket.model_validate(item) for item in payload]
