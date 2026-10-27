"""Offline tests: tools, guardrails and the agent loop with a scripted LLM.

Run:  python -m pytest -q
These do not call any LLM; they check the plumbing the real model relies on.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tam_agent.agent import Context, triage  # noqa: E402
from tam_agent.data import load_accounts, load_tickets, load_usage  # noqa: E402
from tam_agent.kb import load_kb  # noqa: E402
from tam_agent.policy import bump  # noqa: E402
from tam_agent.schema import validate  # noqa: E402


class ScriptedLLM:
    """Plays back a fixed list of tool calls, then submits `decision`."""

    provider, model = "scripted", "scripted"

    def __init__(self, steps, decision):
        self.steps, self.decision, self.calls, self.seen = list(steps), decision, 0, []

    def chat_tools(self, messages, tools, system=None, max_tokens=0, temperature=0.0):
        self.calls += 1
        self.seen.append(messages[-1])
        if self.steps:
            name, args = self.steps.pop(0)
            return {"text": f"Next: {name}", "tool_calls": [{"id": f"c{self.calls}", "name": name, "args": args}]}
        return {"text": "", "tool_calls": [{"id": "final", "name": "submit_triage", "args": self.decision}]}


@pytest.fixture(scope="module")
def ctx():
    import os
    os.environ.setdefault("LLM_PROVIDER", "ollama")
    return Context(load_accounts(), load_tickets(), load_usage(), load_kb())


def ticket(ctx, tid):
    return ctx.tickets.set_index("ticket_id").loc[tid].to_dict() | {"ticket_id": tid}


def base_decision(**kw):
    d = {"category": "how_to", "base_priority": "P4", "route_to": "L1_support", "escalate": False,
         "notify_tam": False, "needs_clarification": False, "kb_references": [], "account_signals": [],
         "customer_reply": "Hello", "internal_note": "note", "confidence": 0.8, "rationale": "test"}
    return d | kw


# ---------------------------------------------------------------- tools
def test_blank_account_resolved_by_domain_not_similar_name(ctx):
    r = ctx.tools.resolve_account("ironleaf-foods-europe.example", "Ironleaf Foods EU")
    assert r["account_id"] == "ACC-4850"  # not ACC-3405 "Ironleaf Foods"


def test_risk_override_rule(ctx):
    assert ctx.tools.evaluate_account_risk("ACC-2431")["override_applies"] is True   # Pinecrest
    assert ctx.tools.evaluate_account_risk("ACC-3751")["override_applies"] is False  # renewal near, healthy usage


def test_duplicate_uses_alias_and_code_in_description(ctx):
    ctx.tools.current_ticket = ticket(ctx, "TKT-192732") | {"_account": "ACC-6031"}
    r = ctx.tools.find_duplicate_candidates("ACC-6031", "analytics", "E-6105")
    assert [c["ticket_id"] for c in r["candidates"]] == ["TKT-122022"]


def test_history_excludes_future_tickets(ctx):
    t = ticket(ctx, "TKT-160477")
    ctx.tools.current_ticket = t | {"_account": "ACC-1633"}
    h = ctx.tools.get_ticket_history("ACC-1633", days=365)
    assert all(x["created_at"] < t["created_at"] for x in h["tickets"])


def test_kb_search_prefers_current_over_deprecated(ctx):
    res = ctx.tools.search_kb("e-signature rejected", module="CLM", error_code="E-2207")["results"]
    assert res[0]["id"] == "KB-1786" and all(r["status"] != "deprecated" for r in res)


def test_active_incident_lookup(ctx):
    assert ctx.tools.check_active_incidents("EINV", "E-4102")["active_incidents"][0]["id"] == "KB-3985"


def test_bump_caps_at_p2():
    assert [bump(p) for p in ("P4", "P3", "P2", "P1")] == ["P3", "P2", "P2", "P1"]


# ---------------------------------------------------------------- agent loop + guardrails
def test_loop_executes_tools_and_applies_risk_override(ctx):
    ctx.llm = ScriptedLLM([("evaluate_account_risk", {"account_id": "ACC-2431"}),
                           ("search_kb", {"query": "approvals slow inbox", "module": "eProcurement"}),
                           ("get_kb_article", {"kb_id": "KB-6082"})],
                          base_decision(kb_references=["KB-6082"]))
    out = triage(ticket(ctx, "TKT-119877"), ctx)
    assert validate(out) == []
    assert out["priority"] == "P3" and out["notify_tam"] and out["escalate"]
    actions = [s["action"] for s in out["reasoning_trace"]]
    assert actions.count("evaluate_account_risk") == 1 and "get_kb_article" in actions
    assert "KB-2914" in out["kb_references"]


def test_guardrails_force_security_handling(ctx):
    ctx.llm = ScriptedLLM([], base_decision(category="security", base_priority="P4", route_to="L1_support"))
    out = triage(ticket(ctx, "TKT-135081"), ctx)
    assert (out["category"], out["priority"], out["route_to"], out["escalate"]) == ("security", "P1", "security", True)
    assert "KB-6401" in out["kb_references"]


def test_injection_flagged_and_deprecated_refs_replaced(ctx):
    ctx.llm = ScriptedLLM([], base_decision(kb_references=["KB-2605", "KB-0000"],
                                            customer_reply="See ACC-2431 and ACC-3600"))
    out = triage(ticket(ctx, "TKT-158764"), ctx)
    assert any(s["action"] == "flag_untrusted_instructions" for s in out["reasoning_trace"])
    assert "KB-1786" in out["kb_references"] and "KB-2605" not in out["kb_references"]
    assert "ACC-2431" not in out["customer_reply"] and "ACC-3600" in out["customer_reply"]


def test_duplicate_enforced_and_not_escalated(ctx):
    ctx.llm = ScriptedLLM([("find_duplicate_candidates", {"account_id": "ACC-6031", "module": "analytics",
                                                          "error_code": "E-6105"})],
                          base_decision(category="bug", base_priority="P2", escalate=True))
    out = triage(ticket(ctx, "TKT-192732"), ctx)
    assert out["duplicate_of"] == "TKT-122022" and out["category"] == "duplicate" and not out["escalate"]


def test_fallback_when_model_never_submits(ctx):
    class Silent(ScriptedLLM):
        def chat_tools(self, *a, **k):
            self.calls += 1
            return {"text": "hmm", "tool_calls": []}
    ctx.llm = Silent([], {})
    out = triage(ticket(ctx, "TKT-128253"), ctx)
    assert validate(out) == [] and out["needs_clarification"] and out["priority"] == "P3"
