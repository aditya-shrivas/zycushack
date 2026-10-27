"""Deterministic policy guardrails applied to the LLM's proposed decision.

The LLM investigates and proposes; this layer enforces the hard rules from the
support policy articles (KB-2914, KB-5930, KB-6401) so a weak or manipulated
model cannot produce a non-compliant triage. Every change is returned as a note
and written to reasoning_trace, so overrides are auditable.
"""
import re

from .schema import CATEGORIES, PRIORITIES, ROUTES

POLICY_KB, DUPLICATE_KB, EXPOSURE_KB = "KB-2914", "KB-5930", "KB-6401"


def bump(priority: str) -> str:
    """Raise one level, but the override never goes above P2 (P1 stays P1)."""
    return {"P4": "P3", "P3": "P2", "P2": "P2", "P1": "P1"}[priority]


def apply_policy(d: dict, facts: dict, toolbox) -> tuple[dict, list[str]]:
    """d: arguments of submit_triage. facts: evidence gathered by tools during the run."""
    notes = []
    out = dict(d)

    def fix(field, value, why):
        if out.get(field) != value:
            notes.append(f"{field}: {out.get(field)!r} -> {value!r} ({why})")
            out[field] = value

    # --- shape / enum hygiene
    if out.get("category") not in CATEGORIES:
        fix("category", "needs_info", "invalid category from model")
    if out.get("base_priority") not in PRIORITIES:
        fix("base_priority", "P3", "invalid priority from model")
    if out.get("route_to") not in ROUTES:
        fix("route_to", "L1_support", "invalid route from model")
    for k in ("escalate", "notify_tam", "needs_clarification"):
        out[k] = bool(out.get(k))
    for k in ("kb_references", "account_signals"):
        v = out.get(k) or []
        out[k] = [str(x) for x in v] if isinstance(v, list) else [str(v)]

    # --- account: trust tool resolution over the model's text
    resolved = facts.get("account_id")
    if resolved and out.get("account_id") != resolved:
        fix("account_id", resolved, "resolved by requester_domain/account_id lookup")
    if not resolved and out.get("account_id") and out["account_id"] not in toolbox.accounts.index:
        fix("account_id", None, "not a known account")

    # --- duplicates (KB-5930): exact rule hit (same module + same error code) is enforced
    dup = facts.get("duplicate")
    if dup and not out.get("duplicate_of"):
        strict = [c for c in dup.get("candidates", []) if c.get("same_error_code")]
        if strict:
            fix("duplicate_of", strict[0]["ticket_id"], "KB-5930 rule match: same account/module/error code within 7 days")
    if out.get("duplicate_of"):
        if out["duplicate_of"] not in toolbox.tickets_by_id.index:
            fix("duplicate_of", None, "unknown ticket id")
        else:
            fix("category", "duplicate", "duplicate_of is set (KB-5930)")
            fix("escalate", False, "KB-5930: keep original priority, do not re-escalate")
    elif out["category"] == "duplicate":
        fix("category", "needs_info" if out["needs_clarification"] else "bug", "no original ticket identified")

    # --- security / data exposure (KB-2914, KB-6401)
    if out["category"] == "security" or out["route_to"] == "security":
        fix("category", "security", "security report")
        fix("route_to", "security", "KB-2914: security never goes to L1")
        fix("base_priority", "P1", "KB-6401: data exposure is P1 regardless of tone")
        fix("escalate", True, "KB-2914: security reports escalate")

    # --- insufficient information
    if out["category"] == "needs_info" or out["needs_clarification"]:
        fix("needs_clarification", True, "needs_info category")
        if out["category"] in ("needs_info",):
            fix("base_priority", "P3", "KB-2914: insufficient information -> P3")
            fix("route_to", "L1_support", "clarification is handled by L1")

    # --- feature requests / by-design limits go to Product
    if out["category"] == "feature_request" and out["route_to"] != "product":
        fix("route_to", "product", "KB-2914: feature requests and by-design limits -> Product")

    # --- active incident linked -> L2
    if facts.get("active_incidents") and out["category"] == "incident" and out["route_to"] != "L2_engineering":
        fix("route_to", "L2_engineering", "KB-2914: known active incident -> L2")

    # --- account risk override (KB-2914); not applied to duplicates (keep original priority)
    priority = out["base_priority"]
    risk = facts.get("risk")
    if risk is None and out.get("account_id"):
        risk = toolbox.evaluate_account_risk(out["account_id"])
        facts["risk"] = risk
        facts.setdefault("guardrail_steps", []).append(("evaluate_account_risk", out["account_id"], risk))
    if risk and risk.get("override_applies") and not out.get("duplicate_of") and out["category"] != "security":
        new = bump(priority)
        if new != priority:
            notes.append(f"priority: {priority} -> {new} (KB-2914 account risk override)")
            priority = new
        fix("notify_tam", True, "KB-2914 account risk override")
        fix("escalate", True, "KB-2914 account risk override")
        sig = f"Account risk override: renewal in {risk['days_to_renewal']} days"
        if risk.get("usage_decline_gt_25pct"):
            sig += f", active users {risk['active_users_change_3m_pct']}% over 3 months"
        if risk.get("two_or_more_escalations"):
            sig += f", {risk['open_escalations']} open escalations"
        if not any("override" in s.lower() for s in out["account_signals"]):
            out["account_signals"].append(sig)
    out["priority"] = priority
    if priority == "P1" and not out.get("duplicate_of"):
        fix("escalate", True, "KB-2914: P1 escalates")

    # --- KB references: only real, non-deprecated articles; cite the policies applied
    refs = []
    for r in out["kb_references"]:
        r = r.strip().upper()
        a = toolbox.kb.articles.get(r)
        if not a:
            notes.append(f"dropped unknown KB reference {r}")
            continue
        if a["meta"].get("status") == "deprecated":
            notes.append(f"replaced deprecated {r} with {a['meta'].get('superseded_by')}")
            r = a["meta"].get("superseded_by") or r
        if r not in refs:
            refs.append(r)
    required = [POLICY_KB]
    if out.get("duplicate_of"):
        required.append(DUPLICATE_KB)
    if out["category"] == "security":
        required.append(EXPOSURE_KB)
    for r in required:
        if r not in refs:
            refs.append(r)
    out["kb_references"] = refs

    # --- never leak other customers' identifiers in the reply
    own = out.get("account_id") or ""
    reply = out.get("customer_reply") or ""
    leaked = sorted({m for m in re.findall(r"ACC-\d{4}", reply) if m != own})
    if leaked:
        reply = re.sub(r"ACC-\d{4}", lambda m: m.group(0) if m.group(0) == own else "[redacted]", reply)
        notes.append(f"redacted other account ids from reply: {leaked}")
    out["customer_reply"] = reply

    try:
        c = float(out.get("confidence", 0.5))
    except (TypeError, ValueError):
        c = 0.5
    out["confidence"] = round(min(max(c, 0.0), 1.0), 2)
    corrections = [n for n in notes if not n.startswith("priority:")]  # the bump itself is by design
    if corrections:  # the model needed correcting: be less sure
        out["confidence"] = round(min(out["confidence"], 0.9 - 0.05 * min(len(corrections), 6)), 2)
    return out, notes
