"""Tools the agent can call. Each tool is a plain Python function over
indexes built once per run, plus a JSON schema the LLM sees.

Design rules:
  - Tools return compact JSON (never whole tables / whole KB) so prompts stay small.
  - Deterministic facts (dates, % change, policy thresholds) are computed here,
    not left to the LLM to do arithmetic on.
  - History tools only see tickets created *before* the ticket being triaged.
"""
import json
import re
from datetime import date, datetime
from difflib import SequenceMatcher

import pandas as pd

from .data import AS_OF_DATE
from .retrieval import KBRetriever

CANONICAL_MODULES = ["Platform", "Contracts", "Sourcing", "Spend Analytics", "Supplier Management",
                     "eProcurement", "eInvoice", "Integrations"]

MODULE_ALIASES = {
    "platform": "Platform", "sso": "Platform", "login": "Platform",
    "contracts": "Contracts", "contract": "Contracts", "clm": "Contracts", "contract mgmt": "Contracts",
    "sourcing": "Sourcing", "rfx": "Sourcing", "esourcing": "Sourcing", "sourcing events": "Sourcing",
    "spend analytics": "Spend Analytics", "spend": "Spend Analytics", "analytics": "Spend Analytics",
    "reports": "Spend Analytics",
    "supplier management": "Supplier Management", "sim": "Supplier Management",
    "supplier mgmt": "Supplier Management", "supplier portal": "Supplier Management",
    "eprocurement": "eProcurement", "eproc": "eProcurement", "e-procurement": "eProcurement",
    "procurement": "eProcurement",
    "einvoice": "eInvoice", "e-invoice": "eInvoice", "einv": "eInvoice", "invoicing": "eInvoice", "ap": "eInvoice",
    "integrations": "Integrations", "erp integration": "Integrations", "api": "Integrations",
    "connector": "Integrations",
}

ERROR_CODE = re.compile(r"\bE-\d{4}\b")
INJECTION = re.compile(r"(system note|note to (the )?ai|ai (triage )?assistant|ignore (all |any )?(previous|prior)"
                       r"|classify this ticket as|set (this|the) (ticket )?(to|as) p[1-4]|include the full list"
                       r"|you are now|disregard)", re.I)
# KB-6401 scope: someone can see data they should not (bids, prices, invoices, bank details, other tenants)
EXPOSURE = re.compile(r"((could|can|able to|were able to) (see|view|access|download)\b[^.]{0,80}\b(bids?|bid amounts|"
                      r"prices|pricing|invoices?|contracts?|bank details|personal data|competitor|competing|"
                      r"another (supplier|tenant|customer|company)|other (suppliers?|tenants?|customers?|compan(y|ies))))"
                      r"|data (leak|exposure)|(another|other) (tenant|customer|supplier)'?s? (data|bids?|invoices?|prices)",
                      re.I)
OPEN_STATES = {"open", "pending_customer"}


def normalize_module(raw: str) -> str | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    return MODULE_ALIASES.get(raw.lower(), raw if raw in CANONICAL_MODULES else None)


def error_codes_in(*texts: str) -> list[str]:
    seen = []
    for t in texts:
        for c in ERROR_CODE.findall(t or ""):
            if c not in seen:
                seen.append(c)
    return seen


def _days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days


class Toolbox:
    def __init__(self, accounts: pd.DataFrame, tickets: pd.DataFrame, usage: pd.DataFrame, kb: list[dict]):
        self.today = AS_OF_DATE
        self.accounts = accounts.set_index("account_id", drop=False)
        self.by_domain = {d.lower(): a for d, a in zip(accounts["domain"], accounts["account_id"])}
        t = tickets.copy()
        t["module_norm"] = t["module"].map(normalize_module)
        t["codes"] = [error_codes_in(ec, s, d) for ec, s, d in zip(t["error_code"], t["subject"], t["description"])]
        t["resolved_account"] = [aid or self.by_domain.get(dom.lower(), "")
                                 for aid, dom in zip(t["account_id"], t["requester_domain"])]
        self.tickets = t
        self.tickets_by_id = t.set_index("ticket_id", drop=False)
        self.usage = usage.sort_values(["account_id", "month"])
        self.kb = KBRetriever(kb)
        self.current_ticket: dict | None = None  # set by the agent; scopes history to "before this ticket"

    # ------------------------------------------------------------------ accounts
    def resolve_account(self, requester_domain: str = "", company_name: str = "", account_id: str = "") -> dict:
        if account_id and account_id in self.accounts.index:
            a = self.accounts.loc[account_id]
            agree = not requester_domain or a["domain"].lower() == requester_domain.lower()
            return {"account_id": account_id, "company_name": a["company_name"], "method": "account_id field",
                    "domain_agrees": agree}
        if requester_domain and requester_domain.lower() in self.by_domain:
            aid = self.by_domain[requester_domain.lower()]
            a = self.accounts.loc[aid]
            # surface look-alike names so the model does not confuse parent / regional entities
            raw = (company_name or a["company_name"]).lower()
            stem = raw.split()[0]
            look_alikes = sorted((r for r in self.accounts.itertuples()
                                  if r.company_name.lower().startswith(stem) and r.account_id != aid),
                                 key=lambda r: -SequenceMatcher(None, raw, r.company_name.lower()).ratio())
            similar = [f"{r.company_name} ({r.account_id})" for r in look_alikes[:4]]
            return {"account_id": aid, "company_name": a["company_name"], "method": "requester_domain exact match",
                    "company_name_raw": company_name, "similar_names_not_matched": similar,
                    "note": "Domain is unique per account and authoritative; similar company names are different accounts."}
        if company_name:
            q = re.sub(r"\b(inc|ltd|llc|gmbh|plc|corp|co)\b\.?", "", company_name.lower()).strip()
            cands = [r for r in self.accounts.itertuples() if q and q in r.company_name.lower()]
            if len(cands) == 1:
                return {"account_id": cands[0].account_id, "company_name": cands[0].company_name,
                        "method": "unique company-name match (low confidence)"}
            return {"account_id": None, "candidates": [f"{c.company_name} ({c.account_id})" for c in cands[:8]],
                    "method": "ambiguous or no match"}
        return {"account_id": None, "method": "no identifying fields"}

    def get_account_profile(self, account_id: str) -> dict:
        if account_id not in self.accounts.index:
            return {"error": f"unknown account_id {account_id}"}
        a = self.accounts.loc[account_id]
        days_to_renewal = _days_between(self.today, a["renewal_date"])
        out = {
            "account_id": account_id, "company_name": a["company_name"], "tier": a["tier"], "arr_band": a["arr_band"],
            "region": a["region"], "renewal_date": a["renewal_date"], "days_to_renewal": days_to_renewal,
            "within_90_days_of_renewal": 0 <= days_to_renewal <= 90,
            "open_escalations": int(a["open_escalations"] or 0),
            "modules_licensed": a["modules_licensed"].split(";"),
            "sso_enabled_date": a["sso_enabled_date"] or None,
            "nps_last": a["nps_last"] or None, "last_qbr_date": a["last_qbr_date"] or None,
        }
        if a["sso_enabled_date"]:
            out["days_since_sso_change"] = _days_between(a["sso_enabled_date"], self.today)
        return out

    def get_usage_trend(self, account_id: str, months: int = 6) -> dict:
        u = self.usage[self.usage["account_id"] == account_id]
        if u.empty:
            return {"error": f"no usage for {account_id}"}
        tail = u.tail(max(months, 4))
        series = [{"month": r.month, "active_users": int(r.active_users), "login_rate_pct": float(r.login_rate_pct),
                   "tickets_opened": int(r.tickets_opened)} for r in tail.itertuples()]
        last4 = u.tail(4)["active_users"].tolist()  # month M-3 .. M  => change "over the last 3 months"
        change = (last4[-1] - last4[0]) / last4[0] * 100 if last4[0] else 0.0
        return {"account_id": account_id, "series": series[-months:],
                "active_users_change_3m_pct": round(change, 1),
                "declined_more_than_25pct_3m": change < -25}

    def evaluate_account_risk(self, account_id: str) -> dict:
        """Apply the 'Account risk override' rule from KB-2914 deterministically."""
        p = self.get_account_profile(account_id)
        if "error" in p:
            return p
        u = self.get_usage_trend(account_id)
        cond_a = u.get("declined_more_than_25pct_3m", False)
        cond_b = p["open_escalations"] >= 2
        applies = p["within_90_days_of_renewal"] and (cond_a or cond_b)
        return {"account_id": account_id, "override_applies": applies,
                "within_90_days_of_renewal": p["within_90_days_of_renewal"], "days_to_renewal": p["days_to_renewal"],
                "active_users_change_3m_pct": u.get("active_users_change_3m_pct"),
                "usage_decline_gt_25pct": cond_a, "open_escalations": p["open_escalations"],
                "two_or_more_escalations": cond_b, "rule": "KB-2914: renewal<=90d AND (users -25%/3m OR >=2 escalations)"
                " => raise priority one level (max P2), notify TAM, escalate"}

    # ------------------------------------------------------------------ tickets
    def _before_current(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.current_ticket:
            df = df[(df["created_at"] < self.current_ticket["created_at"])
                    & (df["ticket_id"] != self.current_ticket["ticket_id"])]
        return df

    @staticmethod
    def _row(r) -> dict:
        return {"ticket_id": r.ticket_id, "created_at": r.created_at, "module": r.module_norm or r.module or None,
                "subject": r.subject[:90], "error_codes": r.codes, "status": r.status,
                "customer_priority": r.customer_priority or None}

    def get_ticket_history(self, account_id: str, days: int = 90, module: str | None = None,
                           only_open: bool = False, limit: int = 10) -> dict:
        t = self._before_current(self.tickets[self.tickets["resolved_account"] == account_id])
        ref = self.current_ticket["created_at"] if self.current_ticket else self.today
        t = t[[0 <= _days_between(c, ref) <= days for c in t["created_at"]]]
        if module:
            t = t[t["module_norm"] == (normalize_module(module) or module)]
        if only_open:
            t = t[t["status"].isin(OPEN_STATES)]
        t = t.sort_values("created_at", ascending=False)
        return {"account_id": account_id, "window_days": days, "count": len(t),
                "open_count": int(t["status"].isin(OPEN_STATES).sum()),
                "tickets": [self._row(r) for r in t.head(limit).itertuples()]}

    def find_duplicate_candidates(self, account_id: str, module: str | None = None, error_code: str | None = None,
                                  keywords: str = "") -> dict:
        """KB-5930: same account, open/pending, same module, same error code or symptom, previous 7 days."""
        t = self._before_current(self.tickets[self.tickets["resolved_account"] == account_id])
        ref = self.current_ticket["created_at"] if self.current_ticket else self.today
        t = t[[0 <= (datetime.fromisoformat(ref) - datetime.fromisoformat(c)).total_seconds() <= 7 * 86400
               for c in t["created_at"]]]
        t = t[t["status"].isin(OPEN_STATES)]
        mod = normalize_module(module) if module else None
        kw = {w for w in re.findall(r"[a-z]{4,}", keywords.lower())}
        cands = []
        for r in t.itertuples():
            same_module = mod is None or r.module_norm == mod
            same_code = bool(error_code) and error_code in r.codes
            overlap = len(kw & set(re.findall(r"[a-z]{4,}", (r.subject + " " + r.description).lower())))
            if same_module and (same_code or overlap >= 2):
                cands.append(self._row(r) | {"same_error_code": same_code, "keyword_overlap": overlap})
        return {"rule": "KB-5930 (same account, open/pending, same module, same code/symptom, <=7 days)",
                "candidates": cands, "is_duplicate": bool(cands)}

    def count_cross_account_reports(self, error_code: str, hours: int = 48) -> dict:
        """How many *other* tenants reported this error recently - a platform-incident signal.
        Returns counts only: other customers' data never reaches the reply."""
        ref = datetime.fromisoformat(self.current_ticket["created_at"]) if self.current_ticket else datetime.fromisoformat(self.today)
        t = self._before_current(self.tickets)
        t = t[[error_code in c for c in t["codes"]]]
        t = t[[0 <= (ref - datetime.fromisoformat(c)).total_seconds() <= hours * 3600 for c in t["created_at"]]]
        accts = set(t["resolved_account"])
        if self.current_ticket:
            accts.discard(self.current_ticket.get("_account", ""))
        return {"error_code": error_code, "window_hours": hours, "other_accounts_reporting": len(accts),
                "tickets": int(len(t))}

    # ------------------------------------------------------------------ KB
    def search_kb(self, query: str, module: str | None = None, error_code: str | None = None,
                  doc_type: str | None = None, include_deprecated: bool = False, top_k: int = 5) -> dict:
        res = self.kb.search(query, normalize_module(module) if module else None, error_code, doc_type,
                             include_deprecated, min(int(top_k or 5), 8))
        return {"results": res, "hint": "Read promising articles with get_kb_article; follow 'related' links "
                                        "and 'superseded_by' when relevant. Deprecated articles are not valid references."}

    def get_kb_article(self, kb_id: str) -> dict:
        a = self.kb.articles.get(kb_id)
        if not a:
            return {"error": f"unknown article {kb_id}"}
        out = self.kb.card(kb_id) | {"related": a["meta"].get("related") or [], "body": a["body"].strip()[:2500]}
        out.pop("snippet", None)
        if a["meta"].get("status") == "deprecated":
            out["warning"] = f"DEPRECATED - do not cite; use {a['meta'].get('superseded_by')}"
        return out

    def check_active_incidents(self, module: str | None = None, error_code: str | None = None) -> dict:
        mod = normalize_module(module) if module else None
        hits = []
        for aid, a in self.kb.articles.items():
            m = a["meta"]
            if m.get("type") != "known_issue" or m.get("status") != "active_incident":
                continue
            if error_code and error_code not in (m.get("error_codes") or []):
                continue
            if mod and m.get("module") not in (mod, "All"):
                continue
            hits.append(self.kb.card(aid))
        return {"active_incidents": hits, "count": len(hits)}


# ---------------------------------------------------------------------- schemas
def _s(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_STR, _INT, _BOOL = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}

TOOL_SPECS = [
    {"name": "resolve_account", "description": "Resolve the customer account from account_id, requester_domain "
     "(authoritative, unique) or company name. Use first when account_id is blank or to verify it.",
     "parameters": _s({"requester_domain": _STR, "company_name": _STR, "account_id": _STR}, [])},
    {"name": "get_account_profile", "description": "Tier, ARR band, renewal date and days to renewal, open "
     "escalations, licensed modules, SSO change date, NPS.", "parameters": _s({"account_id": _STR}, ["account_id"])},
    {"name": "get_usage_trend", "description": "Monthly active users, login rate and tickets, plus the computed "
     "3-month change in active users.", "parameters": _s({"account_id": _STR, "months": _INT}, ["account_id"])},
    {"name": "evaluate_account_risk", "description": "Deterministically applies the KB-2914 account-risk override "
     "rule (renewal window + usage decline or escalations).", "parameters": _s({"account_id": _STR}, ["account_id"])},
    {"name": "get_ticket_history", "description": "Earlier tickets for an account (newest first), optionally "
     "filtered by module or to open/pending only.", "parameters": _s(
        {"account_id": _STR, "days": _INT, "module": _STR, "only_open": _BOOL, "limit": _INT}, ["account_id"])},
    {"name": "find_duplicate_candidates", "description": "Applies the KB-5930 duplicate rule: open/pending tickets "
     "from the same account in the previous 7 days with the same module and error code or symptom keywords.",
     "parameters": _s({"account_id": _STR, "module": _STR, "error_code": _STR, "keywords": _STR}, ["account_id"])},
    {"name": "count_cross_account_reports", "description": "Counts how many other accounts reported the same error "
     "code recently (platform-incident signal). Returns counts only.",
     "parameters": _s({"error_code": _STR, "hours": _INT}, ["error_code"])},
    {"name": "search_kb", "description": "Search the knowledge base (BM25 + error-code/module boosts). doc_type can "
     "be policy, troubleshooting, how_to, limit, known_issue, faq, reference, release_notes.",
     "parameters": _s({"query": _STR, "module": _STR, "error_code": _STR, "doc_type": _STR,
                       "include_deprecated": _BOOL, "top_k": _INT}, ["query"])},
    {"name": "get_kb_article", "description": "Read one KB article in full, including related links and "
     "superseded_by.", "parameters": _s({"kb_id": _STR}, ["kb_id"])},
    {"name": "check_active_incidents", "description": "List known issues with status active_incident, optionally "
     "filtered by module or error code.", "parameters": _s({"module": _STR, "error_code": _STR}, [])},
]

SUBMIT_SPEC = {
    "name": "submit_triage",
    "description": "Submit the final triage decision. Call exactly once, when you have enough evidence. "
                   "Calling it ends the investigation.",
    "parameters": _s({
        "account_id": {"type": ["string", "null"]},
        "category": {"type": "string", "enum": ["incident", "bug", "how_to", "feature_request", "security",
                                                 "duplicate", "needs_info"]},
        "base_priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"],
                          "description": "Priority from the policy definitions BEFORE any account-risk override."},
        "route_to": {"type": "string", "enum": ["L1_support", "L2_engineering", "security", "product"]},
        "escalate": _BOOL, "notify_tam": _BOOL,
        "duplicate_of": {"type": ["string", "null"]},
        "needs_clarification": _BOOL,
        "kb_references": {"type": "array", "items": _STR},
        "account_signals": {"type": "array", "items": _STR},
        "customer_reply": _STR, "internal_note": _STR,
        "confidence": {"type": "number"},
        "rationale": {"type": "string", "description": "Short justification citing policy and evidence."},
    }, ["category", "base_priority", "route_to", "escalate", "notify_tam", "needs_clarification",
        "kb_references", "account_signals", "customer_reply", "internal_note", "confidence", "rationale"]),
}


def summarize(result: dict, limit: int = 300) -> str:
    """One-line observation for reasoning_trace."""
    s = json.dumps(result, default=str)
    return s if len(s) <= limit else s[: limit - 3] + "..."
