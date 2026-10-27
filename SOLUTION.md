# SOLUTION: Agentic ticket triage

## 1. Design
```
ticket ─► pre-parse (module alias, error codes in text, injection markers)
       ─► ReAct loop: LLM ⇄ tools (native tool calling, ≤12 calls) ──► submit_triage
       ─► policy guardrails (KB-2914/5930/6401, deterministic) ─► output + reasoning_trace
```
- **The LLM drives.** Each turn it sees the ticket, the earlier observations and the tool catalogue. It picks the next tool or calls `submit_triage`. Nothing runs in a fixed order. A how-to ticket might take only `search_kb`. An E-3310 bid failure goes `get_account_profile` (SSO changed 6 days ago) → `search_kb` → `get_kb_article KB-4795` → follows `related` → `KB-7889`.
- **Tools** (`tam_agent/tools.py`):
  - `resolve_account`: by domain (authoritative), and lists look-alike names it did *not* match.
  - `get_account_profile`, `get_usage_trend`, `evaluate_account_risk`.
  - `get_ticket_history`, `find_duplicate_candidates`: the KB-5930 rule, after module aliases are normalised and codes are pulled from the description.
  - `count_cross_account_reports`: platform-incident signal. It returns counts only.
  - `search_kb`: BM25 with error-code and module boosts. Deprecated articles are hidden.
  - `get_kb_article`: returns the related article and `superseded_by` links.
  - `check_active_incidents`.

  The tools do the arithmetic (days to renewal, 3-month % change), so the model doesn't compute it. History tools only see tickets created *before* the ticket being triaged, which prevents leakage from the future.
- **Small-model robustness.** Local models sometimes write a tool call as JSON text, or announce "I will call submit_triage" without calling it. The loop parses tool calls written as text. On a stall it offers only `submit_triage`. A data-exposure detector (KB-6401 scope) flags security tickets in code, so they are handled correctly even if the model fails.
- **Stopping.** The model stops when it calls `submit_triage`. Repeated identical calls return a cached result with a "repeated call" note. When the budget runs out, only `submit_triage` is offered. If the model still doesn't submit, a conservative `needs_info` fallback is produced (confidence 0.2, flagged for a human).
- **Guardrails** (`policy.py`). The model proposes a `base_priority`. Code then applies the hard rules:
  - Risk override: +1 level, capped at P2, plus notify the TAM and escalate.
  - Security → P1, security route, escalate.
  - Duplicates keep their priority and are not re-escalated.
  - P1 → escalate.
  - Deprecated KB references are swapped for their replacements.
  - Other customers' account IDs are redacted from replies.

  Each correction is written into `reasoning_trace` and lowers confidence. Ticket text sits inside `<ticket>` tags, is treated as untrusted, and any injection is flagged in the trace.

## 2. Evaluation
- Build a labelled gold set of 200–300 tickets, stratified by category, tier and the known traps: duplicates, blank account, injection, deprecated KB, active incident. Have two TAMs label them and measure their agreement.
- Per-field metrics: category macro-F1; priority accuracy plus a cost-weighted confusion matrix (P1 missed ≫ P4 over-called); routing accuracy; precision/recall for escalate and notify_tam.
- Hard-rule recall must be 100% (security → Security/P1; risk override → TAM). Run these as CI tests (`tests/`).
- Retrieval: recall@5 and MRR of the gold KB article, plus the deprecated-citation rate (target 0).
- Trace quality: tool calls per ticket, how often guardrails fire (a proxy for model error), and an LLM-as-judge rubric for reply quality, spot-checked by humans.
- Shadow mode for 2–4 weeks next to human triage. Track agreement, time to first response and re-route rate before turning on automation.

## 3. Scale (30k tickets, 4k articles)
- **Prompt size doesn't change.** Tools return top-k cards and capped bodies, so prompt tokens per ticket stay flat, roughly 6–15k with 4–8 tool calls. Cost and latency grow with how many tool calls a ticket needs, not with how much data there is. Local 7B: about 20–60 s per ticket. API model: about 10–20 s and about $0.02–0.05 per ticket.
- **Data access.** Pandas filters are O(n) per call. At 30k tickets that's still milliseconds, but I'd move to SQLite/Postgres with indexes on (account_id, created_at) and on error_code.
- **Retrieval is the real risk.** With 4k articles, BM25 alone loses precision on paraphrased symptoms. I'd add hybrid retrieval: BM25 plus embeddings in a vector DB, a cross-encoder re-ranker, metadata filters (module, status, type), and an index refresh whenever an article changes. I'd also cache the policy articles and use prompt caching for the system prompt and tool schemas.
- **Throughput.** Run tickets in parallel (they're independent). Send easy tickets to a small model and only hard or low-confidence ones to a larger model.

## 4. Limitations
- I haven't measured accuracy beyond the 10 review tickets. The keyword thresholds for duplicates are heuristic.
- Small local models sometimes skip reading the policy or format arguments badly. Guardrails catch the hard rules, but category and routing nuance (e.g. config issue → L1 vs defect → L2) still depends on the model.
- An original ticket's priority isn't stored, so a duplicate uses the model's priority.
- No multi-turn clarification and no real incident-system link. Replies are drafts that need a human to send them.

## 5. How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # + `pip install anthropic` or `openai` if used
cp .env.example .env                       # set LLM_PROVIDER / LLM_MODEL (+ API key)
# local & free: install Ollama, then `ollama pull qwen2.5:7b`
python scripts/validate_data.py
python -m pytest -q                        # offline tests (scripted LLM)
python run.py --set review                 # writes outputs/*.json + outputs/run_log.jsonl
python scripts/check_outputs.py            # must show 10/10 valid
```
