"""Lightweight KB retrieval: BM25 over title + body with metadata boosts.

No external dependency. The index is built once per run (~390 articles in
well under a second) and each query only touches the posting lists of its
terms, so cost per query grows with the vocabulary hit, not the KB size.
"""
import math
import re
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-z0-9]+(?:-[0-9]+)?")
_STOP = set("""a an and are as at be but by can do does for from has have how i if in into is it its
of on or our so that the their them there these they this to was we were what when where which who
why will with you your not no any all some been being than then too very just also""".split())


def tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    out = []
    for t in toks:
        if t in _STOP or len(t) < 2:
            continue
        # crude plural / verb stemming keeps "approvals" ~ "approval", "failing" ~ "fail"
        for suf in ("ing", "ies", "es", "s", "ed"):
            if len(t) > 4 and t.endswith(suf) and not t.startswith("e-"):
                t = t[: -len(suf)] + ("y" if suf == "ies" else "")
                break
        out.append(t)
    return out


class BM25Index:
    def __init__(self, docs: dict[str, str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: dict[str, Counter] = {}
        self.len: dict[str, int] = {}
        self.postings: dict[str, set] = defaultdict(set)
        for doc_id, text in docs.items():
            toks = tokenize(text)
            self.tf[doc_id] = Counter(toks)
            self.len[doc_id] = len(toks)
            for t in self.tf[doc_id]:
                self.postings[t].add(doc_id)
        self.n = len(docs)
        self.avgdl = sum(self.len.values()) / max(self.n, 1)

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf(term)
            for doc_id in self.postings.get(term, ()):
                f = self.tf[doc_id][term]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[doc_id] / self.avgdl)
                out[doc_id] += idf * f * (self.k1 + 1) / denom
        return out


class KBRetriever:
    """BM25 + metadata-aware re-ranking.

    - exact error-code match in front matter is a strong signal (codes are precise)
    - module match is a mild boost (labels in tickets are noisy)
    - deprecated articles are demoted and always point at their replacement
    """

    def __init__(self, kb: list[dict]):
        self.articles = {a["id"]: a for a in kb}
        # title is repeated to weight it above body text
        self.index = BM25Index({a["id"]: f"{a['meta'].get('title', '')} " * 3 + a["body"] for a in kb})

    def search(self, query: str, module: str | None = None, error_code: str | None = None,
               doc_type: str | None = None, include_deprecated: bool = False, top_k: int = 5) -> list[dict]:
        scores = self.index.scores(f"{query} {error_code or ''}")
        if error_code:
            for aid, a in self.articles.items():
                if error_code in (a["meta"].get("error_codes") or []):
                    scores[aid] = scores.get(aid, 0) + 15.0
        ranked = []
        for aid, s in scores.items():
            meta = self.articles[aid]["meta"]
            # doc_type is a hard filter only for policy; otherwise a boost, because models often guess the
            # wrong type (e.g. "troubleshooting" for a by-design "limit" article)
            if doc_type == "policy" and meta.get("type") != "policy":
                continue
            if doc_type and meta.get("type") == doc_type:
                s *= 1.3
            if module and meta.get("module") in (module, "All"):
                s *= 1.25
            if meta.get("status") == "deprecated":
                if not include_deprecated:
                    continue
                s *= 0.3
            if meta.get("status") == "active_incident":
                s *= 1.2
            ranked.append((s, aid))
        ranked.sort(reverse=True)
        return [self.card(aid, score) for score, aid in ranked[:top_k]]

    def card(self, aid: str, score: float | None = None) -> dict:
        a = self.articles[aid]
        m = a["meta"]
        body = re.sub(r"\s+", " ", re.sub(r"^#.*$", "", a["body"], flags=re.M)).strip()
        out = {"id": aid, "title": m.get("title"), "module": m.get("module"), "type": m.get("type"),
               "status": m.get("status"), "error_codes": m.get("error_codes") or [],
               "last_updated": m.get("last_updated"), "snippet": body[:220]}
        if m.get("superseded_by"):
            out["superseded_by"] = m["superseded_by"]
        if score is not None:
            out["score"] = round(score, 2)
        return out
