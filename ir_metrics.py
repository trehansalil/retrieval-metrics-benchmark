"""Own implementations of standard IR metrics (E1).

Conventions (matching trec_eval so results can be cross-checked):
- qrels: {qid: {docid: int_grade}}; run: {qid: {docid: float_score}}.
- A document is relevant when grade > 0.
- Ranking: score descending, ties broken by docid descending (trec_eval).
- Metrics are computed per query, then averaged over the queries in qrels
  that also appear in the run (queries with no relevant docs score 0.0).
- AP@k / MAP divides by R (all relevant docs in qrels), not by hits.
- DCG linear gain: rel / log2(i + 1)   (trec_eval, ranx `ndcg`).
- DCG exponential gain: (2**rel - 1) / log2(i + 1)   (Burges, ranx `ndcg_burges`).
- IDCG is built from the full qrels for the query, cut at k.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

Qrels = Dict[str, Dict[str, int]]
Run = Dict[str, Dict[str, float]]


def rank_docs(scores: Dict[str, float]) -> List[str]:
    """Doc ids sorted by score desc, ties by docid desc (trec_eval)."""
    return [d for d, _ in sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)]


def _topk(ranked: List[str], k: Optional[int]) -> List[str]:
    return ranked if k is None else ranked[:k]


def _n_rel(rels: Dict[str, int]) -> int:
    return sum(1 for g in rels.values() if g > 0)


def precision_at_k(rels, ranked, k):
    return sum(1 for d in _topk(ranked, k) if rels.get(d, 0) > 0) / k


def recall_at_k(rels, ranked, k):
    r = _n_rel(rels)
    return sum(1 for d in _topk(ranked, k) if rels.get(d, 0) > 0) / r if r else 0.0


def f1_at_k(rels, ranked, k):
    p, r = precision_at_k(rels, ranked, k), recall_at_k(rels, ranked, k)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def hit_rate_at_k(rels, ranked, k):
    return 1.0 if any(rels.get(d, 0) > 0 for d in _topk(ranked, k)) else 0.0


def rr_at_k(rels, ranked, k=None):
    for i, d in enumerate(_topk(ranked, k), start=1):
        if rels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ap_at_k(rels, ranked, k=None):
    r = _n_rel(rels)
    if not r:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(_topk(ranked, k), start=1):
        if rels.get(d, 0) > 0:
            hits += 1
            total += hits / i
    return total / r


def _gain(grade: int, gain: str) -> float:
    if grade <= 0:
        return 0.0
    if gain == "linear":
        return float(grade)
    if gain == "exp":
        return float(2 ** grade - 1)
    raise ValueError(f"gain must be 'linear' or 'exp', got {gain!r}")


def dcg_at_k(gains: List[float], k: Optional[int] = None) -> float:
    g = gains if k is None else gains[:k]
    return sum(x / math.log2(i + 1) for i, x in enumerate(g, start=1))


def ndcg_at_k(rels, ranked, k, gain="linear"):
    got = dcg_at_k([_gain(rels.get(d, 0), gain) for d in _topk(ranked, k)], k)
    ideal = sorted((_gain(g, gain) for g in rels.values()), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return got / idcg if idcg > 0 else 0.0


def raw_dcg_at_k(rels, ranked, k, gain="linear"):
    return dcg_at_k([_gain(rels.get(d, 0), gain) for d in _topk(ranked, k)], k)


# name -> per-query function(rels, ranked, k)
METRICS: Dict[str, Callable] = {
    "P": precision_at_k,
    "R": recall_at_k,
    "F1": f1_at_k,
    "hit": hit_rate_at_k,
    "MRR": lambda rels, ranked, k: rr_at_k(rels, ranked, k),
    "MAP": lambda rels, ranked, k: ap_at_k(rels, ranked, k),
    "NDCG": lambda rels, ranked, k: ndcg_at_k(rels, ranked, k, "linear"),
    "NDCG_exp": lambda rels, ranked, k: ndcg_at_k(rels, ranked, k, "exp"),
    "DCG": lambda rels, ranked, k: raw_dcg_at_k(rels, ranked, k, "linear"),
    "DCG_exp": lambda rels, ranked, k: raw_dcg_at_k(rels, ranked, k, "exp"),
}


def per_query(qrels: Qrels, run: Run, metric: str, k: Optional[int] = None) -> Dict[str, float]:
    """Per-query scores for `metric` (key of METRICS) at cutoff k."""
    fn = METRICS[metric]
    out = {}
    for qid, rels in qrels.items():
        if qid not in run:
            continue
        out[qid] = fn(rels, rank_docs(run[qid]), k)
    return out


def evaluate(qrels: Qrels, run: Run, metric: str, k: Optional[int] = None) -> float:
    """Mean of the per-query scores."""
    pq = per_query(qrels, run, metric, k)
    return sum(pq.values()) / len(pq) if pq else 0.0
