"""Cross-check ir_metrics against pytrec_eval and ranx (tol 1e-6).

Run: .venv/bin/python -m pytest test_ir_metrics.py -q
The SciFact test needs data/scifact (python load_data.py scifact).
"""
import math

import pytest
import pytrec_eval
from ranx import Qrels, Run, evaluate as ranx_evaluate

import ir_metrics as M

TOL = 1e-6
KS = (1, 5, 10)

# Toy: graded qrels (incl. unretrieved relevant docs, a zero-relevant-free mix).
TOY_QRELS = {
    "q1": {"a": 3, "b": 2, "c": 3, "d": 0, "e": 1, "f": 2, "z": 1},
    "q2": {"a": 1, "x": 1, "y": 1},
    "q3": {"m": 2, "n": 0},
}
TOY_RUN = {
    "q1": {"a": 6.0, "b": 5.0, "c": 4.0, "d": 3.0, "e": 2.0, "f": 1.0, "g": 0.5, "h": 0.4, "i": 0.3, "j": 0.2, "k": 0.1},
    "q2": {"p": 9.0, "a": 8.0, "q": 7.0, "x": 6.0, "r": 5.0, "s": 4.0, "t": 3.0, "u": 2.0, "v": 1.0, "w": 0.5, "y": 0.1},
    "q3": {"n": 5.0, "o": 4.0, "m": 3.0, "p": 2.0, "q": 1.5, "r": 1.0, "s": 0.9, "t": 0.8, "u": 0.7, "v": 0.6, "w": 0.5},
}


def _pytrec_means(qrels, run):
    measures = {"P_1", "P_5", "P_10", "recall_1", "recall_5", "recall_10", "success_1", "success_5",
                "success_10", "recip_rank", "map_cut_10", "ndcg_cut_1", "ndcg_cut_5", "ndcg_cut_10", "map"}
    res = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run)
    out = {}
    for m in measures:
        vals = [res[q][m] for q in qrels if q in run]
        out[m] = sum(vals) / len(vals)
    return out


def _check_against_pytrec(qrels, run):
    pt = _pytrec_means(qrels, run)
    for k in KS:
        assert M.evaluate(qrels, run, "P", k) == pytest.approx(pt[f"P_{k}"], abs=TOL)
        assert M.evaluate(qrels, run, "R", k) == pytest.approx(pt[f"recall_{k}"], abs=TOL)
        assert M.evaluate(qrels, run, "hit", k) == pytest.approx(pt[f"success_{k}"], abs=TOL)
        assert M.evaluate(qrels, run, "NDCG", k) == pytest.approx(pt[f"ndcg_cut_{k}"], abs=TOL)
    assert M.evaluate(qrels, run, "MAP", 10) == pytest.approx(pt["map_cut_10"], abs=TOL)
    assert M.evaluate(qrels, run, "MAP", None) == pytest.approx(pt["map"], abs=TOL)
    assert M.evaluate(qrels, run, "MRR", None) == pytest.approx(pt["recip_rank"], abs=TOL)


def _check_against_ranx(qrels, run):
    rq, rr = Qrels(qrels), Run(run)
    names = {"P": "precision", "R": "recall", "F1": "f1", "hit": "hit_rate", "MRR": "mrr", "MAP": "map",
             "NDCG": "ndcg", "NDCG_exp": "ndcg_burges"}
    for ours, theirs in names.items():
        for k in KS:
            got = M.evaluate(qrels, run, ours, k)
            exp = ranx_evaluate(rq, rr, f"{theirs}@{k}", make_comparable=False)
            assert got == pytest.approx(exp, abs=TOL), (ours, k)


def test_toy_vs_pytrec_eval():
    _check_against_pytrec(TOY_QRELS, TOY_RUN)


def test_toy_vs_ranx():
    _check_against_ranx(TOY_QRELS, TOY_RUN)


def test_hand_values():
    # Wikipedia DCG example: rels 3,2,3,0,1,2 -> DCG6 = 6.861. Its ideal ordering is 3,3,3,2,2,2,1,0, so the
    # judged set holds a 3 and a 2 that the ranking never retrieved: IDCG6 = 8.740, NDCG6 = 0.785.
    rels = {"a": 3, "b": 2, "c": 3, "d": 0, "e": 1, "f": 2, "g": 3, "h": 2}
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert M.raw_dcg_at_k(rels, ranked, 6) == pytest.approx(6.861, abs=1e-3)
    assert M.ndcg_at_k(rels, ranked, 6) == pytest.approx(0.785, abs=1e-3)
    # same ranking judged only on the six ranked docs: ideal 3,3,2,2,1,0 -> IDCG6 = 7.141, NDCG6 = 0.961
    six = {k: v for k, v in rels.items() if k in ranked}
    assert M.ndcg_at_k(six, ranked, 6) == pytest.approx(6.861 / 7.141, abs=1e-3)
    # binary: relevant at ranks 1 and 4 of R=3 -> AP = (1/1 + 2/4)/3
    b = {"r1": 1, "r2": 1, "r3": 1}
    rk = ["r1", "x", "y", "r2", "z"]
    assert M.ap_at_k(b, rk) == pytest.approx((1 + 0.5) / 3)
    assert M.precision_at_k(b, rk, 5) == pytest.approx(0.4)
    assert M.recall_at_k(b, rk, 5) == pytest.approx(2 / 3)
    assert M.f1_at_k(b, rk, 5) == pytest.approx(2 * 0.4 * (2 / 3) / (0.4 + 2 / 3))
    assert M.rr_at_k(b, ["x", "y", "r2"]) == pytest.approx(1 / 3)
    # linear vs exponential gain must differ on graded data
    assert M.ndcg_at_k(rels, ["f", "e", "d", "c", "b", "a"], 6, "linear") != pytest.approx(
        M.ndcg_at_k(rels, ["f", "e", "d", "c", "b", "a"], 6, "exp"), abs=1e-4)


@pytest.fixture(scope="module")
def scifact_bm25():
    import bm25s
    import numpy as np
    from common import load_dataset
    corpus, queries, qrels = load_dataset("scifact")
    ids = list(corpus)
    docs = [(corpus[i].get("title", "") + " " + corpus[i]["text"]) for i in ids]
    ret = bm25s.BM25(k1=1.2, b=0.75)
    ret.index(bm25s.tokenize(docs, stopwords="en", show_progress=False), show_progress=False)
    qids = [q for q in qrels if q in queries]
    res, sc = ret.retrieve(bm25s.tokenize([queries[q] for q in qids], stopwords="en", show_progress=False),
                           k=100, show_progress=False, n_threads=1)
    run = {q: {ids[int(res[i, j])]: float(sc[i, j]) for j in range(res.shape[1])} for i, q in enumerate(qids)}
    return {q: qrels[q] for q in qids}, run


def test_scifact_bm25_vs_pytrec_eval(scifact_bm25):
    qrels, run = scifact_bm25
    _check_against_pytrec(qrels, run)


def test_scifact_bm25_vs_ranx(scifact_bm25):
    qrels, run = scifact_bm25
    _check_against_ranx(qrels, run)
    ndcg10 = M.evaluate(qrels, run, "NDCG", 10)
    print(f"SciFact bm25s NDCG@10={ndcg10:.4f} R@100={M.evaluate(qrels, run, 'R', 100):.4f}")
    assert 0.55 < ndcg10 < 0.75  # loose plausibility vs 0.679 published (not a repro claim)
