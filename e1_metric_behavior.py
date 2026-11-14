"""E1 metric behaviour: same rankings, many metrics, find where metrics disagree.

Run: .venv/bin/python e1_metric_behavior.py
Writes results/e1_<dataset>_<ts>.json via common.write_result (toy sanity rows go into the scifact file).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytrec_eval
from ranx import Qrels, Run, evaluate as ranx_evaluate

sys.path.insert(0, str(Path(__file__).parent))
import ir_metrics as M
from common import load_dataset, set_seed, write_result

SEED = 42
TOL = 1e-6
DEPTH = 100


# ---------- helpers ----------
def err_at_k(rels, ranked, k, gmax):
    """Chapelle et al. 2009 ERR; R_i = (2^g - 1)/2^gmax."""
    p, total = 1.0, 0.0
    for i, d in enumerate(ranked[:k], start=1):
        g = rels.get(d, 0)
        r = (2 ** g - 1) / 2 ** gmax if g > 0 else 0.0
        total += p * r / i
        p *= 1 - r
    return total


def rank_run(ranked_lists):
    """{qid: [docids]} -> {qid: {doc: unique descending score}} so every library sees the SAME order."""
    return {q: {d: float(len(r) - i) for i, d in enumerate(r)} for q, r in ranked_lists.items()}


def our_per_query(qrels, ranked_lists):
    run = rank_run(ranked_lists)
    spec = {
        "P@10": ("P", 10), "R@10": ("R", 10), "R@100": ("R", 100), "Hit@10": ("hit", 10),
        "MRR": ("MRR", None), "MRR@10": ("MRR", 10), "MAP@10": ("MAP", 10), "MAP@100": ("MAP", 100),
        "NDCG@10": ("NDCG", 10), "NDCG@10_exp": ("NDCG_exp", 10),
    }
    return {name: M.per_query(qrels, run, m, k) for name, (m, k) in spec.items()}


def crosscheck(qrels, ranked_lists):
    """Max abs diff of ours vs pytrec_eval (per query) and ranx (per query and mean)."""
    run = rank_run(ranked_lists)
    ours = our_per_query(qrels, ranked_lists)
    pt_map = {"P@10": "P_10", "R@10": "recall_10", "R@100": "recall_100", "Hit@10": "success_10",
              "MRR": "recip_rank", "MAP@10": "map_cut_10", "MAP@100": "map_cut_100", "NDCG@10": "ndcg_cut_10"}
    res = pytrec_eval.RelevanceEvaluator(qrels, set(pt_map.values())).evaluate(run)
    out = {}
    for name, key in pt_map.items():
        out[f"pytrec_eval:{name}"] = max(abs(ours[name][q] - res[q][key]) for q in ours[name])
    rx = {"P@10": "precision@10", "R@10": "recall@10", "R@100": "recall@100", "Hit@10": "hit_rate@10",
          "MRR": "mrr", "MRR@10": "mrr@10", "MAP@10": "map@10", "MAP@100": "map@100",
          "NDCG@10": "ndcg@10", "NDCG@10_exp": "ndcg_burges@10"}
    rq, rr = Qrels(qrels), Run(run)
    qids = rr.get_query_ids()
    for name, m in rx.items():
        mean = ranx_evaluate(rq, rr, m, make_comparable=False)
        pq = rr.scores[m]
        if isinstance(pq, dict):  # ranx stores {qid: score} after evaluate
            out[f"ranx:{name}"] = max(abs(ours[name][q] - float(pq[q])) for q in ours[name])
        else:
            out[f"ranx:{name}"] = max(abs(ours[name][q] - float(s)) for q, s in zip(qids, pq))
        out[f"ranx_mean:{name}"] = abs(sum(ours[name].values()) / len(ours[name]) - mean)
    return out


def means(pq):
    return {m: float(np.mean(list(v.values()))) for m, v in pq.items()}


def labels(rels, ranked, k=10):
    return [rels.get(d, 0) for d in ranked[:k]]


# ---------- Part 1: toy + Wikipedia ----------
def toy_part():
    out = {}
    # 00a toy: judged {3,3,2,2,1}; system top-5 labels [3,2,0,1,2]; so exactly one 3 is never retrieved
    qr = {"d1": 3, "d2": 2, "d3": 0, "d4": 1, "d5": 2, "u3": 3}
    rk = ["d1", "d2", "d3", "d4", "d5"]
    ideal = sorted(qr, key=lambda d: -qr[d])
    out["ndcg5_linear"] = M.ndcg_at_k(qr, rk, 5, "linear")
    out["ndcg5_exp"] = M.ndcg_at_k(qr, rk, 5, "exp")
    out["err5_gmax3"] = err_at_k(qr, rk, 5, 3)
    out["dcg5_linear"] = M.raw_dcg_at_k(qr, rk, 5, "linear")
    out["dcg5_exp"] = M.raw_dcg_at_k(qr, rk, 5, "exp")
    out["idcg5_linear"] = M.raw_dcg_at_k(qr, ideal, 5, "linear")
    out["idcg5_exp"] = M.raw_dcg_at_k(qr, ideal, 5, "exp")
    b = {"r1": 1, "r2": 1, "r3": 1}
    rb = ["x", "r1", "y", "r2", "z"]
    out["P@5"] = M.precision_at_k(b, rb, 5)
    out["R@5"] = M.recall_at_k(b, rb, 5)
    out["F1@5"] = M.f1_at_k(b, rb, 5)
    out["AP"] = M.ap_at_k(b, rb)
    out["AP_wrong_divisor_hits"] = M.ap_at_k(b, rb) * 3 / 2
    sc = {d: float(5 - i) for i, d in enumerate(rk)}
    rq, rr = Qrels({"q": qr}), Run({"q": sc})
    out["ranx_ndcg5_linear"] = ranx_evaluate(rq, rr, "ndcg@5", make_comparable=False)
    out["ranx_ndcg5_exp"] = ranx_evaluate(rq, rr, "ndcg_burges@5", make_comparable=False)
    out["pytrec_ndcg5_linear"] = pytrec_eval.RelevanceEvaluator({"q": qr}, {"ndcg_cut_5"}).evaluate({"q": sc})["q"]["ndcg_cut_5"]
    # Wikipedia DCG example: ranking labels 3,2,3,0,1,2. Two judged sets.
    wk = ["a", "b", "c", "d", "e", "f"]
    wq_six = {"a": 3, "b": 2, "c": 3, "d": 0, "e": 1, "f": 2}
    wq_full = {**wq_six, "g": 3, "h": 2}  # Wikipedia ideal 3,3,3,2,2,2,1,0 implies a 3 and a 2 not in the ranking
    out["wiki_dcg6"] = M.raw_dcg_at_k(wq_six, wk, 6)
    out["wiki_ndcg6_ideal_from_6_ranked_docs_only"] = M.ndcg_at_k(wq_six, wk, 6)
    out["wiki_idcg6_full"] = M.raw_dcg_at_k(wq_full, sorted(wq_full, key=lambda d: -wq_full[d]), 6)
    out["wiki_ndcg6_ideal_from_full_judged_set"] = M.ndcg_at_k(wq_full, wk, 6)
    out["wiki_ndcg6_full_ranx"] = ranx_evaluate(Qrels({"q": wq_full}), Run({"q": {d: float(6 - i) for i, d in enumerate(wk)}}),
                                                "ndcg@6", make_comparable=False)
    return out


# ---------- Part 2: systems ----------
def bm25_rankings(corpus, queries, qids):
    import bm25s
    ids = list(corpus)
    docs = [(corpus[i].get("title", "") + " " + corpus[i]["text"]) for i in ids]
    ret = bm25s.BM25(k1=1.2, b=0.75)
    ret.index(bm25s.tokenize(docs, stopwords="en", show_progress=False), show_progress=False)
    res, sc = ret.retrieve(bm25s.tokenize([queries[q] for q in qids], stopwords="en", show_progress=False),
                           k=DEPTH, show_progress=False, n_threads=1)
    out = {}
    for i, q in enumerate(qids):
        pairs = [(ids[int(res[i, j])], float(sc[i, j])) for j in range(res.shape[1])]
        pairs.sort(key=lambda x: (x[1], x[0]), reverse=True)  # deterministic ties
        out[q] = [d for d, _ in pairs]
    return out


def dense_rankings(corpus, queries, qids, model_name="BAAI/bge-small-en-v1.5"):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    model = SentenceTransformer(model_name, device="cpu")
    ids = list(corpus)
    docs = [(corpus[i].get("title", "") + " " + corpus[i]["text"]) for i in ids]
    D = model.encode(docs, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    prefix = "Represent this sentence for searching relevant passages: "
    Q = model.encode([prefix + queries[q] for q in qids], batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)
    S = Q @ D.T
    out = {}
    for i, q in enumerate(qids):
        top = np.argsort(-S[i], kind="stable")[:DEPTH]
        out[q] = [ids[j] for j in top]
    return out, model_name


def rrf(a, b, k=60):
    out = {}
    for q in a:
        s = {}
        for lst in (a[q], b[q]):
            for r, d in enumerate(lst, start=1):
                s[d] = s.get(d, 0.0) + 1.0 / (k + r)
        out[q] = [d for d, _ in sorted(s.items(), key=lambda x: (x[1], x[0]), reverse=True)][:DEPTH]
    return out


def saved_systems(ds, qids):
    """Low-memory path: reuse saved E2 stemmed BM25 and E3 bge-base runs, truncated to DEPTH, plus RRF."""
    res = Path(__file__).parent / "results"

    def top(run):
        out = {}
        for q in qids:
            s = run[q]
            out[q] = [d for d, _ in sorted(s.items(), key=lambda x: (x[1], x[0]), reverse=True)][:DEPTH]
        return out

    e2 = json.loads((res / f"e2_bm25_{ds}_selected.json").read_text())
    e3 = json.loads((res / f"e3_bge-base_{ds}.json").read_text())
    systems = {"BM25": top(e2["run"]), "dense_bge_base": top(e3["run"])}
    systems["RRF_hybrid"] = rrf(systems["BM25"], systems["dense_bge_base"])
    srcs = {"BM25": f"e2_bm25_{ds}_selected.json meta={ {k: v for k, v in e2['meta'].items() if k in ('k1', 'b', 'analyzer')} }",
            "dense": f"e3_bge-base_{ds}.json", "RRF": "k=60 over the two runs, depth 100"}
    return systems, str(e3.get("model")), srcs


# ---------- Part 3: disagreement analysis ----------
PAIR_METRICS = ["P@10", "R@10", "R@100", "Hit@10", "MRR@10", "MAP@10", "NDCG@10"]


def sign(x, eps=1e-9):
    return 0 if abs(x) < eps else (1 if x > 0 else -1)


def system_level(pq_by_sys):
    out = {}
    for a, b in itertools.combinations(pq_by_sys, 2):
        row = {}
        for m in pq_by_sys[a]:
            ma, mb = np.mean(list(pq_by_sys[a][m].values())), np.mean(list(pq_by_sys[b][m].values()))
            row[m] = {"A": float(ma), "B": float(mb), "winner": a if ma > mb else (b if mb > ma else "tie")}
        out[f"{a} vs {b}"] = row
    return out


def paired_boot(pq_a, pq_b, n=10000, seed=SEED):
    qs = sorted(pq_a)
    d = np.array([pq_a[q] - pq_b[q] for q in qs])
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    bm = d[idx].mean(1)
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}


def pair_disagreement(pq_a, pq_b, metrics):
    qs = sorted(pq_a[metrics[0]])
    signs = {m: {q: sign(pq_a[m][q] - pq_b[m][q]) for q in qs} for m in metrics}
    res = {}
    for m1, m2 in itertools.combinations(metrics, 2):
        both = [q for q in qs if signs[m1][q] != 0 and signs[m2][q] != 0]
        opp = [q for q in both if signs[m1][q] != signs[m2][q]]
        res[f"{m1}|{m2}"] = {"queries_both_nonzero": len(both), "opposite_winner": len(opp),
                              "frac": len(opp) / len(both) if both else 0.0}
    return res, signs


def first_rel(rels, ranked):
    return next((i for i, d in enumerate(ranked, 1) if rels.get(d, 0) > 0), None)


def find_examples(qrels, rk_a, rk_b, pq_a, pq_b, name_a, name_b, queries, signs, pairs, n=2):
    ex = []
    for m1, m2 in pairs:
        cands = [q for q in signs[m1] if signs[m1][q] * signs[m2][q] == -1]
        cands.sort(key=lambda q: -(abs(pq_a[m1][q] - pq_b[m1][q]) + abs(pq_a[m2][q] - pq_b[m2][q])))
        for q in cands[:n]:
            ex.append({
                "metrics": [m1, m2], "qid": q, "query": queries[q][:140], "n_relevant": M._n_rel(qrels[q]),
                f"{name_a}_labels_top10": labels(qrels[q], rk_a[q]),
                f"{name_b}_labels_top10": labels(qrels[q], rk_b[q]),
                f"{name_a}_first_rel_rank": first_rel(qrels[q], rk_a[q]),
                f"{name_b}_first_rel_rank": first_rel(qrels[q], rk_b[q]),
                m1: {name_a: pq_a[m1][q], name_b: pq_b[m1][q]},
                m2: {name_a: pq_a[m2][q], name_b: pq_b[m2][q]},
            })
    return ex


def main():
    set_seed(SEED)
    toy = toy_part()
    print(json.dumps(toy, indent=1))
    paths = []
    for ds in ["scifact", "nfcorpus"]:
        corpus, queries, qrels_all = load_dataset(ds)
        qids = sorted(q for q in qrels_all if q in queries and M._n_rel(qrels_all[q]) > 0)
        qrels = {q: qrels_all[q] for q in qids}
        print(ds, "queries", len(qids), "docs", len(corpus), flush=True)
        systems, mname, srcs = saved_systems(ds, qids)

        pq = {s: our_per_query(qrels, r) for s, r in systems.items()}
        cc = {s: crosscheck(qrels, r) for s, r in systems.items()}
        maxdiff = max(max(v.values()) for v in cc.values())
        print(ds, "max |ours - library| =", maxdiff, flush=True)
        assert maxdiff < TOL, cc

        sysmeans = {s: means(p) for s, p in pq.items()}
        gap = {}
        for s in systems:
            d = [pq[s]["NDCG@10"][q] - pq[s]["NDCG@10_exp"][q] for q in qids]
            gap[s] = {"mean_linear_minus_exp": float(np.mean(d)), "max_abs_gap": float(np.max(np.abs(d))),
                      "n_queries_gap_gt_0.01": int(np.sum(np.abs(d) > 0.01))}
        boots = {f"{a} - {b}": {m: paired_boot(pq[a][m], pq[b][m]) for m in ["MRR@10", "R@100", "NDCG@10", "P@10", "MAP@100"]}
                 for a, b in itertools.combinations(systems, 2)}

        pay = {"n_queries": len(qids), "n_docs": len(corpus), "dense_model": mname, "depth": DEPTH, "sources": srcs,
               "system_means": sysmeans, "system_level_winners": system_level(pq), "paired_bootstrap": boots,
               "ndcg_variant_gap": gap, "crosscheck_max_abs_diff": cc, "crosscheck_overall_max": maxdiff,
               "grade_values": sorted({g for r in qrels.values() for g in r.values()})}
        if ds == "scifact":
            pay["toy_and_wikipedia"] = toy

        mets = PAIR_METRICS + (["NDCG@10_exp"] if ds == "nfcorpus" else [])
        dis, examples = {}, []
        for a, b in itertools.combinations(systems, 2):
            res, signs = pair_disagreement(pq[a], pq[b], mets)
            dis[f"{a} vs {b}"] = res
            top = sorted(res.items(), key=lambda kv: -kv[1]["opposite_winner"])[:4]
            examples += [dict(pair=f"{a} vs {b}", **e) for e in find_examples(
                qrels, systems[a], systems[b], pq[a], pq[b], a, b, queries, signs,
                [tuple(k.split("|")) for k, _ in top])]
        pay["query_level_disagreement"] = dis
        pay["examples"] = examples

        # same system, same P@10 and same n_relevant, very different rank placement
        rk = systems["BM25"]
        best = None
        for q1, q2 in itertools.combinations(qids, 2):
            if M._n_rel(qrels[q1]) == M._n_rel(qrels[q2]) and pq["BM25"]["P@10"][q1] == pq["BM25"]["P@10"][q2] > 0:
                g = abs(pq["BM25"]["NDCG@10"][q1] - pq["BM25"]["NDCG@10"][q2])
                if best is None or g > best[0]:
                    best = (g, q1, q2)
        if best:
            _, q1, q2 = best
            pay["same_P_R_different_rank"] = {
                "system": "BM25", "q1": q1, "q2": q2, "n_relevant": M._n_rel(qrels[q1]),
                "labels_q1": labels(qrels[q1], rk[q1]), "labels_q2": labels(qrels[q2], rk[q2]),
                "P@10": pq["BM25"]["P@10"][q1],
                "R@10": [pq["BM25"]["R@10"][q1], pq["BM25"]["R@10"][q2]],
                "MRR@10": [pq["BM25"]["MRR@10"][q1], pq["BM25"]["MRR@10"][q2]],
                "MAP@10": [pq["BM25"]["MAP@10"][q1], pq["BM25"]["MAP@10"][q2]],
                "NDCG@10": [pq["BM25"]["NDCG@10"][q1], pq["BM25"]["NDCG@10"][q2]],
            }

        if ds == "nfcorpus":
            flips = []
            for a, b in itertools.combinations(systems, 2):
                for q in qids:
                    dl = pq[a]["NDCG@10"][q] - pq[b]["NDCG@10"][q]
                    de = pq[a]["NDCG@10_exp"][q] - pq[b]["NDCG@10_exp"][q]
                    if sign(dl) * sign(de) == -1:
                        flips.append({"pair": f"{a} vs {b}", "qid": q, "query": queries[q][:100],
                                      f"{a}_labels": labels(qrels[q], systems[a][q]),
                                      f"{b}_labels": labels(qrels[q], systems[b][q]),
                                      "linear": [pq[a]["NDCG@10"][q], pq[b]["NDCG@10"][q]],
                                      "exp": [pq[a]["NDCG@10_exp"][q], pq[b]["NDCG@10_exp"][q]]})
            pay["linear_vs_exp_winner_flips"] = {"count": len(flips), "examples": flips[:5]}
            pay["system_order_linear"] = sorted(systems, key=lambda s: -sysmeans[s]["NDCG@10"])
            pay["system_order_exp"] = sorted(systems, key=lambda s: -sysmeans[s]["NDCG@10_exp"])

        paths.append(str(write_result("e1", ds, pay, seed=SEED)))
        print(ds, json.dumps(sysmeans, indent=1), flush=True)
    print(paths)


if __name__ == "__main__":
    main()
