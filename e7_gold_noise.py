"""E7: gold-set noise. Synthetic qrels perturbation on SciFact / NFCorpus; measures NDCG@10 shift,
system-ranking flips and paired-bootstrap significance. No re-encoding, no LLM: reuses saved run files.

Usage: .venv/bin/python e7_gold_noise.py scifact nfcorpus
Systems: BM25 (e2_bm25_<ds>), MiniLM + bge-base (e3_*), RRF k=60 over BM25+bge-base (e4_bge-base_<ds> runs_test).
Scenarios (rate p, 40 seeded repetitions each, rate 0 is deterministic):
  a_drop_random       each relevant label dropped with prob p (incomplete judgments)
  a_drop_pooled_bm25  p*N relevant labels dropped, preferring labels outside BM25's top-10 (single lexical pool)
  b_flip_random       per query add round(p*n_rel) false positives (grade 1) drawn from non-relevant docs in the union of all systems' top-10
  b_flip_dense_biased same but drawn only from bge-base top-10 (a judge that agrees with one system)
  c_grade_jitter      NFCorpus only: each label's grade moves 1<->2 with prob p
Queries left with no relevant label are excluded from all systems (trec_eval convention).
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pytrec_eval
from scipy.stats import kendalltau

import ir_metrics as irm
from common import load_dataset, write_result, lib_versions

RES = Path(__file__).parent / "results"
DEPTH = 100
RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
SEEDS = [42 + i for i in range(40)]
N_BOOT = 1000
SYS = ["BM25", "MiniLM", "bge-base", "RRF60"]
SCEN_ALL = ["a_drop_random", "a_drop_pooled_bm25", "b_flip_random", "b_flip_dense_biased"]
GAPS = [0.02, 0.05]
UNRELIABLE = 0.05  # flip probability threshold called "unreliable"


def load_systems(ds):
    e2 = json.loads((RES / f"e2_bm25_{ds}.json").read_text())["run"]
    e3m = json.loads((RES / f"e3_minilm_{ds}.json").read_text())
    e3b = json.loads((RES / f"e3_bge-base_{ds}.json").read_text())
    e4 = json.loads((RES / f"e4_bge-base_{ds}.json").read_text())
    runs = {"BM25": e2, "MiniLM": e3m["run"], "bge-base": e3b["run"], "RRF60": e4["runs_test"]["rrf60"]}
    ranked = {s: {q: irm.rank_docs(r)[:DEPTH] for q, r in run.items()} for s, run in runs.items()}
    ref = {"MiniLM": e3m["metrics"]["ndcg@10"], "bge-base": e3b["metrics"]["ndcg@10"],
           "BM25_e4": e4["test_ndcg10"]["bm25"], "RRF60_e4": e4["test_ndcg10"]["rrf60"]}
    return ranked, ref


def clean_qrels(qrels):
    out = {}
    for q, d in qrels.items():
        pos = {k: int(v) for k, v in d.items() if v > 0}
        if pos:
            out[q] = pos
    return out


def perturb(scen, p, qrels, ranked, rng):
    """Return (perturbed qrels, info dict)."""
    new = {q: dict(d) for q, d in qrels.items()}
    labels = [(q, d) for q, dd in qrels.items() for d in dd]
    info = {}
    if scen == "a_drop_random":
        drop = [l for l in labels if rng.random() < p]
        for q, d in drop:
            del new[q][d]
        info["frac_dropped"] = len(drop) / len(labels)
    elif scen == "a_drop_pooled_bm25":
        pool = {q: set(ranked["BM25"][q][:10]) for q in qrels}
        unp = [l for l in labels if l[1] not in pool[l[0]]]
        pld = [l for l in labels if l[1] in pool[l[0]]]
        k = int(round(p * len(labels)))
        rng.shuffle(unp)
        drop = unp[:k]
        if k > len(unp):
            idx = rng.permutation(len(pld))[: k - len(unp)]
            drop = drop + [pld[i] for i in idx]
        for q, d in drop:
            del new[q][d]
        info["frac_dropped"] = len(drop) / len(labels)
        info["frac_dropped_unpooled"] = (min(k, len(unp)) / max(len(drop), 1))
    elif scen in ("b_flip_random", "b_flip_dense_biased"):
        n_add_tot = 0
        for q, dd in qrels.items():
            exp = p * len(dd)
            n_add = int(exp) + (1 if rng.random() < exp - int(exp) else 0)
            if scen == "b_flip_random":
                cand = {d for s in SYS for d in ranked[s][q][:10]}
            else:
                cand = set(ranked["bge-base"][q][:10])
            cand = sorted(cand - set(dd))
            if n_add and cand:
                pick = rng.permutation(len(cand))[:n_add]
                for i in pick:
                    new[q][cand[i]] = 1
                n_add_tot += min(n_add, len(cand))
        info["frac_added"] = n_add_tot / len(labels)
    elif scen == "c_grade_jitter":
        n = 0
        for q, d in labels:
            if rng.random() < p:
                new[q][d] = 2 if qrels[q][d] == 1 else 1
                n += 1
        info["frac_jittered"] = n / len(labels)
    else:
        raise ValueError(scen)
    return {q: d for q, d in new.items() if d}, info


def evaluate(qrels, ranked):
    qs = sorted(qrels)
    return qs, np.array([[irm.ndcg_at_k(qrels[q], ranked[s][q], 10) for q in qs] for s in SYS])


def pytrec_check(qrels, ranked, qs, per_q):
    worst = 0.0
    for i, s in enumerate(SYS):
        run = {q: {d: float(len(ranked[s][q]) - j) for j, d in enumerate(ranked[s][q])} for q in qs}
        ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in qs}, {"ndcg_cut_10"}).evaluate(run)
        ref = np.array([ev[q]["ndcg_cut_10"] for q in qs])
        worst = max(worst, float(np.abs(ref - per_q[i]).max()))
    return worst


def run_dataset(ds):
    t0 = time.time()
    _, _, qrels_raw = load_dataset(ds)
    ranked, ref = load_systems(ds)
    qrels = {q: d for q, d in clean_qrels(qrels_raw).items() if all(q in ranked[s] for s in SYS)}
    scens = SCEN_ALL + (["c_grade_jitter"] if ds == "nfcorpus" else [])
    pairs_idx = list(combinations(range(len(SYS)), 2))
    # clean
    qs0, pq0 = evaluate(qrels, ranked)
    qpos = {q: i for i, q in enumerate(qs0)}
    clean = pq0.mean(1)
    # orient pairs so first system is clean-better
    pairs = [(i, j) if clean[i] >= clean[j] else (j, i) for i, j in pairs_idx]
    clean_gap = np.array([clean[i] - clean[j] for i, j in pairs])
    out = {"n_queries_clean": len(qs0), "clean_ndcg10": dict(zip(SYS, clean.tolist())),
           "pairs": [[SYS[i], SYS[j]] for i, j in pairs], "clean_gap": clean_gap.tolist(),
           "reference_check": {"e3_minilm_json": ref["MiniLM"], "e3_bge_json": ref["bge-base"],
                               "e4_bm25_json": ref["BM25_e4"], "e4_rrf60_json": ref["RRF60_e4"]},
           "pytrec_eval_worst_abs_diff": 0.0, "scenarios": {}}
    # clean bootstrap CI per pair (paired, per-query)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(qs0), (2000, len(qs0)))
    out["clean_pair_ci"] = []
    for (i, j) in pairs:
        d = pq0[i] - pq0[j]
        m = d[idx].mean(1)
        out["clean_pair_ci"].append({"pair": [SYS[i], SYS[j]], "delta": float(d.mean()),
                                     "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]})
    worst = pytrec_check(qrels, ranked, qs0, pq0)
    for si, scen in enumerate(scens):
        out["scenarios"][scen] = {}
        for rate in RATES:
            if rate == 0.0 and si > 0:
                out["scenarios"][scen]["0.0"] = out["scenarios"][scens[0]]["0.0"]
                continue
            seeds = [SEEDS[0]] if rate == 0.0 else SEEDS
            rows = []
            for seed in seeds:
                if rate == 0.0:
                    q_p, info, qs, pq = qrels, {}, qs0, pq0
                else:
                    rng = np.random.default_rng([seed, si, int(round(rate * 100))])
                    q_p, info = perturb(scen, rate, qrels, ranked, rng)
                    qs, pq = evaluate(q_p, ranked)
                    if seed in SEEDS[:2] and rate in (0.1, 0.3):
                        worst = max(worst, pytrec_check(q_p, ranked, qs, pq))
                means = pq.mean(1)
                brng = np.random.default_rng([seed, si, int(round(rate * 100)), 7])
                bidx = brng.integers(0, len(qs), (N_BOOT, len(qs)))
                sub = [qpos[q] for q in qs]
                row = {"seed": seed, "n_queries": len(qs), "means": means.tolist(), "info": info, "pairs": [],
                       "means_clean_subset": pq0[:, sub].mean(1).tolist()}
                for (i, j) in pairs:
                    d = pq[i] - pq[j]
                    bm = d[bidx].mean(1)
                    row["pairs"].append([float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))])
                rows.append(row)
            out["scenarios"][scen][str(rate)] = summarize(rows, clean, clean_gap, pairs)
        print(ds, scen, "done", round(time.time() - t0), "s", flush=True)
    out["pytrec_eval_worst_abs_diff"] = worst
    return out


def summarize(rows, clean, clean_gap, pairs):
    M = np.array([r["means"] for r in rows])          # R x S
    P = np.array([r["pairs"] for r in rows])          # R x pairs x 3
    nq = np.array([r["n_queries"] for r in rows])
    M0 = np.array([r["means_clean_subset"] for r in rows])  # clean qrels on the retained queries
    R = len(rows)
    s = {"n_reps": R, "seeds": [r["seed"] for r in rows], "mean_n_queries": float(nq.mean()),
         "system": {}, "pair": [], "tau_vs_clean_mean": None, "p_ranking_identical": None}
    for i, name in enumerate(SYS):
        d = M[:, i] - clean[i]
        s["system"][name] = {"ndcg10_mean": float(M[:, i].mean()), "ndcg10_sd": float(M[:, i].std(ddof=1)) if R > 1 else 0.0,
                             "delta_vs_clean_mean": float(d.mean()),
                             "delta_label_only_mean": float((M[:, i] - M0[:, i]).mean()),
                             "delta_query_subset_only_mean": float((M0[:, i] - clean[i]).mean()),
                             "delta_vs_clean_p2.5_p97.5": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if R > 1 else [float(d[0])] * 2}
    taus, same = [], []
    for r in range(R):
        t = kendalltau(clean, M[r])[0]
        taus.append(1.0 if np.isnan(t) else t)
        same.append(float(np.array_equal(np.argsort(-clean), np.argsort(-M[r]))))
    s["tau_vs_clean_mean"] = float(np.mean(taus))
    s["tau_vs_clean_min"] = float(np.min(taus))
    s["p_ranking_identical"] = float(np.mean(same))
    for k, (i, j) in enumerate(pairs):
        delta, lo, hi = P[:, k, 0], P[:, k, 1], P[:, k, 2]
        s["pair"].append({"pair": [SYS[i], SYS[j]], "clean_gap": float(clean_gap[k]),
                          "delta_mean": float(delta.mean()),
                          "delta_p2.5_p97.5": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
                          "p_flip": float(np.mean(delta < 0) + 0.5 * np.mean(delta == 0)),
                          "p_ci_excludes0_correct": float(np.mean(lo > 0)),
                          "p_ci_excludes0_wrong": float(np.mean(hi < 0)),
                          "mean_ci_width": float(np.mean(hi - lo))})
    major = clean_gap >= 0.03   # pairs with a clean gap of at least 0.03
    s["major_pairs"] = [[SYS[i], SYS[j]] for k, (i, j) in enumerate(pairs) if major[k]]
    s["p_all_major_pairs_preserved"] = float(np.mean(np.all(P[:, major, 0] > 0, axis=1)))
    s["pair_deltas_per_rep"] = P[:, :, 0].round(5).tolist()
    shift = (P[:, :, 0] - clean_gap[None, :]).ravel()
    s["gap_flip"] = {str(g): float(np.mean(shift < -g)) for g in GAPS}
    s["shift_sd"] = float(shift.std())
    s["shift_mean"] = float(shift.mean())
    return s


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        res = run_dataset(ds)
        res["protocol"] = {"rates": RATES, "seeds": SEEDS, "n_boot": N_BOOT, "depth": DEPTH, "gaps": GAPS,
                           "unreliable_flip_prob": UNRELIABLE, "systems": SYS,
                           "versions": lib_versions()}
        print("written", write_result("e7", ds, res, seed=42))
