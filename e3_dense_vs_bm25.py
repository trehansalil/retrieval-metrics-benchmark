"""E3: dense (sentence-transformers) vs BM25 (bm25s, k1=0.9 b=0.4) on BEIR SciFact/NFCorpus/FiQA.

Usage: .venv/bin/python e3_dense_vs_bm25.py <model|bm25> <dataset> [max_seq_length]
Writes results/e3_<model>_<dataset>.json (top-1000 run + metrics + timings + versions)
and caches doc embeddings in data/emb/ for E4/E5.
"""
import json
import os
import sys
import time

import numpy as np

import ir_metrics as M
from common import RESULTS_DIR, DATA_DIR, dataset_hash, lib_versions, load_dataset, set_seed, timer

SEED = 42
TOPK = 1000
MODELS = {
    "minilm": dict(hf="sentence-transformers/all-MiniLM-L6-v2", q_prefix="", d_prefix=""),
    "bge-base": dict(hf="BAAI/bge-base-en-v1.5",
                     q_prefix="Represent this sentence for searching relevant passages: ", d_prefix=""),
}


def doc_text(d):
    return (d.get("title", "") + " " + d.get("text", "")).strip()


def topk_run(qids, scores_fn, dids):
    run, lat = {}, []
    for qid in qids:
        t0 = time.perf_counter()
        s = scores_fn(qid)
        k = min(TOPK, len(dids))
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        lat.append(time.perf_counter() - t0)
        run[qid] = {dids[i]: round(float(s[i]), 6) for i in idx}
    return run, lat


def run_bm25(corpus, queries, qids, dids):
    import bm25s
    import Stemmer
    st = Stemmer.Stemmer("english")
    with timer() as ti:
        toks = bm25s.tokenize([doc_text(corpus[d]) for d in dids], stopwords="en", stemmer=st, show_progress=False)
        r = bm25s.BM25(k1=0.9, b=0.4)
        r.index(toks, show_progress=False)
    run, lat = {}, []
    for qid in qids:
        qt = bm25s.tokenize([queries[qid]], stopwords="en", stemmer=st, show_progress=False)
        t0 = time.perf_counter()
        docs, sc = r.retrieve(qt, k=min(TOPK, len(dids)), show_progress=False, n_threads=1)
        lat.append(time.perf_counter() - t0)
        run[qid] = {dids[int(i)]: round(float(s), 6) for i, s in zip(docs[0], sc[0])}
    return run, lat, ti["seconds"], {"dim": None, "index_mb": None, "note": "bm25s sparse, English stopwords + Snowball stemmer, k1=0.9 b=0.4"}


def run_dense(name, ds, corpus, queries, qids, dids, max_seq):
    import torch
    from sentence_transformers import SentenceTransformer
    cfg = MODELS[name]
    model = SentenceTransformer(cfg["hf"], device="cpu")
    if max_seq:
        model.max_seq_length = max_seq
    emb_path = DATA_DIR / "emb" / f"{name}_{ds}_{model.max_seq_length}.npy"
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    with timer() as ti:
        D = model.encode([cfg["d_prefix"] + doc_text(corpus[d]) for d in dids], batch_size=32,
                         normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    D = D.astype(np.float32)
    np.save(emb_path, D)
    qenc, qvec = [], {}
    for qid in qids:
        t0 = time.perf_counter()
        qvec[qid] = model.encode([cfg["q_prefix"] + queries[qid]], normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)[0].astype(np.float32)
        qenc.append(time.perf_counter() - t0)
    run, lat = topk_run(qids, lambda q: D @ qvec[q], dids)
    lat = [a + b for a, b in zip(qenc, lat)]
    extra = {"dim": int(D.shape[1]), "index_mb": round(D.nbytes / 2**20, 2), "max_seq_length": int(model.max_seq_length),
             "hf_model": cfg["hf"], "q_prefix": cfg["q_prefix"], "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
             "mean_query_encode_ms": round(1000 * float(np.mean(qenc)), 2), "torch_threads": torch.get_num_threads()}
    return run, lat, ti["seconds"], extra


def crosscheck(qrels, run):
    import pytrec_eval
    ev = pytrec_eval.RelevanceEvaluator({q: {d: int(g) for d, g in r.items()} for q, r in qrels.items()}, {"ndcg_cut_10", "recall_100"})
    res = ev.evaluate({q: {d: float(s) for d, s in r.items()} for q, r in run.items()})
    n = np.mean([v["ndcg_cut_10"] for v in res.values()])
    r = np.mean([v["recall_100"] for v in res.values()])
    return float(n), float(r)


def main(name, ds, max_seq=None):
    set_seed(SEED)
    corpus, queries, qrels = load_dataset(ds, "test")
    cap = int(os.environ.get("E3_CAP", 0))
    label = ds
    if cap and cap < len(corpus):
        # capped corpus: all judged-relevant docs for test queries + seeded random distractors up to `cap`
        import random
        rel = {d for r in qrels.values() for d, g in r.items() if g > 0}
        rest = sorted(d for d in corpus if d not in rel)
        random.Random(SEED).shuffle(rest)
        keep = rel | set(rest[: max(0, cap - len(rel))])
        corpus = {d: corpus[d] for d in corpus if d in keep}
        label = f"{ds}-cap{cap}"
    dids = list(corpus)
    qids = [q for q in qrels if q in queries]
    if name == "bm25":
        run, lat, t_index, extra = run_bm25(corpus, queries, qids, dids)
    else:
        run, lat, t_index, extra = run_dense(name, label, corpus, queries, qids, dids, max_seq)
    ndcg = M.evaluate(qrels, run, "NDCG", 10)
    rec = M.evaluate(qrels, run, "R", 100)
    pn, pr = crosscheck({q: qrels[q] for q in run}, run)
    assert abs(ndcg - pn) < 1e-6 and abs(rec - pr) < 1e-6, (ndcg, pn, rec, pr)
    out = {
        "experiment": "e3", "model": name, "dataset": label, "seed": SEED, "split": "test",
        "dataset_sha256": dataset_hash(ds), "libraries": lib_versions(),
        "n_docs": len(dids), "n_queries": len(run),
        "metrics": {"ndcg@10": ndcg, "recall@100": rec, "mrr@10": M.evaluate(qrels, run, "MRR", 10),
                    "pytrec_eval_ndcg_cut_10": pn, "pytrec_eval_recall_100": pr},
        "timing": {"index_seconds": round(t_index, 2), "latency_ms_per_query_mean": round(1000 * float(np.mean(lat)), 3),
                   "latency_ms_p95": round(1000 * float(np.percentile(lat, 95)), 3)},
        "model_info": extra, "cpu_count": os.cpu_count(), "run_top": TOPK, "run": run,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"e3_{name}_{label}.json"
    p.write_text(json.dumps(out, sort_keys=True))
    print(name, ds, {k: round(v, 4) for k, v in out["metrics"].items()}, out["timing"], extra, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
