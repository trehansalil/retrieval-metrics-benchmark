"""E2: BM25 baseline reproduction + k1/b sweep (SciFact, NFCorpus, FiQA).

Engine: own sparse BM25 (scipy) so one tf matrix serves the whole k1/b grid; cross-checked
against bm25s ("lucene" method) in `crosscheck`. Pyserini/Lucene is unavailable (no Java).

Usage:
  .venv/bin/python e2_bm25.py gap
  .venv/bin/python e2_bm25.py cross
  .venv/bin/python e2_bm25.py sweep '<analyzer json>' scifact nfcorpus fiqa
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter

import numpy as np
import scipy.sparse as sp
import Stemmer  # PyStemmer
import bm25s

from common import RESULTS_DIR, load_dataset, set_seed, write_result, lib_versions, dataset_hash
from ir_metrics import per_query

SEED = 42
MAXQ = 1000  # cap on selection-split queries (seeded sample) to bound FiQA compute
# Lucene EnglishAnalyzer.ENGLISH_STOP_WORDS_SET (33 words)
LUCENE_STOP = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their "
    "then there these they this to was will with".split()
)
RE_W2 = re.compile(r"(?u)\b\w\w+\b")  # bm25s default token pattern (drops 1-char tokens)
RE_W1 = re.compile(r"(?u)\b\w+\b")  # keeps 1-char tokens (closer to Lucene StandardTokenizer)


def make_analyzer(stem=None, stop=None, pattern="w2", possessive=False):
    """stem in {None,'porter','english'(Snowball/Porter2)}; stop in {None,'lucene','bm25s_en'}."""
    stemmer = Stemmer.Stemmer(stem) if stem else None
    rx = RE_W2 if pattern == "w2" else RE_W1
    if stop == "lucene":
        sw = LUCENE_STOP
    elif stop == "bm25s_en":
        from bm25s.stopwords import STOPWORDS_EN
        sw = frozenset(STOPWORDS_EN)
    else:
        sw = frozenset()
    poss = re.compile(r"(?i)['’]s\b")

    def analyze(text):
        if possessive:
            text = poss.sub("", text)
        toks = rx.findall(text.lower())
        if sw:
            toks = [t for t in toks if t not in sw]
        if stemmer:
            toks = stemmer.stemWords(toks)
        return toks

    return analyze


# Lucene BM25Similarity length-norm quantisation (SmallFloat.intToByte4, lossy)
def _long_to_int4(i):
    nb = i.bit_length()
    if nb < 4:
        return i
    shift = nb - 4
    return ((i >> shift) & 0x07) | ((shift + 1) << 3)


def _int4_to_long(i):
    bits = i & 0x07
    shift = (i >> 3) - 1
    return ((bits | 0x08) << shift) if shift >= 0 else bits


_NUM_FREE = 255 - _long_to_int4(2**31 - 1)


def _quant(n):
    b = n if n < _NUM_FREE else _NUM_FREE + _long_to_int4(n - _NUM_FREE)
    return b if b < _NUM_FREE else _NUM_FREE + _int4_to_long(b - _NUM_FREE)


def small_float_len(dl):
    cache = {}
    return np.array([cache.setdefault(int(n), _quant(int(n))) for n in dl], dtype=np.float64)


class SparseBM25:
    def __init__(self, corpus_tokens, doc_ids):
        self.doc_ids = doc_ids
        vocab = {}
        rows, cols, vals = [], [], []
        for d, toks in enumerate(corpus_tokens):
            for t, c in Counter(toks).items():
                j = vocab.setdefault(t, len(vocab))
                rows.append(d)
                cols.append(j)
                vals.append(c)
        self.vocab = vocab
        self.N = len(corpus_tokens)
        self.dl = np.array([len(t) for t in corpus_tokens], dtype=np.float64)
        tf = sp.coo_matrix((np.array(vals, dtype=np.float64), (rows, cols)), shape=(self.N, len(vocab)))
        self.coo = tf
        self.df = np.bincount(tf.col, minlength=len(vocab)).astype(np.float64)
        self.order_key = np.argsort(np.argsort(np.array(doc_ids)))  # docid string rank, for ties

    def weights(self, k1, b, idf="lucene", quantize_len=False):
        dl = small_float_len(self.dl) if quantize_len else self.dl
        avgdl = dl.mean()
        c = self.coo
        w = c.data / (c.data + k1 * (1 - b + b * dl[c.row] / avgdl))  # (k1+1) dropped: rank-neutral
        if idf == "lucene":
            idfv = np.log(1 + (self.N - self.df + 0.5) / (self.df + 0.5))
        else:  # robertson (may be negative)
            idfv = np.log((self.N - self.df + 0.5) / (self.df + 0.5))
        return sp.csr_matrix((w * idfv[c.col], (c.col, c.row)), shape=(c.shape[1], c.shape[0]))  # V x N

    def query_matrix(self, q_tokens, dedup=False):
        rows, cols, vals = [], [], []
        for i, toks in enumerate(q_tokens):
            for t, c in Counter(toks).items():
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(1.0 if dedup else float(c))
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(q_tokens), len(self.vocab)))

    def search(self, Q, k1, b, topk=100, chunk=200, **kw):
        W = self.weights(k1, b, **kw)
        out = []
        for s in range(0, Q.shape[0], chunk):
            S = (Q[s:s + chunk] @ W).toarray()
            for row in S:
                idx = np.argpartition(-row, topk * 2)[: topk * 2]
                idx = idx[np.lexsort((-self.order_key[idx], -row[idx]))][:topk]
                out.append([(self.doc_ids[j], float(row[j])) for j in idx])
        return out


def doc_text(d):
    return (d.get("title", "") + " " + d.get("text", "")).strip()


def score(qrels, run):
    n = per_query(qrels, run, "NDCG", 10)
    r = per_query(qrels, run, "R", 100)
    return {"ndcg10": round(sum(n.values()) / max(1, len(n)), 5), "r100": round(sum(r.values()) / max(1, len(r)), 5)}


def build(corpus, queries, qrels, analyzer):
    doc_ids = list(corpus)
    idx = SparseBM25([analyzer(doc_text(corpus[d])) for d in doc_ids], doc_ids)
    return idx


def qprep(idx, analyzer, queries, qrels):
    qids = [q for q in qrels if q in queries]
    return qids, idx.query_matrix([analyzer(queries[q]) for q in qids])


def eval_cfg(idx, qids, Q, qrels, k1, b, **kw):
    res = idx.search(Q, k1, b, **kw)
    run = {q: dict(r) for q, r in zip(qids, res)}
    return score(qrels, run), run


def gap(ds="scifact"):
    set_seed(SEED)
    corpus, queries, qrels = load_dataset(ds, "test")
    rows = []
    for stem in (None, "porter", "english"):
        for stop in (None, "bm25s_en", "lucene"):
            for pat, poss in (("w2", False), ("w1", False), ("w1", True)):
                an = make_analyzer(stem, stop, pat, poss)
                idx = build(corpus, queries, qrels, an)
                qids, Q = qprep(idx, an, queries, qrels)
                for qz in (False, True):
                    s, _ = eval_cfg(idx, qids, Q, qrels, 0.9, 0.4, quantize_len=qz)
                    rows.append({"stem": stem, "stop": stop, "pattern": pat, "possessive": poss,
                                 "lucene_len_quant": qz, **s})
                    print(rows[-1], flush=True)
    return rows


def crosscheck(ds="scifact"):
    """Our sparse engine vs bm25s variants on identical tokenisation (porter, lucene stop, w1, possessive)."""
    corpus, queries, qrels = load_dataset(ds, "test")
    an = make_analyzer("porter", "lucene", "w1", True)
    idx = build(corpus, queries, qrels, an)
    qids, Q = qprep(idx, an, queries, qrels)
    out = {"ours_lucene_k0.9_b0.4": eval_cfg(idx, qids, Q, qrels, 0.9, 0.4)[0],
           "ours_robertson_idf": eval_cfg(idx, qids, Q, qrels, 0.9, 0.4, idf="robertson")[0]}
    doc_ids = list(corpus)
    ctoks = [an(doc_text(corpus[d])) for d in doc_ids]
    v = {}
    cid = [[v.setdefault(t, len(v)) for t in toks] for toks in ctoks]
    for method in ("lucene", "robertson", "atire", "bm25l", "bm25+"):
        ret = bm25s.BM25(method=method, k1=0.9, b=0.4)
        ret.index(bm25s.tokenization.Tokenized(ids=cid, vocab=v), show_progress=False)
        run = {}
        for q in qids:
            qt = [v[t] for t in an(queries[q]) if t in v]
            sc = ret.get_scores(qt) if qt else np.zeros(len(doc_ids))
            top = np.argsort(-sc, kind="stable")[:100]
            run[q] = {doc_ids[j]: float(sc[j]) for j in top}
        out["bm25s_" + method] = score(qrels, run)
        print(method, out["bm25s_" + method], flush=True)
    return out


K1_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
B_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]


def pick(grid, metric="ndcg10"):
    # max metric; ties broken toward the Anserini default for determinism
    return max(grid, key=lambda k: (grid[k][metric], -abs(float(k.split("|")[0]) - 0.9), -abs(float(k.split("|")[1]) - 0.4)))


def subsample(qrels, n):
    if len(qrels) <= n:
        return qrels
    rng = random.Random(SEED)
    keep = rng.sample(sorted(qrels), n)
    return {q: qrels[q] for q in keep}


def run_sweep(ds, an_cfg):
    set_seed(SEED)
    an = make_analyzer(**an_cfg)
    corpus, queries, qrels_test = load_dataset(ds, "test")
    splits, qtext = {}, {"test": queries}
    for name in ("train", "dev"):
        try:
            _, qs, qr = load_dataset(ds, name)  # BEIR loader returns only that split's queries
            splits[name], qtext[name] = subsample(qr, MAXQ), qs
        except Exception:
            pass
    splits["test"] = qrels_test
    idx = build(corpus, queries, qrels_test, an)
    prep = {n: (*qprep(idx, an, qtext[n], qr), qr) for n, qr in splits.items()}
    grids = {}
    for name, (qids, Q, qr) in prep.items():
        if name == "test":
            continue
        grids[name] = {f"{k1}|{b}": eval_cfg(idx, qids, Q, qr, k1, b)[0] for k1 in K1_GRID for b in B_GRID}
        print(ds, name, "done", len(qids), flush=True)
    best = pick(grids["train"])  # selection on train only; dev is a check; test touched after selection
    qids, Q, qr = prep["test"]
    cfgs = {"selected": best, "anserini": "0.9|0.4", "lucene_default": "1.2|0.75"}
    test_scores, runs = {}, {}
    for tag, key in cfgs.items():
        k1, b = map(float, key.split("|"))
        test_scores[tag], runs[tag] = eval_cfg(idx, qids, Q, qr, k1, b)
    test_grid = {f"{k1}|{b}": eval_cfg(idx, qids, Q, qr, k1, b)[0] for k1 in K1_GRID for b in B_GRID}
    payload = {"analyzer": an_cfg, "k1_grid": K1_GRID, "b_grid": B_GRID, "grids": {**grids, "test": test_grid},
               "n_queries": {n: len(v[0]) for n, v in prep.items()}, "selected_on_train": best,
               "test_at": {t: {"cfg": cfgs[t], **test_scores[t]} for t in cfgs},
               "oracle_best_on_test_REFERENCE_ONLY": {"cfg": pick(test_grid), **test_grid[pick(test_grid)]}}
    if "dev" in grids:
        payload["dev_at_selected"] = grids["dev"][best]
        payload["dev_at_anserini"] = grids["dev"]["0.9|0.4"]
        payload["selected_on_dev_for_stability"] = pick(grids["dev"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for tag, fn in (("anserini", f"e2_bm25_{ds}.json"), ("selected", f"e2_bm25_{ds}_selected.json")):
        k1, b = map(float, cfgs[tag].split("|"))
        (RESULTS_DIR / fn).write_text(json.dumps({
            "meta": {"dataset": ds, "split": "test", "k1": k1, "b": b, "analyzer": an_cfg, "seed": SEED,
                     "tag": tag, "libraries": lib_versions(), "dataset_sha256": dataset_hash(ds)},
            "run": runs[tag]}))
    return write_result("e2sweep", ds, payload, seed=SEED)


def main(argv):
    cmd = argv[1]
    if cmd == "gap":
        print(write_result("e2gap", "scifact", {"rows": gap("scifact")}, seed=SEED))
    elif cmd == "cross":
        print(write_result("e2cross", "scifact", crosscheck(), seed=SEED))
    elif cmd == "sweep":
        an_cfg = json.loads(argv[2])
        for ds in argv[3:]:
            print(run_sweep(ds, an_cfg))
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main(sys.argv)
