# Retrieval-eval experiment results

Label rule: every number is tagged "I ran, <dataset>, <library versions>" or "reference (source)". Raw JSON lives in `experiments/results/`.

## E1 Metric behaviour

Label for every "ours" number: **I ran, BEIR SciFact (300 test queries) and NFCorpus (323), own `ir_metrics.py` cross-checked with pytrec-eval-terrier 0.5.11 and ranx 0.3.21, numpy 2.5.4, python 3.12.3, seed 42 (PYTHONHASHSEED=42).** Rankings are reused from saved runs to keep memory low: BM25 = `results/e2_bm25_<ds>_selected.json` (own scipy BM25, Porter; SciFact k1 1.2 b 0.8, NFCorpus k1 2.5 b 0.6; selected on train), dense = `results/e3_bge-base_<ds>.json` (bge-base-en-v1.5, sentence-transformers 6.2.0), hybrid = RRF k=60 over those two, all truncated to top 100. Code: `experiments/e1_metric_behavior.py`. Raw: `results/e1_scifact_20261103T065340.json`, `results/e1_nfcorpus_20261103T065346.json`. Every system has one fixed ranking, and each ranking is fed to all three implementations as unique descending scores, so tie handling cannot differ.

### 1. Sanity rows: ours vs reference vs libraries

| Quantity | Reference | Ours (I ran) | pytrec_eval | ranx |
|---|---|---|---|---|
| Toy P@5 | 0.4 | 0.4000 | n/a | n/a |
| Toy R@5 | 0.667 | 0.6667 | n/a | n/a |
| Toy F1@5 | 0.5 | 0.5000 | n/a | n/a |
| Toy AP (divide by R=3) | 0.333 | 0.3333 (0.5000 if divided by hits) | n/a | n/a |
| Toy NDCG@5 linear | 0.7655 | 0.7655 | 0.7655 | 0.7655 (`ndcg`) |
| Toy NDCG@5 exponential | 0.7183 | 0.7183 | not offered | 0.7183 (`ndcg_burges`) |
| Toy ERR@5 (gmax 3) | 0.906 | 0.9060 | n/a | n/a |
| Wikipedia DCG6 | 6.861 | 6.8611 | n/a | n/a |
| Wikipedia NDCG6, ideal from full judged set | 0.785 (Wikipedia) | 0.7850 (IDCG6 8.7403) | n/a | 0.7850 |
| Wikipedia NDCG6, ideal from the six ranked docs only | 0.961 (setup agent) | 0.9608 (IDCG6 7.141) | n/a | n/a |

Toy references are from 00a (computed by hand there), not an external source. All matched to 4 dp.

**0.785 vs 0.961: both numbers are right for different judged sets.** I fetched the Wikipedia DCG page (2026-11-02): DCG6 = 6.861, ideal ordering 3,3,3,2,2,2,1,0, IDCG6 = 8.740, nDCG6 = 0.785. The ranking has only two 3s and two 2s, so Wikipedia's ideal assumes a 3 and a 2 exist in the judged set but were never retrieved. The old test judged just the six ranked docs, so its ideal was 3,3,2,2,1,0 (IDCG6 7.141) and gave 0.961. The 00g "0.785" row is correct. `test_hand_values` now covers both (5 tests pass).

### 2. Cross-check on real runs
Ours vs pytrec_eval (per query) and ranx (per query and mean), 10 metrics x 3 systems x 2 datasets: **max absolute difference 5.6e-16** (tolerance 1e-6). Metrics: P@10, R@10, R@100, Hit@10 (`success`), MRR (full list and @10), MAP@10, MAP@100, NDCG@10 linear; exponential NDCG only against ranx (`ndcg_burges`).

### 3. System-level numbers (ours, I ran)

SciFact, 300 queries:

| System | P@10 | R@10 | R@100 | Hit@10 | MRR@10 | MAP@100 | NDCG@10 (lin = exp) |
|---|---|---|---|---|---|---|---|
| BM25 (stemmed) | 0.0893 | 0.8138 | 0.9280 | 0.8333 | 0.6483 | 0.6433 | 0.6838 |
| bge-base | 0.0987 | 0.8742 | 0.9667 | 0.8833 | 0.7034 | 0.6978 | 0.7404 |
| RRF hybrid | 0.0973 | 0.8729 | 0.9727 | 0.8867 | 0.7030 | 0.6978 | 0.7399 |

NFCorpus, 323 queries (graded 0-2):

| System | P@10 | R@10 | R@100 | Hit@10 | MRR@10 | MAP@100 | NDCG@10 linear | NDCG@10 exp |
|---|---|---|---|---|---|---|---|---|
| BM25 (stemmed) | 0.2356 | 0.1566 | 0.2464 | 0.7028 | 0.5314 | 0.1468 | 0.3271 | 0.3282 |
| bge-base | 0.2799 | 0.1784 | 0.3367 | 0.7461 | 0.5626 | 0.1841 | 0.3735 | 0.3747 |
| RRF hybrid | 0.2675 | 0.1757 | 0.3286 | 0.7368 | 0.5628 | 0.1759 | 0.3630 | 0.3639 |

Paired bootstrap on per-query deltas (10,000 resamples, seed 42): BM25 minus bge-base on SciFact NDCG@10 -0.057 [-0.089, -0.024], MRR@10 -0.055 [-0.092, -0.018], R@100 -0.039 [-0.069, -0.010]. On NFCorpus NDCG@10 -0.046 [-0.065, -0.029], but MRR@10 -0.031 [-0.068, +0.004] (CI spans zero). bge-base minus RRF is inside noise on SciFact for every metric.

### 4. Where metrics disagree

**System level.**
- BM25 vs either dense or hybrid: every metric names the same winner on both datasets. The only thing that varies is whether the CI excludes zero (NFCorpus MRR@10 does not).
- SciFact, bge-base vs RRF: 7 of 9 distinct metrics (P@10, R@10, MRR, MRR@10, MAP@10, MAP@100, NDCG@10; exponential NDCG is identical here) favour bge-base, while Hit@10 (0.8867 vs 0.8833) and R@100 (0.9727 vs 0.9667) favour RRF. Winner depends on the metric you pick; every difference is within noise (NDCG@10 diff +0.0005, CI [-0.021, +0.021]).
- NFCorpus, bge-base vs RRF: MRR and MRR@10 name RRF (by 0.0002), everything else names bge-base.

**Query level, SciFact, BM25 vs bge-base (concrete).**
- Query 261 ("Chronic aerobic exercise alters endothelial function..."), 2 relevant docs. BM25 top-10 labels `[0,0,1,0,0,0,0,0,0,0]`, bge-base `[0,0,0,0,0,1,0,1,0,0]`. P@10 0.1 vs 0.2 and R@10 0.5 vs 1.0 and MAP@10 0.167 vs 0.208 all pick bge-base, but MRR@10 0.333 vs 0.167 picks BM25.
- Query 1041 ("Replacement of histone H2A with H2A.Z..."), 2 relevant. BM25 `[0,0,0,1,0,0,0,0,0,0]`, bge-base `[0,0,0,0,0,0,1,0,0,1]`. P@10 0.1 vs 0.2, R@10 0.5 vs 1.0, MAP@10 0.125 vs 0.171 pick bge-base; MRR@10 0.25 vs 0.143 picks BM25.
- Frequency: P@10 vs MRR@10 name opposite winners on 2 of 42 queries where both are non-zero (BM25 vs bge-base); MRR@10 vs NDCG@10 on 2 of 124; MAP@10 vs NDCG@10 on 0 of 131. On SciFact, MAP and NDCG practically never disagree.
- Same system, same query set, same P@10 = 0.1, R@10 = 1.0, one relevant doc: query 100 (BM25 puts it at rank 1: MRR 1.0, NDCG 1.0) vs query 1279 (rank 10: MRR 0.1, NDCG 0.289). P@10 and R@10 cannot tell them apart.
- Depth vs head on SciFact: R@100 vs NDCG@10 name opposite winners on 1 of 15 usable queries (BM25 vs bge-base), too few to say anything.

**Query level, NFCorpus (graded, many relevant docs), where it is much worse.**
- R@100 vs NDCG@10 name opposite winners on 56 of 169 queries (33%) for BM25 vs bge-base, 40 of 129 (31%) for BM25 vs RRF, and 50 of 111 (45%) for bge-base vs RRF. R@100 vs MRR@10: 33 of 99 (33%).
- Query PLAIN-468 ("Is vitamin D3 (cholecalciferol) preferable to D2 (ergocalciferol)?"), 10 relevant. BM25 top-10 `[0,0,0,0,2,2,0,0,0,0]`, bge-base `[2,0,0,0,0,0,0,0,0,2]`. R@100 0.7 vs 0.2 picks BM25; NDCG@10 0.241 vs 0.418 (linear), 0.286 vs 0.495 (exponential) picks bge-base.
- Query PLAIN-371 ("Any update on the scary in vitro avocado data?"), 2 relevant. BM25 puts the first hit at rank 4 and bge-base at rank 1 (NDCG@10 0.164 vs 0.380, MAP@10 0.125 vs 0.500), but R@100 is 1.0 for BM25 vs 0.5 for bge-base.
- Query PLAIN-2019 ("rickets"), 3 relevant: BM25 R@100 0.33 vs bge-base 1.0, yet BM25's top hit is at rank 1 (MAP@10 0.333 vs 0.278).

### 5. Linear vs exponential NDCG
- SciFact: qrels are binary (all grade 1), so linear = exponential exactly (gap 0.0 on all 300 queries). Any "0.765 vs 0.718" effect needs graded labels.
- NFCorpus (grades 1 and 2): system means differ by only 0.001 (BM25 0.3271 vs 0.3282, bge-base 0.3735 vs 0.3747), and the system ordering is the same (bge-base, RRF, BM25) under both. Per query the gap is up to 0.099 (BM25), 0.117 (RRF) and 0.132 (bge-base); 66 to 80 of 323 queries differ by more than 0.01 (bge-base: 80 with a strict >0.01 test; an 81st query sits at 0.00996, which rounds to 0.010 but does not exceed it).
- Winner flips: 6 query-pair cases across the three system pairs where linear and exponential NDCG@10 name opposite winners. Example, PLAIN-2 ("Do Cholesterol Statin Drugs Cause Breast Cancer?"), BM25 `[2,2,2,0,2,1,0,2,1,0]` vs bge-base `[2,2,2,2,1,1,1,0,1,0]`: linear 0.773 vs 0.795 picks bge-base; exponential 0.775 vs 0.767 picks BM25.

### 6. Conclusions you can share
- Every metric agreed on BM25 vs dense, on two datasets. The metric choice mattered for close calls, not for the 0.05 gaps.
- On SciFact (about one relevant doc per query) MRR, MAP and NDCG are near-interchangeable (opposite winner on 0 to 2% of queries), because they all reduce to "where is the one relevant doc". P@10 is not an informative metric there: with about one relevant doc per query it cannot exceed roughly 0.1 (observed 0.089 to 0.099).
- On graded, many-relevant data (NFCorpus) the depth metric R@100 and the head metric NDCG@10 name opposite winners on about a third of queries. A first-stage retriever and a final ranker should not be compared on the same metric.
- Linear vs exponential NDCG did not change any system ordering in our runs and moved means by 0.001. It matters per query (up to 0.13) and only on graded labels. State the variant, but do not expect it to flip headline conclusions.
- Hit@10 and R@100 can crown a different hybrid-vs-dense winner than NDCG@10, but only when the difference is already inside noise.

### 7. Caveats
- Two datasets, three systems, one deterministic run, one seed; bootstraps cover query sampling only. No multiple-comparison correction on the disagreement counts (they are descriptive, not tests).
- SciFact has about 1.1 relevant docs per query, so rank-aware metrics collapse to one another there; NFCorpus qrels are graded 0-2 only (no grade 3), so the exponential variant is the mildest possible.
- "Disagreement" counts only queries where both metrics register a non-zero difference; ties are excluded, which inflates the percentages relative to all queries. Denominators are printed above.
- BM25 params were selected on train queries (E2), bge-base is the E3 run; neither was tuned for E1. RRF uses k=60 at depth 100, not tuned.
- Reused E2/E3 run files rather than recomputing, because the host is memory-bound. The first attempt (own bge-small run) was killed for memory and discarded; nothing from it is reported.
- Licences: SciFact/NFCorpus non-commercial: numbers only, no document text quoted beyond query strings truncated to 140 chars.

### 8. What surprised me
- On SciFact the metrics barely disagree at all: MAP@10 vs NDCG@10 never named opposite winners across 131 queries. I expected a visible rank-metric split and found it only with P@10 or R@100 against MRR.
- The R@100 vs NDCG@10 split on NFCorpus is a third of queries, far more than any head-metric pair. "Better retriever" depends on whether you mean depth or head.
- Wikipedia's 0.785 and our 0.961 were both right; the entire difference is two documents nobody retrieved. That is the "IDCG from full qrels" pitfall, live.

## Change log
- 2026-11-03: E3 section 2 SciFact bge-base minus BM25 CI upper bound corrected from +0.092 to +0.094 (recomputed from `e2_bm25_scifact.json` and `e3_bge-base_scifact.json`: +0.0601 [+0.0258, +0.0937], 2000 resamples, seed 42; matches E7 JSON and the E7 text). Also clarified the NFCorpus bge-base linear-vs-exp gap count (80 strict) in E1 section 5, and added the BM25 R@100 0.926 (k1=0.9, b=0.4) vs 0.928 (k1=1.2, b=0.8) note in E3 section 2.


## E2: BM25 baseline reproduction and k1/b sweep

Label for every number below: **I ran, BEIR <dataset> test, own scipy sparse BM25 engine (cross-checked vs bm25s 0.3.16 method=lucene, identical to 5 dp), PyStemmer 3.1.0 Porter, numpy 2.5.4, scipy 1.18.1, python 3.12.3, seed 42.** Pyserini/Lucene not run (no Java). Dataset sha256 first-8: SciFact 8ad90358, NFCorpus c10ce6eb, FiQA dddfa29e. Code: `experiments/e2_bm25.py`. Raw: `results/e2gap_scifact_*.json`, `e2cross_scifact_*.json`, `e2sweep_<ds>_*.json` (full 12x12 grids for train, dev, test = heatmap-ready: keys `"k1|b"`, values `{ndcg10, r100}`, plus `k1_grid`, `b_grid`).

### 1. Closing the SciFact gap (k1=0.9, b=0.4, title+text as one field)
Reference (Anserini flat, 00y row 2a/2b): NDCG@10 0.679, R@100 0.925.

| Analyzer (ours) | NDCG@10 | R@100 |
|---|---|---|
| Setup agent: bm25s, no stemmer, English stopwords, tokens >= 2 chars | 0.6625 | 0.876 |
| Ours: no stem, Lucene stopwords, tokens >= 1 char | 0.663 | 0.886 |
| Porter, Lucene stopwords, tokens >= 2 chars | 0.671 | 0.919 |
| Porter, Lucene stopwords, tokens >= 1 char | 0.677 | 0.929 |
| Snowball (Porter2), no stopwords, tokens >= 1 char | 0.682 | 0.920 |
| **Porter, no stopwords, tokens >= 1 char (chosen)** | **0.680** | **0.926** |

What explains the 0.017 / 0.049 gap, by size of effect (from the 36-config ablation in `e2gap`):
1. **Stemming** (Lucene EnglishAnalyzer uses the original Porter stemmer): +0.014 NDCG, +0.033 R@100. Largest single factor, and most of the R@100 gap.
2. **Keeping 1-char tokens** (bm25s default regex drops them; Lucene's tokenizer keeps them, e.g. "t", "b", digits): +0.005 NDCG, +0.010 R@100.
3. **Stopwords**: removing them costs about 0.003 NDCG; with them removed or kept both are inside the tolerance, so this run cannot tell which Anserini used. I did not verify the analyzer source; the no-stopword choice matched best and is an empirical match, not a confirmed fact.
4. Porter vs Snowball: Porter matches R@100 better (0.926 vs 0.920), Snowball NDCG slightly higher (0.682). Under 0.003.
5. **No effect**: Lucene length-norm byte quantisation (SmallFloat, +-0.001), possessive stripping (0), idf variant lucene vs Robertson (-0.0003), bm25s variants atire / bm25+ / bm25l (all 0.678 vs 0.677 for lucene, same tokenisation; see `e2cross`).

Result: **0.6803 / 0.9263 vs reference 0.679 / 0.925. Gap +0.001 / +0.001, inside the +-0.01 same-implementation tolerance.** Caveat: the analyzer was chosen by matching the published test number, so this is a reproduction target and not a tuned-for-quality choice; it is not independent confirmation that Anserini does exactly this.

### 2. Sweep protocol (anti test-tuning)
k1 in {0.1,0.2,0.3,0.5,0.7,0.9,1.2,1.5,1.8,2.0,2.5,3.0} x b in {0,0.1,...,0.7,0.75,0.8,0.9,1.0}. Selection by NDCG@10 on **train queries only** (SciFact: all 809 train; NFCorpus and FiQA: seeded sample of 1000 train queries). NFCorpus/FiQA dev (324/500 queries) used as a check. Test (300/323/648 queries) evaluated after selection; the test grid is stored only so you can plot it, and its oracle best is labelled reference-only. Ties broken toward (0.9, 0.4).

### 3. Ours vs reference (test, NDCG@10 / R@100)

| Dataset | Ours at Anserini k1=0.9, b=0.4 | Reference NDCG@10 (flat / multifield) | Delta (flat) | Ours at Lucene 1.2 / 0.75 | Selected on train (k1, b) | Ours at selected |
|---|---|---|---|---|---|---|
| SciFact | 0.680 / 0.926 | 0.679 / 0.665; R@100 0.925 | +0.001 | 0.684 / 0.928 | 1.2, 0.8 | 0.684 / 0.928 |
| NFCorpus | 0.322 / 0.246 | 0.322 / 0.325 | 0.000 | 0.322 / 0.247 | 2.5, 0.6 | 0.327 / 0.246 |
| FiQA | 0.234 / 0.537 | 0.236 / 0.236 | -0.002 | 0.248 / 0.558 | 1.2, 0.7 | 0.247 / 0.555 |

Reference values: Pyserini 2CR flat/multifield (00y row 2a, verified) and BEIR paper Table 2 (NFCorpus 0.325, FiQA 0.236, SciFact 0.665; row 1a, verified). Our three reproduction deltas are within +-0.01.

Dev check for the selected config: NFCorpus 0.292 vs 0.287 at Anserini params; FiQA 0.270 vs 0.259. Dev-only selection would choose NFCorpus 1.5/0.7 and FiQA 0.9/0.75, so the optimum is flat and noisy (see below).

### 4. Conclusions you can share
- A BM25 you build yourself matches published Lucene BM25 to within 0.002 NDCG@10 on SciFact, NFCorpus and FiQA, but only once the analyzer matches (Porter stemming and 1-char tokens). A bare bm25s tokenizer with stopwords sits 0.017 NDCG / 0.05 R@100 below on SciFact. When you quote a "BM25 baseline", state the analyzer.
- The k1/b optimum is shallow. Within the plausible region (k1 1.2 to 2.5, b 0.6 to 0.8) NDCG@10 moves by under about 0.01, and moving from Anserini 0.9/0.4 to Lucene 1.2/0.75 gains 0.004 (SciFact), 0.000 (NFCorpus) and 0.014 (FiQA). The FiQA gain is the only one clearly larger than split noise, and b matters most there (higher b helps long answers).
- Train-selected parameters beat the Anserini defaults on test by at most 0.005 NDCG (0.684 vs 0.680, 0.327 vs 0.322, 0.247 vs 0.234 for FiQA 0.013). Gains on NFCorpus and SciFact are small relative to the spread across query subsets; no significance test was run (this is one deterministic run), so do not claim them as real.
- Test oracle best (reference only, not a result): SciFact 0.698 at 2.0/0.9, NFCorpus 0.331 at 2.5/0.3, FiQA 0.251 at 1.2/0.8. The gap between oracle and train-selected (0.014 SciFact) is what tuning on the test set would have falsely claimed.

### 5. Caveats
- Not Pyserini: same formula and Porter analyzer, not the same code path. Lucene's StandardTokenizer (UAX#29) differs from `\b\w+\b` on hyphens, apostrophes and punctuation edge cases; stopword handling in Anserini unverified.
- NFCorpus R@100 and the FiQA/NFCorpus R@100 columns have no verified reference row here; I did not check them against BEIR tables.
- SciFact has no dev split in BEIR; selection used train only. NFCorpus/FiQA train selection used a 1000-query sample, so those picks carry sampling noise.
- Train distributions differ from test (e.g. NFCorpus train NDCG 0.32 vs dev 0.29 vs test 0.32), so absolute levels are not comparable across splits; only the argmax is used.
- SciFact/NFCorpus licences are non-commercial/unverified: numbers only, no redistribution of data.

### 6. Reusable run files for E3/E4/E5
`results/e2_bm25_<scifact|nfcorpus|fiqa>.json` = Anserini params 0.9/0.4, top-100, test split, format `{"meta": {...}, "run": {qid: {docid: score}}}`. `results/e2_bm25_<ds>_selected.json` = train-selected params (same format). Note `results/e3_bm25_*.json` already existed from E3 setup (unstemmed bm25s) and was not touched; E3 should switch to the e2 files for a stemmed baseline.

## E3: dense vs BM25 (SciFact, NFCorpus, FiQA)

Label for every "ours" number: **I ran, BEIR <dataset> test split, sentence-transformers 6.2.0, torch 2.15.0+cpu, transformers 5.22.1, numpy 2.5.4, bm25s 0.3.16, pytrec-eval-terrier 0.5.11, python 3.12.3, CPU only (4 cores, shared host), seed 42.** Dataset sha256 first-8: NFCorpus c10ce6eb, FiQA dddfa29e (SciFact in the run JSONs). Code: `experiments/e3_dense_vs_bm25.py`. Run files (top-1000 per query, `{"run": {qid: {docid: score}}}` plus metrics, timings, versions): `results/e3_<bm25|minilm|bge-base>_<dataset>.json`. Doc embeddings cached in `data/emb/*.npy` (L2-normalised float32, for E4/E5 reuse). Every NDCG@10 and R@100 was computed with `ir_metrics.py` and asserted equal to pytrec_eval to 1e-6 inside the run.

Models: all-MiniLM-L6-v2 (22.7M params, 384-d, max_seq 256, no prefix) and bge-base-en-v1.5 (109.5M params, 768-d, max_seq 512, query prefix "Represent this sentence for searching relevant passages: "). Doc text = title + " " + text, cosine on normalised vectors, brute-force exact search (no ANN, so no index recall loss). BM25 baseline = the E2 run files `results/e2_bm25_<ds>.json` (Porter, k1=0.9, b=0.4), as the coordinator asked.

### 1. Ours vs reference (NDCG@10 / R@100, test)

| Dataset | System | Ours (I ran) | Reference | Delta NDCG | Reference source |
|---|---|---|---|---|---|
| SciFact | BM25 (E2) | 0.680 / 0.926 | 0.679 / 0.925 | +0.001 | Pyserini 2CR flat (00y 2a) |
| SciFact | MiniLM-L6 | 0.6451 / 0.925 | 0.64508 / 0.925 | 0.0000 | MTEB results repo, all-MiniLM-L6-v2 rev 8b3219a, mteb 1.12.75 |
| SciFact | bge-base | 0.7404 / 0.9667 | 0.74039 / 0.96667 | 0.0000 | BAAI/bge-base-en-v1.5 model card MTEB metadata |
| NFCorpus | BM25 (E2) | 0.322 / 0.247 | 0.322 / 0.246 | 0.000 | Pyserini flat |
| NFCorpus | MiniLM-L6 | 0.3159 / 0.3115 | 0.31594 / 0.31151 | 0.0000 | MTEB results repo (same rev) |
| NFCorpus | bge-base | 0.3735 / 0.3367 | 0.373 / 0.337 (card 0.37389 / 0.33672) | 0.000 | Pyserini BGE-base row (00g #10); model card |
| FiQA (57,638 docs) | BM25 (E2) | 0.234 / 0.537 | 0.236 / 0.539 | -0.002 | Pyserini / BEIR paper |
| FiQA (57,638 docs) | MiniLM-L6 | 0.3687 / 0.7061 | 0.36867 / 0.70606 | 0.0000 | MTEB results repo (same rev) |
| FiQA | bge-base | NOT RUN on full corpus (see 4) | 0.406 / 0.742 (card 0.40646 / 0.74224) | n/a | Pyserini; model card |

Our dense numbers reproduce the published checkpoints to the 4th decimal on all 5 runs we could compare, well inside the +-0.01 tolerance. Sources fetched 2026-11-02/03: raw MTEB result JSONs at github.com/embeddings-benchmark/results (results/sentence-transformers__all-MiniLM-L6-v2/8b3219a9.../{SciFact,NFCorpus,FiQA2018}.json), raw README of huggingface.co/BAAI/bge-base-en-v1.5.

### 2. The head-to-head (ours, test, E2 stemmed BM25 as baseline; paired bootstrap on per-query NDCG@10, 2000 resamples, seed 42)

| Dataset | BM25 | MiniLM | bge-base | MiniLM minus BM25 [95% CI] (W/L/T) | bge-base minus BM25 [95% CI] (W/L/T) |
|---|---|---|---|---|---|
| SciFact | 0.680 | 0.645 | **0.740** | -0.035 [-0.071, -0.001] (67/69/164) | +0.060 [+0.026, +0.094] (90/43/167) |
| NFCorpus | 0.322 | 0.316 | **0.374** | -0.006 [-0.029, +0.015] (108/108/107) | +0.051 [+0.032, +0.071] (134/81/108) |
| FiQA full | 0.234 | **0.369** | not run | +0.134 [+0.109, +0.159] (313/106/229) | n/a |
| FiQA capped 5,000 docs | 0.425 | 0.598 | **0.623** | +0.173 [+0.146, +0.199] (374/110/164) | +0.197 [+0.173, +0.221] (399/81/168) |

Note on BM25 R@100 = 0.926 vs 0.928: 0.926 is BM25 at Anserini k1=0.9, b=0.4 (E2 and E3, and the E4 tables); 0.928 is BM25 at the train-selected k1=1.2, b=0.8 (E1, which reuses `e2_bm25_scifact_selected.json`). Same engine, different parameters, not a discrepancy.

Recall@100: SciFact 0.926 / 0.925 / 0.967; NFCorpus 0.247 / 0.312 / 0.337; FiQA full 0.537 / 0.706 / n/a; FiQA capped 0.724 / 0.900 / 0.895 (BM25 capped = own bm25s run, Snowball+stopwords, since E2 ran the full corpus only).

### 3. Cost (I ran, 4 shared CPU cores; treat as order of magnitude)

| Dataset | System | Index time | Latency per query (mean, batch 1) | Dim | Index size |
|---|---|---|---|---|---|
| SciFact (5,183) | bm25s | 1.3 s | 1.0 ms | sparse | n/a |
| SciFact | MiniLM | 309 s | 27 ms (26 ms is query encoding) | 384 | 7.6 MB |
| SciFact | bge-base | 3,171 s | 144 ms | 768 | 15.2 MB |
| NFCorpus (3,633) | bm25s | 0.9 s | 1.1 ms | sparse | n/a |
| NFCorpus | MiniLM | 198 s | 17 ms | 384 | 5.3 MB |
| NFCorpus | bge-base | 2,367 s | 96 ms | 768 | 10.6 MB |
| FiQA (57,638) | bm25s | 14 s | mean 95 ms, p95 25 ms (mean inflated by host contention; capped run 1.1 ms) | sparse | n/a |
| FiQA | MiniLM | 2,267 s | 31 ms | 384 | 84.4 MB |
| FiQA capped 5,000 | bge-base | 1,818 s | 115 ms | 768 | 14.7 MB |

Dense latency is dominated by encoding the query with the transformer, not by the 5k to 58k vector dot products (exact search over 57,638 x 384 floats adds only a few ms). Index and latency timings were taken while another agent's jobs shared the host (load average up to ~100 during the first runs), so they overstate real times and are only comparable within about 2x. The ratio is the usable fact: bge-base cost roughly 10 to 12x MiniLM to index and about 5x per query, and 1,000 to 3,000x BM25 to index.

### 4. FiQA cap (deviation, read this)
bge-base on full FiQA would be 57.6k docs at roughly 2 docs/s on this machine (SciFact 1.6/s, capped FiQA 2.8/s), around 6 to 10 hours, and a first run was killed for low memory. So bge-base on FiQA was run on a **capped corpus of 5,000 docs: all judged-relevant docs for the 648 test queries plus seeded random distractors (seed 42)**. All three systems were re-run on that same subset (`results/e3_*_fiqa-cap5000.json`). A 10x smaller haystack makes everything easier (BM25 jumps 0.234 to 0.425, MiniLM 0.369 to 0.598), so **capped numbers are NOT comparable to published FiQA numbers and must not be quoted next to them**; use them only for the within-subset ordering. The full-corpus bge-base FiQA reference (0.406) is quoted from Pyserini / the model card, not run by me.

### 5. Why ours match the reference (gap explanation)
- Same checkpoint, same pooling and normalisation, same title+text concatenation, same BGE query prefix, same test qrels, exact (not approximate) search. The MTEB harness and Pyserini differ in tooling but score the same ranking function, so agreement to 4 decimals is expected rather than lucky. If the BGE prefix were dropped or the E5 prefix scheme mixed, this is where a silent drop would appear (00c section 8); we did not run that ablation.
- BM25 drift (-0.002 to +0.001) comes from E2's analyzer choice (see E2 section 5), not from the dense side.
- Doc truncation: MiniLM at 256 tokens vs bge-base at 512 is each model's default and matches the published setting. Long SciFact abstracts can be truncated for MiniLM; I did not measure that effect separately.

### 6. Conclusions you can share
- "Dense beats BM25" depends on both the model and the dataset. A good base-size embedder (bge-base) beat tuned-default BM25 on SciFact (+0.060) and NFCorpus (+0.051) with CIs clear of zero. The small MiniLM lost to BM25 on SciFact (-0.035, CI just excludes zero) and tied on NFCorpus (-0.006, CI spans zero). On FiQA, both dense models won big (MiniLM +0.134 on the full corpus).
- Recall tells a different story from NDCG: MiniLM matched BM25's Recall@100 on SciFact (0.925 vs 0.926) while losing 3.5 NDCG points, so it finds the right documents but ranks them lower. As a first-stage candidate generator it is as good as BM25 on SciFact, as a final ranker it is not. On NFCorpus and FiQA dense recall@100 is clearly higher (0.31 to 0.34 vs 0.25; 0.71 vs 0.54).
- Cost: bge-base gave +0.05 to +0.06 NDCG for ~100 ms per query on 4 CPU cores and tens of minutes of indexing for a 5k-doc corpus, versus 1 ms and a second for BM25. That price decides whether it is worth it, and it is why E4 (hybrid) matters.
- Small models are not "BM25 plus a bit". Choose by measuring on your own domain.

### 7. Caveats
- One run each, deterministic, one seed; the bootstrap covers query sampling only, not model or training variance. 300 to 648 queries; SciFact CIs are wide.
- Different tokenisers and truncation lengths per model are confounds against any "dense vs sparse" generalisation; this is three models on three datasets, not a law.
- FiQA bge-base is capped (section 4). Timings are from a contended shared host. BM25 latency uses bm25s, not Lucene.
- BEIR numbers on NFCorpus/FiQA/SciFact predate or may overlap MTEB training data for BGE (00g notes contamination risk from post-2023 models); I did not check. A model that was trained on in-domain data (BGE) vs one that was not is part of why bge-base wins.
- Licences: SciFact/NFCorpus non-commercial, FiQA unverified: numbers only.

### 8. What surprised me
- The reproduction was exact, not approximate: 0.64508 published vs 0.6451 ours on SciFact, 0.74039 vs 0.7404. I expected a drift of a point or two from library versions.
- MiniLM lost to BM25 on SciFact on NDCG yet tied it on Recall@100. The right paper is in the pool as often as with BM25; it is just ranked lower.
- Wall-clock: indexing 5k short scientific abstracts with bge-base took ~53 minutes on CPU, so an "easy local dense baseline" is not free; the machine contention doubled that, but even at half it is not seconds.
- Shrinking FiQA's corpus 10x lifted BM25 from 0.23 to 0.43. A "capped for speed" benchmark can silently change the answer to "is BM25 weak here".

## E4: hybrid fusion, RRF vs convex combination (SciFact, NFCorpus, FiQA)

Label for every "ours" number: **I ran, BEIR <dataset> test, own scipy BM25 (Porter, k1=0.9, b=0.4; E2 run files), all-MiniLM-L6-v2 and bge-base-en-v1.5 via sentence-transformers 6.2.0 (E3 cached doc embeddings, no re-encoding), torch 2.15.0+cpu, numpy 2.5.4, pytrec-eval-terrier 0.5.11, python 3.12.3, seed 42.** Code `experiments/e4_hybrid.py`; run files `results/e4_<minilm|bge-base>_<dataset>.json` (fused test runs for RRF k=60 and train-alpha convex, per-query NDCG@10, full train/test alpha grids, bootstrap, cross-check). NDCG@10 and R@100 from `ir_metrics.py`, asserted equal to pytrec_eval to 1e-6 for BM25, dense, RRF60 and the convex run on every dataset (max abs diff 6e-17). Dense@100 NDCG@10 was also asserted equal to the E3 top-1000 value.

### 1. Protocol
- Fusion depth: top-100 of each system (BM25 file was top-100; dense E3 top-1000 truncated to 100 so both lists have the same depth). Fused list cut to 100 for R@100.
- RRF: sum 1/(k+rank), k in {10, 60, 100}; k=60 is the a-priori headline.
- Convex: per-query min-max over each system's top-100, score = alpha*dense + (1-alpha)*bm25, missing doc = 0. alpha 0 = BM25 only, 1 = dense only, step 0.1. (Our alpha is the weight on dense; Bruch et al. put alpha on the other side in places, so do not compare alpha values directly.)
- Alpha selection: NDCG@10 on TRAIN queries (SciFact 809 train queries, NFCorpus and FiQA 1000-query seeded sample, same sample rule as E2); train queries encoded fresh with the same model, train BM25 from the same engine. Test scored once with the selected alpha. Test-oracle alpha reported as reference only.
- Bootstrap: paired, 2000 resamples, seed 42, per-query NDCG@10 on test. "Best single" = the single system with higher NDCG@10 on train (BM25 for MiniLM on SciFact/NFCorpus, dense otherwise).

### 2. Results (test NDCG@10; R@100 in section 3)

| Dataset / dense | BM25 | Dense | RRF k=10 | RRF k=60 | RRF k=100 | Convex, train alpha | Test-oracle convex (reference only) |
|---|---|---|---|---|---|---|---|
| SciFact / MiniLM | 0.680 | 0.645 | 0.718 | 0.716 | 0.715 | **0.727** (a=0.5) | 0.727 (a=0.4) |
| SciFact / bge-base | 0.680 | 0.740 | 0.750 | 0.740 | 0.740 | **0.764** (a=0.7) | 0.764 (a=0.7) |
| NFCorpus / MiniLM | 0.322 | 0.316 | 0.350 | 0.345 | 0.344 | **0.358** (a=0.5) | 0.360 (a=0.4) |
| NFCorpus / bge-base | 0.322 | 0.374 | 0.368 | 0.362 | 0.359 | **0.385** (a=0.6) | 0.385 (a=0.6) |
| FiQA full / MiniLM | 0.234 | 0.369 | 0.382 | 0.372 | 0.370 | **0.401** (a=0.6) | 0.401 (a=0.7) |
| FiQA / bge-base | not run (capped only in E3; not compared) | | | | | | |

Train selection of RRF k picked k=10 everywhere, but k=10 vs 60 vs 100 differ by at most 0.01 except SciFact/bge-base (0.750 vs 0.740) and FiQA (0.382 vs 0.372); no k was tested for significance.

### 3. Fusion vs best single system (paired bootstrap, test, diff in NDCG@10 [95% CI], W/L/T queries)

| Dataset / dense | RRF60 minus best single | Convex(train alpha) minus best single | Convex minus RRF60 |
|---|---|---|---|
| SciFact / MiniLM (vs BM25) | +0.036 [+0.012, +0.058] 80/26/194 | +0.046 [+0.027, +0.065] 68/16/216 | +0.011 [-0.003, +0.024] |
| SciFact / bge-base (vs dense) | 0.000 [-0.023, +0.023] 60/50/190 | +0.024 [+0.010, +0.039] 52/31/217 | +0.024 [+0.006, +0.042] |
| NFCorpus / MiniLM (vs BM25) | +0.023 [+0.009, +0.037] 114/67/142 | +0.036 [+0.024, +0.049] 117/63/143 | +0.014 [+0.006, +0.022] |
| NFCorpus / bge-base (vs dense) | -0.012 [-0.025, +0.001] 96/104/123 | +0.011 [+0.003, +0.020] 96/92/135 | +0.023 [+0.014, +0.033] |
| FiQA full / MiniLM (vs dense) | +0.003 [-0.016, +0.022] 204/191/253 | +0.032 [+0.019, +0.045] 183/139/326 | +0.029 [+0.018, +0.040] |

Recall@100 (BM25 / dense / RRF60 / convex): SciFact-MiniLM 0.926 / 0.925 / 0.953 / 0.953; SciFact-bge 0.926 / 0.967 / 0.971 / 0.969; NFCorpus-MiniLM 0.246 / 0.312 / 0.312 / 0.314; NFCorpus-bge 0.246 / 0.337 / 0.327 / 0.336; FiQA-MiniLM 0.537 / 0.706 / 0.704 / 0.703.

Alpha curves (test NDCG@10, a=0..1): SciFact/bge 0.680 0.696 0.714 0.729 0.737 0.750 0.757 0.764 0.759 0.745 0.740 (train curve peaks at the same 0.7). The curve is a broad hill: +-0.1 around the optimum loses under 0.015 on SciFact and NFCorpus; +-0.2 can lose up to 0.022 (SciFact/MiniLM at a=0.7). On FiQA, alpha too low (toward BM25) is costly: a=0.4 loses 0.046 and a=0.3 loses 0.085.

### 4. Ours vs reference (all secondary preprints, no code, per 00j)

| Row | Ours (I ran) | Reference | Verdict |
|---|---|---|---|
| RRF k=60, BM25 + dense, SciFact | 0.716 (MiniLM), 0.740 (bge-base) | AgentIR BM25 (k1 1.2, b .75) + BGE-small: 0.690; Pati and Patil equal RRF bge-base + BM42 (test split) 0.707 | Ours, no direct reference (different dense model, different BM25 or BM42). Same direction: RRF beats BM25 and, for weak dense, dense. |
| RRF k=60, NFCorpus | 0.345 (MiniLM), 0.362 (bge-base) | AgentIR 0.352; Pati and Patil 0.364 | Within 0.01 of both; consistent. |
| RRF k=60, FiQA | 0.372 (MiniLM, full corpus) | AgentIR 0.321 (BGE-small); Pati and Patil 0.348 | Ours higher by 0.02 to 0.05. Our dense alone is 0.369 (AgentIR BGE-small alone 0.279), so the gap follows the stronger dense leg, not a fusion difference. |
| Tuned min-max, train alpha, bge-base | SciFact 0.764 (a=0.7), NFCorpus 0.385 (a=0.6) | Pati and Patil (BM42 sparse): 0.753 (a=0.8), 0.379 (a=0.7) | Loose anchor only: BM42 not BM25, alpha convention unverified. Same ordering (tuned > equal RRF > dense for SciFact) and similar alpha. |
| Convex vs RRF | convex above RRF60 by +0.011 to +0.029 on all five runs (CI excludes 0 on 4 of 5) | Bruch et al.: convex beats RRF on all 9 datasets at NDCG@100/1000 with PISA BM25 + MiniLM | Direction agrees. Not comparable numerically (metric depth differs, they fix alpha=0.8 from in-domain data). Our MiniLM SciFact diff is the one CI that includes 0. |

### 5. Conclusion you can share
- Hybrid is not automatically better than the best leg. With a weak dense model (MiniLM) next to BM25, fusion helped everywhere: RRF60 +0.023 to +0.036 over BM25 on SciFact and NFCorpus. With a strong dense model, plain RRF k=60 was a wash: SciFact/bge-base tied dense (0.000, CI [-0.023, +0.023]), NFCorpus/bge-base was -0.012 (CI includes 0), FiQA/MiniLM +0.003 (CI includes 0). Equal-rank-weight fusion drags the stronger system toward the weaker one.
- Tuning one scalar fixes that. Train-selected convex alpha beat the best single system with a CI clear of zero on all five runs (+0.011 to +0.046) and beat RRF60 by +0.011 to +0.029 (4 of 5 CIs clear of zero). Gains over the best single leg are +0.011 (NFCorpus/bge) to +0.046 (SciFact/MiniLM).
- Train-chosen alpha transferred: it matched the test-oracle alpha or was one step off, and the test score lost at most 0.0006 against oracle (SciFact/MiniLM 0.7268 vs 0.7274; NFCorpus/MiniLM 0.3585 vs 0.3597; FiQA 0.4006 vs 0.4010). Alpha lands at 0.5 to 0.7 in every run, i.e. the dense leg gets somewhat more weight than BM25 after min-max, higher when dense is stronger.
- RRF k barely matters (k 10 to 100 moves 0.001 to 0.01), consistent with Cormack's flat curve; k=60 is fine without labels.
- Practical rule supported by the data: no labels -> RRF k=60 (never much worse than the better leg, always better than the worse one); a few hundred labelled queries -> tune one alpha on a normalised convex combination.
- Hybrid raised recall only when the legs were complementary: SciFact/MiniLM R@100 0.926 and 0.925 fused to 0.953; NFCorpus and FiQA hybrid R@100 is no higher than dense alone.

### 6. Caveats
- One deterministic run, one seed; bootstrap covers query sampling only. 300 (SciFact), 323 (NFCorpus), 648 (FiQA) test queries; "convex beats RRF" CIs are narrow on the lower-bound side (+0.006 on two rows).
- Fusion depth is 100 per leg with missing = 0 after min-max. A different normaliser window (top-1000 dense), z-score, or treating missing as the list minimum could move the convex numbers; not tested. RRF and convex use the same candidates.
- Train queries for NFCorpus and FiQA are a 1000-query sample; train queries differ in distribution from test (SciFact train MiniLM/BM25 mix peaks higher than test), so only the argmax was used.
- bge-base on FiQA was not run (E3 only has the 5,000-doc cap; capped numbers are not comparable with published FiQA and were excluded here).
- References are preprints with no code (00j); AgentIR uses BGE-small and its own BM25 at k1 1.2, b .75, so no row is a clean replication. Bruch et al. is NDCG@100/1000. No NDCG@10 BM25+MiniLM RRF/convex row exists.
- bge-base may have seen in-domain BEIR-style data (E3 caveat), which can inflate the dense leg and shift the best alpha.
- Licences: SciFact/NFCorpus non-commercial, FiQA unverified: numbers only.

### 7. What surprised me
- RRF k=60 gave exactly the dense score on SciFact/bge-base (0.7400 vs 0.7404). Equal weighting of a 0.68 leg and a 0.74 leg cancelled, then alpha 0.7 pulled out +0.024.
- Train-selected alpha was essentially the oracle on every run, even from a 1000-query sample, which is better transfer than the BM25 k1/b selection in E2.
- On FiQA, a mis-set alpha is expensive (0.316 at a=0.3 against 0.401 at best) but RRF60 at 0.372 sits safely between, a fair argument for RRF as a no-label default.
- Our simple RRF numbers landed within 0.01 of the preprint on NFCorpus even though the dense models and BM25 variants differ.

## E5: cross-encoder rerank depth vs NDCG@10 gain vs latency (SciFact, NFCorpus, FiQA)

Label for every "ours" number: **I ran, BEIR <dataset> test, 100-query seeded subsample (50 for the depth-200 run), `cross-encoder/ms-marco-MiniLM-L-6-v2` (the BEIR Table 6 checkpoint, 00y row 7a; BEIR top-k = 100 believed, not re-read), sentence-transformers 6.2.0, torch 2.15.0+cpu, transformers 5.22.1, numpy 2.5.4, pytrec-eval-terrier 0.5.11, python 3.12.3, CPU only (4 shared cores, 3 torch threads), seed 42, max_length 512, input = (query, title + " " + text).** Code `experiments/e5_rerank.py` (modes score / latency / report). Run files `results/e5_<bm25|bm25e3|hybrid-minilm|hybrid-bge-base>_<dataset>_*.json` (per-depth, per-query NDCG@10, bootstrap, hurt counts) and `results/e5_latency_*.json`; CE score cache `data/e5_ce_cache_<dataset>.json`. First stage = saved runs only, no re-encoding: `bm25` = E2 stemmed BM25 top-100 (k1 .9, b .4); `bm25e3` = E3 `e3_bm25_scifact.json` top-200 (**unstemmed bm25s**, not the E2 engine, SciFact only); `hybrid-*` = the E4 convex train-alpha test run (top-100). Reranked list at depth d = top-d reordered by CE score, ranks d+1..100 unchanged. All NDCG@10 and R@100 from `ir_metrics.py`, asserted equal to pytrec_eval to 1e-6 on every first-stage and reranked run (max abs diff 2.2e-16). Bootstrap: paired, 2000 resamples, seed 42, per-query NDCG@10 vs the first stage, 95% CI.

### 0. Deviation you must read first: a query subsample
The CPU scored about 7 query-doc pairs/s (SBERT's docs list 1,800 docs/s for this model on a GPU), so a full run (about 400k pairs) would take about 16 hours. I used a **seeded random subsample of 100 test queries per dataset (seed 42, same 100 for every first stage of a dataset), 50 for the depth-200 BM25 run**. Consequences: CIs are wide (half-width about 0.02 to 0.05 at n=100, up to about 0.08 at n=50), and first-stage numbers on the subsample differ from the full-test numbers in E2/E4 (SciFact BM25 0.700 vs 0.680 full; NFCorpus BM25 0.331 vs 0.322). Compare gains within a row, not absolute levels with E2/E4 or with BEIR.

### 1. Ours vs reference, depth 100 (NDCG@10)
| Row | First stage | + CE d=100 (I ran) | Gain | Reference | Verdict |
|---|---|---|---|---|---|
| SciFact, BM25 (E2), n=100 | 0.700 | 0.742 | +0.042 [-0.004, +0.087] | BEIR Table 6: BM25 0.665 to BM25+CE 0.688 (+0.023), full test | Same direction, similar size (the reference gain +0.023 lies inside our CI). Gains only: our absolute levels are on a 100-query subsample and are not comparable with BEIR's full-test levels. Loose sanity pass, not a reproduction. |
| NFCorpus, BM25 (E2), n=100 | 0.331 | 0.339 | +0.008 [-0.017, +0.037] | BEIR Table 6: BM25 0.325 to 0.350 (+0.025) | Same direction, smaller point gain; our CI includes both 0 and the reference gain +0.025, so not distinguishable from it. Absolute levels not compared (subsample vs full test). |
| FiQA | BM25 not reranked (only MiniLM-hybrid candidates by the brief) | n/a | n/a | BEIR 0.236 to 0.347 | No like-for-like row. |
| SciFact, hybrid-MiniLM (E4), n=100 | 0.745 | 0.741 | -0.004 [-0.042, +0.034] | none | ours, no reference |
| SciFact, hybrid-bge-base, n=100 | 0.795 | 0.744 | -0.051 [-0.100, -0.008] | none | ours, no reference |
| NFCorpus, hybrid-MiniLM / bge-base, n=100 | 0.364 / 0.401 | 0.345 / 0.348 | -0.019 [-0.041, +0.003] / -0.054 [-0.085, -0.024] | none | ours, no reference |
| FiQA full, hybrid-MiniLM, n=100 | 0.374 | 0.339 | -0.035 [-0.075, +0.005] | BEIR BM25+CE 0.347 (different first stage) | ours, no reference (BEIR's row starts from BM25, ours from a hybrid on a 100-query subsample, so neither the levels nor the gains are comparable). |

Loose anchor, vendor only (Qdrant "When Is a Reranker Worth It?", SciFact, 200 queries, tuned BM25+MiniLM-L6 fusion 0.7656, best of 4 rerankers, different models): rerank NDCG@10 .782/.797/.794/.796/.798 at depth 10/25/50/100/200. Their shape (most gain by 10 to 25, flat afterwards) matches our SciFact BM25 curve; absolute numbers and the gain over fusion (+0.016 to +0.033) are not comparable (their best reranker was jina-v2, their fusion is strong). NFCorpus and FiQA depth curves below are ours, no reference.

### 2. Depth curves (NDCG@10, I ran; gain vs first stage with paired-bootstrap 95% CI; help/hurt/tie queries)
SciFact, E2 stemmed BM25 first stage, n=100 (first stage 0.700; first-stage R@10/20/50/100 = 0.82/0.85/0.885/0.895):
| Depth | NDCG@10 | Gain [CI] | Help/Hurt/Tie | Latency mean / p95 per query (this contended CPU only, not portable) |
|---|---|---|---|---|
| 10 | 0.742 | +0.042 [+0.007, +0.078] | 17/4/79 | 1.39 s / 1.60 s |
| 20 | 0.740 | +0.040 [-0.001, +0.083] | 18/4/78 | 2.75 s / 3.14 s |
| 50 | 0.751 | +0.050 [+0.007, +0.093] | 21/4/75 | 6.92 s / 7.80 s |
| 100 | 0.742 | +0.042 [-0.004, +0.087] | 20/5/75 | 13.96 s / 15.67 s |

SciFact, E3 unstemmed bm25s first stage, n=50 (first stage 0.615; R@10/20/50/100/200 = 0.68/0.71/0.79/0.79/0.83):
| Depth | NDCG@10 | Gain [CI] | Help/Hurt/Tie | R@100 of reranked list | Latency mean / p95 (10 queries, this contended CPU only, not portable) |
|---|---|---|---|---|---|
| 10 | 0.621 | +0.006 [-0.037, +0.049] | 6/2/42 | 0.79 | 1.71 s / 2.34 s |
| 20 | 0.637 | +0.022 [-0.048, +0.090] | 7/2/41 | 0.79 | 3.41 s / 4.70 s |
| 50 | 0.661 | +0.046 [-0.027, +0.119] | 10/2/38 | 0.79 | 9.31 s / 12.10 s |
| 100 | 0.647 | +0.032 [-0.046, +0.110] | 9/2/39 | 0.79 | 18.66 s / 22.33 s |
| 200 | 0.646 | +0.031 [-0.048, +0.108] | 9/2/39 | 0.83 | 42.59 s / 71.87 s |

(R@100 of a reranked list equals first-stage R@100 for d <= 100 because the top-100 set is only permuted; only depth 200 can change it. Depth-200 latency is noisy: E7 shared the host.)

NFCorpus, E2 stemmed BM25, n=100 (first stage 0.331; first-stage R@10/20/50/100 = 0.178/0.205/0.245/0.279):
| Depth | NDCG@10 | Gain [CI] | Help/Hurt/Tie | Latency mean / p95 (this contended CPU only, not portable) |
|---|---|---|---|---|
| 10 | 0.346 | +0.015 [-0.001, +0.036] | 29/20/51 | 1.38 s / 1.61 s |
| 20 | 0.345 | +0.014 [-0.008, +0.039] | 28/22/50 | 2.76 s / 3.10 s |
| 50 | 0.343 | +0.012 [-0.012, +0.039] | 30/23/47 | 6.93 s / 7.54 s |
| 100 | 0.339 | +0.008 [-0.017, +0.037] | 28/23/49 | 13.83 s / 14.43 s |

Hybrid candidates (E4 convex, train alpha), gain vs the hybrid first stage:
| Dataset / dense | First stage | d=10 | d=20 | d=50 | d=100 | Hurt queries at d=100 |
|---|---|---|---|---|---|---|
| SciFact / MiniLM (n=100) | 0.745 | +0.012 [-0.015, +0.042] | +0.012 [-0.023, +0.047] | +0.000 [-0.038, +0.039] | -0.004 [-0.042, +0.034] | 13 (13 help) |
| SciFact / bge-base | 0.795 | -0.015 [-0.050, +0.019] | -0.028 [-0.070, +0.009] | -0.046 [-0.094, -0.003] | -0.051 [-0.100, -0.008] | 19 (9 help) |
| NFCorpus / MiniLM | 0.364 | -0.000 [-0.011, +0.012] | -0.010 [-0.028, +0.009] | -0.019 [-0.040, +0.002] | -0.019 [-0.041, +0.003] | 39 (25 help) |
| NFCorpus / bge-base | 0.401 | -0.022 [-0.042, -0.005] | -0.037 [-0.065, -0.014] | -0.052 [-0.082, -0.024] | -0.054 [-0.085, -0.024] | 41 (24 help) |
| FiQA full / MiniLM (n=100) | 0.374 | -0.014 [-0.038, +0.010] | -0.010 [-0.042, +0.023] | -0.027 [-0.065, +0.010] | -0.035 [-0.075, +0.005] | 36 (19 help) |

Latency (same for every candidate set, scoring cost only; first-stage cost excluded: BM25 about 1 ms, MiniLM query encode about 26 ms, bge-base about 144 ms from E3): FiQA 1.43/2.86/7.36/15.16 s mean and 1.96/3.78/9.03/19.17 s p95 at depth 10/20/50/100 (15 queries each). Cost is linear: about 0.14 s per pair on this host, so depth 100 costs 10x depth 10. **CPU, 4 shared cores, load average 4 to 5, batch of 10 pairs, max_length 512: absolute seconds are not portable; the ratio (depth 100 = 10x depth 10 = 1.0x per doc) is the usable fact.** Latency per depth is the summed time of the first d/10 chunks of 10 pairs, measured in a separate fresh timing pass on the first 15 queries of the same seeded draw (first 10 for the depth-200 run), not derived from the cached scoring pass. Depth 100 / depth 10 mean-latency ratio per run: 10.0 (SciFact BM25), 10.0 (NFCorpus BM25), 10.6 (FiQA hybrid), 10.9 (SciFact unstemmed, contended).

### 3. Cost/gain knee
- **BM25 first stage: knee at depth 10 to 20.** SciFact: depth 10 already gives +0.042 (the whole gain; 50 gives +0.050, +0.009 more for 5x the cost; paired bootstrap of depth 50 vs depth 10, QA recompute: +0.009 [-0.014, +0.034], CI covers 0). NFCorpus: depth 10 gives +0.015 and gain shrinks with depth (+0.008 at 100). Unstemmed SciFact (n=50) is the only curve that rises to a peak at 50 (+0.046) and then flattens, but its CIs include 0 at every depth (n=50, weaker first stage), so this curve, including depth 200, is inconclusive.
- **Hybrid first stage: no knee, the cheapest depth is the least harmful.** Gain is at best a non-significant +0.012 (SciFact/MiniLM, depth 10 to 20, CI includes 0) and drifts negative as depth grows (strongest on bge-base, where the first stage was already the best system we have).
- Ceiling check: reranking can only reorder what the first stage found. SciFact BM25 R@10 is 0.82 and R@100 0.895, so a perfect reranker would reach about 0.90 at depth 100 (QA recompute: oracle reorder of the top-100 gives NDCG@10 0.896 on the subsample); we reach 0.74 NDCG@10. NFCorpus R@100 is 0.28 (graded, many relevant docs), but an oracle reorder of the same top-100 still reaches NDCG@10 0.558 against our 0.339, so on NFCorpus the reranker, not only recall, leaves most of the headroom unused.

### 4. Failure cases: queries where reranking hurts (counts above, depth 100)
BM25 stemmed: 5 of 100 on SciFact, 23 of 100 on NFCorpus. Hybrid: 13 to 41 of 100 per run (SciFact bge-base 19 hurt vs 9 helped; NFCorpus bge-base 41 vs 24). At depth 10 the hurt count is lower than at depth 100 in every row except the unstemmed SciFact run, where it is equal (2 and 2) (e.g. SciFact BM25 4 vs 5, SciFact bge-base 13 vs 19). On the graded NFCorpus data the CE hurts and helps about equally often even from BM25 (helped 28, hurt 23 at d=100), so the mean gain is small relative to churn. Worst individual drops per run are in each JSON (`worst_drops`).

### 5. Conclusion you can share
All bullets: I ran, SciFact / NFCorpus / FiQA, 100-query seeded subsample, one MiniLM-L6 checkpoint on CPU.
- A MiniLM cross-encoder over stemmed BM25 at depth 10 raised NDCG@10 on SciFact (+0.042 [+0.007, +0.078], CI excludes 0) and nudged it on NFCorpus (+0.015 [-0.001, +0.036], CI includes 0), the same direction as BEIR's depth-100 gains (0.665 to 0.688 SciFact, 0.325 to 0.350 NFCorpus). Going deeper bought nothing measurable: SciFact depth 50 vs depth 10 is +0.009 [-0.014, +0.034], at 5x the scoring cost, and depth 100 costs 10x depth 10.
- **Rerank a strong hybrid and it can lose.** Over the E4 bge-base hybrid the reranker lowered NDCG@10 on both datasets, with the CI excluding 0 on both at depth 50 and 100 (depth 100: SciFact -0.051 [-0.100, -0.008], NFCorpus -0.054 [-0.085, -0.024]). At depth 100 the point gain was negative on all five hybrid runs, but on the three MiniLM-hybrid runs every CI includes 0. A plausible reading (not tested here) is that this MS MARCO-trained MiniLM-L6 is a weaker ranker on these domains than the bge-base fusion it replaces, and a full reorder discards the first stage's signal. On our data, a reranker helped only when the first stage was BM25 alone.
- If you must rerank, depth 10 to 20 is the cheapest, and on our data no deeper setting was significantly better than depth 10 (point estimates at depth 50 were higher on the two SciFact BM25 curves, inside the noise).

### 6. Caveats
- 100-query subsample (50 for depth 200, 10 to 15 for latency); CI half-widths are about 0.02 to 0.05 at n=100 and up to 0.08 at n=50, so most per-depth differences in the BM25 rows are not distinguishable from each other. One run, one seed, one reranker, one dataset snapshot.
- Latency is from a contended 4-core CPU with batch 10 and max_length 512; only ratios between depths are usable. A GPU will be much faster (SBERT's published GPU throughput for this model, 1,800 docs/s, against our 7 pairs/s suggests a large factor), but that factor was not measured here. Depth-200 latency was inflated by host contention (load average up to 32) and its depth curve is inconclusive anyway.
- **What not to conclude.** Not that rerankers are useless or harmful in general: this is one small MS MARCO checkpoint (MiniLM-L6, 22M parameters), on CPU, 100 queries per dataset, CIs about +-0.04, no larger or domain-tuned reranker, no score fusion of CE and first-stage scores. Not that depth 200 is or is not worth it: that curve is n=50 on a weaker unstemmed first stage and every CI includes 0. Not absolute NDCG levels: subsample levels must not be compared with the full-test E2/E4 numbers or with BEIR's absolute numbers, only gains. Not absolute seconds: they belong to this contended host.
- The depth-200 BM25 candidates are E3's unstemmed bm25s run, a different (weaker) first stage than the E2 curve; do not stitch the two curves into one.
- Hybrid candidates inherit E4's train-selected alpha and bge-base's possible in-domain data exposure. BEIR Table 6's top-k (believed 100) was not re-read. FiQA was only run on MiniLM-hybrid candidates.
- Qdrant sweep is a vendor blog (200 scored queries, 37% not held out, different models), used only as a shape anchor.
- Licences: SciFact/NFCorpus non-commercial, FiQA unverified: numbers only.

### 7. What surprised me
- The cross-encoder scored below the bge-base hybrid on both datasets I tried (SciFact, NFCorpus) at every depth, and the drop deepened with depth (CI excludes 0 from depth 10 on NFCorpus, from depth 50 on SciFact). I expected a flat line, not a downhill one.
- CPU throughput: 7 pairs/s for a 22M-parameter model. A 100-deep rerank is 14 s a query here; the cost curve is the story, not the gain curve.
- Almost all of the SciFact gain arrived at depth 10, which matches the vendor sweep (their depth 10 is already +0.016 of +0.033).
- On NFCorpus the reranker helps and hurts nearly the same number of queries, so a +0.015 mean hides 20 of 100 queries getting worse.

### 8. QA (2026-11-03)
Independent recompute from the saved first-stage runs plus the CE score cache, scored directly with pytrec_eval (not `ir_metrics.py`), paired bootstrap 2000 resamples seed 42: every NDCG@10, gain, CI bound and help/hurt/tie count in sections 1 to 4 matches the `e5_*.json` files to 1e-9 (8 runs, 33 depth rows); latency means and p95 match `e5_latency_*.json`. Same 100 query ids for every first stage of a dataset (the n=50 depth-200 set is a subset of them); candidates come only from the E2/E3/E4 run files; depths are a fixed list and the reranker is off the shelf, so nothing was tuned on test. Wording fixed in this pass: absolute-level comparisons with BEIR removed from section 1, CI-width statement, hurt-count claim, NFCorpus ceiling (oracle 0.558, so recall does not explain the gap), conclusion bullets (NFCorpus depth-10 CI includes 0; "never worse than deeper" was false on point estimates), GPU speed-up projection, latency sample description.

## E7: gold-set noise (synthetic label errors vs NDCG@10 and system rankings)

Label for every number: **I ran, BEIR SciFact test (300 queries, 339 relevant labels, all grade 1) and NFCorpus test (323 queries, 12,334 labels: 11,758 grade 1, 576 grade 2), own `ir_metrics.py` NDCG@10 (linear), numpy 2.5.4, scipy 1.18.1, pytrec-eval-terrier 0.5.11, python 3.12.3, CPU only.** Code `experiments/e7_gold_noise.py`; raw `results/e7_scifact_20261103T065919.json`, `results/e7_nfcorpus_20261103T065945.json` (per-rep system means, pair deltas, bootstrap CIs, versions, dataset sha256). No encoding, no LLM judge: systems are reused run files: BM25 `e2_bm25_<ds>.json` (Porter, k1 0.9, b 0.4), MiniLM and bge-base `e3_*`, RRF k=60 = `e4_bge-base_<ds>.json` `runs_test.rrf60` (BM25 + bge-base). Rates 0, 5, 10, 20, 30%; 40 repetitions per scenario and rate with seeds 42..81 (`default_rng([seed, scenario, rate])`); 1,000-resample paired bootstrap on per-query deltas inside every repetition; 0% is deterministic (1 run).

### 1. Sanity rows (0% noise) and cross-check
| System | SciFact ours (I ran) | E2/E3/E4 value | NFCorpus ours | E2/E3/E4 value |
|---|---|---|---|---|
| BM25 | 0.6803 | 0.6803 (E4 json; E2 0.680) | 0.3221 | 0.3221 (E2 0.322) |
| MiniLM | 0.6451 | 0.64508 (E3) | 0.3159 | 0.31594 (E3) |
| bge-base | 0.7404 | 0.74039 (E3) | 0.3735 | 0.37355 (E3) |
| RRF60 | 0.7400 | 0.74002 (E4) | 0.3617 | 0.36166 (E4) |

Worst per-query |ours - pytrec_eval ndcg_cut_10| over the clean qrels and 2 seeds x 2 rates x all scenarios x 4 systems of perturbed qrels: 2.2e-16 (SciFact), 3.3e-16 (NFCorpus); tolerance 1e-6.

Noise definitions (all synthetic): (a1) `a_drop_random`: each relevant label dropped with probability p. (a2) `a_drop_pooled_bm25`: p of all relevant labels dropped, outside-BM25-top-10 labels first (a judged pool made by one lexical system, i.e. pooling bias). (b1) `b_flip_random`: per query add round(p x n_rel) false grade-1 labels drawn from non-relevant docs in the union of all four systems' top-10. (b2) `b_flip_dense_biased`: same count, drawn only from bge-base's top-10 (a labeller that agrees with one system). (c) `c_grade_jitter` (NFCorpus only): each label's grade swaps 1<->2 with probability p. Queries left with no relevant label are dropped for all systems (trec_eval rule); the JSON separates `delta_label_only` from `delta_query_subset_only`.

### 2. Change in NDCG@10 vs clean (mean over 40 reps; BM25 / MiniLM / bge-base / RRF60)
SciFact (clean 0.680 / 0.645 / 0.740 / 0.740):
| Scenario | 10% | 30% |
|---|---|---|
| a1 drop random | -0.003 / -0.003 / -0.003 / -0.003 | -0.017 / -0.010 / -0.011 / -0.013 |
| a2 drop, BM25-pooled | +0.060 / +0.039 / +0.039 / +0.051 | +0.151 / +0.097 / +0.097 / +0.129 |
| b1 flip random | -0.015 / -0.013 / -0.018 / -0.018 | -0.043 / -0.040 / -0.052 / -0.052 |
| b2 flip dense-biased | -0.016 / -0.012 / -0.002 / -0.012 | -0.052 / -0.038 / -0.008 / -0.040 |

NFCorpus (clean 0.322 / 0.316 / 0.374 / 0.362):
| Scenario | 10% | 30% |
|---|---|---|
| a1 drop random | -0.023 / -0.023 / -0.026 / -0.025 | -0.072 / -0.070 / -0.083 / -0.081 |
| a2 drop, BM25-pooled | +0.006 / -0.004 / -0.005 / +0.002 | +0.023 / -0.010 / -0.011 / +0.009 |
| b1 flip random | +0.079 / +0.080 / +0.071 / +0.070 | +0.176 / +0.173 / +0.153 / +0.156 |
| b2 flip dense-biased | +0.042 / +0.056 / +0.149 / +0.084 | +0.076 / +0.107 / +0.285 / +0.155 |
| c grade jitter | -0.053 / -0.053 / -0.058 / -0.058 | -0.062 / -0.061 / -0.069 / -0.068 |

Per-rep sd of a system's NDCG@10 is small (under 0.01 for random scenarios; see JSON). Random noise moves every system by almost the same amount, so absolute NDCG@10 is the thing that drifts; gaps move far less (sd of pair-gap shift 0.002 to 0.01 for random scenarios at 30%).

### 3. Do system rankings flip?
Clean order: SciFact bge-base (0.7404) ~ RRF60 (0.7400) > BM25 > MiniLM; NFCorpus bge-base > RRF60 > BM25 > MiniLM. Fraction of repetitions where every pair with a clean gap of at least 0.03 keeps its order: **1.00 in every scenario and rate except SciFact pooled drop at 30% (0.72, bge-base vs BM25 flips in 28% of reps)**. Pairs inside noise flip early and often:
- SciFact bge-base vs RRF60 (clean gap 0.0004): a coin flip under random noise (flip prob 0.35 to 0.53 for random drop and random flip). Under BM25-pooled drop at 20% and 30% the paired-bootstrap CI shows RRF60 significantly above bge-base in 88% and 100% of reps, a significant reversal manufactured by the pool.
- NFCorpus BM25 vs MiniLM (clean gap 0.006): flip prob 0.10 to 0.15 under random drop at 10 to 30%; under dense-biased flips 0.65 at 5%, 1.00 at 10% and 20%, and at 30% the CI says MiniLM is significantly better in 65% of reps.
- NFCorpus bge-base vs RRF60 (gap 0.012): flips in 85% (20%) and 100% (30%) of reps under BM25-pooled drop.
- Kendall tau (clean vs noisy system means, 4 systems) averages 0.94 to 1.0 for grade jitter and random drop on NFCorpus, but 0.58 to 0.88 on SciFact (1.0 for dense-biased flips there) and 0.67 for NFCorpus dense-biased flips, almost entirely due to the near-tie pairs above. Treat tau on 4 systems with a tied pair as a poor summary.

### 4. When does a 0.05 or 0.02 gap become unreliable?
Method: for each repetition and each of the six system pairs, take (noisy gap - clean gap) and pool (240 values per cell, not independent); `gap_flip[g]` = fraction below -g. "Unreliable" = flip probability of at least 0.05. This is a pooled-over-pairs empirical estimate, not a per-pair test.
| Gap | Scenario | SciFact first unreliable rate (flip prob) | NFCorpus first unreliable rate (flip prob) |
|---|---|---|---|
| 0.05 | random drop / random flip / dense-biased flip / jitter | none up to 30% (0 to 0) | none up to 30% (0) |
| 0.05 | BM25-pooled drop | 30% (0.12) | none up to 30% (0) |
| 0.02 | random drop | none up to 30% (0.03) | none up to 30% (0.01) |
| 0.02 | random flip | 30% (0.054) | 20% (0.10) |
| 0.02 | dense-biased flip | none up to 30% (0.03) | 20% (0.14) |
| 0.02 | BM25-pooled drop | 10% (0.12) | 20% (0.14) |
| 0.02 | grade jitter | n/a | none up to 30% (0) |
Real pairs agree: bge-base vs BM25 (gap 0.060 SciFact, 0.051 NFCorpus) never flips in point estimate except SciFact pooled 30% (0.28), and the CI excludes zero in 97 to 100% of reps for most settings (SciFact pooled 20%: 0%, because the query set shrinks to 253 and BM25-missed queries vanish; NFCorpus pooled 30%: 35%). Clean paired-bootstrap CIs (2,000 resamples, seed 42): SciFact bge-base minus BM25 +0.060 [+0.026, +0.094]; NFCorpus +0.051 [+0.033, +0.070]; NFCorpus RRF60 minus BM25 +0.040 [+0.029, +0.051]; SciFact BM25 minus MiniLM +0.035 [-0.000, +0.071].

### 5. Conclusion you can share
- Random label noise at up to 30% did not flip any gap of 0.03 or more, on either dataset, and a 0.05 gap held up to 30%. It does move the absolute number: NFCorpus NDCG@10 falls 0.02 (10% drop) to 0.08 (30% drop), SciFact barely (0.003 to 0.017).
- A 0.02 gap is where it breaks: under biased noise (a BM25-built pool, or a labeller who agrees with one system) the pooled flip probability reaches 0.12 to 0.14 at 10 to 20% noise (random flips also cross 5%, later: 0.10 at 20% on NFCorpus, 0.054 at 30% on SciFact), while random drop and jitter never reach 5% within 30%.
- The dangerous noise is systematic, not random. BM25-pooled drops on SciFact add +0.06 to +0.15 to BM25 and made a hybrid beat its own dense leg significantly; dense-biased false positives on NFCorpus lifted bge-base by +0.15 (10%) to +0.29 (30%) against BM25's +0.04 to +0.08, which turns a 0.05 lead into a 0.16 (10%) to 0.26 (30%) lead that no labeller disagreement would justify.
- Direction of absolute drift depends on the noise type and the data: drops lower NDCG, false positives raise it on graded NFCorpus (many relevant docs, top-10 easy to hit) and lower it on SciFact (one relevant doc per query, extra "relevant" docs inflate the ideal DCG). Do not quote an NDCG drift sign without naming the noise.
- Reference anchors (from 00f, not re-verified here): Voorhees 2000 reports that system rankings stay very stable under assessor disagreement, with a Kendall tau figure of about 0.90 for MAP; per 00y the figure and the "tau 0.9 / 0.8" convention remain UNVERIFIED (00f section 1.4: threshold attribution not confirmed, primary source not opened). The bin we measured agrees in spirit (random noise leaves coarse order alone), but do not cite 0.90 as fact. Pooling-bias claims (Buckley et al., SIGIR 2006; BEIR annotation-bias study) are also M-confidence from search summaries.

### 6. Caveats
- Synthetic, not real annotator noise. Real disagreement is correlated with document type, query type and grade, which random flips do not model. The two systematic scenarios are stylised bounds, not estimates of any real pool.
- Rate meanings differ between scenarios: drops are a fraction of all labels, flips are a fraction of each query's relevant count (so on SciFact 30% adds about 0.3 labels per query, on NFCorpus about 11), jitter is a fraction of labels. Rates are comparable within a scenario, not across.
- Dropped queries are removed, so the SciFact pooled-drop "gain" is mostly a query-subset effect (label-only shift is within 0.01; subset-only shift +0.026 to +0.142 for BM25): the queries BM25 fails on lose their only label and leave the set. Counting them as zero instead would change the sign of that row.
- Grade jitter is lopsided by construction: 95.3% of NFCorpus grades are 1 (11,758 of 12,334), so 1->2 swaps dominate and inflate the ideal DCG; the -0.043 at 5% followed by only -0.062 at 30% is that saturation, not gradual degradation.
- Four systems, two datasets, three of them correlated (RRF contains BM25 and bge-base). Pooled-gap flip probabilities mix real pairs with a 0.0004 tie; they are descriptive. 40 reps gives a resolution of 0.025 on flip probabilities. Hybrid parameters come from E4 (RRF60 only, convex combination not used). No length-biased judge scenario (00f 4.1 item ii) and no LLM judge, by choice.
- Licences: SciFact and NFCorpus non-commercial; numbers only.

### 7. What surprised me
- On SciFact, randomly deleting 30% of the labels barely moved NDCG@10 (-0.01 to -0.02) because single-label queries vanish instead of being scored lower; the cost is a smaller query set (300 to about 213), so wider CIs rather than a lower score.
- A pool built from one lexical system flipped the verdict between two non-lexical-looking systems: it made RRF60 significantly better than bge-base in 100% of reps at 30%, when their clean gap is 0.0004.
- Grade jitter hurt NFCorpus by 0.043 at only 5% noise and then plateaued at 0.062, because there are 20x more grade-1 than grade-2 labels.
- Even 40 reps of 30% random noise never overturned a 0.05 gap; the effective risk to a ranking decision is bias, not variance.
