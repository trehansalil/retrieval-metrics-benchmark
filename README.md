# Retrieval metrics benchmark

Code and results behind the blog post [Retrieval Evaluation Metrics: When Each One Misleads](https://www.saliltrehan.com/post/retrieval-evaluation-metrics). It scores BM25, bge-base-en-v1.5 and a reciprocal rank fusion hybrid on BEIR SciFact and NFCorpus with recall, precision, hit rate, MRR, MAP and nDCG, and shows where the metrics disagree.

## Headline result

On NFCorpus (323 test queries), BM25 against bge-base-en-v1.5: recall@100 and nDCG@10 named opposite winners on 56 of the 169 queries where both systems scored above zero on both metrics (33%). On SciFact (300 queries), MAP@10 and nDCG@10 never disagreed across 131 queries. These counts are descriptive, with no correction for multiple comparisons.

## What is here

| File | What it does |
|---|---|
| ir_metrics.py | Metric code: recall, precision, hit rate, MRR, MAP, nDCG (linear and exponential gain) |
| test_ir_metrics.py | 5 tests, including both Wikipedia nDCG values (0.785 and 0.961) and cross-checks against pytrec_eval and ranx |
| e1_metric_behavior.py | E1: same rankings scored with many metrics; query-level disagreement; nDCG variants; paired bootstrap |
| e2_bm25.py | E2: BM25 with a k1 and b sweep (selected settings are in results/) |
| e3_dense_vs_bm25.py | E3: dense models against BM25 |
| e7_gold_noise.py | E7: synthetic label noise (random, pooled, biased) and how it changes system order |
| common.py, load_data.py | Dataset download, hashing, result writing |
| RESULTS.md, DATASETS.md | Written results and dataset notes |
| results/ | The saved JSON from the run used in the post |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python load_data.py          # downloads SciFact and NFCorpus from BEIR
pytest
```

The first three tests need only numpy, pytrec_eval and ranx. The two SciFact cross-check tests also need the full requirements (bm25s) and the downloaded SciFact data. Dataset files are not included because of their licences; the result JSON holds numbers and query text only, no document text.

## Read this before quoting numbers

Everything is one deterministic run, seed 42, on two datasets and three systems, with bootstraps over queries only. The E1 table uses BM25 with train-selected k1 and b. E3 and E7 use an Anserini-style BM25 (k1 0.9, b 0.4), so their gaps differ slightly from E1's. Noise results are synthetic: there is no human or LLM-judge noise and no online metrics. See the post for the full list of what was not tested.

## Versions

numpy 2.5.4, pytrec-eval-terrier 0.5.11, ranx 0.3.21, sentence-transformers 6.2.0, pytest 9.2.0. Exact pins are in requirements.txt.

## Licence

MIT, see LICENSE. SciFact and NFCorpus have their own terms.
