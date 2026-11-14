# Datasets for local runs

Downloaded 2026-11-02 with `python load_data.py scifact nfcorpus fiqa` into `experiments/data/` (gitignored). CPU only; total ~76 MB on disk, no download near the 500 MB flag.

| Dataset | Role | Zip size | Unzipped | Corpus docs | Test queries | URL |
|---|---|---|---|---|---|---|
| SciFact | primary | 2.8 MB (2,816,079 B) | 8 MB | 5,183 | 300 | https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip |
| NFCorpus | secondary (graded qrels) | 2.4 MB (2,448,432 B) | 9.3 MB | 3,633 | 323 | https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip |
| FiQA-2018 | optional | 18 MB (17,948,027 B) | 47 MB | 57,638 | 648 | https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip |

Sizes are from HTTP `Content-Length` before download. Corpus and query counts are from the BEIR loader (BEIR README lists 5K / 3.6K / 57K corpus).

## Licence facts
- BEIR code: Apache-2.0 (BEIR README, fetched 2026-11-02).
- BEIR README gives no per-dataset licence; it says it is the user's responsibility to determine permission to use each dataset.
- Per-dataset terms are NOT verified here. Working assumption from TOPIC-MAP T56: SciFact and NFCorpus are academic/non-commercial, FiQA unverified. Rule: publish numbers only, never redistribute the data or quote documents.

## Layout and integrity
BEIR format: `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`. `common.dataset_hash(name)` is a sha256 over those three files and is stored in every results JSON. First-8 sha256: NFCorpus c10ce6eb, FiQA dddfa29e.
