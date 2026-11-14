"""Download BEIR datasets into experiments/data/ (gitignored).

Usage: python load_data.py [scifact] [nfcorpus] [fiqa]   (default: scifact nfcorpus)
Sizes (zip): scifact 2.8 MB, nfcorpus 2.4 MB, fiqa 18 MB. See DATASETS.md.
"""
import sys

from common import dataset_hash, download_dataset, load_dataset

if __name__ == "__main__":
    names = sys.argv[1:] or ["scifact", "nfcorpus"]
    for n in names:
        folder = download_dataset(n)
        corpus, queries, qrels = load_dataset(n)
        print(f"{n}: {folder} docs={len(corpus)} test_queries={len(qrels)} sha256={dataset_hash(n)[:16]}")
