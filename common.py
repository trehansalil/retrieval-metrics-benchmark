"""Shared helpers for the retrieval-eval experiments (E1-E7)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
LIBS = ["numpy", "scipy", "pytrec_eval-terrier", "ranx", "ir_measures", "rank-bm25",
        "bm25s", "sentence-transformers", "torch", "transformers", "beir", "matplotlib"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def lib_versions() -> dict:
    out = {"python": platform.python_version()}
    for lib in LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = None
    return out


def download_dataset(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Download+unzip a BEIR dataset (idempotent). Returns dataset folder."""
    from beir import util
    folder = Path(data_dir) / name
    if not (folder / "corpus.jsonl").exists():
        util.download_and_unzip(BEIR_URL.format(name=name), str(data_dir))
    return folder


def dataset_hash(name: str, data_dir: Path = DATA_DIR) -> str:
    """sha256 over corpus.jsonl, queries.jsonl and qrels/test.tsv bytes."""
    folder = Path(data_dir) / name
    h = hashlib.sha256()
    for rel in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        h.update(rel.encode())
        h.update((folder / rel).read_bytes())
    return h.hexdigest()


def load_dataset(name: str, split: str = "test", data_dir: Path = DATA_DIR):
    """Return (corpus, queries, qrels) as BEIR dicts; downloads if absent.

    corpus: {id: {"title", "text"}}, queries: {id: text}, qrels: {qid: {docid: int}}.
    """
    from beir.datasets.data_loader import GenericDataLoader
    folder = download_dataset(name, data_dir)
    return GenericDataLoader(data_folder=str(folder)).load(split=split)


@contextmanager
def timer():
    """with timer() as t: ...; t['seconds'] holds wall time afterwards."""
    box = {"seconds": None}
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - t0


def write_result(experiment: str, dataset: str, payload: dict, seed: int | None = None,
                 results_dir: Path = RESULTS_DIR) -> Path:
    """Write results/<experiment>_<dataset>_<timestamp>.json with versions + dataset hash."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    doc = {
        "experiment": experiment,
        "dataset": dataset,
        "dataset_sha256": dataset_hash(dataset),
        "seed": seed,
        "timestamp": stamp,
        "libraries": lib_versions(),
        "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "results": payload,
    }
    path = results_dir / f"{experiment}_{dataset}_{stamp}.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return path
