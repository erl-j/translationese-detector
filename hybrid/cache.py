#!/usr/bin/env python3
"""Cached expert-feature matrices on the trn split (fit data for the hybrid models).

    python3 -m hybrid.cache --gran 500 100 sent      # ~70 CPU-seconds for all three

For each trn.<gran>.bal record: the expert-feature counts (hybrid/features.py)
on the raw text, plus a sparse bag of words (all word types; train.py keeps
the top-N). Only trn is cached: val and test are scored live through the
harness (hybrid/eval_val.py), never read here.

Output: data/cache/hybrid/trn.<gran>.npz (+ .vocab.json), gitignored, tagged
with a hash of features.py (stale caches are refused) and the data sha256.
A cache built under the old name dev.<gran>.npz (same bytes: trn == old dev)
is migrated automatically.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy import sparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evals"))
sys.path.insert(0, ROOT)
import run  # noqa: E402
from hybrid import features as F  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "cache", "hybrid")
SPLIT = "trn"


def features_hash():
    with open(os.path.join(ROOT, "hybrid", "features.py"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def cache_path(gran):
    return os.path.join(CACHE_DIR, "%s.%s.npz" % (SPLIT, gran))


def vocab_path(gran):
    return os.path.join(CACHE_DIR, "%s.%s.vocab.json" % (SPLIT, gran))


def data_file(gran):
    return run.data_path(run.data_dir(), SPLIT, gran, True)


def build(gran):
    t0 = time.time()
    path = data_file(gran)
    recs = run.load_records(path)
    vocab = {}
    C, br, bc, bv = [], [], [], []
    for i, r in enumerate(recs):
        counts, words = F.raw_counts(r["text"], return_words=True)
        C.append(counts)
        c = {}
        for w in words:
            j = vocab.get(w)
            if j is None:
                j = vocab[w] = len(vocab)
            c[j] = c.get(j, 0) + 1
        for j, v in c.items():
            br.append(i)
            bc.append(j)
            bv.append(v)
    bow = sparse.csr_matrix((bv, (br, bc)), shape=(len(recs), len(vocab)), dtype=np.float32)
    words = [None] * len(vocab)
    for w, j in vocab.items():
        words[j] = w
    save(gran, np.asarray(C, np.float32), np.asarray([r["label"] for r in recs], np.int8),
         np.asarray([r["date"] for r in recs]), np.asarray([r["id"] for r in recs]), bow, words,
         run.sha256_file(path))
    print("gran=%s n=%d vocab=%d  %.1fs" % (gran, len(recs), len(vocab), time.time() - t0))


def save(gran, C, labels, dates, ids, bow, words, sha):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path(gran), C=C, labels=labels, dates=dates, ids=ids,
                        bow_data=bow.data, bow_indices=bow.indices, bow_indptr=bow.indptr, bow_shape=bow.shape,
                        count_names=np.asarray(F.COUNT_NAMES), features_hash=features_hash(), data_sha256=sha)
    with open(vocab_path(gran), "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)


def migrate_old(gran):
    """Reuse a dev.<gran>.npz cache (old protocol) if it was built from the same bytes."""
    old = os.path.join(CACHE_DIR, "dev.%s.npz" % gran)
    if not os.path.exists(old):
        return False
    z = np.load(old, allow_pickle=False)
    if str(z["features_hash"]) != features_hash() or str(z["data_sha256"]) != run.sha256_file(data_file(gran)):
        return False
    bow = sparse.csr_matrix((z["bow_data"], z["bow_indices"], z["bow_indptr"]), shape=tuple(z["bow_shape"]))
    with open(os.path.join(CACHE_DIR, "dev.%s.vocab.json" % gran), encoding="utf-8") as f:
        words = json.load(f)
    save(gran, z["C"], z["labels"], z["dates"], z["ids"], bow, words, str(z["data_sha256"]))
    print("gran=%s: migrated %s -> %s (same data sha256, same features.py)" % (gran, old, cache_path(gran)))
    return True


def load(gran):
    if not os.path.exists(cache_path(gran)) and not migrate_old(gran):
        build(gran)
    z = np.load(cache_path(gran), allow_pickle=False)
    if str(z["features_hash"]) != features_hash():
        sys.exit("cache %s is stale (features.py changed); rebuild: python3 -m hybrid.cache --gran %s"
                 % (cache_path(gran), gran))
    d = {k: z[k] for k in z.files}
    d["bow"] = sparse.csr_matrix((d.pop("bow_data"), d.pop("bow_indices"), d.pop("bow_indptr")),
                                 shape=tuple(d.pop("bow_shape")))
    with open(vocab_path(gran), encoding="utf-8") as f:
        d["vocab"] = json.load(f)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gran", nargs="+", default=["500", "100", "sent"])
    ap.add_argument("--migrate", action="store_true", help="only migrate old dev.* caches, never rebuild")
    a = ap.parse_args()
    for g in a.gran:
        g = run.GRANS[g]
        if not migrate_old(g) and not a.migrate:
            build(g)


if __name__ == "__main__":
    main()
