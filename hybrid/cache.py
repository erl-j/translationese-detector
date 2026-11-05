#!/usr/bin/env python3
"""Build the cached feature matrices for the hybrid models (run once per granularity).

    python3 -m hybrid.cache --gran 500        # ~20 CPU-seconds
    python3 -m hybrid.cache --gran 100 sent

For each dev.<gran>.bal record: expert-feature counts (hybrid/features.py) on
the raw text; for dev-val records also on the topic-masked text
(evals/topicmask.py, exactly what `evals/run.py --mask-topic` feeds a
classifier). Also a sparse bag of words (all word types; the models select
the top-N on dev-train). dev-train is never masked: models are always fitted
on unmasked text, as a deployed classifier would be.

Output: data/cache/hybrid/dev.<gran>.npz (+ .vocab.json), gitignored. Stores
a hash of features.py so stale caches are refused by hybrid/train.py.
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
import devsplit  # noqa: E402
import run  # noqa: E402
import topicmask  # noqa: E402
from hybrid import features as F  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "cache", "hybrid")


def features_hash():
    with open(os.path.join(ROOT, "hybrid", "features.py"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def cache_path(gran):
    return os.path.join(CACHE_DIR, "dev.%s.npz" % gran)


def build(gran):
    t0 = time.time()
    path = run.data_path(run.data_dir(), "dev", gran, True)
    recs = run.load_records(path)
    vocab = {}

    def bow_row(words, rows, cols, vals, i):
        c = {}
        for w in words:
            j = vocab.get(w)
            if j is None:
                j = vocab[w] = len(vocab)
            c[j] = c.get(j, 0) + 1
        for j, v in c.items():
            rows.append(i)
            cols.append(j)
            vals.append(v)

    C, Cm, val_idx = [], [], []
    br, bc, bv, mr, mc, mv = [], [], [], [], [], []
    for i, r in enumerate(recs):
        counts, words = F.raw_counts(r["text"], return_words=True)
        C.append(counts)
        bow_row(words, br, bc, bv, i)
        if devsplit.subset_of(r) == "dev-val":
            k = len(val_idx)
            val_idx.append(i)
            mcounts, mwords = F.raw_counts(topicmask.mask(r["text"]), return_words=True)
            Cm.append(mcounts)
            bow_row(mwords, mr, mc, mv, k)
    V = len(vocab)
    bow = sparse.csr_matrix((bv, (br, bc)), shape=(len(recs), V), dtype=np.float32)
    bowm = sparse.csr_matrix((mv, (mr, mc)), shape=(len(val_idx), V), dtype=np.float32)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        cache_path(gran), C=np.asarray(C, np.float32), C_masked=np.asarray(Cm, np.float32),
        val_idx=np.asarray(val_idx, np.int64), labels=np.asarray([r["label"] for r in recs], np.int8),
        dates=np.asarray([r["date"] for r in recs]), ids=np.asarray([r["id"] for r in recs]),
        is_val=np.asarray([devsplit.subset_of(r) == "dev-val" for r in recs]),
        bow_data=bow.data, bow_indices=bow.indices, bow_indptr=bow.indptr, bow_shape=bow.shape,
        bowm_data=bowm.data, bowm_indices=bowm.indices, bowm_indptr=bowm.indptr, bowm_shape=bowm.shape,
        count_names=np.asarray(F.COUNT_NAMES), features_hash=features_hash(),
        data_sha256=run.sha256_file(path), val_start=devsplit.VAL_START)
    words = [None] * V
    for w, j in vocab.items():
        words[j] = w
    with open(os.path.join(CACHE_DIR, "dev.%s.vocab.json" % gran), "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)
    with open(os.path.join(CACHE_DIR, "dev.%s.speakers.json" % gran), "w", encoding="utf-8") as f:
        json.dump([r.get("speakers") or [r.get("speaker")] for r in recs], f, ensure_ascii=False)
    print("gran=%s n=%d (val %d) vocab=%d  %.1fs" % (gran, len(recs), len(val_idx), V, time.time() - t0))


def load(gran):
    z = np.load(cache_path(gran), allow_pickle=False)
    if str(z["features_hash"]) != features_hash():
        sys.exit("cache %s is stale (features.py changed); rebuild: python3 -m hybrid.cache --gran %s"
                 % (cache_path(gran), gran))
    d = {k: z[k] for k in z.files}
    d["bow"] = sparse.csr_matrix((d.pop("bow_data"), d.pop("bow_indices"), d.pop("bow_indptr")),
                                 shape=tuple(d.pop("bow_shape")))
    d["bow_masked"] = sparse.csr_matrix((d.pop("bowm_data"), d.pop("bowm_indices"), d.pop("bowm_indptr")),
                                        shape=tuple(d.pop("bowm_shape")))
    with open(os.path.join(CACHE_DIR, "dev.%s.vocab.json" % gran), encoding="utf-8") as f:
        d["vocab"] = json.load(f)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gran", nargs="+", default=["500"])
    for g in ap.parse_args().gran:
        build(run.GRANS[g])


if __name__ == "__main__":
    main()
