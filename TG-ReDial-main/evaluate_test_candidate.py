#!/usr/bin/env python3
"""Candidate-set test-split evaluation (CRS standard metric).

Scores each test example against its target + 64 hard/sampled negatives,
matching the validation protocol used during training. This is the metric
reported in conversational recommender literature.

Output: Hit@10, NDCG@10, MRR on the official test split using candidate sets.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
_CODE_DIR = PROJECT_ROOT / "TG-ReDial-main"
if _CODE_DIR.is_dir() and str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from chinese_recommender import ChineseRanker, build_context_text  # noqa: E402

TEST_PATH = PROJECT_ROOT.parent / "TG-ReDial-Data" / "out" / "test.jsonl"
ORDER_PATH = PROJECT_ROOT.parent / "TG-ReDial-Data" / "out" / "movie_id_order.json"
K = 10
MAX_NEG = 64
SEED = 42


def load_test_examples(path):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cids = d.get("target_movie_ids") or []
            tgt = int(cids[0]) if cids else None
            examples.append((d["context"], tgt))
    return examples


def sample_candidates(tgt_idx, n_all, max_neg, rng, exclude=None):
    """Sample max_neg negatives by rejection (O(max_neg) per example)."""
    exclude = exclude or set()
    negs = []
    attempts = 0
    max_attempts = max_neg * 10 + 50
    while len(negs) < max_neg and attempts < max_attempts:
        attempts += 1
        j = rng.randrange(n_all)
        if j != tgt_idx and j not in exclude and j not in negs:
            negs.append(j)
    while len(negs) < max_neg:
        j = rng.randrange(n_all)
        if j != tgt_idx and j not in negs:
            negs.append(j)
    return negs


def ndcg_at_k(rel, k):
    n = len(rel)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    order = np.argsort(-rel)
    rank = int(np.flatnonzero(order == 0)[0])
    if rank >= k:
        return 0.0
    return float(1.0 / np.log2(rank + 2))


def hit_at_k(rel, k):
    n = len(rel)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    order = np.argsort(-rel)
    return 1.0 if int(np.flatnonzero(order == 0)[0]) < k else 0.0


def mrr(rel):
    order = np.argsort(-rel)
    rank = int(np.flatnonzero(order == 0)[0])
    return 1.0 / (rank + 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Candidate-set test-split evaluation (CRS standard).")
    ap.add_argument("--checkpoint", type=Path,
                    default=PROJECT_ROOT.parent / "out_probe" / "best.pt")
    ap.add_argument("--test", type=Path, default=TEST_PATH)
    ap.add_argument("--order", type=Path, default=ORDER_PATH)
    ap.add_argument("--max-neg", type=int, default=MAX_NEG)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ranker = ChineseRanker.load(args.checkpoint)
    order = ranker.order
    id_to_idx = {str(mid): i for i, mid in enumerate(order)}
    n_all = len(order)
    print(f"Catalog: {n_all} titles | Checkpoint: {args.checkpoint}", flush=True)

    # Cache projected title matrix (frozen model).
    with torch.no_grad():
        t = ranker.model.head(ranker.model.proj(
            torch.from_numpy(ranker.title_emb).to(ranker.model.proj.weight.device)))
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    examples = load_test_examples(args.test)
    n = len(examples)
    print(f"Test examples: {n}", flush=True)

    ndcg_sum = hit_sum = mrr_sum = 0.0
    n_missing = 0
    t0 = time.time()

    for i, (ctx, tgt) in enumerate(examples):
        if tgt is None:
            n_missing += 1
            continue
        tgt_idx = id_to_idx.get(str(tgt))
        if tgt_idx is None:
            n_missing += 1
            continue

        # Candidate set: target + sampled negatives.
        negs = sample_candidates(tgt_idx, n_all, args.max_neg, rng)
        cand_idxs = [tgt_idx] + negs  # index 0 is always the target
        cand_emb = ranker.title_emb[cand_idxs]

        # Encode context (same as serving).
        q = ranker._encode_texts([build_context_text(ctx)])
        ctx_vec = torch.from_numpy(q)
        tv = torch.from_numpy(cand_emb)
        scores = ranker.model(ctx_vec, tv).squeeze(0).detach().numpy()

        ndcg_sum += ndcg_at_k(scores, K)
        hit_sum += hit_at_k(scores, K)
        mrr_sum += mrr(scores)

        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{n}", flush=True)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s ({n/elapsed:.1f} ex/s)", flush=True)
    print(f"Targets missing from catalog: {n_missing}", flush=True)
    print(f"\nCandidate-set test metrics (target + {args.max_neg} negatives)", flush=True)
    print(f"  Hit@{K} : {hit_sum / n:.4f}", flush=True)
    print(f"  NDCG@{K}: {ndcg_sum / n:.4f}", flush=True)
    print(f"  MRR     : {mrr_sum / n:.4f}", flush=True)

    results = {
        "checkpoint": str(args.checkpoint),
        "n_test_examples": n,
        "catalog_size": n_all,
        "max_neg": args.max_neg,
        "hit10": float(hit_sum / n),
        "ndcg10": float(ndcg_sum / n),
        "mrr": float(mrr_sum / n),
        "targets_missing": n_missing,
        "elapsed_sec": round(elapsed, 2),
    }
    out_path = PROJECT_ROOT.parent / "out_probe" / "test_candidate_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nSaved metrics to {out_path}", flush=True)


if __name__ == "__main__":
    main()
