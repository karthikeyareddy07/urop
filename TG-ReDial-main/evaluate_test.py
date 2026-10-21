#!/usr/bin/env python3
"""Official test-split evaluation for the frozen-ALBERT Chinese movie recommender.

Loads the trained probe (``out_probe/best.pt``), encodes the held-out test split
(``test.jsonl``) over the full 33,834-title catalog, and reports
NDCG@10 / Hit@10 / Recall@20 / MRR.

CPU-only.  The projected title matrix is computed once (model frozen); every
batch of contexts is scored in a single torch matmul, so the whole test pass is
a handful of large GEMMs instead of a loop of per-example forward calls.
"""
from __future__ import annotations

import argparse
import json
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
MAX_CTX_TOKENS = 256
CATALOG_BATCH = 64
K_NDCG = 10


def encode_contexts_batch(enc, tok, contexts, max_length=MAX_CTX_TOKENS):
    """Mean-pooled, L2-normalized context embeddings for a batch of contexts."""
    texts = [build_context_text(c) for c in contexts]
    toks = tok(texts, padding=True, truncation=True, max_length=max_length,
               return_tensors="pt")
    h = enc(**toks).last_hidden_state
    mask = toks["attention_mask"].unsqueeze(-1).to(h.dtype)
    v = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def load_test_examples(path):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cids = d.get("target_movie_ids") or []
            tgt = int(cids[0]) if cids else None
            examples.append((d["context"], tgt))
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the frozen probe on the official test split.")
    ap.add_argument("--checkpoint", type=Path,
                    default=PROJECT_ROOT.parent / "out_probe" / "best.pt")
    ap.add_argument("--test", type=Path, default=TEST_PATH)
    ap.add_argument("--batch", type=int, default=CATALOG_BATCH,
                    help="titles per catalog scoring batch")
    args = ap.parse_args()

    ranker = ChineseRanker.load(args.checkpoint)
    order = ranker.order
    id_to_idx = {str(mid): i for i, mid in enumerate(order)}
    n_titles = len(order)
    print(f"Ranker: {ranker.model.__class__.__name__}"
          f" | title_emb {n_titles} titles | checkpoint {args.checkpoint}", flush=True)

    # Cache the frozen model's projected, L2-normalized title matrix once.
    with torch.no_grad():
        t = ranker.model.head(ranker.model.proj(
            torch.from_numpy(ranker.title_emb).to(ranker.model.proj.weight.device)))
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    print(f"Projected title matrix cached: {t.shape} ({t.element_size()*t.numel()/1e6:.1f} MB)",
          flush=True)

    examples = load_test_examples(args.test)
    n = len(examples)
    print(f"Test examples: {n}", flush=True)

    ndcg10_sum = hit10_sum = recall20_sum = mrr_sum = 0.0
    n_missing_target = 0
    t0 = time.time()

    for start in range(0, n, args.batch):
        chunk = examples[start:start + args.batch]
        ctxs, tgts = zip(*chunk)
        ctx_vecs = encode_contexts_batch(ranker.enc, ranker.tok, list(ctxs))
        # Mirror the model's serving projection+head (enc -> proj_dim, then head),
        # matching how ``ChineseRanker.recommend`` builds the query vector.
        q = ranker.model.head(ranker.model.proj(ctx_vecs))
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        scores = q @ t.T                              # (B, N)
        scores = scores * ranker.model.temperature.exp().clamp(1.0, 20.0)
        scores = scores.detach().numpy()
        for i, tgt in enumerate(tgts):
            if tgt is None:
                n_missing_target += 1
                print(f"  [warn] target id {tgt!r} not in catalog; skipping",
                      flush=True)
                continue
            tgt_idx = id_to_idx.get(str(tgt))
            if tgt_idx is None:
                n_missing_target += 1
                print(f"  [warn] target id {tgt} not in catalog; skipping",
                      flush=True)
                continue
            rel = scores[i]
            rank = int(np.flatnonzero(np.argsort(-rel) == tgt_idx)[0])
            if rank < K_NDCG:
                ndcg10_sum += 1.0 / np.log2(rank + 2)
            if rank < K_NDCG:
                hit10_sum += 1.0
            if rank < 20:
                recall20_sum += 1.0
            mrr_sum += 1.0 / (rank + 1)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s ({n/elapsed:.1f} ex/s)", flush=True)
    print(f"Targets not in catalog: {n_missing_target}", flush=True)
    print(f"\nTest-split metrics (full {n_titles}-title catalog)", flush=True)
    print(f"  NDCG@{K_NDCG} : {ndcg10_sum / n:.4f}", flush=True)
    print(f"  Hit@{K_NDCG}: {hit10_sum / n:.4f}", flush=True)
    print(f"  Recall@20   : {recall20_sum / n:.4f}", flush=True)
    print(f"  MRR         : {mrr_sum / n:.4f}", flush=True)

    results = {
        "checkpoint": str(args.checkpoint),
        "n_test_examples": n,
        "title_count": n_titles,
        "ndcg10": float(ndcg10_sum / n),
        "hit10": float(hit10_sum / n),
        "recall20": float(recall20_sum / n),
        "mrr": float(mrr_sum / n),
        "targets_not_in_catalog": n_missing_target,
        "elapsed_sec": round(elapsed, 2),
        "throughput_per_sec": round(n / elapsed, 2),
    }
    out_path = PROJECT_ROOT / "out_probe" / "test_metrics.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to {out_path}", flush=True)


if __name__ == "__main__":
    main()
