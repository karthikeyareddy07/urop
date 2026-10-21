#!/usr/bin/env python3
"""Ablation A: current-only vs history/profile-only vs current+history vs full.

Four cells required by the research plan (Sec 7 ablation table):
  A current-only:        build_context_text (last-8 turns), no profile/history
  B history-only:        profile + history only, no dialogue
  C current+history:     enriched context (dialogue + profile + history)
  D full (C + elicited): C plus simulated elicited-preference turn appended

Metric: candidate-set Hit@10 / NDCG@10 / MRR on VALID split (never test),
seed-fixed uniform negatives, shared across cells (controlled comparison).
Usage:
  python TG-ReDial-main/ablation_context.py --n 400 --seed 7
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch

FILE_DIR = Path(__file__).resolve().parent
PROJECT = FILE_DIR.parent
sys.path.insert(0, str(FILE_DIR))
from chinese_recommender import ChineseRanker, build_context_text, build_enriched_context_text  # noqa: E402

DATA = PROJECT / "TG-ReDial-Data" / "out"
CKPT = PROJECT / "tools" / "out_probe" / "best.pt"


def metrics(scores_list):
    def h(s, k=10):
        return 1.0 if int(np.flatnonzero(np.argsort(-np.asarray(s)) == 0)[0]) < k else 0.0
    def n(s, k=10):
        r = int(np.flatnonzero(np.argsort(-np.asarray(s)) == 0)[0])
        return float(1.0 / np.log2(r + 2)) if r < k else 0.0
    def m(s):
        return 1.0 / (int(np.flatnonzero(np.argsort(-np.asarray(s)) == 0)[0]) + 1)
    n_ex = len(scores_list)
    return (sum(h(s) for s in scores_list) / n_ex, sum(n(s) for s in scores_list) / n_ex,
            sum(m(s) for s in scores_list) / n_ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-neg", type=int, default=64)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    movies = json.load(open(DATA / "movies.json", encoding="utf-8"))
    order = json.load(open(DATA / "movie_id_order.json", encoding="utf-8"))["ids"]
    pos = {m: i for i, m in enumerate(order)}
    exs = []
    with open(DATA / "valid.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = (d.get("target_movie_ids") or [None])[0]
            if t is not None and str(t) in pos:
                exs.append(d)
            if len(exs) >= args.n:
                break
    print(f"loaded {len(exs)} valid examples", flush=True)
    ranker = ChineseRanker.load(str(CKPT), movies=movies,
                                title_emb=np.load(DATA / "title_embeddings_albert_tiny.npz")["embeddings"],
                                order=order)
    # shared negative sets per example (controlled)
    negsets = []
    for d in exs:
        ti = pos[str((d.get("target_movie_ids") or [None])[0])]
        ns, att = [], 0
        while len(ns) < args.max_neg and att < args.max_neg * 10 + 50:
            att += 1
            j = rng.randrange(len(order))
            if j != ti and j not in ns:
                ns.append(j)
        negsets.append((ti, ns))
    cells = {"A_current_only": [], "B_history_only": [], "C_current_history": [], "D_full_elicited": []}
    t0 = time.time()
    for k, d in enumerate(exs):
        ti, ns = negsets[k]
        prof = d.get("user_profile", []) or []
        hist = []
        for mid in (d.get("watch_history", []) or [])[-8:]:
            m = movies.get(str(mid))
            if m and m.get("title"):
                hist.append(m["title"])
        ctxs = {
            "A_current_only": build_context_text(d.get("context", [])),
            "B_history_only": build_enriched_context_text([], prof, hist),
            "C_current_history": build_enriched_context_text(d.get("context", []), prof, hist),
            "D_full_elicited": build_enriched_context_text(
                d.get("context", []) + [{"speaker": "user", "text": "我喜欢" + "、".join(hist[:3] + [str(p) for p in prof[:2]])}],
                prof, hist),
        }
        texts = [ctxs["A_current_only"], ctxs["B_history_only"], ctxs["C_current_history"], ctxs["D_full_elicited"]]
        emb = ranker._encode_texts(texts)
        tv = torch.from_numpy(ranker.title_emb)
        for name, e in zip(cells.keys(), emb):
            cand = [ti] + ns
            s = ranker.model(torch.from_numpy(e).unsqueeze(0), tv[cand]).detach().squeeze(0).numpy()
            cells[name].append(s)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(exs)}", flush=True)
    print(f"scored in {time.time() - t0:.0f}s", flush=True)
    rows = {}
    for name, sl in cells.items():
        h, nd, mr = metrics(sl)
        rows[name] = {"hit10": round(h, 4), "ndcg10": round(nd, 4), "mrr": round(mr, 4), "n": len(sl)}
        print(f"{name}: Hit@10={h:.4f} NDCG@10={nd:.4f} MRR={mr:.4f}", flush=True)
    out = PROJECT / "TG-ReDial-Data" / "out" / "experiments" / f"ablation_context_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"cells": rows, "seed": args.seed, "max_neg": args.max_neg,
               "split": "valid", "checkpoint": str(CKPT)}, open(out, "w", encoding="utf-8"), indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
