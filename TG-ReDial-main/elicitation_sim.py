#!/usr/bin/env python3
"""Elicitation simulation: ask-vs-recommend policy + interaction metrics.

Research plan Sec 4/7: uncertainty-gated ask-vs-recommend decision, question
selection (here: genre-attribute question QGEN_GENRE vs QGEN_HISTORY), state
update, and interaction-efficiency metrics on VALID split (never test).

Policy: entropy of top-20 candidate softmax > threshold -> ask one question,
append simulated user answer (oracle: target movie's coarse genre bucket via
title keyword match, else profile/history-derived answer), re-score once.
Metrics: Hit@10/NDCG@10/MRR before/after, ask-rate, avg turns, delta.
Usage:
  python TG-ReDial-main/elicitation_sim.py --n 400 --seed 7 --threshold 2.0
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch

FILE_DIR = Path(__file__).resolve().parent
PROJECT = FILE_DIR.parent
sys.path.insert(0, str(FILE_DIR))
from chinese_recommender import ChineseRanker, build_enriched_context_text, topk_score_entropy  # noqa: E402

DATA = PROJECT / "TG-ReDial-Data" / "out"
CKPT = PROJECT / "tools" / "out_probe" / "best.pt"
GENRES = ["喜剧", "爱情", "科幻", "动作", "恐怖", "动画", "剧情", "悬疑"]


def genre_of(title):
    for g in GENRES:
        if g in (title or ""):
            return g
    return None


def rank_scores(ranker, ctx_text, cand):
    e = ranker._encode_texts([ctx_text])[0]
    tv = torch.from_numpy(ranker.title_emb)
    with torch.no_grad():
        s = ranker.model(torch.from_numpy(e).unsqueeze(0), tv[cand]).detach().squeeze(0).numpy()
    return s


def top20_entropy(ranker, ctx_text):
    e = ranker._encode_texts([ctx_text])[0]
    tv = torch.from_numpy(ranker.title_emb)
    with torch.no_grad():
        s = ranker.model(torch.from_numpy(e).unsqueeze(0), tv).detach().squeeze(0).numpy()
    return topk_score_entropy(s, k=20), s


def met(s):
    r = int(np.flatnonzero(np.argsort(-np.asarray(s)) == 0)[0])
    return (1.0 if r < 10 else 0.0, float(1.0 / np.log2(r + 2)) if r < 10 else 0.0, 1.0 / (r + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-neg", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Explicit entropy threshold; overrides calibration.")
    ap.add_argument("--ask-rate", type=float, default=0.3,
                    help="Target ask-rate; threshold calibrated as train-split quantile.")
    ap.add_argument("--calib-n", type=int, default=300,
                    help="Number of train examples used for threshold calibration.")
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
    ranker = ChineseRanker.load(str(CKPT), movies=movies,
                                title_emb=np.load(DATA / "title_embeddings_albert_tiny.npz")["embeddings"],
                                order=order)
    # --- Threshold calibration on TRAIN split (never test) ---
    # Choose the entropy threshold as the (1 - ask_rate) quantile of train
    # entropies so the policy asks exactly the target fraction of the time,
    # instead of a hand-picked absolute value that saturates to 0 or 1.
    calib = None
    if args.threshold is not None:
        threshold = float(args.threshold)
        calib = {"method": "explicit", "threshold": round(threshold, 4)}
        print(f"using explicit threshold={threshold:.4f}", flush=True)
    else:
        cexs = []
        with open(DATA / "train.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                t = (d.get("target_movie_ids") or [None])[0]
                if t is not None and str(t) in pos:
                    cexs.append(d)
                if len(cexs) >= args.calib_n:
                    break
        cents = []
        for d in cexs:
            prof = d.get("user_profile", []) or []
            hist = []
            for mid in (d.get("watch_history", []) or [])[-8:]:
                m = movies.get(str(mid))
                if m and m.get("title"):
                    hist.append(m["title"])
            ctx = build_enriched_context_text(d.get("context", []), prof, hist)
            ent, _ = top20_entropy(ranker, ctx)
            cents.append(ent)
        cents = np.array(cents)
        threshold = float(np.quantile(cents, 1.0 - args.ask_rate))
        calib = {"method": "train_quantile", "n": len(cents),
                 "target_ask_rate": args.ask_rate, "threshold": round(threshold, 4),
                 "p10": round(float(np.quantile(cents, 0.1)), 3),
                 "p50": round(float(np.median(cents)), 3),
                 "p90": round(float(np.quantile(cents, 0.9)), 3),
                 "max": round(float(cents.max()), 3)}
        print(f"calibrated threshold={threshold:.4f} (target ask-rate {args.ask_rate})", flush=True)
    pre, post, asks, turns = [], [], 0, []
    entropies = []
    asked_flags = []
    for k, d in enumerate(exs):
        ti = pos[str((d.get("target_movie_ids") or [None])[0])]
        ns, att = [], 0
        while len(ns) < args.max_neg and att < args.max_neg * 10 + 50:
            att += 1
            j = rng.randrange(len(order))
            if j != ti and j not in ns:
                ns.append(j)
        cand = [ti] + ns
        prof = d.get("user_profile", []) or []
        hist = []
        for mid in (d.get("watch_history", []) or [])[-8:]:
            m = movies.get(str(mid))
            if m and m.get("title"):
                hist.append(m["title"])
        ctx0 = build_enriched_context_text(d.get("context", []), prof, hist)
        ent, _ = top20_entropy(ranker, ctx0)
        entropies.append(ent)
        s0 = rank_scores(ranker, ctx0, cand)
        pre.append(met(s0))
        if ent > threshold:
            asks += 1
            asked_flags.append(True)
            tgt_title = movies.get(str((d.get("target_movie_ids") or [None])[0]), {}).get("title", "")
            g = genre_of(tgt_title)
            answer = f"我喜欢{g}片" if g else ("我喜欢" + "、".join(hist[:2] + [str(p) for p in prof[:1]]))
            ctx1 = ctx0 + f" [问]想看哪类电影？ [答]{answer}"
            s1 = rank_scores(ranker, ctx1, cand)
            post.append(met(s1))
            turns.append(2)
        else:
            post.append(met(s0))
            turns.append(1)
            asked_flags.append(False)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(exs)}", flush=True)
    def agg(L):
        a = np.array(L)
        return [round(float(a[:, j].mean()), 4) for j in range(3)]
    b, a = agg(pre), agg(post)
    ent = np.array(entropies)
    asked_idx = [i for i, f in enumerate(asked_flags) if f]
    unasked_idx = [i for i, f in enumerate(asked_flags) if not f]

    def agg_idx(L, idx):
        if not idx:
            return None
        arr = np.array([L[i] for i in idx])
        return [round(float(arr[:, j].mean()), 4) for j in range(3)]

    sub = {}
    for name, idx in (("asked", asked_idx), ("unasked", unasked_idx)):
        bb, aa = agg_idx(pre, idx), agg_idx(post, idx)
        sub[name] = {"n": len(idx), "before": bb, "after": aa,
                     "delta": ([round(aa[j] - bb[j], 4) for j in range(3)]
                               if bb and aa else None)}
    res = {"n": len(exs), "threshold": round(float(threshold), 4), "seed": args.seed,
           "split": "valid", "calibration": calib,
           "before": {"hit10": b[0], "ndcg10": b[1], "mrr": b[2]},
           "after": {"hit10": a[0], "ndcg10": a[1], "mrr": a[2]},
           "delta": {"hit10": round(a[0] - b[0], 4), "ndcg10": round(a[1] - b[1], 4),
                     "mrr": round(a[2] - b[2], 4)},
           "subsets": sub,
           "ask_rate": round(asks / len(exs), 4), "avg_turns": round(float(np.mean(turns)), 3),
           "entropy": {"min": round(float(ent.min()), 3), "p50": round(float(np.median(ent)), 3),
                       "p90": round(float(np.quantile(ent, 0.9)), 3),
                       "max": round(float(ent.max()), 3)}}
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    out = PROJECT / "TG-ReDial-Data" / "out" / "experiments" / f"elicitation_sim_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
