#!/usr/bin/env python3
"""Probe which uncertainty measure has dynamic range + predictive power (audit gap).

The catalog-wide softmax entropy is saturated (all scores near-identical ->
entropy pinned at log(20)), so the elicitation gate was effectively random.
Here we measure, on 150 VALID examples (never test):
  a) m12: top1-top2 margin in catalog-std units
  b) ent_within: entropy of softmax over top-20 z-scored WITHIN top-20
  c) spread: top1-top20 spread in catalog-std units
and correlate each with candidate-set Hit@10 (seed 42, 64 negatives).
A usable uncertainty signal must have dynamic range AND anti-correlate with
ranking success (higher uncertainty -> lower hit rate).
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np
import torch

FILE_DIR = Path(__file__).resolve().parent
PROJECT = FILE_DIR.parent
sys.path.insert(0, str(FILE_DIR))
from chinese_recommender import ChineseRanker, build_enriched_context_text  # noqa: E402

DATA = PROJECT / "TG-ReDial-Data" / "out"
CKPT = PROJECT / "tools" / "out_probe" / "best.pt"


def main():
    n, max_neg = 150, 64
    rng = random.Random(42)
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
            if len(exs) >= n:
                break
    ranker = ChineseRanker.load(str(CKPT), movies=movies,
                                title_emb=np.load(DATA / "title_embeddings_albert_tiny.npz")["embeddings"],
                                order=order)
    tv = torch.from_numpy(ranker.title_emb)
    rows = []
    for d in exs:
        ti = pos[str((d.get("target_movie_ids") or [None])[0])]
        prof = d.get("user_profile", []) or []
        hist = []
        for mid in (d.get("watch_history", []) or [])[-8:]:
            m = movies.get(str(mid))
            if m and m.get("title"):
                hist.append(m["title"])
        ctx = build_enriched_context_text(d.get("context", []), prof, hist)
        e = ranker._encode_texts([ctx])[0]
        with torch.no_grad():
            s = ranker.model(torch.from_numpy(e).unsqueeze(0), tv).detach().squeeze(0).numpy()
        sd = float(s.std()) + 1e-12
        t20 = s[np.argsort(-s)[:20]]
        m12 = float((t20[0] - t20[1]) / sd)
        z20 = (t20 - t20.mean()) / (t20.std() + 1e-12)
        ex_ = np.exp(z20 - z20.max())
        p20 = ex_ / ex_.sum()
        ent_within = float(-(p20 * np.log(p20.clip(min=1e-12))).sum())
        spread = float((t20[0] - t20[-1]) / sd)
        ns, att = [], 0
        while len(ns) < max_neg and att < max_neg * 10 + 50:
            att += 1
            j = rng.randrange(len(order))
            if j != ti and j not in ns:
                ns.append(j)
        cand = [ti] + ns
        with torch.no_grad():
            sc = ranker.model(torch.from_numpy(e).unsqueeze(0), tv[cand]).detach().squeeze(0).numpy()
        r = int(np.flatnonzero(np.argsort(-sc) == 0)[0])
        rows.append({"m12": m12, "ent_within": ent_within, "spread": spread,
                     "hit10": 1.0 if r < 10 else 0.0, "rank": r})
    def summarize(key):
        v = np.array([r[key] for r in rows], float)
        hits = np.array([r["hit10"] for r in rows], float)
        return {"min": round(float(v.min()), 4), "p10": round(float(np.quantile(v, 0.1)), 4),
                "p50": round(float(np.median(v)), 4), "p90": round(float(np.quantile(v, 0.9)), 4),
                "max": round(float(v.max()), 4), "std": round(float(v.std()), 4),
                "corr_hit10": round(float(np.corrcoef(v, hits)[0, 1]), 4)}
    out = {"n": len(rows), "split": "valid", "seed": 42,
           "hit10_rate": round(float(np.mean([r["hit10"] for r in rows])), 4),
           "m12_margin": summarize("m12"),
           "ent_within_top20": summarize("ent_within"),
           "spread_top20": summarize("spread")}
    json.dump(out, open(FILE_DIR / "uncertainty_probe.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
