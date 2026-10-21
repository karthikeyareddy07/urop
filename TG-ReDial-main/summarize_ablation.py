#!/usr/bin/env python3
"""Summarize ablation and elicitation results across all seeds."""

import json
import numpy as np
from pathlib import Path

BASE = Path("d:/TG-ReDial-main/TG-ReDial-main")
DATA = BASE.parent / "TG-ReDial-Data" / "out"

seeds = [7, 42, 123, 2026]
ablation = {}
for s in seeds:
    with open(DATA / "experiments" / f"ablation_context_s{s}.json", encoding="utf-8") as f:
        ablation[s] = json.load(f)["cells"]

print("=" * 70)
print("ABLATION SUMMARY (4 seeds: 7, 42, 123, 2026)")
print("=" * 70)
print(f"{'Cell':<22} {'N':>5} {'Hit@10':>10} {'NDCG@10':>10} {'MRR':>10} {'Stdh':>10}")
print("-" * 70)
for cell in ["A_current_only", "B_history_only", "C_current_history", "D_full_elicited"]:
    hits = [ablation[s][cell]["hit10"] for s in seeds]
    ndcs = [ablation[s][cell]["ndcg10"] for s in seeds]
    mrrs = [ablation[s][cell]["mrr"] for s in seeds]
    print(f"{cell:<22} {400:>5} {np.mean(hits):>10.4f} {np.mean(ndcs):>10.4f} {np.mean(mrrs):>10.4f} {np.std(hits):>10.4f}")

print()
print("=== PAIRED DIFFERENCES (C - A) ===")
for s in seeds:
    diff = ablation[s]["C_current_history"]["hit10"] - ablation[s]["A_current_only"]["hit10"]
    print(f"  Seed {s}: {diff:+.4f}  (NDCG: {ablation[s]['C_current_history']['ndcg10'] - ablation[s]['A_current_only']['ndcg10']:+.4f})")
mean_ca = np.mean([ablation[s]["C_current_history"]["hit10"] - ablation[s]["A_current_only"]["hit10"] for s in seeds])
print(f"  Mean: {mean_ca:+.4f}")

print()
print("=== PAIRED DIFFERENCES (D - A) ===")
for s in seeds:
    diff = ablation[s]["D_full_elicited"]["hit10"] - ablation[s]["A_current_only"]["hit10"]
    print(f"  Seed {s}: {diff:+.4f}")
mean_da = np.mean([ablation[s]["D_full_elicited"]["hit10"] - ablation[s]["A_current_only"]["hit10"] for s in seeds])
print(f"  Mean: {mean_da:+.4f}")

# Elicitation results
print()
print("=" * 70)
print("ELICITATION SIMULATION SUMMARY")
print("=" * 70)
for s in [7, 42]:
    with open(DATA / "experiments" / f"elicitation_sim_s{s}.json", encoding="utf-8") as f:
        e = json.load(f)
    print(f"\nSeed {s}: ask_rate={e['ask_rate']:.3f}, avg_turns={e['avg_turns']:.2f}")
    print(f"  Before: hit10={e['before']['hit10']:.4f}, ndcg10={e['before']['ndcg10']:.4f}, mrr={e['before']['mrr']:.4f}")
    print(f"  After:  hit10={e['after']['hit10']:.4f}, ndcg10={e['after']['ndcg10']:.4f}, mrr={e['after']['mrr']:.4f}")
    print(f"  Delta:  hit10={e['delta']['hit10']:+.4f}, ndcg10={e['delta']['ndcg10']:+.4f}, mrr={e['delta']['mrr']:+.4f}")
    print(f"  Asked subset n={e['subsets']['asked']['n']}, unasked n={e['subsets']['unasked']['n']}")

# Uncertainty probe
print()
print("=" * 70)
print("UNCERTAINTY PROBE (seed 42, 150 valid examples)")
print("=" * 70)
with open(BASE / "uncertainty_probe.json", encoding="utf-8") as f:
    up = json.load(f)
for k, v in up.items():
    if k == "hit10_rate":
        print(f"  {k}: {v:.4f}")
    elif isinstance(v, dict):
        print(f"  {k}: corr_hit10={v['corr_hit10']:+.4f}, p50={v['p50']:.4f}, std={v['std']:.4f}")
    else:
        print(f"  {k}: {v}")

