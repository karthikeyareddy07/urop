# Experiment Log (2026-10-18/19, CPU-only workstation)

All runs: frozen `voidful/albert_chinese_tiny` encoder, checkpoint
`tools/out_probe/best.pt` (965 KiB, MD5 `48C83FA5642A5660659E24DF0793BC7A`)
unless noted. Protocol: `tools/docs/protocol.md`.

## 1. Baseline reproduction (2026-10-18)

- Candidate-set test (target + 64 uniform negatives, seed 42, 2,244
  examples): Hit@10 0.2972 / NDCG@10 0.1393 / MRR 0.1210 in 48 s
  (`tools/out_probe/test_candidate_metrics.json`). Matches the previously
  reported figures exactly.
- Full-catalog test (33,834 titles): Hit@10 0.00089 / NDCG@10 0.00033 /
  MRR 0.00059, 52.3 examples/s (`out_probe/test_metrics.json`).
- Analytic random baselines: candidate-set Hit@10 0.1538 / NDCG@10 0.0699 /
  MRR 0.0732 → ratios 1.9× / 2.0× / 1.7×.

## 2. Data fix (2026-10-18)

- Root cause of empty profiles: string `user_id` keys in
  `user2TopicSent.pkl` vs int joins. Fixed in `preprocess_tgredial.py`;
  regenerated `TG-ReDial-Data/out/`. Coverage: profile nonempty
  25,485/25,485 train, 2,271/2,271 valid, 2,244/2,244 test; watch_history
  24,996 (98.1%) / 2,199 (96.8%) / 2,199 (98.0%). Backups:
  `*.jsonl.bak-empty-profile`.

## 3. Uncertainty validation (2026-10-19, `uncertainty_probe.py`, valid n=150, seed 42)

| Measure | Range (p10–p90) | corr with Hit@10 |
| --- | --- | --- |
| top1–top2 margin (std units) | 0.002–0.073 | +0.056 (wrong sign, ties) |
| **top-20 entropy, z-scored within top-20** | **2.00–2.51 (full 1.55–2.67)** | **−0.115 ✓** |
| top-20 spread (std units) | 0.10–0.25 | −0.084 |

Catalog-wide softmax entropy is saturated at log(20)=2.996 for all examples
(raw scores near-identical) — retired. Served uncertainty is now
within-top-20 standardized entropy (`chinese_recommender.topk_score_entropy`).

## 4. Multi-seed ablation (2026-10-19, `ablation_context.py`, valid n=400)

Cells: A current-only (last-8 turns), B history/profile-only, C
current+profile+history, D C+simulated elicited turn. Shared negative sets
per example; seeds vary negative sampling. Mean±population-std over seeds
{7, 42, 123, 2026}:

| Cell | Hit@10 | NDCG@10 | MRR |
| --- | --- | --- | --- |
| A current-only | 0.2900±0.0129 | 0.1415±0.0035 | 0.1298±0.0063 |
| B history-only | 0.1413±0.0126 | 0.0635±0.0069 | 0.0698±0.0051 |
| C current+history | 0.2588±0.0297 | 0.1262±0.0095 | 0.1162±0.0035 |
| D full+elicited | 0.2550±0.0290 | 0.1237±0.0105 | 0.1139±0.0046 |

**Honest finding:** C/D ≤ A on 3 of 4 seeds. The probe was *trained* on
dialogue-only contexts; injecting profile/history text at inference is a
distribution shift that hurts. Per-cell matched retraining is required
(`train_probe_v2.py`; open TODO). Per-seed JSON:
`TG-ReDial-Data/out/experiments/ablation_context_s{7,42,123,2026}.json`.
Cost: ~33 s scoring per 1,600 context encodes + full-catalog scoring pass.

## 5. Elicitation simulation (`elicitation_sim.py`, valid n=400)

- v1 (superseded): hand-picked entropy threshold; entropy saturated
  (2.991–2.996) → ask-rate 1.0 at threshold 2.99; gate was noise.
- v2 (current, seeds 7 & 42): threshold calibrated on train-quantile
  (n=300, target ask-rate 0.3) → threshold 2.4468; valid-split ask_rate
  0.22 (88/400 asked), avg_turns 1.22.
  - **Gate validity:** asked subset baseline Hit@10 0.193 vs unasked 0.269
    (seed 7) — the gate correctly targets hard cases.
  - **Elicitation effect (oracle genre answer appended):** delta Hit@10
    0.000, NDCG@10 0.000, MRR +0.0003 on asked subset; 0.000 across all 400.
    Same at seed 42. Honest conclusion: appending answers at inference to a
    dialogue-only-trained probe does not help; matched training needed.
  - Artifacts: `TG-ReDial-Data/out/experiments/elicitation_sim_s{7,42}.json`
    (v2 overwrote v1; v1 numbers preserved in this log).

## 6. Cost summary (all measured on the same CPU-only workstation)

| Operation | Cost |
| --- | --- |
| Probe training | ~22 s/epoch × 10 epochs (seed 42) |
| Candidate-set test eval (2,244 ex) | ~48 s (~47 ex/s) |
| Full-catalog test eval (2,244 ex) | ~43 s (~52 ex/s) |
| Ablation run (400 ex × 4 cells) | ~40 s |
| Elicitation run (400 ex + 300 calib) | ~90 s |
| Checkpoint | 965 KiB (988,111 bytes), ~246k trainable params |

## 7. Reproduction

```powershell
python TG-ReDial-main/ablation_context.py --n 400 --seed 42    # repeat for 7/123/2026
python TG-ReDial-main/elicitation_sim.py --n 400 --seed 7 --ask-rate 0.3
python TG-ReDial-main/uncertainty_probe.py
python TG-ReDial-main/evaluate_test_candidate.py
python TG-ReDial-main/evaluate_test.py
```
