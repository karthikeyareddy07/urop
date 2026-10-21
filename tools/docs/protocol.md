# Evaluation Protocol (FROZEN — 2026-10-18)

This document freezes the evaluation protocol used for all headline numbers.
Any change to any item below requires a new protocol document, not an edit here.

## 1. Data

- Dataset: TG-ReDial v1.0, Chinese-recommendation variant (COLING 2020).
- Processed splits (from `TG-ReDial-Data/out/dataset_stats.json`, which is the
  single source of truth):
  - Train: **25,485** examples
  - Valid: **2,271** examples
  - Test: **2,244** examples (official split, held out)
- Catalog: **33,834** unique movie titles
  (`TG-ReDial-Data/out/movie_id_order.json`), each with a precomputed frozen
  ALBERT-tiny title embedding (`title_embeddings_albert_tiny.npz`).

## 2. Two distinct ranking protocols (never conflate them)

### 2.1 Candidate-set protocol (the comparable headline number)

- For each test example: rank the single target item against **64 uniform
  random negatives** sampled from the catalog (excluding the target),
  generated with `random.Random(seed=42)`.
- 2,244 test examples; 65 candidates each; target always included at index 0
  before scoring.
- Random-baseline expectations (computed analytically, 65 candidates):
  - Hit@10 = 10/65 = **0.1538**
  - NDCG@10 = (1/65) · Σ_{r=0..9} 1/log2(r+2) = **0.0699**
  - MRR = H_65/65 = **0.0732**
- **Frozen baseline result** (checkpoint MD5
  `48C83FA5642A5660659E24DF0793BC7A`, `tools/out_probe/best.pt`, 988,111
  bytes): Hit@10 **0.2972**, NDCG@10 **0.1393**, MRR **0.1210**
  (`tools/out_probe/test_candidate_metrics.json`; runtime ~48 s CPU).
  Ratios vs. random: 1.9× / 2.0× / 1.7×.

### 2.2 Full-catalog protocol (much harder; report separately, never mix)

- Rank the target against **all 33,834 titles**.
- Frozen baseline result: Hit@10 **0.00089**, NDCG@10 **0.00033**, MRR
  **0.00059**, Recall@20 0.00134 (2,244 examples, ~43 s CPU,
  `out_probe/test_metrics.json`).
- Reference point: the official TG-ReDial COLING 2020 paper reports NDCG@10
  **0.0348** on its own (differently-preprocessed) protocol; numbers are not
  directly comparable and are never presented as such.

## 3. Model selection & seeds

- Checkpoint selection: validation split candidate-set metrics only. The test
  split is evaluated once per final configuration.
- Multi-seed reporting: evaluation-seed sensitivity via seeds {7, 42, 123,
  2026} for ablations (valid split, n=400); training seed is 42.
- Thresholds (elicitation ask-rate) are calibrated on the **train** split by
  quantile, never tuned on valid/test.

## 4. Metrics

- Hit@10, NDCG@10 (log2 discount), MRR — standard definitions, single target
  per example.
- Ablations/elicitation report on valid n=400; final numbers on full test.

## 5. Hardware & cost

- CPU-only workstation (no GPU used anywhere); encoder frozen ALBERT-tiny
  (312-dim, `voidful/albert_chinese_tiny`).
- Probe checkpoint: 965 KiB (988,111 bytes), ~246k trainable parameters.

## 6. Reproduction commands

```powershell
# Preprocess (regenerates splits + embeddings under TG-ReDial-Data/out/)
python TG-ReDial-main/preprocess_tgredial.py

# Train frozen probe (seed 42)
python TG-ReDial-main/train_frozen_probe.py --epochs 10 --lr 1e-3 --batch 64 --seed 42

# Headline candidate-set test evaluation
python TG-ReDial-main/evaluate_test_candidate.py

# Full-catalog test evaluation
python TG-ReDial-main/evaluate_test.py

# Ablations (valid, n=400, seeds 7/42/123/2026)
python TG-ReDial-main/ablation_context.py --n 400 --seed 42

# Elicitation simulation (train-calibrated ask-rate threshold)
python TG-ReDial-main/elicitation_sim.py --n 400 --seed 7 --ask-rate 0.3
```
