# Model Card — TG-ReDial Frozen-ALBERT Linear-Probe Recommender

## Model Details

- **Model name:** TG-ReDial frozen-ALBERT linear-probe movie recommender
- **Model date:** 2026-10-18
- **Model version:** 1.0
- **Model author:** TG-ReDial project
- **License:** See LICENSE
- **Finetuned from:** `voidful/albert_chinese_tiny` (encoder frozen)

## Intended Use

Conversational movie recommendation for Chinese-language dialogue. Given a multi-turn
conversation context (Seeker / Recommender turns), the model scores the full 33,834-title
catalog and returns the top-K most relevant movies.

**Primary use case:** Retrieval-stage candidate generation in a conversational
recommender system (CRS). Outputs are ranked movie titles with similarity scores;
they are not purchase/viewing advice and should not be treated as medical, legal, or
financial recommendations.

**Out-of-scope use cases:** Cold-start users with no dialogue context; non-Chinese
languages; non-movie content (books, music, etc.); real-time personalization without
fine-tuning on user history.

## Architecture

```text
Context text → ALBERT-tiny (frozen) → Mean-pool → LayerNorm → MLP head → 256-dim query
Title text   → ALBERT-tiny (frozen) → Mean-pool → LayerNorm → MLP head → 256-dim key
Query · Key  → cosine similarity × temperature + log-frequency popularity bias
```

- **Encoder:** `voidful/albert_chinese_tiny` (312-dim hidden, frozen)
- **Projection:** `Linear(312 → 256)`
- **Head:** `LayerNorm(256) → Linear(256→256) → GELU → Linear(256→256)`
- **Scoring:** Cosine similarity between projected context and title embeddings,
  scaled by a learned temperature parameter (clamped to [1.0, 20.0]), plus a
  learned log-frequency popularity bias
- **Trainable parameters:** Projection + head + temperature + popularity weight
  (encoder frozen; ~2.0M trainable params out of ~14M total)
- **Checkpoint:** `tools/out_probe/best.pt` (965 KiB / 988,111 bytes,
  MD5 `48C83FA5642A5660659E24DF0793BC7A`)

## Training Data

- **Dataset:** TG-ReDial v1.0 (topic-guided conversational recommendation)
- **Splits:** 25,485 train / 2,271 valid / 2,244 test dialogue examples
- **Catalog:** 33,834 unique movie titles with precomputed ALBERT title embeddings
- **Preprocessing:** `preprocess_tgredial.py` (converts raw PKL → JSONL splits)
- **Context encoding:** Last 8 turns, speaker-prefixed text, max 1500 chars,
  max 256 ALBERT tokens, mean-pooled with attention mask

## Training Hyperparameters

| Hyperparameter | Value |
| --- | --- |
| Seed | 42 |
| Epochs | 10 |
| Batch size | 64 |
| Learning rate | 1e-3 (AdamW) |
| Scheduler | Cosine annealing (T_max=10) |
| Max negatives per example | 64 (rejection sampling) |
| Negative sampling | O(max_neg) per example via rejection |
| Gradient clipping | 1.0 |
| Early stopping patience | 2 |
| Encoder | Frozen (no gradient) |
| Training time | ~22 s/epoch on CPU |

## Evaluation Metrics

### Validation (candidate-set ranking: target + 64 sampled negatives)

| Metric | Value | vs. random |
| --- | --- | --- |
| Hit@10 | 31.57% | 2.05× random (~15.4%) |
| NDCG@10 | 0.1536 | ~2× random |

### Official Test Split (candidate-set: target + 64 uniform negatives, seed 42)

| Metric | Value | Random (analytic) | Ratio |
| --- | --- | --- | --- |
| Hit@10 | 0.2972 | 0.1538 | 1.9× |
| NDCG@10 | 0.1393 | 0.0699 | 2.0× |
| MRR | 0.1210 | 0.0732 | 1.7× |
| Examples | 2,244 | — | — |
| Runtime | ~48 s (CPU) | — | ~47 examples/sec |

Artifact: `tools/out_probe/test_candidate_metrics.json`.

### Official Test Split (full 33,834-title catalog ranking)

| Metric | Value | vs. random |
| --- | --- | --- |
| NDCG@10 | 0.000333 | ~3× random (0.000296) |
| Hit@10 | 0.000891 | ~3× random (0.000296) |
| Recall@20 | 0.001337 | ~2.3× random (0.000591) |
| MRR | 0.000591 | — |
| Examples | 2,244 | — |
| Runtime | ~43 s (CPU) | ~52 examples/sec |

Artifact: `out_probe/test_metrics.json`.

**Interpretation:** Validation metrics use candidate-set ranking (random Hit@10 ≈ 15.4%),
while test metrics use full-catalog ranking (random Hit@10 ≈ 0.030%). Full-catalog
ranking is substantially harder, so absolute test metrics are lower even though the
model retains ~2–3× random relative signal. All 2,244 test targets are present in the
catalog (0 missing).

## Limitations

1. **Full-catalog ranking is the bottleneck:** The model scores all 33,834 titles per
   context (O(N) scan). Absolute Hit@10 on the full catalog is <0.1% even though
   candidate-set Hit@10 is 31.6%. A two-stage recall+rerank architecture is recommended
   for production use.
2. **Frozen encoder:** The ALBERT encoder is not fine-tuned; domain adaptation to
   movie-recommendation dialogue is limited. End-to-end fine-tuning is supported via
   `train_frozen_probe.py --unfreeze-encoder` but has not been evaluated on the full
   split.
3. **Chinese-only:** Trained and evaluated on Chinese dialogue; performance on other
   languages is untested.
4. **Popularity bias:** A learned popularity weight can amplify popular titles; this may
   reduce diversity in recommendations.
5. **User modeling (partially addressed):** `recommend_with_context` builds
   the served context from the full dialogue + user-profile sentences +
   watch-history titles (`build_enriched_context_text`), and gates between
   asking and recommending via top-20 entropy (z-scored within top-20,
   threshold calibrated on train quantiles). Multi-seed ablations show that
   injecting profile/history at *inference time* into a probe trained on
   dialogue-only contexts does **not** improve metrics (Hit@10 drops ~3 pts
   on 3 of 4 seeds); matched training with enriched contexts is future work
   (`TG-ReDial-main/train_probe_v2.py`). Elicitation with oracle answers is
   metric-neutral-to-slightly-negative under the same distribution shift.

## How to Reproduce

### 1. Install dependencies

```powershell
pip install -r requirements.lock.txt
```

### 2. Preprocess data

```powershell
python TG-ReDial-main/preprocess_tgredial.py
```

Processed splits land in `TG-ReDial-Data/out/`.

### 3. Train (frozen probe)

```powershell
python TG-ReDial-main/train_frozen_probe.py --epochs 10 --lr 1e-3 --batch 64 --seed 42
```

Winner checkpoint saved to `out_probe/best.pt`.

### 4. Evaluate on official test split

```powershell
python TG-ReDial-main/evaluate_test.py
```

Writes `out_probe/test_metrics.json`.

### 5. Serve

```powershell
python serve_chinese.py --host 0.0.0.0 --port 8000 \
    --checkpoint out_probe/best.pt \
    --movies TG-ReDial-Data/out/movies.json
```

Endpoints: `GET /api/health`, `GET /` (chat UI), `POST /api/chat`.

### 6. Optional: end-to-end fine-tune encoder

```powershell
python TG-ReDial-main/train_frozen_probe.py --unfreeze-encoder --encoder-lr 1e-5 --lr 1e-3
```

## Ethical Considerations

- Recommendations are based solely on dialogue context and title embeddings; no
  sensitive user attributes are modeled.
- The model may reflect biases present in the TG-ReDial training data (e.g., genre,
  popularity, or release-year distributions).
- Do not use for high-stakes decisions (medical, legal, financial).
- Output titles are retrieved from the catalog; the model does not hallucinate movie
  names, but similarity scores are not calibrated probabilities.

## References

- Zhou et al., "Towards Topic-Guided Conversational Recommender System", COLING 2020.
  <https://arxiv.org/abs/2010.04125>
- TG-ReDial dataset: <https://drive.google.com/drive/folders/1sWD6urkwyZo8ZyZBJoJw40eKK0jDNEni?usp=sharing>
