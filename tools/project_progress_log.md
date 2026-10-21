# TG-ReDial Project Work Log

This log records the project work in chronological order for review and presentation. Exact calendar dates were not retained for the early preprocessing and cleanup work; the model evaluation was completed on 2026-10-10.

## Project Summary

The TG-ReDial conversations were cleaned and converted into files recommendation models can use. Five methods were compared. **The Movie-context centroid ranker performed best:** it put the correct movie in its first ten recommendations for 5.72% of validation conversations and 4.81% of test conversations. This is the best result among the five methods tested, but it still needs improvement before it would be a strong user-facing recommender.

## 1. Dataset Review and Data-Quality Investigation

- Reviewed the TG-ReDial conversation, movie-mention, user, and identity/history data to understand its schema and suitability for conversational movie recommendation.
- The raw splits contain 10,000 conversations and 129,404 dialogue messages: 8,495 train, 757 validation, and 748 test conversations.
- Investigated missing identity/history lookup keys and malformed or missing conversation IDs. Earlier processed examples could contain placeholder `None` IDs, which produced lookup keys that did not match the source identity/history data.

## 2. Preprocessing Corrections and Processed Dataset

- Updated `preprocess_tgredial.py` to accept several possible conversation-ID field names and create stable split-specific fallback IDs when an ID is missing.
- Normalized dialogue text, ordered messages by local turn ID, and created a recommendation example from the dialogue context before each movie-mention turn.
- Earlier pipeline validation recorded zero missing identity-to-history keys after the fallback-ID correction and no conversation-ID overlap among the train, validation, and test splits.
- Generated the model-ready files under `TG-ReDial-Data/out/`: 25,485 train examples, 2,271 validation examples, and 2,244 test examples, plus `dataset_stats.json`, `vocab.json`, `users.json`, and `movies.json`.
- Each processed example has one target movie. Therefore, for these results, Recall@10 equals Hit@10.

## 3. CSV Recovery and Storage Cleanup

- Exported the raw conversation splits, identity/history mappings, conversation-to-user mapping, and user-topic information to CSV files in `TG-ReDial-Data/csv_branch` for inspection and retention.
- Removed the original PKL files to reduce storage after exporting the CSV branch.
- Removed one-time diagnostic/sample utilities (`analyze_missing_keys_more.py`, `find_missing_keys.py`, `inspect_missing_detail.py`, `inspect_none_conv.py`, `inspect_pkl.py`, `show_samples.py`), the optional batch creator (`batch_and_tokenize.py`), and the two optional Markdown-to-PDF helpers.
- Retained the preprocessing, pipeline-check, and model-benchmark scripts, along with the processed JSONL files.
- **Rebuild limitation:** the current preprocessor and some raw-data checks still expect PKL input; they do not yet rebuild the processed JSONL from `csv_branch`. The existing JSONL files remain usable for benchmarking, but CSV-to-JSONL support is needed for a complete rebuild from the retained source data.
- The earlier preprocessing summary refers to `show_samples.py`, which was removed during cleanup. Its sample-display command is no longer available.

## 4. PyTorch Environment Investigation

- Initial GRU and BERT candidates could not import PyTorch. Importing PyTorch failed with Windows `WinError 4551` while loading `torch/lib/c10.dll`.
- Windows Code Integrity logs recorded events 3077 and 3033 for the blocked DLL and reported that it did not meet the enterprise signing requirement. The DLL existed but was reported as unsigned by Authenticode.
- This is a machine App Control policy restriction, not a preprocessing-data error. No policy bypass was attempted.

## 5. Five-Model Comparison

- Because Windows policy blocked PyTorch, the benchmark was changed to five methods that run on the CPU. SciPy and scikit-learn were added to `requirements.txt`.
- All five methods learned from the same training data and were compared on the same validation data. The validation results chose the winner; that winner was then checked on the separate test data.

### What the Scores Mean

- **Accuracy@1:** how often the correct movie is ranked first. Higher is better.
- **Precision@10:** among the top ten recommendations, how many are relevant. With a single target movie per example, this is the share of the top ten slots that are correct.
- **Recall@10:** how often the target movie appears in the first ten. Each example has one target movie here, so Recall@10 equals Hit@10.
- **F1@10:** the harmonic mean of Precision@10 and Recall@10; it balances how many correct movies are found versus how many of the top ten suggestions are relevant.
- **Hit@5 / Hit@10:** how often the correct movie appears in the first 5 / 10 recommendations. Higher is better.
- **NDCG@10:** a score from 0 to 1 for how high the correct movie appears; a higher position earns a higher score. Higher is better.

The benchmark script includes Accuracy@1, Precision@10, Recall@10, and F1@10; the values below are from the completed validation run.

### Full Validation Results

| Model | Accuracy@1 | Precision@10 | Recall@10 | F1@10 | Hit@5 | Hit@10 | NDCG@10 |

| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

| Popularity (most frequent movies) | 0.00% | 0.05% | 0.48% | 0.09% | 0.18% | 0.48% | 0.0019 |

| TF-IDF retrieval (similar past conversations) | 0.00% | 0.02% | 0.18% | 0.03% | 0.09% | 0.18% | 0.0007 |

| Complement Naive Bayes (learns text/movie patterns) | 0.88% | 0.39% | 3.92% | 0.71% | 2.82% | 3.92% | 0.0216 |

| **Movie-context centroid (compares conversation to movie examples)** | **0.79%** | **0.57%** | **5.72%** | **1.04%** | **3.65%** | **5.72%** | **0.0291** |

| User-item SVD (learns user/movie patterns) | 0.04% | 0.04% | 0.40% | 0.07% | 0.22% | 0.40% | 0.0018 |

For example, 5.72% Hit@10 means this model recommended the correct movie in its first ten suggestions for about 130 of the 2,271 validation conversations.

### How the Models Differ

| Model | What it looks at | In simple terms | Validation Hit@10 |

| --- | --- | --- | ---: |

| Popularity | Which movies appeared most often in training | Always recommends the most frequent movies; it does not read the conversation | 0.48% |

| TF-IDF retrieval | Words in the current and past conversations | Finds a past conversation with similar words and reuses its movie | 0.18% |

| Complement Naive Bayes | Text patterns linked to movies | Learns which character and word patterns are associated with each movie | 3.92% |

| **Movie-context centroid** | Conversation text and examples grouped by movie | Recommends movies whose example conversations are most like the current one | **5.72%** |

| User-item SVD | Which users and movies appear together | Learns user/movie patterns, but does not read conversation meaning | 0.40% |

### Model Selection Rationale

**Selected candidate: Movie-context centroid ranker.** It achieved the best result on all four validation measures. It compares the current conversation with the typical conversations associated with each movie. This uses the dialogue directly and worked better here than recommending popular movies or relying only on user/movie history.

Complement Naive Bayes was second. The centroid model found the target in its top ten for 5.72% of validation conversations, compared with 3.92% for Naive Bayes: a difference of 1.8 percentage points. It also ranked targets better overall (NDCG@10 0.0291 vs. 0.0216).

The other candidates ranked lower because:

- **Popularity:** 0.48% Hit@10. It ignores what the user is saying, so it rarely finds the target movie.
- **TF-IDF retrieval:** 0.18% Hit@10. Copying recommendations from a past conversation with similar words was least effective.
- **Complement Naive Bayes:** 3.92% Hit@10. It learned useful text patterns, but the centroid method did better on every validation measure.
- **User-item SVD:** 0.40% Hit@10. User/movie history patterns alone were not enough for this dialogue-focused task.

### Held-Out Test Result for the Selected Model

| Metric | Test score |

| --- | ---: |

| Hit@5 | 2.76% |

| Hit@10 | 4.81% |

| NDCG@10 | 0.0245 |

| Recall@10 | 4.81% |

On separate test conversations, the selected model put the correct movie in its top ten for about 108 of 2,244 conversations (4.81%). This is 0.91 percentage points below its validation result of 5.72%. The other four models were not run on test, so we cannot claim that the winner beats them on test data.

## 6. Current Status and Remaining Work

- The processed JSONL files and five-model CPU benchmark are usable for model comparison.
- The Movie-context centroid ranker is the current best candidate among the five evaluated CPU-compatible models; it is not yet established as a production-quality recommender.
- GRU and BERT were not compared because the machine policy blocks PyTorch. Their previous benchmark implementations are not part of the active CPU comparison.
- The benchmark reports ranking metrics but does not record training/validation loss curves, so overfitting and underfitting have not been formally assessed.
- `TG-ReDial-Data/out` currently contains the processed splits and metadata files. The earlier `processing_check_report.json` and `processing_check_report.md` are not present in the current output folder listing; reproduce or restore them if they are required as supporting evidence.
- Recommended next work: make preprocessing read the retained CSV branch, add movie-title/metadata features to the recommender, record training/validation diagnostics, and compare all frozen candidates on test only after model selection is complete.

## Reproduction Command

Run from the workspace root:

```powershell
.\.venv\Scripts\python.exe .\TG-ReDial-main\benchmark_5_models.py

```

## 7. Chatbot Recommendation Prototype

- Wrapped the selected Movie-context centroid ranker in `TG-ReDial-main/movie_recommender.py` so the chatbot can submit conversation text and receive movie IDs, Chinese titles, and similarity values.
- Confirmed the catalog maps all unique target movie IDs in the train, validation, and test splits to titles (100% ID coverage).
- Trained the recommender on the full training split and saved the reusable artifact to `TG-ReDial-Data/out/movie_centroid_model.joblib`.
- Tested live inference with a Chinese preference message; the CLI returned five ranked movie titles and similarity values.
- The returned similarity is a text-vector similarity score, **not** a probability or a guarantee that the user will like the movie.

Train or update the saved model:

```powershell
.\.venv\Scripts\python.exe .\TG-ReDial-main\movie_recommender.py train

```

Get recommendations for a conversation:

```powershell
.\.venv\Scripts\python.exe .\TG-ReDial-main\movie_recommender.py recommend --context "我想看一部轻松有趣的家庭喜剧电影" --top-k 5

```

## 8. Neural Model Investigation (2026-10-17)

- Attempted multiple neural model architectures to improve upon TF-IDF baseline
- All neural attempts failed to achieve meaningful performance (Hit@10 ~0.0%)
- Root causes identified:
  - Training-evaluation mismatch (sampled vs full-catalog)
  - Character-level representation insufficient for Chinese semantic understanding
  - CPU-only training too slow for effective experimentation
  - Loss function may not be optimal for recommendation
- Documentation created in `NEURAL_MODEL_ISSUES.md`
- Decision: TF-IDF code removed; neural model is now the primary recommender with attention encoding, margin loss, and LR scheduling

## 9. Preprocessing CSV Migration (2026-10-17)

- Updated `preprocess_tgredial.py` to read from CSV branch instead of PKL files
- Successfully processed all splits from `TG-ReDial-Data/csv_branch/`
- Output saved to `TG-ReDial-Data/out_csv_test/`
- Statistics match original processing (25,485 train, 2,271 valid, 2,244 test examples)
- Pipeline now reproducible without original PKL files

## 10. Azure Deployment Preparation (2026-10-17)

- Created `requirements.txt` with all dependencies
- Created `Dockerfile` for containerized deployment (configured to use TF-IDF model)
- Created `.dockerignore` to exclude unnecessary files
- Created `azure.yaml` for Azure Developer CLI
- Updated `chat_app.py` to support environment variables (HOST, PORT)
- Created comprehensive `DEPLOYMENT.md` with deployment instructions
- Ready for Azure App Service or Container Apps deployment

## 11. Neural Recommender Rebuild and Baseline (2026-10-18)

- Fixed blockers in `neural_movie_recommender.py`: a stray `5` artifact in
  `_evaluate_arrays` (the module did not import at all), `save()` crashing on
  a missing `mean_pooling` attribute, and `load()` passing misaligned
  positional arguments (every checkpoint load raised `TypeError`).
- Made anonymous mode the coherent default everywhere: module docstring,
  `_NeuralRanker` constructor, `train()` signature, and CLI (`--split` now
  defaults to `valid`, keeping the test split for final reporting only).
  Validation now uses exactly the features chat serving sends (no user ID,
  no profile text).
- Added per-epoch diagnostics stored in the checkpoint and logged: sampled
  win rate, positive/negative logit means, popularity weight, top-1
  concentration, catalog-only top-1 share, target coverage, and
  seen/unseen-target Hit@10 (audit finding 3).
- Checkpoint lineage: the 2026-10-13 checkpoint (SHA256 `8605d382...`) is
  architecturally incompatible with the current code (32-dim 1-layer BiLSTM,
  no transformer/temperature/popularity parameters) and was archived
  byte-identical as `neural_movie_model_legacy_8605d382.pt`; the 2026-10-16/17
  experiment files share that pre-transformer layout. New checkpoints
  round-trip cleanly; incompatible ones fail with an actionable retrain
  error (tests cover all three cases).
- Environment findings: default `--batch-size 256` with 512-char contexts
  OOMs (~2.1 GB attention map) on this 16 GB machine; the full-size config at
  batch 64 measures 19.4 s per batch (~2.2 h per epoch). The baseline
  therefore runs `--hidden-size 128 --max-context-chars 256 --batch-size 128`
  (about 9-13 s per batch); `--max-context-chars` is stored in the
  checkpoint so training, validation, and serving truncate identically.
- Rewrote `test_neural_movie_recommender.py` around five audit targets
  (anonymous/personalized parity, checkpoint compatibility, candidate scoring
  equivalence, hand-computed metric diagnostics, cold-start eligibility):
  29 tests, all passing (1 conditional skip until the new checkpoint ships).
- Genre features were evaluated as an evidence-based decision and rejected:
  the catalog CSV has only titles and the raw pickles hold no item metadata,
  so no genre ablation is possible; see `NEURAL_MODEL_PROTOCOL.md`.
- Baseline run (anonymous, seed 42, validation-selected): **RUN CRASHED — NO CHECKPOINT PRODUCED**.
  PID 4204 suffered a native access violation (0xc0000005) in `VCRUNTIME140.dll`/`c10.dll`
  (PyTorch C++ backend) at ~13:41:51, about 5 minutes into epoch 3, immediately after
  hard-negative mining logged `mined_hard_negatives=10`. No epoch-3 batch logs were ever
  emitted, and `recommender.save()` (which runs only at end of training) never executed.
  Windows Event Log confirms APPCRASH for PID 0x106C (4204). Last completed epoch: epoch 2,
  Val Hit@10=0.0022, Val NDCG@10=0.0012, top1_max_share=1.0000, unique=1.
  Epoch 1 was best: Val Hit@10=0.0031, Val NDCG@10=0.0014. Restart planned with same config;
  if the crash repeats, reduce batch size further (64) or disable hard-negative mining for
  epoch 3+. Results table and exact commands live in `NEURAL_MODEL_PROTOCOL.md`.
- **Superseded (2026-10-18):** the BiLSTM/transformer stack above was retired in §12 and
  replaced in §13 by a frozen-encoder (`voidful/albert_chinese_tiny`) + linear-probe
  architecture (`chinese_recommender.py`) that trains cleanly on CPU (~22 s/epoch, zero
  crashes) and ships checkpoint `out_probe/best.pt` (SHA256 `750d0f9b...`).

## 12. Neural Backend Removal (2026-10-18)

- Removed all neural-model traces: `neural_movie_recommender.py` (+ tests),
  `improved/fast_neural_trainer` (`experimental/`), `evaluate_ckpt.py`,
  `evaluate_candidate_policy.py`, `build_semantic_cache.py`,
  `run_baseline_restart.bat`, `NEURAL_MODEL_{AUDIT,ISSUES,PROTOCOL}.md`,
  all `neural_*.pt` checkpoints, `*.npz` caches, helper scripts
  (`_launch_b1.vbs`, `_smoke_chat.py`, ...), and `__pycache__`/`.pytest_cache`.
- `chat_app.py` is now a model-agnostic shell exposing
  `make_handler(recommender)`; `DEFAULT_MODEL_PATH` is `None`.
- Cleaned live references in `README.md` (root + TG-ReDial-main),
  `DEPLOYMENT.md`, `Dockerfile` (no model baked into the image).
  `requirements.txt` is torch-free. Remaining `neural/torch` mentions are
  history-only in `project_progress_log.md` and dataset-notes in READMEs.
- Verified: `test_chat_app` 5/5 pass, `chat_app` imports with `make_handler`.
  Remaining code: `chat_app.py`, `check_pipeline.py`,
  `preprocess_tgredial.py`, `test_chat_app.py`, `static/chat.html`.

## 13. Frozen ALBERT-Tiny Linear Probe Recommender (2026-10-18)

### Architecture

- **Encoder (frozen):** `voidful/albert_chinese_tiny` (312-dim hidden state,
  mean-pooled over attention mask and L2-normalized).
- **Offline title cache:** `TG-ReDial-main/build_encoder_cache.py` encodes all
  33,834 catalog titles once into
  `TG-ReDial-Data/out/title_embeddings_albert_tiny.npz` (`33834 × 312`,
  `float32`) plus `movie_id_order.json`.
- **Trainable probe (`chinese_recommender.py`):** shared linear projection
  (`312 → 256`) + `LayerNorm → Linear → GELU → Linear` MLP head (`256 → 256`),
  L2-normalized cosine similarity scaled by learnable `exp(temperature)`
  (clamped to `[1, 20]`) plus optional popularity log-frequency bias.
- **Serving wrapper (`ChineseRanker`):** loads the frozen encoder once and the
  trained probe from checkpoint; exposes `recommend(context, top_k)` and
  `recommend_with_context(context, top_k, profile, watch_history)` matching
  the `chat_app.make_handler` contract (returns a Chinese clarification prompt
  when confidence `< 0.05`).

### Bugs Diagnosed and Fixed During Stage 6 Training

1. **Missing `__main__` entrypoint** in `train_frozen_probe.py`: script defined
   `main()` at the top and helper classes below it without calling `main()`,
   exiting 0 with zero work done.
2. **Per-line context precompute bottleneck:** replaced line-by-line encoder
   calls with batched `encode_ctx_batch(batch_size=32, padding=True)` and
   added `CTX_CACHE` (`out_ctx_cache.pt`) reuse (`25485 × 312` train,
   `2271 × 312` valid).
3. **PyTorch `IterableDataset` + `shuffle=True` crash:** moved line-order
   shuffling inside `ProbeExample.__iter__` and replaced the O(N) per-example
   negative-pool list comprehension with O(`max_neg`) rejection sampling
   (~22 s/epoch on CPU).
4. **`np.ndarray` passed to `nn.Linear`:** converted `load_cache()` output to
   contiguous `torch.float32` tensor.
5. **3-D candidate scoring in `ChineseRecommender.forward`:** branched on
   `t.dim() == 3` to use `torch.einsum('bd,bkd->bk', c, t)` for per-example
   candidate sets `(B, K, D)` while preserving `c @ t.T` for 2-D full-catalog
   serving `(N, D)`.
6. **`if not rel` truth-value crash** in `topk_variance`: replaced with
   `getattr(rel, 'size', 0) == 0`.
7. **Degenerate `ndcg_at_k` and `hit_at_k`:** rewrote both against ground-truth
   target index 0 (`1/log2(rank+2)` for `rank < k`, binary `rank < k` for
   Hit@k), fixing false early-stopping after epoch 3 and 10× Hit@10
   understatement.
8. **`ChineseRanker.load` keyword bug:** removed the unexpected
   `max_context_tokens` kwarg from `ChineseRecommender(movie_count=mc)`.

### Stage 6 Training Results (65-Candidate Sampled Ranking: 1 Positive + 64 Negatives)

Seed 42, `batch=64`, `lr=1e-3`, cosine annealing over 10 epochs:

| Epoch | Train NDCG@10 | Val NDCG@10 | Val Hit@10 | Top-1 Max (batch) | Top-1 Unique (batch) | LR | Duration |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Untrained | — | — | 15.00% | — | — | — | — |
| 1 | 0.1043 | 0.1148 | 23.60% | 3.94 | 40 | 9.76e-04 | 22.5 s |
| 2 | 0.1139 | 0.1126 | 24.88% | 3.94 | 40 | 9.05e-04 | 22.4 s |
| 3 | 0.1227 | 0.1189 | 25.32% | 3.83 | 40 | 7.94e-04 | 22.9 s |
| 4 | 0.1346 | 0.1304 | 26.82% | 3.97 | 40 | 6.55e-04 | 23.6 s |
| 5 | 0.1331 | 0.1307 | 27.04% | 4.28 | 40 | 5.00e-04 | 24.1 s |
| 6 | 0.1448 | 0.1391 | 28.01% | 4.17 | 40 | 3.45e-04 | 24.1 s |
| 7 | 0.1525 | 0.1431 | 29.24% | 4.19 | 39 | 2.06e-04 | 23.9 s |
| 8 | 0.1589 | 0.1426 | 28.89% | 3.89 | 40 | 9.55e-05 | 22.2 s |
| 9 | 0.1646 | 0.1484 | 30.38% | 4.08 | 40 | 2.45e-05 | 21.7 s |
| **10 (winner)** | **0.1665** | **0.1536** | **31.57%** | **3.94** | **40** | **0.00e+00** | **21.8 s** |

- **Collapse gate:** PASSED (`top1_unique = 40 ≫ 2` out of 64-example batches;
  `top1_max = 3.94 / 64 ≈ 6.2% ≪ 90%`).
- **Lift over random (10/65 = 15.38%):** Val Hit@10 reaches **31.57%**
  (**2.05×** random / **2.10×** untrained initialization).
- **Stage 7 decision (tiny-encoder-only):** kept the frozen ALBERT-tiny +
  linear-probe winner without unfreezing encoder layers, preserving fast CPU
  inference and the precomputed 33,834-title embedding cache.
- **Winner checkpoint:** `out_probe/best.pt` (988,111 bytes, SHA256
  `750d0f9bee786067998b8542b210a50719dc809cabc7ac3eb27428cd2b30e98e`),
  archived to `TG-ReDial-Data/out/frozen_probe_checkpoint_750d0f9b.pt` with
  training log `frozen_probe_train_750d0f9b.log` and metadata
  `frozen_probe_checkpoint_meta.json`.

### Integration, Serving, and Verification (Stages 9–12)

- **Chat wiring (`test_chat_app.py`):** preserved `DEFAULT_MODEL_PATH = None`
  and all 5 unit tests (`ChatApiTests`), and added `RealCheckpointChatTests`
  (3 integration tests loading `out_probe/best.pt` into `ChineseRanker` and
  exercising `GET /api/health`, 33,834-movie catalog load, and `POST /api/chat`
  with a Chinese query). All **12/12** tests pass (`test_chat_app.py` +
  `test_chinese_recommender.py`).
- **Entrypoint (`serve_chinese.py`) & Docker (`Dockerfile`, `.dockerignore`):**
  created `serve_chinese.py` (loads `out_probe/best.pt` via `ChineseRanker`
  and serves `chat_app.make_handler` on `0.0.0.0:8000`, including
  `GET /api/health`), updated `Dockerfile` `CMD` to run `serve_chinese.py`,
  and added `!.pt` whitelist rules in `.dockerignore` so the checkpoint ships
  in the image. Live-verified `serve_chinese.py` locally (`GET /api/health` →
  `200 {"status": "ok"}`, `POST /api/chat` → `200` with 5 Chinese movie
  recommendations). Docker CLI is not installed on this workstation, so
  `docker build` must be run in CI or on a host with Docker Desktop.
