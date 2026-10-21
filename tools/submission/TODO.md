# TODO — Stage 13 / Future Work

This file tracks remaining items for the TG-ReDial frozen-ALBERT linear-probe
recommender. Items are roughly ordered by priority.

## Reproducibility / CI (in progress)

- [ ] Add `--eval-test` flag to `train_frozen_probe.py` so the official
  test-split evaluation runs automatically after training (currently a
  separate `evaluate_test.py` script).
- [ ] Pin `requirements.lock.txt` in CI (`pip install -r requirements.lock.txt`)
  instead of `requirements.txt` for fully reproducible builds.
- [ ] Add `requirements.lock.txt` regeneration to the CI workflow on dependency
  changes.
- [x] Add `.pre-commit-config.yaml` (black, isort, flake8, check-yaml/json/toml, end-of-file-fixer, etc.)

## Serving / Demo

- [ ] Verify `serve_chinese.py` + Docker build in CI (GitHub Actions `docker`
  job). Docker CLI is not available on the local Windows workstation, so this
  can only be validated in CI.
- [ ] Wire `static/chat.html` to `/api/chat` via a JS fetch wrapper (frontend
  already exists; server verified live).
- [ ] Add OpenAPI/swagger docs or at least an `/api/docs` endpoint describing
  `/api/chat` request/response schema.
- [ ] Add rate limiting / request size validation to `chat_app.py`.

## Accuracy improvements (optional)

- [x] Unfreeze the ALBERT encoder and fine-tune the whole model (probe +
  encoder) instead of keeping the encoder frozen. Implemented via
  ``--unfreeze-encoder`` / ``--encoder-lr`` flags in
  ``train_frozen_probe.py``. Expect better accuracy at the cost of longer
  training and higher memory. Not yet evaluated on the full split.
- [ ] Implement a two-stage recall+rerank architecture: Stage 1 retrieves top-K
  candidates via approximate nearest neighbor on title embeddings; Stage 2
  reranks with the full probe scorer. This reduces the O(N) catalog scan.
- [ ] Try a larger encoder (e.g. `voidful/albert_chinese_base` or a
  `bert-base-chinese`) if latency allows (~22 s/epoch on CPU with tiny).
- [ ] Add popularity bias tuning: the `pop_weight` parameter is currently
  trained; consider a separate popularity reweighting schedule.
- [ ] Experiment with different loss functions (margin ranking loss, sampled
  softmax) instead of cross-entropy over candidate sets.

## Data / preprocessing

- [ ] `preprocess_tgredial.py` still consumes PKL and cannot rebuild JSONL from
  `csv_branch` (noted limitation in `project_progress_log.md` §3). Add JSONL
  regeneration from the raw CSV branch.
- [ ] Add data versioning (DVC or simple checksums) for `TG-ReDial-Data/out/`
  artifacts so data changes are detectable.

## Testing

- [ ] Increase integration test coverage: add a full-catalog test (not just
  candidate-set) to `test_chat_app.py`.
- [ ] Add a CI job that runs `evaluate_test.py` and asserts metrics stay above
  a regression threshold (e.g. Hit@10 > 0.0005 on full catalog).
- [ ] Add unit tests for `evaluate_test.py` metric functions.

## Docs

- [x] Add a `docs/` folder with a training guide, serving guide, and model
  card (architecture, metrics, limitations, intended use).
  - Created `docs/model_card.md` (176 lines: architecture, metrics, limitations,
    reproduction steps, ethical considerations, references).
- [ ] Document expected latency: ~22 s/epoch on CPU for training; ~2 s per
  chat request (encoder forward + full-catalog scoring).
- [ ] Add a `CONTRIBUTING.md` with setup and PR guidelines.
