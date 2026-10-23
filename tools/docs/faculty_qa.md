# Anticipated Faculty Q&A — TG-ReDial Context-Aware Personalization Project

Prepared from the repository's audited artifacts. All numbers below are taken from those artifacts; nothing is invented.

## A. Problem & Motivation

Q1. What is the exact problem statement?
Conversational Recommender Systems (CRS) interact with users over multiple turns and recommend items from the information given. As a conversation progresses, the user's preferences may become clearer, change, or need further questioning. We investigate whether a CRS can (i) maintain a useful representation of the user's current preferences, (ii) combine it with the user's historical preferences, and (iii) use that to decide whether to ask for more information or proceed to recommendation. Research question: Can a context-aware current preference state, informed by historical user preferences and updated through preference elicitation, improve the selection of the next conversational action and recommendation quality?

Q2. What is the research gap?
Recent work covers dynamic preference tracking (TUT4CRS), elicitation strategy (Makarova et al.), contrasting preferences (Kook et al.), and target-free evaluation. The gap we target is their interaction: using current conversational context + historical user preferences to maintain a current preference state, estimate whether enough information is available, and decide whether to elicit another preference or recommend. We explicitly do not claim the sub-problems are unexplored.

Q3. Why TG-ReDial over ReDial?
TG-ReDial provides multi-turn conversations plus user profiles and historical watching/interaction data. ReDial is secondary validation only, introduced only if the pipeline is stable by late October.

Q4. Is this a product or research?
Offline research experiment. No website/mobile/chatbot required. Evaluated offline on TG-ReDial.

## B. Dataset

Q5. Dataset size and splits?
TG-ReDial v1.0: 10,000 conversations, 129,404 dialogue turns, avg 12.94 turns. Our processed splits: 25,485 train / 2,271 valid / 2,244 test dialogue examples. Catalog: 33,834 unique movie titles with precomputed ALBERT title embeddings. All 2,244 test targets present in catalog (0 missing).

Q6. What data quality issue did you fix?
Empty user profiles/history. Root cause: string user_id keys in user2TopicSent.pkl vs int joins. Fixed in preprocess_tgredial.py. Coverage after fix: profile nonempty 25,485/25,485 (train), 2,271/2,271 (valid), 2,244/2,244 (test); watch history 98.1% / 96.8% / 98.0%. Old files backed up as *.jsonl.bak-empty-profile.

Q7. What preprocessing is done?
Raw PKL to JSONL splits (train.jsonl, valid.jsonl, test.jsonl) plus movies.json, users.json, vocab.json, dataset_stats.json. Context encoding: last 8 turns, speaker-prefixed text, max 1500 chars, max 256 ALBERT tokens, mean-pooled with attention mask.

## C. Architecture & Model

Q8. Describe the model architecture.
Frozen voidful/albert_chinese_tiny encoder (312-dim hidden) to mean-pool to LayerNorm to MLP head to 256-dim query/key. Scoring = cosine similarity between projected context and title embeddings, scaled by learned temperature (clamped [1.0, 20.0]), plus learned log-frequency popularity bias. Encoder frozen; trainable params ~246k-2.0M. Checkpoint 965 KiB, MD5 48C83FA5642A5660659E24DF0793BC7A.

Q9. Why freeze the encoder?
Efficiency: CPU-only training/inference. Frozen ALBERT-tiny + 246k-param probe vs full fine-tuning in every baseline. Training ~22 s/epoch on CPU. End-to-end unfreezing supported (--unfreeze-encoder) but not yet evaluated on the full split.

Q10. Why ALBERT-tiny (Chinese)?
Chinese-language dialogues in TG-ReDial. Tiny variant keeps CPU feasibility. Larger encoders (bert-base-chinese) are future work if latency allows.

Q11. How does serving work?
chat_app.py is model-agnostic: make_handler(recommender); pass any object exposing recommend(context, top_k). recommend_with_context encodes full dialogue + user-profile sentences + last-8 watched titles (build_enriched_context_text) into the query. FastAPI server (serve_chinese.py) exposes /api/health, /, /api/chat.

## D. Training & Evaluation

Q12. Training hyperparameters?
Seed 42, 10 epochs, batch 64, LR 1e-3 AdamW, cosine annealing (T_max=10), max 64 negatives/example (rejection sampling), grad clip 1.0, early stopping patience 2. ~22 s/epoch on CPU.


Q13. How do you evaluate? Two protocols?
Candidate-set: target + 64 sampled negatives (seed 42). Test: Hit@10 0.2972, NDCG@10 0.1393, MRR 0.1210 (2,244 examples, ~48 s). Analytic random: Hit@10 0.1538, NDCG@10 0.0699, MRR 0.0732 -> ratios 1.9x / 2.0x / 1.7x.
Full-catalog: 33,834 titles. Test: Hit@10 0.00089, NDCG@10 0.00033, MRR 0.00059 (~52 ex/s). Harder; absolute metrics lower but retains ~2-3x random relative signal.
We always report which protocol. Never conflate candidate-set with full-catalog numbers (a prior "20x random" claim was corrected to 1.9-2.0x).

Q14. Why is full-catalog Hit@10 so low (<0.1%)?
Scoring 33,834 titles per context is an O(N) scan with a tiny probe; ranking is substantially harder. Recommended fix: two-stage recall+rerank (ANN on title embeddings to rerank with probe). Candidate-set Hit@10 31.6% shows the model does learn signal.

Q15. Did you compare against KBRD/CRSLab baselines?
No, not reproduced. KBRD requires GPU-scale training; CRSLab expects its own environment. We compare within one audited protocol and record the external baseline as outstanding (TODO.md), rather than quoting unreproducible numbers. This is a documented limitation, not a hidden gap.

## E. Uncertainty & Adaptive Elicitation (core contribution)

Q16. How do you estimate uncertainty?
Retired the saturated topscore/10 heuristic. Implemented top-20 softmax entropy, z-scored within top-20 (topk_score_entropy). Validated on 150 valid examples: dynamic range 1.55-2.67 (vs saturated 2.996) and anti-correlation with Hit@10 (r = -0.115). Catalog-wide softmax entropy is saturated at log(20)=2.996 (raw scores near-identical) - retired. Fixed a missing @torch.no_grad() bug found by smoke test.

Q17. How does the ask-vs-recommend policy work?
Entropy gate with threshold calibrated as a train-split quantile: elicitation_sim.py --ask-rate 0.3 -> threshold 2.4468, valid ask-rate 0.22 (88/400 asked), avg 1.22 turns. Question type selected by missing signal (interests / history / genre). Gate validity: asked subset baseline Hit@10 0.193 vs unasked 0.269 - gate correctly targets hard cases.

Q18. Does elicitation actually improve recommendations?
Honest negative result: oracle genre answer appended -> delta Hit@10 0.000, NDCG@10 0.000, MRR +0.0003 on asked subset; 0.000 across all 400. Same at seed 42. Appending answers at inference to a dialogue-only-trained probe does not help; matched training needed. We report this honestly.

Q19. What is the multi-seed ablation?
4 cells x 4 seeds {7, 42, 123, 2026}, valid n=400, shared negative sets per example:
- A current-only (last-8 turns): Hit@10 0.2900+-0.0129
- B history/profile-only: 0.1413+-0.0126
- C current+profile+history: 0.2588+-0.0297
- D C + simulated elicited turn: 0.2550+-0.0290
Honest finding: C/D <= A on 3 of 4 seeds. The probe was trained on dialogue-only contexts; injecting profile/history text at inference is a distribution shift that hurts. Matched retraining required (train_probe_v2.py; open TODO).

## F. Literature & Citations (handle carefully)

Q20. Which citations were verified / retracted?
Verified: KBRD (Chen et al., arXiv:1908.05391), CRSLab (Zhou et al., arXiv:2101.00939), BARS (Zhu et al., arXiv:2205.09626), TG-ReDial (Zhou et al., COLING 2020, arXiv:2010.04125). Retracted on 2026-10-19 (arXiv 0 hits, no DOI/web match, no local PDF): KGSAS, Shu et al. (constraint-aware CARS), DURec - removed from references.bib as presumed fabricated. We documented the retraction in literature_review.md sections 3/4/7 and the verification log.

Q21. Why mention retracted citations at all?
Transparency. We tried to verify; they failed; we removed them rather than keeping unverifiable claims. The ideas (constraints slot, ask-vs-recommend) survive as design choices, not literature claims.


## G. Limitations & Future Work

Q22. Top limitations?
1. Full-catalog ranking bottleneck (O(N) scan; <0.1% Hit@10). Two-stage recall+rerank recommended.
2. Frozen encoder - no domain adaptation; unfreezing supported but unevaluated.
3. Chinese-only.
4. Popularity bias may reduce diversity.
5. User modeling partially addressed: inference-time injection doesn't help without matched training (open: train_probe_v2.py).
6. External CRSLab/KBRD baseline not reproduced (GPU needed).

Q23. What is future work?
Matched-training ablation (retrain probe per cell), CRSLab/KBRD reproduction, --eval-test flag, lock file pinning, serve_chinese + Docker CI, frontend wiring, OpenAPI docs, rate limiting, two-stage recall+rerank, larger encoder, popularity reweighting, alternative losses, JSONL regeneration from CSV branch, data versioning, full-catalog integration tests.

## H. Likely "Trap" Questions

Q24. "Your enriched context hurts metrics - isn't that a failure?"
It's an honest, reported negative result. The probe was trained on dialogue-only contexts; injecting profile/history at inference is a distribution shift. The scientific contribution is identifying this and tracking matched retraining as the next step, not claiming an unverified improvement.

Q25. "Why report candidate-set numbers that look much better than full-catalog?"
We report both, always labeled, with analytic random baselines. Candidate-set is the standard CRS evaluation (target + sampled negatives); full-catalog is the production-relevant, harder setting. Conflating them is a known pitfall we explicitly avoid (corrected a prior "20x random" claim).

Q26. "Did you touch the test set during tuning?"
No. Test set untouched during audit fixes; only valid/train used for calibration (threshold quantile on train, ablation on valid).

Q27. "How reproducible is this?"
Pinned seeds, frozen protocol doc (tools/docs/protocol.md), published run logs with costs, checkpoint MD5, multi-seed ablations. Reproduction commands in experiment_log.md section 7. Lock file pinning still open (TODO).

## Suggested 30-second pitch
"We build a CPU-feasible conversational movie recommender on TG-ReDial: a frozen ALBERT-tiny encoder with a 246k-param probe, achieving 31.6% candidate-set Hit@10 (1.9x random) and <0.1% full-catalog Hit@10. The novel part is an adaptive ask-vs-recommend gate using within-top-20 entropy (r = -0.115 with Hit@10), threshold-calibrated on train quantiles. We honestly report that inference-time profile/history injection and oracle elicitation do not help without matched training, and we document retracted citations rather than keeping unverifiable claims."

