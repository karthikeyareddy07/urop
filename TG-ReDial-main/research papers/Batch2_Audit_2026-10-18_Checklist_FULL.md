# Batch 2 Research Plan â€” Compliance Audit (2026-10-18)

**Plan:** `TG-ReDial-main/research papers/Batch_2_Context_Aware_Personalization_TG_ReDial_Research_Plan.pdf`
**Full-text extraction:** `TG-ReDial-main/research papers/Batch2_Audit_2026-10-18_Plan_FullText.txt`
**Mode:** inspection only â€” no training run, no code/data modified (2 new audit files only).
**Key:** [x]=Followed/evidenced Â· [~]=Partly Â· [ ]=Not followed/no evidence Â· [?]=Cannot verify

## 1. Dataset, baseline, ReDial (plan Â§Â§1,5,8)

- [x] **TG-ReDial is primary dataset.** `TG-ReDial-Data/out/train.jsonl` (25,485) / `valid.jsonl` (2,271) / `test.jsonl` (2,244); `movies.json` (33,834); `movie_id_order.json`; `title_embeddings_albert_tiny.npz`; `dataset_stats.json`. Verified by read 2026-10-18.
- [~] **Reproducible ranking baseline exists â€” but NOT the plan's toolkit.** `TG-ReDial-main/train_frozen_probe.py` + `evaluate_test.py` + `evaluate_test_candidate.py`; ckpt `tools/out_probe/best.pt`. No CRSLab/UniCRS/CORAL/UserSimCRS usage found in any `.py` (grep 2026-10-18: zero hits). Plan Â§5 names CRSLab as primary baseline toolkit.
- [~] **Baseline reproduced, numbers verified â€” protocol caveat.** `tools/out_probe/test_candidate_metrics.json`: Hit@10=0.2972, NDCG@10=0.1393, MRR=0.1210 (target+64 uniform negs, seed 42, n=2244). Re-ran 2026-10-18: identical (0.2972/0.1393/0.1210). Full-catalog `TG-ReDial-main/out_probe/test_metrics.json`: Hit@10=0.00089, NDCG@10=0.00033, MRR=0.00059 â€” keep separate; paper draft conflates them (claims 29.72% "over a 33,834-item catalog", `tools/paper/paper.md:70-74`).
- [ ] **ReDial secondary validation.** No ReDial data/pipeline found. (Optional per plan â€” absence is compliant if main pipeline unstable.)
- [x] **Offline experiment focus.** Offline train/eval pipeline is the real workhorse. `chat_app.py` demo exists but plan explicitly says web/chatbot not required â€” its presence is fine.

## 2. Literature review & gap (plan Â§Â§2â€“3)

- [ ] **Initial 7-paper set studied/documented** (Li18, CRSLab21, Makarova24, TUT4CRS24, Ziegfeld25, Kook25/CORAL, target-free sim25). Grep over `tools/paper/paper.md/.tex`, `tools/docs/model_card.md`, `TODO.md` (2026-10-18): zero mentions of CRSLab/UniCRS/CORAL/Makarova/TUT4CRS/Kook/Ziegfeld. Paper Related Work has only 2 generic subsections, `\bibliography{references}` with no `.bib` in repo.
- [ ] **2024â€“2026 extension + cited-paper tracing.** No evidence.
- [ ] **Interaction-focused gap statement finalized post-review.** Paper frames contribution as frozen-encoder efficiency, not the plan's currentÃ—historyÃ—uncertaintyÃ—elicitation interaction (plan Â§3).

## 3. Preference state: interests/dislikes/constraints/changes (plan Â§Â§4,6: Student 1)

- [ ] **Current preference state (interests, dislikes, constraints, changes).** No state object/code found; grep `elicit|uncertainty|ablat|next.action|threshold|fusion|fuse` over `chinese_recommender.py/chat_app.py/train_probe_v2.py/evaluate_test_candidate.py` (2026-10-18) hit only `attention_mask` lines. `train_probe_v2.py:build_enriched_context` concatenates profile+history strings into the encoder input â€” a feature-string enrichment, not a structured state with update semantics.
- [ ] **Preference-change tracking (TUT4CRS-style).** No evidence.
- [ ] **Contrasting preferences (CORAL-style).** No evidence.

## 4. Profiles & watch history: present AND causal (plan Â§4)

- [ ] **Present in processed examples.** Measured 2026-10-18 over all 25,485 train examples: `user_profile` non-empty = **0**, `watch_history` non-empty = **0**. (Prior checklist Â§3 claimed preprocessor attaches them â€” code path exists at `preprocess_tgredial.py:268-285` but output is empty; likely `user2TopicSent.pkl` key/format mismatch. `users.json` HAS 1,482 profiles, e.g. user 0 = 10 sentences; `train_identity2history.pkl` HAS 25,485 keys e.g. `0/7â†’[14644]` â€” sources exist, join fails.)
- [ ] **Affect recommendations.** `ChineseRanker.recommend()` (`chinese_recommender.py:120-131`) scores `build_context_text(context)` only. `recommend_with_context()` (`:133-141`) accepts `profile`/`watch_history` params but ignores them (identical return path). V2 training script reads `users.json`/`watch_history` at encode time, but no trained V2 checkpoint exists (`out_probe_v2/` empty; `v2_seed42.log` tail = still encoding train contexts) â€” so no evaluated effect.
- [~] **Topic-guide attachment.** `TG-ReDial-Data/topic_to_id_2k5.json` (2,571 topics) exists, matching "topic guide" resourcing; no evidence it drives ranking or elicitation.

## 5. Uncertainty, ask-vs-recommend, elicitation, updates (plan Â§4: Students 2+3)

- [ ] **Preference uncertainty estimator.** `recommend_with_context` confidence = `recs[0].similarity/10` (`:136`) â€” a rescaled top score, not uncertainty over the preference state.
- [ ] **Ask-vs-recommend decision.** Only a fixed rule: `conf<0.05` â†’ hardcoded Chinese clarification string (`:137-139`); else recommend. No threshold study, no learned policy.
- [ ] **Informative question selection (info-gain / candidate-reduction).** No code; `chat_app.py:213-215` rerank/clarification path is display logic, not selection.
- [ ] **Preference-state update after elicitation.** No update mechanism found.
- Note: per instructions, the fixed clarification message + chat UI are NOT counted as adaptive elicitation.

## 6. CRS baseline integration (plan Â§Â§4â€“5: M.Tech)

- [~] **Existing-CRS-then-extend sequence.** A working ranker baseline exists and is reproduced, but it is a custom frozen-ALBERT probe, not CRSLab/UniCRS as Â§5 prescribes; fusion-before-CRS (`Updated Preference State â†’ Fusion â†’ Existing CRS`) cannot exist without the CRS.
- [ ] **Fusion of current+historical+elicited before CRS.** No fusion module; nothing to fuse into.

## 7. Ablations: current-only / history-only / current+history / full+elicited (plan Â§Â§7â€“8)

- [ ] **All four cells.** Zero ablation code, configs, or results files found (`TG-ReDial-main/testdata/` holds only `train.jsonl/valid.jsonl/out_probe2`; `out/experiments/` listing empty). History-only is currently unbuildable given Â§4 emptiness.
- [ ] **Shared difficult/failure-case log.** No experiment log found (only `train.log` 2-line unfreeze stub + `v2_seed42.log` in-progress).

## 8. Metrics: quality / elicitation / efficiency / state / errors (plan Â§8)

- [~] **Recommendation quality.** Candidate-set Hit@10/NDCG@10/MRR verified on held-out test (see Â§1); Recall@10/20 + full-catalog also computed (`recall20=0.00134`). BUT: single seed (42) only, no CIs; valid split numbers in paper table (22,138/2,244/2,244) contradict `dataset_stats.json` (25,485/2,271/2,244) â€” protocol doc error.
- [ ] **Elicitation quality** (info-gain / uncertainty-reduction / space-reduction). Nothing.
- [ ] **Interaction efficiency** (questions/turns to quality). Nothing.
- [ ] **Preference-state quality** (Acc/F1 or downstream+errors). Nothing.
- [ ] **Error analysis** (unnecessary questions, missed changes, bad updates, rec errors). Nothing.

## 9. Reproducibility & controls (plan Â§8 + your Â§9)

- [~] **Splits/commands logged.** `dataset_stats.json` + `model_card.md` + `TODO.md` exist; candidate eval pins seed 42 + `max_neg` (`evaluate_test_candidate.py:32-33,99-103`). Gaps: 3 different `out_probe/` dirs across scripts (`tools/` vs `TG-ReDial-main/` vs `PROJECT/out_probe`); `train.log` is a 2-line stub; no locked requirements cited in audit scope; paper split table wrong (see Â§8).
- [~] **Controlled comparisons.** Test set held out (test user_ids disjoint: 748 test vs 1482 train, verified 2026-10-18); selection on valid NDCG@10 in both trainers. Gap: no compared conditions exist yet beyond baseline-vs-nothing.
- [x] **No test tuning.** Single test eval per protocol; V2 not yet evaluated on test. Clean.

## 10. Team, timeline, manuscript (plan Â§Â§6â€“7,9â€“10)

- [?] **Student 1/2/3 + M.Tech roles & integration.** No authorship/task mapping in repo (no git â€” `git` binary absent, verified 2026-10-18). Cannot verify from artifacts.
- [?] **Sept baseline / mid-Oct components / end-Oct fusion / Nov manuscript.** Baseline exists now; no timestamps establish Sept completion; components/fusion absent as of 2026-10-18. Forward milestones unverifiable.
- [~] **Manuscript.** `tools/paper/paper.tex/.md` (~166 lines) drafts a frozen-probe efficiency paper â€” real artifact, but wrong contribution vs plan; `docs/model_card.md` + `TODO.md` support reproducibility, not the personalization claim.

## Overall assessment

**The project has solid TG-ReDial ranking groundwork but does not implement the plan's central contribution.** Verified: primary dataset present; ranking baseline reproduced exactly (candidate Hit@10=0.2972/NDCG@10=0.1393/MRR=0.1210, seed 42, n=2244; full-catalog 0.00089/0.00033/0.00059 â€” report separately); held-out test discipline clean. Missing: plan-mandated CRS toolkit baseline, literature review, preference state, uncertainty, adaptive elicitation, fusion, all four ablations, and 4 of 5 metric families. Two doc-level hazards: paper draft misattributes candidate-set 29.72% to full-catalog ranking, and its split table disagrees with `dataset_stats.json`. Prior checklist overstated profile/history availability â€” measured 0/25,485 non-empty.

## Most important gaps (ranked)

1. Profile/history join yields **zero** populated examples â€” blocks Student 1, M.Tech fusion, and history ablations.
2. No CRSLab/UniCRS baseline â€” plan Â§5's prescribed foundation.
3. No uncertainty / ask-vs-recommend / elicitation machinery beyond a fixed string.
4. Paper draft claims wrong contribution + wrong protocol attribution.
5. No ablation harness, multi-seed CIs, or error log.

## Prioritized next steps

1. Fix `preprocess_tgredial.py:268-285` join (`user2TopicSent.pkl` keys are `dialog/turn` like `0/7`, not `user_id`); re-emit splits; assert >95% profile/history coverage.
2. Stand up CRSLab TG-ReDial baseline (or document why custom probe substitutes); freeze protocol doc (splits, seeds, candidate vs catalog).
3. Add current-only/history-only/current+history training configs + eval; multi-seed (42/123/2026) valid-gated, single test eval.
4. Define preference-state schema + log; implement threshold ask/recommend + info-gain question policy; simulate and measure elicitation/interaction metrics.
5. Rewrite manuscript around the interaction question; correct split table + protocol attribution; add ablations, CIs, error analysis.

## Evidence index (all verified 2026-10-18)

`TG-ReDial-Data/out/{train,valid,test}.jsonl,movies.json,movie_id_order.json,title_embeddings_albert_tiny.npz,users.json,topic_to_id_2k5.json,dataset_stats.json}` Â· `TG-ReDial-Data/train_identity2history.pkl` Â· `TG-ReDial-main/{preprocess_tgredial.py:268-285,chinese_recommender.py:120-141,chat_app.py:194-223,train_frozen_probe.py,evaluate_test.py,evaluate_test_candidate.py:32-103,train_probe_v2.py,testdata/,research papers/}` Â· `tools/{out_probe/best.pt,test_candidate_metrics.json,train.log,paper/paper.{tex,md},docs/model_card.md}` Â· `TG-ReDial-main/out_probe/test_metrics.json` Â· `TODO.md` Â· `v2_seed42.log` Â· `research_plan_full_text.txt` (root, 15,928 chars â€” prior research-papers txt is stale/truncated at 24 chars).
