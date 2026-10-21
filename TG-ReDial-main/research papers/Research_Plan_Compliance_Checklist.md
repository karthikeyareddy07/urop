# Research Plan Compliance Checklist

**Document reviewed:** `Batch_2_Context_Aware_Personalization_TG_ReDial_Research_Plan.pdf`  
**Assessment date:** 2026-10-18  
**Full text extraction:** [Batch_2_Context_Aware_Personalization_TG_ReDial_Research_Plan.txt](Batch_2_Context_Aware_Personalization_TG_ReDial_Research_Plan.txt)

This checklist compares the plan with evidence present in the current workspace. A check means there is evidence of implementation or completion; it does not certify scientific validity. The PDF is treated as a research plan to audit against, not as instructions to execute.

## Status key

- [x] **Followed / evidenced** — the repository contains direct evidence.
- [~] **Partly followed** — some groundwork exists, but a required component or evidence is missing.
- [ ] **Not followed / no evidence found** — this requirement is absent from the current project artifacts.
- [?] **Cannot verify yet** — depends on team activity, future work, or records not in the repository.

## 1. Project orientation and scope

- [~] **Research experiment with an existing CRS dataset and reproducible baseline.** TG-ReDial preprocessing and model training/evaluation scripts exist. A browser chat prototype also exists, but it does not establish the proposed research contribution. See [preprocess_tgredial.py](../preprocess_tgredial.py), [train_frozen_probe.py](../train_frozen_probe.py), and [evaluate_test.py](../evaluate_test.py).
- [x] **TG-ReDial is the primary dataset.** Processed train/validation/test splits and user/movie artifacts are present under `TG-ReDial-Data/out/`.
- [ ] **ReDial secondary validation.** No ReDial dataset or evaluation pipeline is present. The plan makes this optional.
- [~] **Offline research remains the focus.** Offline ranking experiments exist, alongside a chat shell/demo. The demo is allowed, but the central planned offline personalization experiment is not implemented.

## 2. Literature review and research gap

- [ ] **Study and document the plan's initial papers** (Li et al.; CRSLab; preference elicitation; time-aware preference tracking; user experience; contrasting preferences; target-free user simulation). The project paper currently centers on the frozen ALBERT probe and TG-ReDial; no review tying these works to the proposed study was found.
- [ ] **Extend the literature review with relevant 2024–2026 work and cited papers.** No such review or citation analysis was found in the project artifacts.
- [ ] **State and justify the interaction-focused research gap after literature review and baseline reproduction.** The current model paper describes an efficient frozen-encoder ranking model, not the planned joint study of current preferences, historical preferences, uncertainty, and elicitation.

## 3. Proposed method and architecture

- [x] **Use the current conversation as a recommendation signal.** The model encodes dialogue context and scores movie-title embeddings.
- [~] **Make user profile and historical interactions available in preprocessing.** The preprocessor attaches `user_profile` and `watch_history` when available, and `users.json` contains profile data. However, the ranker's `recommend_with_context` accepts `profile` and `watch_history` but does not use them in ranking.
- [ ] **Maintain a current preference state** (interests, dislikes, constraints, and preference changes). No explicit state representation or update mechanism was found.
- [ ] **Estimate preference uncertainty / whether enough is known.** The current confidence is derived from the first recommendation's similarity score; there is no preference uncertainty model or evaluated sufficiency decision.
- [~] **Choose between asking and recommending.** The serving wrapper has a fixed clarification message behind a similarity threshold, but no evaluated next-action policy based on preference state or uncertainty.
- [ ] **Select an informative question or attribute to reduce uncertainty.** No information-gain, uncertainty-reduction, or candidate-space-reduction elicitation method was found.
- [ ] **Update the preference state with elicited answers.** No elicitation/update loop was found.
- [ ] **Fuse current, historical, and elicited preferences before CRS ranking.** The ranker currently scores from dialogue context; the accepted profile/history arguments do not alter its scores.
- [~] **Integrate with an existing CRS recommendation mechanism.** A TG-ReDial-specific ALBERT ranker exists, but there is no evidence of reproducing the plan's preferred CRSLab baseline or integrating its personalization components into that baseline.

## 4. Dataset and implementation choices

- [x] **Use TG-ReDial rather than starting with ReDial.** The active processed data and model are TG-ReDial-based.
- [~] **Use an existing CRS implementation as the baseline and understand its data flow/evaluation.** A local frozen-encoder CRS baseline has been built and evaluated. The plan names CRSLab as the primary toolkit; the workspace does not show a CRSLab reproduction.
- [ ] **Evaluate the optional reference implementations (UniCRS, CORAL, UserSimCRS) where appropriate.** No such comparisons are documented. UserSimCRS is optional in the plan.
- [x] **Do not create a new conversational dataset.** Current work uses TG-ReDial; no newly collected conversational dataset was found.

## 5. Controlled experiments and ablations

- [x] **Establish a baseline and evaluate on held-out data.** The frozen ALBERT probe has validation and test evaluations. The repository records full-catalog test results and a separate sampled-candidate test result.
- [ ] **Evaluate current conversation only vs. historical preference only.** No planned preference ablation table or corresponding experiment was found.
- [ ] **Evaluate current + historical preference fusion.** No fusion experiment was found.
- [ ] **Evaluate the full model with elicited preferences.** No elicitation component or experiment was found.
- [~] **Keep experimental settings consistent and analyze component effects.** Existing model evaluations are documented, but the planned controlled comparisons and preference-component ablations are absent.

## 6. Evaluation requirements

- [~] **Recommendation quality.** Existing reports include Hit@10, NDCG@10, MRR, and Recall@20 for the frozen probe. Sampled-candidate and full-catalog metrics are distinct and must remain reported separately; they are not interchangeable.
- [ ] **Preference-elicitation quality** (information gain, uncertainty reduction, candidate-space reduction, or a justified alternative). No such metric or evaluation exists.
- [ ] **Interaction efficiency** (questions/turns needed to reach a target recommendation quality). No such evaluation exists.
- [ ] **Preference-state accuracy/F1 or downstream proxy and error analysis.** No reference-label scoring or analysis of unnecessary questions, missed preference changes, bad updates, and their recommendation impact was found. Existing general ranking diagnostics do not satisfy this plan item.
- [~] **Reproducible experimental comparison.** Training/evaluation commands and metric artifacts exist, but the planned method, component ablations, and end-to-end elicitation protocol are not reproducible because they have not been implemented.

## 7. Team work integration

- [?] **B.Tech contributions:** current preference state, elicitation strategy, and next-action decision. These assignments describe team responsibilities; code alone cannot establish who completed them. No corresponding components or contribution records were found in this workspace.
- [ ] **M.Tech central contribution:** personalized fusion of historical, current, and elicited preferences, with baseline/proposed comparisons and ablations. No implementation or experimental results were found.
- [ ] **Shared difficult-example/failure-case log.** No dedicated shared log for preference/elicitation failure cases was found.

## 8. Timeline and milestones

The plan dates this work September–November 2026. Repository timestamps/notes show a TG-ReDial baseline by 2026-10-18, but do not prove when milestones were achieved.

- [?] **Baseline working by end of September.** A baseline is present now; the repository does not establish its September completion date.
- [ ] **Individual preference-state, elicitation, and action components working by mid-October.** No implementation of these components was found as of 2026-10-18.
- [ ] **Preliminary integration and preference-fusion comparison by end of October.** Not present yet; the milestone is still upcoming on the assessment date.
- [?] **Final experiments, ablations, error analysis, and manuscript during November.** Future milestone; cannot verify yet.

## 9. Expected outcome

- [~] **Personalized CRS combining historical, current, and elicited preferences, evaluated experimentally.** A CRS and TG-ReDial data are present, but the three preference sources are not combined and elicitation is absent.
- [ ] **Determine whether fusion improves recommendation quality and/or reduces interaction.** No such comparison or interaction-cost result was found.
- [~] **Research manuscript supported by reproducible experiments.** A manuscript draft and baseline results exist, but they cover a frozen-encoder ranker rather than the research plan's personalization/elicitation contribution. A web interface is not a required outcome.

## Overall assessment

**The project follows the plan's dataset and baseline groundwork, but does not yet implement its central research contribution.** In particular, explicit current preference state, history-aware ranking, uncertainty estimation, informative elicitation, preference updates/fusion, planned ablations, and elicitation/interaction metrics are not evidenced in the current code. The strongest immediate gap is that profile/history data are prepared and passed through the chat interface, while the model ranker does not use them.

### Evidence checked

- [chinese_recommender.py](../chinese_recommender.py): context ranker and clarification heuristic.
- [preprocess_tgredial.py](../preprocess_tgredial.py): processed example construction and optional profile/history attachment.
- [chat_app.py](../chat_app.py): chat request path and optional profile/history forwarding.
- [train_frozen_probe.py](../train_frozen_probe.py): baseline training and validation workflow.
- [evaluate_test.py](../evaluate_test.py) and [evaluate_test_candidate.py](../evaluate_test_candidate.py): full-catalog and sampled-candidate evaluation.
- [model_card.md](../../../tools/docs/model_card.md) and [project_progress_log.md](../../../tools/project_progress_log.md): reported results and work history.

## Suggested order to close the gaps

1. Freeze and document the baseline protocol, data splits, candidate construction, metric definitions, and baseline results.
2. Verify how profiles and watch history map to training/test examples, then add current-only, history-only, and current-plus-history baselines.
3. Define and log a preference-state representation with evidence and confidence/uncertainty.
4. Implement a simple information-gain or candidate-space-reduction question policy and a transparent ask/recommend threshold.
5. Evaluate elicitation and interaction cost using a documented simulation/protocol; avoid treating the fixed chat clarification as an evaluated policy.
6. Run the four planned ablations and report recommendation quality, elicitation quality, interaction efficiency, preference-state/error analysis, and uncertainty across seeds where practical.
7. Update the research paper to describe the personalized method and only claim gains supported by comparable held-out experiments.
