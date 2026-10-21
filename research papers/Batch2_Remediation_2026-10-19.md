# Batch-2 Audit Remediation (2026-10-19)

Maps each gap from `Batch2_Audit_2026-10-18_Checklist_FULL.md` to the action
taken and the evidence artifact. "Outstanding" items are honestly listed as
open with a concrete plan; nothing is claimed fixed that is not.

## 1. Data / profiles & history (was: empty user profiles)

- **Action:** fixed string-vs-int join keys + history join in
  `TG-ReDial-main/preprocess_tgredial.py`; regenerated
  `TG-ReDial-Data/out/` (old files backed up as `*.jsonl.bak-empty-profile`).
- **Evidence:** profile nonempty 25,485/25,485 / 2,271/2,271 / 2,244/2,244;
  watch history 98.1% / 96.8% / 98.0% (train/valid/test). See
  `tools/docs/experiment_log.md` §2.

## 2. Profile/history must affect serving (was: `recommend_with_context` ignored them)

- **Action:** `build_enriched_context_text` + `recommend_with_context` in
  `chinese_recommender.py` now encode profile sentences and last-8 watched
  titles into the query; smoke-tested end-to-end (both entropy paths).
- **Evidence:** `TG-ReDial-main/smoke_rc.py` output; multi-seed ablation
  in `tools/docs/experiment_log.md` §4 (injection at inference *hurts*
  without matched training — reported honestly, matched retraining tracked
  in TODO).

## 3. Real uncertainty (was: `confidence = topscore/10`)

- **Action:** retired `topscore/10`; implemented top-20 softmax entropy with
  z-scoring *within* top-20 (`topk_score_entropy`); validated on 150 valid
  examples that it has dynamic range (1.55–2.67 vs saturated 2.996) and
  anti-correlates with Hit@10 (r = −0.115). Fixed a real
  missing-`@torch.no_grad()` bug in `recommend_with_context` found by the
  smoke test.
- **Evidence:** `TG-ReDial-main/uncertainty_probe.py` →
  `uncertainty_probe.json`; `tools/docs/experiment_log.md` §3.

## 4. Adaptive ask-vs-recommend policy (was: one hardcoded string)

- **Action:** entropy gate with threshold calibrated as a train-split
  quantile (`elicitation_sim.py --ask-rate 0.3` → threshold 2.4468, valid
  ask-rate 0.22); question type selected by missing signal (interests /
  history / genre). Gate validity: asked subset baseline Hit@10 0.193 vs
  unasked 0.269.
- **Evidence:** `elicitation_sim_s{7,42}.json`;
  `tools/docs/experiment_log.md` §5.

## 5. Ablations (was: single-seed, n=400)

- **Action:** 4 cells × 4 seeds {7, 42, 123, 2026}; mean±std reported.
- **Evidence:** `ablation_context_s{7,42,123,2026}.json`;
  `tools/docs/experiment_log.md` §4; paper Table (ablation).

## 6. Protocol / reporting errors (was: misattributed metrics, wrong splits)

- **Action:** rewrote `tools/paper/paper.md` (+ `paper.tex`): correct split
  table (25,485/2,271/2,244), candidate-set vs full-catalog separated with
  *analytic* random baselines (previous "20× random" claim was wrong;
  correct is 1.9–2.0×), official COLING NDCG@10 0.0348 only as a
  non-comparable reference, structural LaTeX corruption fixed.
  `tools/docs/model_card.md` updated to match. Frozen protocol:
  `tools/docs/protocol.md`.
- **Evidence:** the rewritten paper/model card/protocol themselves; run
  logs in `tools/docs/experiment_log.md`.

## 7. Literature review / citations (was: none)

- **Action:** `tools/paper/references.bib` with arXiv-verified entries for
  KBRD (Chen et al., EMNLP 2019), CRSLab (Zhou et al., arXiv:2101.00939),
  BARS (Zhu et al., SIGIR 2022, arXiv:2205.09626); KGSAS / Shu et al. /
  DURec failed verification on 2026-10-19 (no arXiv/DOI/web match, no local
  PDF) and were removed as fabricated; `tools/docs/literature_review.md`
  documents the retraction (§3, §4, §7) and reviews the remaining papers
  with a gap statement.
- **Evidence:** literature review §verification log.

## 8. External baseline (CRSLab/KBRD toolkit) — OUTSTANDING

- Not reproduced: toolkit expects GPU-scale training; environment is
  CPU-only. Recorded in `TODO.md` with the exact reason; we compare only
  within one audited protocol instead of quoting unverifiable numbers.

## 9. Multi-seed protocol & cost reporting — DONE

- Seeds, hardware, per-run costs, and reproduction commands:
  `tools/docs/protocol.md` + `tools/docs/experiment_log.md` §6–7.

## 10. Headline metric improvement (Hit@10 > 29.72% candidate-set) — OUTSTANDING

- No configuration has yet beaten the frozen baseline on the test
  candidate-set protocol; `train_probe_v2.py` (enriched-context training,
  hard negatives, cosine LR, validation-gated checkpoints) has not completed
  a CPU run. No improvement is claimed. Test set untouched during these
  audit fixes (only valid/train used).
