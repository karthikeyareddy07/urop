# Literature Review — Batch-2 Seven-Paper Set (2026-10-18)

This review covers the seven papers named in the Batch-2 research plan and
states, for each, what it contributes and how it relates to this project's
frozen-encoder probing approach. Bibliographic details were verified against
arXiv on 2026-10-18 where noted. The three entries previously flagged
TODO-VERIFY (KGSAS, Shu et al., DURec) failed verification on 2026-10-19 —
no arXiv/DOI/web match and no source PDF exists locally — and were removed
as presumed fabricated; sections 3, 4, and 7 record that outcome.

## 1. Zhou et al., COLING 2020 — TG-ReDial (dataset & task)

"Towards Topic-Guided Conversational Recommender System" (arXiv:2010.04125).
Introduces TG-ReDial: topic-guided multi-turn dialogues where a recommender
must suggest movies mid-conversation; proposes KBRD-style neural baselines
plus topic annotations. **Relation:** defines the dataset we use (Chinese
variant, 33,834-title catalog) and the official evaluation setting. Our
full-catalog numbers are only loosely comparable to the paper's because our
preprocessing/splits differ; we always report our protocol explicitly
(`tools/docs/protocol.md`).

## 2. Chen et al., EMNLP 2019 — KBRD

"Towards Knowledge-Based Recommender Dialog System" (arXiv:1908.05391,
VERIFIED). Jointly trains a knowledge-grounded dialog generator and a
recommender over entity (movie) embeddings; recommendation and generation
mutually benefit. **Relation:** the canonical neural CRS baseline. It requires
fine-tuning a full encoder — precisely the cost our frozen-probe design
avoids. Not reproduced here (GPU-oriented training); tracked as future work
in `TODO.md`.

## 3. KGSAS — retracted citation (verification failed 2026-10-19)

**Citation removed from `tools/paper/references.bib`.** No paper with this
acronym or title exists: the arXiv API returns 0 hits, web searches (ACM,
MDPI, Semantic Scholar, CiteSeerX) find nothing, and no CIKM 2021 PDF is
present in `research papers/`. The idea — topic-guided KG attention as
complementary side information to a text-only probe — survives only as a
design note, not as a literature claim.

## 4. Constraint-aware conversational RS — retracted citation (2026-10-19)

**Citation removed from `tools/paper/references.bib`.** No AAAI 2020 paper
matching "constraint-aware and path-exploratory" by Shu et al. could be
found (arXiv 0 hits for "path-exploratory"; no DOI/web match; no local
PDF). Our explicit "constraints" slot in the preference-state dict
(`recommend_with_context`) remains a design choice motivated by the
documented extraction gap — not by this citation.

## 5. Zhou et al., 2021 — CRSLab toolkit (VERIFIED: arXiv:2101.00939)

Open-source toolkit standardizing CRS datasets, 18 models, and automatic
evaluation protocols. **Relation:** the Batch-2 plan names CRSLab/UniCRS as
the external baseline. Not installed/trained in this repository: it expects
GPU-scale training and its own environment; we instead compare against our
frozen-probe baselines under one audited protocol and record the CRSLab
baseline as outstanding work (`TODO.md`), rather than reporting an
unreproducible number.

## 6. Zhu et al., SIGIR 2022 — BARS benchmarking (VERIFIED: arXiv:2205.09626)

Standardized, reproducible benchmarking pipelines for recommender systems
(datasets, code, hyperparameters, logs published together). **Relation:**
directly motivates this repository's frozen protocol doc, pinned seeds,
published run logs (`tools/docs/protocol.md`, `tools/docs/experiment_log.md`),
and the insistence on separating candidate-set from full-catalog reporting.

## 7. DURec — retracted citation (verification failed 2026-10-19)

**Citation removed from `tools/paper/references.bib`.** The arXiv API
returns 0 hits for "dual-network architecture" + "task-oriented" and no
such paper by Yajiao You exists; no local PDF. Our ask-vs-recommend
elicitation policy remains motivated by mixed-goal dialog design reasoning
in general, not by this citation.

## Gap statement (what this project adds, honestly scoped)

1. **Efficiency:** frozen ALBERT-tiny encoder + 246k-param probe (965 KiB
   checkpoint) versus full fine-tuning in every baseline above; CPU-only
   training/inference with measured costs.
2. **Uncertainty & elicitation:** a *calibrated* ask-vs-recommend gate —
   top-20 entropy z-scored within top-20 (measured dynamic range and
   anti-correlation with Hit@10; `uncertainty_probe.py`), threshold set by
   train-split quantile. Honest negative result: oracle-answer elicitation
   does not improve ranking metrics under inference-time injection.
3. **Protocol hygiene:** candidate-set vs. full-catalog separation with
   analytic random baselines, multi-seed ablations, and immutable protocol —
   addressing the reproducibility concerns BARS raises.

## Verification log

- 2026-10-18: arXiv abs pages fetched for 1908.05391 (KBRD), 2205.09626
  (BARS); arXiv API queries for "CRSLab" (found 2101.00939), "KGSAS"
  (0 results), "Dual-Network Architecture for Task-Oriented" (0 results).
- 2026-10-19: full re-verification of KGSAS / Shu et al. / DURec via arXiv
  API (0 hits each), web search (no exact-title match on any index), and
  local PDFs (none exist). All three presumed fabricated; removed from
  `references.bib`, `paper.md`, and this review (sections 3, 4, 7).
