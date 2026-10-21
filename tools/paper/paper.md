\documentclass[conference]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{hyperref}

\title{Efficient Conversational Movie Recommendation via Frozen-Encoder
Probing: Efficiency Gains and Honest Negative Results on TG-ReDial}

\author{
  \IEEEauthorblockN{Project Authors}
  \IEEEauthorblockA{TG-ReDial RecSys Project\\
    \texttt{\{paper@email\}}}
}

\begin{document}

\maketitle
\thispagestyle{plain}
\pagestyle{plain}

\begin{abstract}
Conversational recommender systems (CRS) increasingly rely on fine-tuning
large pretrained language models, which is costly to train and serve. We
propose a \emph{frozen-encoder linear probe}: a fixed ALBERT-tiny encoder
(312-dim, frozen) produces dialogue-context and title embeddings, and a
246k-parameter MLP probe (965\,KiB checkpoint) ranks titles against them. On
the Chinese-recommendation variant of TG-ReDial we carefully separate two
evaluation protocols that are often conflated: \emph{candidate-set ranking}
(target among 64 sampled negatives) and \emph{full-catalog ranking} (all
33{,}834 titles). On the official 2{,}244-example test split the probe
achieves \textbf{29.72\% Hit@10}, \textbf{13.93\% NDCG@10}, and
\textbf{12.10\% MRR} under candidate-set ranking (1.9--2.0$\times$ the
analytic random baseline), and 0.09\% Hit@10 / 0.033\% NDCG@10 under the far
harder full-catalog protocol, at \textbf{47--52 examples/second on a
CPU-only workstation}. We further contribute (i) a calibrated
ask-vs-recommend policy whose top-20 score entropy measurably identifies
hard cases (baseline Hit@10 0.19 vs 0.27 on gated subsets), and (ii)
multi-seed ablations with two \emph{honest negative results}: injecting
user profiles and watch history at inference time \emph{degrades} ranking
for a probe trained on dialogue-only contexts, and simulating preference
elicitation with oracle answers is metric-neutral. Code, frozen protocol,
run logs, and seeds are released for reproducibility.
\end{abstract}

\section{Introduction}
Conversational recommender systems model user preferences through dialogue
and recommend specific items mid-conversation. Modern CRS models fine-tune
large pretrained language models, but this is expensive: fine-tuning takes
hours of accelerator time and serving requires substantial memory. In
production, cost and latency constraints force tradeoffs between quality
and deployment feasibility.

We propose a \emph{frozen-encoder linear probe} that decouples
representation learning from ranking:
\begin{enumerate}
  \item \textbf{Frozen encoder:} ALBERT \texttt{chinese-tiny} (312-dim
    hidden) encodes dialogue context and movie titles; no gradients flow
    through it.
  \item \textbf{Trainable probe:} a 2-layer MLP (312 $\rightarrow$ 256)
    projects both sides into a shared space scored by temperature-scaled
    cosine similarity plus a learned log-frequency popularity term.
\end{enumerate}

Our contributions:
\begin{itemize}
  \item A CPU-only, reproducible CRS pipeline with a 965\,KiB checkpoint and
    measured costs (training $\sim$22\,s/epoch; inference 47--52 ex/s).
  \item Separated evaluation of \emph{both} candidate-set and full-catalog
    protocols with analytic random baselines, plus multi-seed ablations.
  \item A calibrated uncertainty gate (top-20 entropy standardized within
    the top-20) that measurably targets hard cases, and an honest report
    that eliciting preferences at inference time does not improve ranking
    without matched retraining.
\end{itemize}

\section{Related Work}
\label{sec:related}
Knowledge-grounded CRS: KBRD \cite{chen2019kbrd} jointly trains dialog
generation and recommendation over entity embeddings. Our dataset,
TG-ReDial \cite{zhou2020tgredial}, was proposed with topic-guided neural
baselines. Tooling and benchmarking:
CRSLab \cite{zhou2021crslab} standardizes CRS implementations and
evaluation, and BARS \cite{zhu2022bars} campaigns for open, reproducible
recsys benchmarking---the standard this repository aspires to (frozen
protocol, published seeds/logs). Unlike all of the above, we freeze the
encoder and quantify what a 246k-parameter probe can recover, including
where it \emph{cannot} (Sections~\ref{sec:ablation}--\ref{sec:elicitation}).

\section{Method}
\label{sec:method}
\textbf{Context encoding.} The served context is built from the full
dialogue (speaker-prefixed turns, truncated to 1{,}500 chars / 256 ALBERT
tokens), optionally enriched with user-profile sentences and the titles of
the last eight watched movies (\texttt{build\_enriched\_context\_text}).
Frozen-encoder outputs are mean-pooled with attention masking and
L2-normalized.

\textbf{Scoring.} Context and title embeddings are projected by the shared
MLP head, cosine similarity is scaled by a learned temperature (clamped to
$[1,20]$), and a learned log-frequency popularity term is added. Training
uses cross-entropy over \{target + up to 64 sampled negatives\} per example
with a learnable temperature $\tau$:
\begin{equation}
    \mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B} \log
    \frac{\exp(s(c_b, y_b) / \tau)}
    {\exp(s(c_b, y_b) / \tau) + \sum_{n=1}^{N_n} \exp(s(c_b, y_{b,n}) / \tau)}.
\end{equation}
Optimization: AdamW with cosine annealing ($10^{-3}$, batch 64, gradient
clipping 1.0, early-stopping patience 2).

\textbf{Uncertainty and the ask-vs-recommend gate.} A naive softmax over
catalog scores saturates: top-20 entropy is pinned at $\log 20 = 2.996$ for
every example because raw scores are near-identical. Instead we standardize
the top-20 scores \emph{within the top-20} and compute softmax entropy,
which restores real dynamic range (1.55--2.67 nats on validation) and
anti-correlates with ranking success (Pearson $r = -0.115$ with Hit@10 over
150 validation examples). The service asks a clarification question when
entropy exceeds a threshold calibrated as the 70th percentile of
\emph{train-split} entropies (target ask-rate 0.3); the question type
(interests / watch history / genre) is selected from the missing signal.

\section{Data and Evaluation Protocol}
\label{sec:protocol}
Table~\ref{tab:split} reports the split statistics used throughout (single
source of truth: \texttt{TG-ReDial-Data/out/dataset\_stats.json}).

\begin{table}[h]
\centering
\caption{Dataset splits. All splits rank against the same 33{,}834-title
catalog.}
\label{tab:split}
\begin{tabular}{lrr}
\toprule
Split & Examples & Profile / History coverage \\
\midrule
Train & 25,485 & 100\% / 98.1\% \\
Valid & 2,271  & 100\% / 96.8\% \\
Test  & 2,244  & 100\% / 98.0\% \\
\bottomrule
\end{tabular}
\end{table}

We report two protocols separately and never conflate them.
\textbf{(P1) Candidate-set ranking:} each target is ranked against 64
uniform random negatives (seed 42); analytic random expectations are
Hit@10 $=10/65=0.1538$, NDCG@10 $=0.0699$, MRR $=H_{65}/65=0.0732$.
\textbf{(P2) Full-catalog ranking:} each target is ranked against all
33{,}834 titles. The official TG-ReDial COLING 2020 paper reports NDCG@10
$0.0348$ under its own (different) preprocessing and is not directly
comparable. Test-split evaluation happens once per final configuration;
all tuning uses the validation split; elicitation thresholds are calibrated
on train only. Full frozen protocol: \texttt{tools/docs/protocol.md}.

\section{Results}
\label{sec:results}

\subsection{Test-split results}
Table~\ref{tab:test} reports the official test split. The probe reaches
1.9--2.0$\times$ random under P1 and $\sim$3$\times$ random Hit@10 under
P2, at 47--52 examples/second on CPU.

\begin{table}[h]
\centering
\caption{Official test split (2,244 examples). Ratios are against analytic
random baselines for each protocol.}
\label{tab:test}
\begin{tabular}{lccc}
\toprule
Protocol & Hit@10 & NDCG@10 & MRR \\
\midrule
P1 candidate-set & \textbf{0.2972} & \textbf{0.1393} & \textbf{0.1210} \\
\quad random (analytic) & 0.1538 & 0.0699 & 0.0732 \\
\quad ratio & 1.9$\times$ & 2.0$\times$ & 1.7$\times$ \\
\midrule
P2 full-catalog & 0.00089 & 0.00033 & 0.00059 \\
\quad random (analytic) & 0.000296 & 0.000134 & 0.000231 \\
\quad ratio & 3.0$\times$ & 2.5$\times$ & 2.6$\times$ \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Ablation: what should the probe condition on?}
\label{sec:ablation}
We ablate the served context on validation (400 examples) across four
negative-sampling seeds \{7, 42, 123, 2026\}, reporting mean$\pm$
std (Table~\ref{tab:ablation}). The released probe \emph{was trained on
dialogue-only contexts}; consequently, injecting profile/history at
inference (C, D) \emph{degrades} ranking on 3 of 4 seeds, and
history/profile-only input (B) is far worse. This is a distribution-shift
negative result: profile/history must be present at \emph{training} time to
help (matched-training retraining is ongoing work,
\texttt{train\_probe\_v2.py}).

\begin{table}[h]
\centering
\caption{Ablation on validation (400 examples), mean$\pm$std over four
negative-sampling seeds.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Served context & Hit@10 & NDCG@10 & MRR \\
\midrule
A current (last-8 turns) & \textbf{0.2900±0.0129} & \textbf{0.1415±0.0035} & \textbf{0.1298±0.0063} \\
B history/profile only   & 0.1413±0.0126 & 0.0635±0.0069 & 0.0698±0.0051 \\
C A + profile + history  & 0.2588±0.0297 & 0.1262±0.0095 & 0.1162±0.0035 \\
D C + elicited turn      & 0.2550±0.0290 & 0.1237±0.0105 & 0.1139±0.0046 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Elicitation: a gate that works, an effect that does not}
\label{sec:elicitation}
With the calibrated threshold (2.4468, target ask-rate 0.3, achieved 0.22 on
validation), the gate selects genuinely harder cases: asked examples have
baseline Hit@10 0.193 vs 0.269 for unasked (seed 7; consistent at seed 42).
However, appending an \emph{oracle} genre answer to the context---an
optimistic upper bound on elicitation value---is metric-neutral
($\Delta$ Hit@10 $= 0.000$; $\Delta$ MRR $= +0.0003$ on asked examples).
Interpretation: the uncertainty signal is usable for policy decisions
(e.g., deferring to humans or collecting training data), but inference-time
answer injection without matched training cannot move ranking metrics.
Artifacts: \texttt{TG-ReDial-Data/out/experiments/}.

\section{Conclusion and Future Work}
We presented a frozen-encoder linear probe for conversational movie
recommendation that trains in minutes on CPU, serves at 47--52 examples/s
from a 965\,KiB checkpoint, and reaches 29.7\% candidate-set Hit@10
(1.9$\times$ random) on TG-ReDial. We deliberately reported both protocols,
analytic random baselines, multi-seed ablations, and two negative results.

\noindent\textbf{Limitations:} (1) Candidate-set and full-catalog numbers
are protocol-specific; full-catalog ranking remains far below specialized
two-stage systems---a recall+rerank architecture is recommended. (2)
Inference-time profile/history injection hurts without matched retraining;
the retrained ablation and the CRSLab/KBRD toolkit baselines
\cite{zhou2021crslab,chen2019kbrd} are outstanding (CPU-only budget).
(3) Elicitation uses simulated (oracle) answers, not real user interaction.

\noindent\textbf{Future work:} matched-training context ablation
(\texttt{train\_probe\_v2.py}), recall+rerank for full-catalog serving,
hard-negative mining, and real-user elicitation studies.

{
\footnotesize
\bibliographystyle{IEEEtran}
\bibliography{references}
}

\end{document}
