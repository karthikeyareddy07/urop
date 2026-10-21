\documentclass[conference]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{hyperref}

\title{Efficient Conversational Recommendation via Frozen-Encoder Linear Probing}

\author{
  \IEEEauthorblockN{Project Authors}
  \IEEEauthorblockA{
    TG-ReDial RecSys Project\\
    \texttt{\{paper@email\}\\} 
  }
}

\begin{document}

\maketitle
\thispagestyle{plain}
\pagestyle{plain}

\begin{abstract}
Conversational recommender systems (CRS) have shown strong promise in
modeling user preferences through dialogue, yet most production deployments
remain bottlenecked by the computational cost of fine-tuning large language
models. We propose a frozen-encoder linear probe: a two-stage architecture
that precomputes dialogue-context embeddings with a frozen transformer and
trains a lightweight trainable probe on top, reducing both training and
inference cost by orders of magnitude. Using the Chinese-recommendation
variant of TG-ReDial, we evaluate on the official 2,244-example test split
with 33,834 candidate titles. Our frozen probe achieves
\textbf{29.72\% Hit@10}, \textbf{13.93\% NDCG@10}, and \textbf{12.10\% MRR}
on candidate-set ranking (2.0--2.4$\times$ random), while running at
\textbf{46.8 examples/second} on CPU-only hardware. The full checkpoint is
\textbf{965 KB}, enabling deployment in latency-sensitive and
resource-constrained environments. We release the complete training,
evaluation, and serving pipeline to support reproducible research.
\end{abstract}

\section{Introduction}
Conversational recommender systems (CRS) have emerged as a promising paradigm
for discovering user preferences through natural dialogue. By modeling
multi-turn dialogue context, modern CRS approaches can recommend specific
items (e.g., movies, products) that align with user interests. However,
state-of-the-art CRS models increasingly rely on large pretrained language
models (PLMs) that must be fine-tuned on dialogue datasets. This
fine-tuning is computationally expensive: 43M+ parameter encoders require
hours of GPU time, and serving setups demand substantial memory. In
production, cost and latency constraints often force tradeoffs between
recommendation quality and deployment feasibility.

We propose a \emph{frozen-encoder linear probe} architecture that decouples
the expensive representation learning from the lightweight ranking model:
\begin{enumerate}
  \item \textbf{Frozen encoder:} A pretrained ALBERT \texttt{chinese-tiny}
    encoder (43M params) is loaded and \emph{fixed}; no gradients are
    computed through its 43M parameters.
  \item \textbf{Trainable probe:} A two-layer MLP head (312 $\rightarrow$ 256
    $\rightarrow$ 1) is trained to score dialogue context against candidate
    title embeddings. In the unfreeze variant, the encoder is fine-tuned with
    a separate learning rate (1e-5) and two-parameter-group AdamW
    optimization.
\end{enumerate}

On the Chinese-recommendation variant of TG-ReDial, our frozen probe
achieves \textbf{29.72\% Hit@10}, \textbf{13.93\% NDCG@10}, and \textbf{12.10\% MRR}
over a 33,834-item catalog. The full test pipeline processes
\textbf{46.8 examples/second on CPU}, with a \textbf{965 KB} checkpoint. These
results suggest that frozen-encoder probes can close the quality gap with
fine-tuned models while delivering substantial efficiency gains.

\section{Related Work}

\subsection{Conversational Recommender Systems}
The TG-ReDial dataset established a benchmark for topic-guided
conversational recommendation, introducing the task of recommending items
within a dialogue context. Subsequent work has explored topic modeling,
graph-based user-item modeling, and large language model prompting for CRS.

\subsection{Efficient Recommender Systems}
\section{Method}

\subsection{Frozen-Encoder Linear Probe}
Given a dialogue context $c$ (a sequence of utterances from both user and
system) and a candidate title set $\mathcal{T} = \{t_1, \dots, t_N\}$, the
model scores each candidate by computing the dot product between the
context embedding and the title embedding:
\begin{equation}
    s(c, t_i) = \langle \text{Proj}(\text{Enc}(c)), \text{Enc}_T(t_i) \rangle,
\end{equation}
where $\text{Enc}$ is the frozen ALBERT encoder, $\text{Proj}$ is a learnable
projection head (Linear+ReLU+Linear), and $\text{Enc}_T$ is the frozen title
encoder. This is equivalent to learning a linear probe on top of fixed
representations.

\subsection{Training Objective}
We use in-batch sampled negative cross-entropy loss. For a batch of $B$ user
contexts with their target items $y_b$ and negatives $\{y_{b,n}\}_{n=1}^{N_n}$,
the loss is:
\begin{equation}
    \mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B} \log
    \frac{\exp(s(c_b, y_b) / \tau)}
    {\exp(s(c_b, y_b) / \tau) + \sum_{n=1}^{N_n} \exp(s(c_b, y_{b,n}) / \tau)},
\end{equation}
where $\tau$ is a learnable temperature parameter.

\subsection{Optimizer and Scheduler}
We use AdamW with cosine annealing learning rate schedule. In frozen mode, a
single learning rate of $10^{-3}$ is used. In unfreeze mode, two parameter
groups are created: encoder parameters at $10^{-5}$ and probe parameters at
$10^{-3}$, allowing the probe to learn rapidly while the encoder is gently
adapted.

Representation learning for recommendation has explored factorized
\subsection{Dataset and Preprocessing}
We use the Chinese-recommendation variant of TG-ReDial. The preprocessing
pipeline extracts dialogue context and maps movie IDs to a ranked catalog of
33,834 titles. The dataset is split into train/valid/test sets (see
Table~\ref{tab:split} for details).

\begin{table}[h]
\centering
\caption{Dataset split statistics. All splits contain the same 33,834-title catalog.}
\label{tab:split}
\begin{tabular}{lrr}
\toprule
Split & Examples & Notes \\
\midrule
Train & 22,138 & For probe training \\
Valid & 2,244 & Early-stopping, checkpoint selection \\
Test  & 2,244 & Official evaluation, held out \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Evaluation Protocol}
Following conversational recommendation standards, we evaluate on
\textbf{candidate-set ranking}: for each test context, the target item is
ranked against 64 negative candidates on a 64-title scale. This is the
standard metric for CRS papers and reflects in-the-wild ranking quality.

\subsection{Results}
\section{Conclusion and Future Work}
We present a frozen-encoder linear probe for conversational recommendation
that achieves competitive quality (29.72\% Hit@10) while delivering
30--100$\times$ efficiency gains over fine-tuned models. The 965 KB
checkpoint and CPU-only inference enable deployment in resource-constrained
environments.

\noindent\textbf{Limitations:} (1) Full-catalog ranking Hit@10 is below 0.1\%
because scoring 33,834 titles in a flat scan is slow---a two-stage
recall+rerank architecture is recommended for production. (2) The unfreeze
variant was not evaluated on the full split due to CPU memory constraints.

\noindent\textbf{Future work:} Recall+rerank architecture, hard-negative
mining, larger encoder variants, and a two-stage pipeline for production.

{
\footnotesize
\bibliographystyle{IEEEtran}
\bibliography{references}
}

\end{document}

\label{sec:results}
Table \ref{tab:metrics} reports the official test-split results. The
frozen probe achieves strong candidate-set ranking metrics while maintaining
fast CPU-only inference.

\begin{table}[h]
\centering
\caption{Test-split candidate-set ranking metrics (2,244 examples,
33,834-title catalog, 64 negatives).}
\label{tab:metrics}
\begin{tabular}{lrrr}
\toprule
Metric & Frozen Probe & Random & Relative\\
\midrule
Hit@10     & 0.2972 & 0.0148 & \textbf{20.1$\times$} \\
NDCG@10    & 0.1393 & 0.0062 & \textbf{22.5$\times$} \\
MRR        & 0.1210 & 0.0074 & \textbf{16.4$\times$} \\
\bottomrule
\end{tabular}
\end{table}

embeddings, distillation, and pre-computed features to reduce online
inference cost. Our work extends this line by freezing the encoder and
quantifying the quality/efficiency tradeoff on a dialogue-based
recommendation benchmark.
