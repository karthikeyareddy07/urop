#!/usr/bin/env python3
"""Chinese-encoder conversational movie recommender.

Frozen ALBERT-tiny encoder -> mean-pool -> L2-norm -> shared trainable
projection (enc-dim -> 256) + 2-layer MLP head.
Score = cosine * temperature  +  log-frequency popularity bias.

Two classes:
  * ChineseRecommender  -- the trainable nn.Module (encoder-agnostic probe).
  * ChineseRanker       -- serving/inference wrapper with recommend() and
                           recommend_with_context(); loads checkpoint + catalog.

Checkpoint format (saved by train_frozen_probe.py / train_probe_v2.py):
  v1: {version:1, state, encoder, max_context_tokens, unfreeze_encoder}
  v2: {version:2, state, encoder, proj_dim, seed, epoch, valid}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

FILE_DIR = Path(__file__).resolve().parent
from chat_state import ConversationState  # noqa: E402

ENCODER_NAME = "voidful/albert_chinese_tiny"
PROJ_DIM = 256
MAX_CONTEXT_TOKENS = 256
VERSION = 1

# Default data directory for auto-discovery in ChineseRanker.load().
DATA_DIR = FILE_DIR.parent / "TG-ReDial-Data" / "out"


# ---------------------------------------------------------------------------
# Text / context helpers
# ---------------------------------------------------------------------------

def _norm_text(t: str) -> str:
    """Trim whitespace; return empty string for None / falsy input."""
    return (t or "").strip()


def build_context_text(context: List[Dict[str, str]], max_chars: int = 1500) -> str:
    """Legacy baseline context: last-8 turns only (kept for reproducibility)."""
    parts: List[str] = []
    for turn in context[-8:]:
        sp = turn.get("speaker", "")
        tx = _norm_text(turn.get("text", ""))
        if tx:
            parts.append(f"{sp}:{tx}")
    s = " ".join(parts)
    return s[-max_chars:]


def build_enriched_context_text(context, profile=None, history_titles=None,
                                max_chars: int = 2500) -> str:
    """Full-dialogue + profile + history context (audit-gap #1/#3 fix).

    ``profile`` may be a single string, a list of profile sentences, or None.
    ``history_titles`` may be a list of title strings or None.
    """
    parts: List[str] = []
    for turn in context or []:
        tx = _norm_text(turn.get("text", ""))
        if tx:
            parts.append(f"{turn.get('speaker', '')}:{tx}")
    if profile:
        prof = profile if isinstance(profile, (list, tuple)) else [profile]
        prof = [str(p) for p in prof if str(p).strip()][:10]
        if prof:
            parts.append("Users:" + ";".join(prof))
    if history_titles:
        hist = [str(h) for h in history_titles if str(h).strip()][-8:]
        if hist:
            parts.append("Watched:" + ";".join(hist))
    return " ".join(parts)[-max_chars:]


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def score_entropy(probs) -> float:
    """Shannon entropy of a candidate distribution (nats)."""
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum().clip(min=1e-12)
    return float(-(p * np.log(p.clip(min=1e-12))).sum())


def topk_score_entropy(scores, k: int = 20) -> float:
    """Entropy (nats) of a softmax over the top-k scores, z-scored WITHIN top-k."""
    s = np.asarray(scores, dtype=np.float64)
    t = s[np.argsort(-s)[:k]]
    if t.size == 0:
        return 0.0
    if t.std() < 1e-12:
        return 0.0
    z = (t - t.mean()) / (t.std() + 1e-9)
    ex = np.exp(z - z.max())
    p = ex / ex.sum().clip(min=1e-12)
    return float(-(p * np.log(p.clip(min=1e-12))).sum())


def softmax(x) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    ex = np.exp(x)
    return ex / ex.sum()


# ---------------------------------------------------------------------------
# Trainable model
# ---------------------------------------------------------------------------

class ChineseRecommender(nn.Module):
    """Trainable probe: proj + MLP head + temperature + pop bias.

    In frozen-probe mode (default) the ALBERT encoder lives outside the
    module -- contexts are pre-encoded to (enc_dim,) vectors and fed in.
    In end-to-end mode pass encoder/tokenizer and unfreeze_encoder=True.
    """

    def __init__(self, enc_dim=312, proj_dim=PROJ_DIM, movie_count=0,
                 temperature=1.0, encoder=None, tokenizer=None,
                 unfreeze_encoder=False, max_context_tokens=MAX_CONTEXT_TOKENS):
        super().__init__()
        self.proj = nn.Linear(enc_dim, proj_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, proj_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.pop_weight = nn.Parameter(torch.zeros(1))
        self.register_buffer("pop_log", torch.zeros(max(movie_count, 1)))
        self.enc_dim = enc_dim
        self.proj_dim = proj_dim
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.unfreeze_encoder = bool(unfreeze_encoder)
        self.max_context_tokens = max_context_tokens
        if encoder is not None:
            encoder.requires_grad_(self.unfreeze_encoder)


    def encode_contexts(self, contexts: List[str]) -> torch.Tensor:
        """Mean-pooled, L2-normalised context embeddings via attached encoder."""
        if self.encoder is None or self.tokenizer is None:
            raise RuntimeError(
                "ChineseRecommender was constructed without encoder/tokenizer; "
                "cannot encode contexts.")
        toks = self.tokenizer(
            contexts, padding=True, truncation=True,
            max_length=self.max_context_tokens, return_tensors="pt")
        device = next(self.parameters()).device
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.set_grad_enabled(self.unfreeze_encoder and self.training):
            h = self.encoder(**toks).last_hidden_state
            mk = toks["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * mk).sum(1) / mk.sum(1).clamp_min(1e-6)
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    def forward(self, ctx_vec, title_vecs, pop_idx=None):
        """Cosine-similarity score between context and title vectors.

        ctx_vec    : (B, enc_dim)    raw encoder embeddings
        title_vecs : (N, enc_dim)    or  (B, N, enc_dim)
        Returns    : (B, N)          similarity scores
        """
        c = self.head(self.proj(ctx_vec))
        t = self.head(self.proj(title_vecs))
        c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        if t.dim() == 3:
            s = torch.einsum("bd,bkd->bk", c, t)
        else:
            s = c @ t.T
        s = s * self.temperature.exp().clamp(1.0, 20.0)
        if pop_idx is not None:
            s = s + self.pop_weight * self.pop_log[pop_idx].unsqueeze(0)
        return s


# ---------------------------------------------------------------------------
# Serving wrapper
# ---------------------------------------------------------------------------

class ChineseRanker:
    """Serving wrapper: loads a trained probe + encoder/tokenizer + catalog.

    Attributes (stable API expected by callers):
        model        -- ChineseRecommender (probe head nn.Module)
        enc          -- ALBERT encoder (transformers AutoModel)
        tok          -- ALBERT tokenizer
        title_emb    -- np.ndarray (N, enc_dim) raw L2-normalised embeddings
        order        -- list[str] movie ids, same order as rows in title_emb
        movies       -- dict str(movie_id) -> {title, ...}
    """

    def __init__(self, model, movies, title_emb, order,
                 tokenizer=None, encoder=None,
                 max_context_tokens=MAX_CONTEXT_TOKENS,
                 conversation_state=None):
        self.model = model
        self.movies = movies
        self.title_emb = title_emb
        self.order = order
        self.tok = tokenizer
        self.enc = encoder
        self.max_context_tokens = max_context_tokens
        self.conversation_state = conversation_state

    @classmethod
    def load(cls, checkpoint, movies=None, title_emb=None, order=None):
        """Load from a checkpoint; auto-discover catalog data if not supplied."""
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        enc_name = ckpt.get("encoder", ENCODER_NAME)
        max_ctx = ckpt.get("max_context_tokens", MAX_CONTEXT_TOKENS)
        state = ckpt["state"]
        proj_w = state["proj.weight"]
        enc_dim = int(proj_w.shape[1])
        proj_dim = int(proj_w.shape[0])
        movie_count = int(state.get("pop_log", torch.zeros(1)).shape[0])
        model = ChineseRecommender(
            enc_dim=enc_dim, proj_dim=proj_dim,
            movie_count=movie_count, max_context_tokens=max_ctx)
        model.load_state_dict(state, strict=False)
        model.eval()
        from transformers import AutoModel, AutoTokenizer
        enc = AutoModel.from_pretrained(enc_name)
        tok = AutoTokenizer.from_pretrained(enc_name)
        enc.eval()
        if movies is None or title_emb is None or order is None:
            d = DATA_DIR
            if movies is None:
                with open(d / "movies.json", encoding="utf-8") as f:
                    movies = json.load(f)
            if order is None:
                with open(d / "movie_id_order.json", encoding="utf-8") as f:
                    order = json.load(f)["ids"]
            if title_emb is None:
                title_emb = np.load(d / "title_embeddings_albert_tiny.npz")["embeddings"]
        return cls(model, movies, title_emb, order,
                   tokenizer=tok, encoder=enc,
                   max_context_tokens=max_ctx)


    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Mean-pool + L2-normalise. Returns (N, enc_dim). Does NOT apply probe."""
        if not texts:
            dim = self.title_emb.shape[1] if self.title_emb is not None else 312
            return np.zeros((0, dim), dtype=np.float32)
        toks = self.tok(
            texts, padding=True, truncation=True,
            max_length=self.max_context_tokens, return_tensors="pt")
        h = self.enc(**toks).last_hidden_state
        mk = toks["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * mk).sum(1) / mk.sum(1).clamp_min(1e-6)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return v.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _score(self, ctx_emb: np.ndarray) -> np.ndarray:
        """Score a single embedding against full catalog. Returns (N,)."""
        tv = torch.from_numpy(np.ascontiguousarray(self.title_emb, dtype=np.float32))
        ctx = torch.from_numpy(np.ascontiguousarray(ctx_emb, dtype=np.float32))
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        scores = self.model(ctx, tv).squeeze(0)
        return scores.detach().cpu().numpy()

    def _format_scores(self, scores: np.ndarray, top_k: int) -> list:
        """Return structured top-k recommendation dicts from raw scores."""
        idx = np.argsort(-scores)[:top_k]
        out = []
        for i in idx:
            mid = self.order[int(i)]
            movie = self.movies.get(str(mid), {}) if isinstance(self.movies, dict) else {}
            title = movie.get("title") if isinstance(movie, dict) else None
            try:
                mid_out = int(mid)
            except (TypeError, ValueError):
                mid_out = mid
            out.append({
                "movie_id": mid_out,
                "title": title if title is not None else str(mid),
                "similarity": float(scores[int(i)]),
            })
        return out


    def recommend(self, context, top_k: int = 20):
        """Score dialogue context against the full title catalog.

        context : list[dict]  turn dicts with speaker and text keys
        top_k   : int         number of results to return
        Returns : list[dict]  each dict has movie_id, title, similarity
        """
        ctx_text = build_context_text(context)
        q = self._encode_texts([ctx_text])[0]
        scores = self._score(q)
        return self._format_scores(scores, top_k)


    def recommend_with_context(self, context, top_k: int = 20,
                               profile="", watch_history="",
                               entropy_threshold: float = 2.0):
        """One-shot recommendation with entropy-gated clarification.

        context           : list[dict]  dialogue turns
        profile           : str | list[str]  user profile
        watch_history     : str | list[str]  watched titles or IDs
        entropy_threshold : float  top-20 entropy above which to ask a question

        Returns dict with keys:
            recommendations, clarification, confidence, preferences,
            entropy, margin, and optionally question/state.
        """
        # Normalise profile into a list of non-empty strings.
        if isinstance(profile, str):
            prof = [profile] if profile.strip() else []
        elif isinstance(profile, (list, tuple)):
            prof = [str(p) for p in profile if str(p).strip()]
        else:
            prof = []
        # Normalise watch_history into a list of title strings.
        if isinstance(watch_history, str):
            hist_titles = [w.strip() for w in
                           watch_history.replace(";", "\n").replace(",", "\n").split("\n")
                           if w.strip()]
        elif isinstance(watch_history, (list, tuple)):
            hist_titles = []
            for mid in list(watch_history)[-8:]:
                m = self.movies.get(str(mid))
                if m and m.get("title"):
                    hist_titles.append(m["title"])
                elif isinstance(mid, str) and mid.strip():
                    hist_titles.append(mid)
        else:
            hist_titles = []
        # Encode enriched context.
        ctx_text = build_enriched_context_text(context, prof, hist_titles)
        q = self._encode_texts([ctx_text])[0]
        tv = torch.from_numpy(np.ascontiguousarray(self.title_emb, dtype=np.float32))
        scores = self.model(torch.from_numpy(q).unsqueeze(0), tv).squeeze(0).detach().cpu().numpy()
        # Entropy & margin over top-20.
        entropy = topk_score_entropy(scores, k=20)
        z = (scores - scores.mean()) / (scores.std() + 1e-9)
        ez = np.exp(np.clip(z - z.max(), -40, 40))
        probs_full = ez / ez.sum().clip(min=1e-12)
        order_idx = np.argsort(-z)
        margin = float(z[order_idx[0]] - z[order_idx[1]]) if len(order_idx) > 1 else 0.0
        prefs = {"interests": list(prof), "history_titles": hist_titles,
                 "uncertainty": entropy, "dislikes": [], "constraints": []}
        # --- Multi-turn conversation flow ---
        if (self.conversation_state and
                not self.conversation_state.completed and
                self.conversation_state.question_count <
                self.conversation_state.max_questions):
            q_obj = self.conversation_state.get_next_question()
            if q_obj:
                return {
                    "recommendations": [],
                    "clarification": q_obj["type"],
                    "question": q_obj["question"],
                    "choices": q_obj.get("choices", []),
                    "confidence": float(1.0 - min(entropy / np.log(20), 1.0)),
                    "preferences": prefs, "entropy": entropy, "margin": margin,
                    "state": self.conversation_state.to_dict(),
                }
        # If state completed: re-score with accumulated answers.
        if self.conversation_state and self.conversation_state.completed:
            answers_str = self.conversation_state.summarize_answers()
            final_ctx = build_enriched_context_text(context, prof, hist_titles)
            if answers_str != "No preferences recorded yet.":
                final_ctx += f" [Preference Summary] {answers_str}"
            q_final = self._encode_texts([final_ctx])[0]
            scores_final = self._score(q_final)
            return {
                "recommendations": self._format_scores(scores_final, top_k),
                "clarification": None,
                "confidence": float(probs_full.max()),
                "preferences": prefs, "entropy": entropy, "margin": margin,
            }
        # Default: single-shot entropy check.
        if entropy > entropy_threshold:
            if not prof:
                qtype = "QGEN_INTEREST"
            elif not hist_titles:
                qtype = "QGEN_HISTORY"
            else:
                qtype = "QGEN_GENRE"
            return {
                "recommendations": [],
                "clarification": qtype,
                "confidence": float(1.0 - min(entropy / np.log(20), 1.0)),
                "preferences": prefs, "entropy": entropy, "margin": margin,
            }
        return {
            "recommendations": self._format_scores(scores, top_k),
            "clarification": None,
            "confidence": float(probs_full.max()),
            "preferences": prefs, "entropy": entropy, "margin": margin,
        }


if __name__ == "__main__":
    print("ChineseRecommender module loaded successfully")
    print(f"VERSION: {VERSION}")
    print(f"ENCODER: {ENCODER_NAME}")