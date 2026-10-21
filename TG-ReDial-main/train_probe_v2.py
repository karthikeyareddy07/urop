#!/usr/bin/env python3
"""V2 improved probe training: enriched context + hard negatives + stronger loss.

Frozen encoder; precomputed enriched contexts; 512-dim probe + dropout;
mixed 32 uniform + 32 hard negatives; cosine schedule; multi-seed.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModel, AutoTokenizer

FILE_DIR = Path(__file__).resolve().parent
PROJECT = FILE_DIR.parent
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))
from chinese_recommender import ChineseRecommender, ENCODER_NAME  # noqa: E402

MAX_NEG = 64
N_HARD = 32
LR = 1e-3
BATCH = 64
EPOCHS = 10
PATIENCE = 3
PROJ_DIM = 512
DROPOUT = 0.1
DEFAULT_DATA_DIR = PROJECT / "TG-ReDial-Data" / "out"


def norm_text(t: str) -> str:
    return (t or "").strip()


def build_enriched_context(context, profile_sents=None, history_titles=None,
                           max_chars: int = 2500) -> str:
    parts: list[str] = []
    for turn in context:  # ALL turns, no last-8 cut
        tx = norm_text(turn.get("text", ""))
        if tx:
            parts.append(f"{turn.get('speaker', '')}:{tx}")
    if profile_sents:
        parts.append("用户画像:" + ";".join(profile_sents[:10]))
    if history_titles:
        parts.append("看过:" + ";".join(history_titles[-8:]))
    return " ".join(parts)[-max_chars:]


def encode_texts(enc, tok, texts, batch_size=32, max_length=256):
    vecs = []
    enc.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            toks = tok(chunk, truncation=True, max_length=max_length,
                       padding=True, return_tensors="pt", add_special_tokens=True)
            h = enc(**toks).last_hidden_state
            mk = toks["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * mk).sum(1) / mk.sum(1).clamp_min(1e-6)
            vecs.append((v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)).cpu())
    return torch.cat(vecs, dim=0)


def ndcg_at_k(scores, k=10):
    order = np.argsort(-np.asarray(scores))
    rank = int(np.flatnonzero(order == 0)[0])
    return float(1.0 / np.log2(rank + 2)) if rank < k else 0.0


def hit_at_k(scores, k=10):
    order = np.argsort(-np.asarray(scores))
    return 1.0 if int(np.flatnonzero(order == 0)[0]) < k else 0.0


def mrr(scores):
    order = np.argsort(-np.asarray(scores))
    return 1.0 / (int(np.flatnonzero(order == 0)[0]) + 1)


class EnrichedProbeDataset(IterableDataset):
    def __init__(self, ctx_vecs, idxs_list, n_all, max_neg=64,
                 shuffle=False, seed=0, hard_pool=None, n_hard=32):
        self.ctx_vecs = ctx_vecs
        self.idxs_list = idxs_list
        self.n_all = n_all
        self.max_neg = max_neg
        self.shuffle = shuffle
        self.seed = seed
        self.hard_pool = hard_pool
        self.n_hard = n_hard

    def __iter__(self):
        rng = random.Random(self.seed)
        order = list(range(len(self.ctx_vecs)))
        if self.shuffle:
            rng.shuffle(order)
        for i in order:
            pos = self.idxs_list[i]
            pos_set = set(pos)
            negs = []
            if self.hard_pool and i in self.hard_pool:
                for h in self.hard_pool[i]:
                    if h not in pos_set and h not in negs:
                        negs.append(h)
                    if len(negs) >= self.n_hard:
                        break
            need = self.max_neg - len(negs)
            att = 0
            while need > 0 and att < need * 10 + 50:
                att += 1
                j = rng.randrange(self.n_all)
                if j not in pos_set and j not in negs:
                    negs.append(j)
                    need -= 1
            while need > 0:
                j = rng.randrange(self.n_all)
                if j not in pos_set:
                    negs.append(j)
                    need -= 1
            yield (self.ctx_vecs[i], torch.tensor(pos + negs[:self.max_neg], dtype=torch.long))


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


def load_split(path, users, movies, id2pos):
    texts, idxs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tgt = (d.get("target_movie_ids") or [None])[0]
            if tgt is None or str(tgt) not in id2pos:
                continue
            prof = users.get(str(d.get("user_id")), [])
            if isinstance(prof, (set, tuple)):
                prof = list(prof)
            hist = []
            for mid in (d.get("watch_history") or [])[-8:]:
                m = movies.get(str(mid))
                if m and m.get("title"):
                    hist.append(m["title"])
            texts.append(build_enriched_context(d.get("context", []), prof, hist))
            idxs.append([id2pos[str(tgt)]])
    return texts, idxs


def evaluate(model, ctx_vecs, idxs_list, title_emb, device, max_neg=64, seed=0):
    model.eval()
    rng = random.Random(seed)
    n_all = title_emb.shape[0]
    ndcg = hit = rr = 0.0
    n = len(ctx_vecs)
    with torch.no_grad():
        for i in range(n):
            pos_set = set(idxs_list[i])
            negs, att = [], 0
            while len(negs) < max_neg and att < max_neg * 10 + 50:
                att += 1
                j = rng.randrange(n_all)
                if j not in pos_set and j not in negs:
                    negs.append(j)
            cand = idxs_list[i] + negs
            tv = title_emb[cand].to(device)
            cv = ctx_vecs[i].unsqueeze(0).to(device)
            s = model(cv, tv).squeeze(0).cpu().numpy()
            ndcg += ndcg_at_k(s)
            hit += hit_at_k(s)
            rr += mrr(s)
    return hit / n, ndcg / n, rr / n


@torch.no_grad()
def mine_hard_negatives(model, ctx_vecs, idxs_list, title_emb, device,
                        top_k=200, n_hard=32, batch=256):
    model.eval()
    t = model.head(model.proj(title_emb.to(device)))
    t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    hard = {}
    for s in range(0, len(ctx_vecs), batch):
        chunk = ctx_vecs[s:s + batch].to(device)
        c = model.head(model.proj(chunk))
        c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        scores = (c @ t.T).cpu().numpy()
        for k in range(chunk.shape[0]):
            i = s + k
            order = np.argsort(-scores[k])[:top_k + 4]
            pos_set = set(idxs_list[i])
            hard[i] = [int(j) for j in order if int(j) not in pos_set][:n_hard]
    return hard


def main():
    ap = argparse.ArgumentParser(description="V2 improved probe training.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROJECT / "out_probe_v2")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--hard-from-epoch", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    order = json.load(open(args.data_dir / "movie_id_order.json", encoding="utf-8"))["ids"]
    id2pos = {str(mid): i for i, mid in enumerate(order)}
    n_all = len(order)
    title_emb = torch.from_numpy(np.ascontiguousarray(
        np.load(args.data_dir / "title_embeddings_albert_tiny.npz")["embeddings"],
        dtype=np.float32))
    movies = json.load(open(args.data_dir / "movies.json", encoding="utf-8"))
    users = json.load(open(args.data_dir / "users.json", encoding="utf-8"))

    print("Loading encoder...", flush=True)
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    enc = AutoModel.from_pretrained(ENCODER_NAME)
    enc.eval()

    for split in ["train", "valid"]:
        cache = args.out_dir / f"ctx_{split}_s{args.seed}.pt"
        meta = args.out_dir / f"idxs_{split}_s{args.seed}.json"
        if cache.exists() and meta.exists():
            m = json.load(open(meta, encoding="utf-8"))
            ctx = torch.load(cache, map_location="cpu", weights_only=False)
            idxs = m["idxs"]
            print(f"{split}: cached {len(ctx)}", flush=True)
        else:
            texts, idxs = load_split(args.data_dir / f"{split}.jsonl", users, movies, id2pos)
            print(f"{split}: encoding {len(texts)} enriched contexts", flush=True)
            ctx = encode_texts(enc, tok, texts)
            torch.save(ctx, cache)
            json.dump({"n": len(texts), "idxs": idxs}, open(meta, "w", encoding="utf-8"))
        if split == "train":
            train_ctx, train_idxs = ctx, idxs
        else:
            valid_ctx, valid_idxs = ctx, idxs
    print(f"train={len(train_ctx)} valid={len(valid_ctx)} catalog={n_all}", flush=True)
    del enc, tok

    model = ChineseRecommender(enc_dim=title_emb.shape[1], proj_dim=PROJ_DIM,
                               movie_count=n_all, temperature=1.0)
    model.proj = nn.Linear(title_emb.shape[1], PROJ_DIM)
    model.head = nn.Sequential(
        nn.LayerNorm(PROJ_DIM), nn.Linear(PROJ_DIM, PROJ_DIM),
        nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(PROJ_DIM, PROJ_DIM))
    with torch.no_grad():
        model.temperature.fill_(float(np.log(5.0)))
        model.pop_weight.data.zero_()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"V2 probe params: {n_params}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_ndcg, bad, best_ep = -1.0, 0, -1
    hard_pool = None
    t0 = time.time()
    logf = open(args.out_dir / f"train_s{args.seed}.log", "w", encoding="utf-8")
    for ep in range(1, args.epochs + 1):
        if ep >= args.hard_from_epoch:
            print(f"epoch {ep}: mining hard negatives...", flush=True)
            hard_pool = mine_hard_negatives(model, train_ctx, train_idxs, title_emb, device)
        ds = EnrichedProbeDataset(train_ctx, train_idxs, n_all, shuffle=True,
                                  seed=args.seed * 1000 + ep, hard_pool=hard_pool)
        dl = DataLoader(ds, batch_size=args.batch, collate_fn=collate)
        model.train()
        tot_loss, nb = 0.0, 0
        for ctx_b, idx_b in dl:
            ctx_b = ctx_b.to(device)
            opt.zero_grad()
            c = model.head(model.proj(ctx_b))
            c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            tv = title_emb[idx_b].to(device)
            t = model.head(model.proj(tv))
            t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            tau = model.temperature.exp().clamp(1.0, 20.0)
            s = (c.unsqueeze(1) @ t.transpose(1, 2)).squeeze(1) * tau
            tgt = torch.zeros(s.shape[0], dtype=torch.long)
            loss = nn.functional.cross_entropy(s, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()
            nb += 1
        sched.step()
        vh, vn, vm = evaluate(model, valid_ctx, valid_idxs, title_emb, device, seed=12345)
        line = (f"epoch {ep}: loss={tot_loss / max(nb, 1):.4f} "
                f"valid Hit@10={vh:.4f} NDCG@10={vn:.4f} MRR={vm:.4f}")
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
        if vn > best_ndcg:
            best_ndcg, best_ep, bad = vn, ep, 0
            torch.save({"version": 2, "encoder": ENCODER_NAME, "proj_dim": PROJ_DIM,
                        "seed": args.seed, "epoch": ep,
                        "valid": {"hit10": vh, "ndcg10": vn, "mrr": vm},
                        "state": model.state_dict()},
                       args.out_dir / "best.pt")
            print(f"  -> new best (epoch {ep})", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep} (best {best_ep})", flush=True)
                break
    dt = time.time() - t0
    print(f"done in {dt:.0f}s, best valid NDCG@10={best_ndcg:.4f} (ep {best_ep})", flush=True)
    logf.write(f"BEST valid NDCG@10={best_ndcg:.4f} ep={best_ep} time_s={dt:.0f} params={n_params}\n")
    logf.close()


if __name__ == "__main__":
    main()
