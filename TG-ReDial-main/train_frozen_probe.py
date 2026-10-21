#!/usr/bin/env python3
"""Frozen linear-probe probe training for chinese_recommender (Stage 6)."""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModel, AutoTokenizer

from chinese_recommender import ChineseRecommender, ENCODER_NAME, build_context_text

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

PROJECT = Path(__file__).resolve().parent.parent

MAX_NEG = 64
LR = 1e-3
BATCH = 64
EPOCHS = 4
PATIENCE = 2
CTX_CACHE = Path('out_ctx_cache.pt')

DEFAULT_DATA_DIR = PROJECT / 'TG-ReDial-Data' / 'out'
DEFAULT_TRAIN = DEFAULT_DATA_DIR / 'train.jsonl'
DEFAULT_VALID = DEFAULT_DATA_DIR / 'valid.jsonl'
DEFAULT_TEST = DEFAULT_DATA_DIR / 'test.jsonl'
DEFAULT_CACHE = DEFAULT_DATA_DIR / 'title_embeddings_albert_tiny.npz'
DEFAULT_ORDER = DEFAULT_DATA_DIR / 'movie_id_order.json'
DEFAULT_OUT_DIR = PROJECT / 'out_probe'


def encode_ctx_batch(enc, tokenizer, ctxs):
    toks = tokenizer(ctxs, truncation=True, max_length=256, padding=True, return_tensors='pt', add_special_tokens=True)
    with torch.no_grad():
        h = enc(**toks).last_hidden_state
        mk = toks['attention_mask'].unsqueeze(-1).to(h.dtype)
        v = (h * mk).sum(1) / mk.sum(1).clamp_min(1e-6)
    return v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)


def precompute_contexts(enc, tokenizer, data_path, batch_size=32):
    vecs = []
    batch_ctxs = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            batch_ctxs.append(build_context_text(d.get('context', [])))
            if len(batch_ctxs) >= batch_size:
                batch_vecs = encode_ctx_batch(enc, tokenizer, batch_ctxs)
                vecs.extend([v.numpy() for v in batch_vecs])
                batch_ctxs = []
    if batch_ctxs:
        batch_vecs = encode_ctx_batch(enc, tokenizer, batch_ctxs)
        vecs.extend([v.numpy() for v in batch_vecs])
    return torch.tensor(np.stack(vecs), dtype=torch.float32)


def ndcg_at_k(rel, k):
    # `rel` is a 1-D array of model scores for one candidate list. The
    # ground-truth relevant item is index 0 (all others are sampled negatives).
    # NDCG@k = 1/log2(rank+1) when the target lands in the top-k, else 0.
    # (The old implementation checked `score > 0`, which saturates to 1.0 and
    #  makes early-stopping / best-checkpoint selection meaningless.)
    n = len(rel)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    order = np.argsort(-rel)
    rank = int(np.flatnonzero(order == 0)[0])  # 0-based rank of the target
    if rank >= k:
        return 0.0
    return float(1.0 / np.log2(rank + 2))


def hit_at_k(rel, k):
    # Binary hit@k: 1.0 if the relevant item (index 0) appears in the top-k
    # ranked candidates, else 0.0. (The old `mean(argsort(-rel)[:k] == 0)`
    # returned hit/k, understating the metric by a factor of k.)
    n = len(rel)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    order = np.argsort(-rel)
    return 1.0 if int(np.flatnonzero(order == 0)[0]) < k else 0.0


def topk_variance(rel):
    # `rel` is a 2-D numpy array; `if not rel` is ambiguous for arrays, so
    # test .size explicitly for the empty case.
    if getattr(rel, 'size', 0) == 0:
        return 0.0, 0
    top1 = np.argmax(rel, axis=1)
    vals, counts = np.unique(top1, return_counts=True)
    return float(np.max(counts)), int(len(vals))


def evaluate(ds, model, title_emb, unfreeze=False, model_ctx_fn=None):
    model.eval()
    batch_size = BATCH
    rels, losses, logit_means, top1_max, top1_unique = [], [], [], [], []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_unfreeze if unfreeze else collate):
            if unfreeze:
                ctx_texts, idxs = batch
                ctx_vecs = model.encode_contexts(ctx_texts)
            else:
                ctx_vecs, idxs = batch
            scores = model(ctx_vecs, title_emb[idxs])
            tgt = torch.zeros(len(idxs), device=scores.device, dtype=torch.long)
            losses.append(nn.CrossEntropyLoss()(scores, tgt).item() * len(idxs))
            rel = scores.cpu().numpy()
            rels.append(rel)
            logit_means.append(float(rel.mean()))
            m, u = topk_variance(rel)
            top1_max.append(m)
            top1_unique.append(u)
    rels = np.concatenate(rels, axis=0)
    return dict(
        loss=np.mean(losses),
        ndcg10=np.mean([ndcg_at_k(rel, 10) for rel in rels]),
        hit10=np.mean([hit_at_k(rel, 10) for rel in rels]),
        logit_mean=np.mean(logit_means),
        top1_max=np.mean(top1_max),
        top1_unique=np.mean(top1_unique),
    )


def main():
    ap = argparse.ArgumentParser(
        description='Train a frozen-encoder linear-probe movie recommender.')
    ap.add_argument('--epochs', type=int, default=EPOCHS,
                    help='Number of training epochs.')
    ap.add_argument('--lr', type=float, default=LR,
                    help='Learning rate for AdamW.')
    ap.add_argument('--batch', type=int, default=BATCH,
                    help='Training batch size.')
    ap.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR,
                    help='Directory containing train/valid/test.jsonl and catalog files.')
    ap.add_argument('--train', type=Path, default=None,
                    help='Train split JSONL (default: <data-dir>/train.jsonl).')
    ap.add_argument('--valid', type=Path, default=None,
                    help='Validation split JSONL (default: <data-dir>/valid.jsonl).')
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR,
                    help='Directory for checkpoints and logs.')
    ap.add_argument('--seed', type=int, default=SEED,
                    help='Random seed (torch/numpy/python).')
    ap.add_argument('--patience', type=int, default=PATIENCE,
                    help='Early-stopping patience in epochs.')
    ap.add_argument('--max-neg', type=int, default=MAX_NEG,
                    help='Max negative samples per example.')
    ap.add_argument('--ctx-cache', type=Path, default=CTX_CACHE,
                    help='Context embedding cache file (frozen-probe mode).')
    ap.add_argument('--no-reuse-cache', action='store_true',
                    help='Force fresh context embedding precompute (ignore cache).')
    ap.add_argument('--eval-test', action='store_true',
                    help='Run official test-split evaluation after training and '
                         'write metrics to out_probe/test_metrics.json.')
    ap.add_argument('--unfreeze-encoder', action='store_true',
                    help='Unfreeze the ALBERT encoder and fine-tune it jointly '
                         'with the probe head (end-to-end training). Disables '
                         'the precomputed context cache; contexts are encoded '
                         'on the fly each batch.')
    ap.add_argument('--encoder-lr', type=float, default=1e-5,
                    help='Learning rate for the encoder when --unfreeze-encoder '
                         'is set (probe head uses --lr).')
    args = ap.parse_args()

    # Resolve derived paths.
    train_path = args.train or (args.data_dir / 'train.jsonl')
    valid_path = args.valid or (args.data_dir / 'valid.jsonl')
    order_path = args.data_dir / 'movie_id_order.json'
    cache_path = args.data_dir / 'title_embeddings_albert_tiny.npz'
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Re-seed for reproducibility.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    pidfile = args.out_dir / '.pid'
    pidfile.write_text(str(os.getpid()))
    log_f = open(args.out_dir / 'train.log', 'w', buffering=1)
    sys.stdout = log_f
    ids, title_emb = load_cache(order_path, cache_path)
    pos_pos = {str(m): i for i, m in enumerate(ids)}
    neg_pool = list(range(len(ids)))

    if args.unfreeze_encoder:
        # End-to-end fine-tuning: load the encoder and attach it to the model
        # so gradients flow into ALBERT. Contexts are encoded on the fly each
        # batch; the precomputed context cache is not used.
        print('Loading encoder for end-to-end fine-tuning...', flush=True)
        enc = AutoModel.from_pretrained(ENCODER_NAME)
        tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
        model = ChineseRecommender(
            enc_dim=int(title_emb.shape[1]), movie_count=len(ids),
            encoder=enc, tokenizer=tokenizer,
            unfreeze_encoder=True)
        model.train()
        # Two parameter groups: encoder gets a smaller LR, probe head gets args.lr.
        probe_params = list(model.proj.parameters()) + list(model.head.parameters()) + \
            [model.temperature, model.pop_weight]
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        opt = torch.optim.AdamW([
            {'params': enc_params, 'lr': args.encoder_lr},
            {'params': probe_params, 'lr': args.lr},
        ])
        print(f'Unfreeze mode: {len(enc_params)} encoder params, '
              f'{len(probe_params)} probe params', flush=True)
    else:
        # Frozen-probe mode: precompute context embeddings once, train only the probe head.
        if not args.no_reuse_cache and args.ctx_cache.exists():
            print('reusing context cache', flush=True)
        else:
            enc = AutoModel.from_pretrained(ENCODER_NAME)
            tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
            enc.eval()
            train_ctx = precompute_contexts(enc, tokenizer, train_path)
            val_ctx = precompute_contexts(enc, tokenizer, valid_path)
            torch.save({'train': train_ctx, 'valid': val_ctx}, args.ctx_cache)
            print('precomputed context cache', flush=True)

        model = ChineseRecommender(enc_dim=int(title_emb.shape[1]),
                                   movie_count=len(ids))
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    if args.unfreeze_encoder:
        # Contexts are encoded on the fly from raw text.
        train_ds = TextProbeExample(train_path, ids, pos_pos, neg_pool,
                                    max_neg=args.max_neg, shuffle=True,
                                    tokenizer=tokenizer, max_ctx_tokens=256)
        val_ds = TextProbeExample(valid_path, ids, pos_pos, neg_pool,
                                  max_neg=args.max_neg, shuffle=False,
                                  tokenizer=tokenizer, max_ctx_tokens=256)
    else:
        ctx_cache = torch.load(args.ctx_cache)
        train_ds = ProbeExample(train_path, ids, pos_pos, neg_pool,
                                ctx_cache['train'], max_neg=args.max_neg,
                                shuffle=True)
        val_ds = ProbeExample(valid_path, ids, pos_pos, neg_pool,
                              ctx_cache['valid'], max_neg=args.max_neg)

    best = None
    wait = 0
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        tot, steps = 0.0, 0
        for batch in DataLoader(train_ds, batch_size=args.batch,
                                collate_fn=collate_unfreeze
                                if args.unfreeze_encoder else collate):
            if args.unfreeze_encoder:
                ctx_texts, idxs = batch
                ctx_vecs = model.encode_contexts(ctx_texts)
            else:
                ctx_vecs, idxs = batch
            pos_idx = idxs[:, 0]
            scores = model(ctx_vecs, title_emb[idxs])
            tgt = torch.zeros(len(idxs), device=scores.device, dtype=torch.long)
            loss = criterion(scores, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idxs)
            steps += 1
        scheduler.step()
        ev_tr = evaluate(train_ds, model, title_emb,
                         unfreeze=args.unfreeze_encoder, model_ctx_fn=None)
        ev_val = evaluate(val_ds, model, title_emb,
                          unfreeze=args.unfreeze_encoder, model_ctx_fn=None)
        dur = time.time() - t0
        enc_lr = scheduler.get_last_lr()[0]
        print(f"ep {ep+1}/{args.epochs} loss={tot/steps:.4f} "
              f"train_ndcg10={ev_tr['ndcg10']:.4f} val_ndcg10={ev_val['ndcg10']:.4f} "
              f"val_hit10={ev_val['hit10']:.4f} top1_max={ev_val['top1_max']:.2f} "
              f"top1_unique={ev_val['top1_unique']:.0f} lr={enc_lr:.2e} "
              f"dur={dur:.1f}s", flush=True)
        torch.save({'version': 1, 'state': model.state_dict(),
                    'encoder': ENCODER_NAME, 'max_context_tokens': 256,
                    'unfreeze_encoder': args.unfreeze_encoder},
                   args.out_dir / f'ckpt_ep{ep+1}.pt')
        if best is None or ev_val['ndcg10'] > best['ndcg10']:
            best = ev_val
            wait = 0
            torch.save({'version': 1, 'state': model.state_dict(),
                        'encoder': ENCODER_NAME, 'max_context_tokens': 256,
                        'unfreeze_encoder': args.unfreeze_encoder},
                       args.out_dir / 'best.pt')
        else:
            wait += 1
            if wait >= args.patience:
                print('early stop', flush=True)
                break
    print('TRAINED', flush=True)

    # Optional: run the official test-split evaluation after training.
    if args.eval_test:
        test_path = args.data_dir / 'test.jsonl'
        eval_script = Path(__file__).resolve().parent / 'evaluate_test.py'
        if not test_path.exists():
            print(f'[eval-test] skipping: {test_path} not found', flush=True)
        elif not eval_script.exists():
            print(f'[eval-test] skipping: {eval_script} not found', flush=True)
        else:
            print('[eval-test] running official test-split evaluation...', flush=True)
            result = subprocess.run(
                [sys.executable, str(eval_script),
                 '--checkpoint', str(args.out_dir / 'best.pt'),
                 '--test', str(test_path)],
                capture_output=True, text=True,
            )
            # Stream eval output so it appears in the training log.
            if result.stdout:
                print(result.stdout, end='', flush=True)
            if result.stderr:
                print(result.stderr, end='', flush=True, file=sys.stderr)
            if result.returncode == 0:
                print('[eval-test] evaluation complete', flush=True)
            else:
                print(f'[eval-test] evaluation failed (exit {result.returncode})',
                      flush=True, file=sys.stderr)

EPOCHS = 4
PATIENCE = 2

class ProbeExample(IterableDataset):
    def __init__(self, path, ids, pos_pos, neg_pool, ctx_vecs, max_neg=MAX_NEG,
                 shuffle=False):
        self.path = path
        self.ids = ids
        self.pos_pos = pos_pos
        self.neg_pool = neg_pool
        self.ctx_vecs = ctx_vecs
        self.max_neg = max_neg
        self.shuffle = shuffle

    def __iter__(self):
        with open(self.path, encoding='utf-8') as f:
            lines = f.readlines()
        order = list(range(len(lines)))
        if self.shuffle:
            random.shuffle(order)
        n_all = len(self.ids)
        for i in order:
            d = json.loads(lines[i])
            pids = [str(p) for p in d.get('target_movie_ids', [])]
            pos_i = [self.pos_pos[p] for p in pids if p in self.pos_pos]
            if not pos_i:
                continue
            pos_i = pos_i[:1]
            pos_set = set(pos_i)
            nneed = self.max_neg - len(pos_i)
            # Sample negatives by rejection instead of materialising the full
            # complement (O(N) per example -> O(max_neg)).
            neg_i = []
            attempts = 0
            max_attempts = nneed * 10 + 50
            while len(neg_i) < nneed and attempts < max_attempts:
                attempts += 1
                j = random.randrange(n_all)
                if j not in pos_set and j not in neg_i:
                    neg_i.append(j)
            while len(neg_i) < nneed:
                j = random.randrange(n_all)
                if j not in pos_set:
                    neg_i.append(j)
            idxs = torch.tensor(pos_i + neg_i, dtype=torch.long)
            yield (self.ctx_vecs[i], idxs)


def collate(batch):
    ctx = torch.stack([b[0] for b in batch])
    idxs = torch.stack([b[1] for b in batch])
    return ctx, idxs


def collate_unfreeze(batch):
    # Each item is (context_string, idxs_tensor). Aggregate into a list of
    # strings and a stacked idxs tensor; contexts are encoded on the fly by
    # the model during training/evaluation.
    ctx_texts = [item[0] for item in batch]
    idxs = torch.stack([item[1] for item in batch])
    return ctx_texts, idxs


class TextProbeExample(IterableDataset):
    """Like ProbeExample but yields raw context strings instead of precomputed
    vecs, for end-to-end fine-tuning where the encoder is trainable."""

    def __init__(self, path, ids, pos_pos, neg_pool, max_neg=MAX_NEG,
                 shuffle=False, tokenizer=None, max_ctx_tokens=256):
        self.path = path
        self.ids = ids
        self.pos_pos = pos_pos
        self.neg_pool = neg_pool
        self.max_neg = max_neg
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.max_ctx_tokens = max_ctx_tokens

    def __iter__(self):
        with open(self.path, encoding='utf-8') as f:
            lines = f.readlines()
        order = list(range(len(lines)))
        if self.shuffle:
            random.shuffle(order)
        n_all = len(self.ids)
        for i in order:
            d = json.loads(lines[i])
            # Build the raw context text (same normalization as serving).
            ctx_text = build_context_text(d.get('context', []))
            pids = [str(p) for p in d.get('target_movie_ids', [])]
            pos_i = [self.pos_pos[p] for p in pids if p in self.pos_pos]
            if not pos_i:
                continue
            pos_i = pos_i[:1]
            pos_set = set(pos_i)
            nneed = self.max_neg - len(pos_i)
            neg_i = []
            attempts = 0
            max_attempts = nneed * 10 + 50
            while len(neg_i) < nneed and attempts < max_attempts:
                attempts += 1
                j = random.randrange(n_all)
                if j not in pos_set and j not in neg_i:
                    neg_i.append(j)
            while len(neg_i) < nneed:
                j = random.randrange(n_all)
                if j not in pos_set:
                    neg_i.append(j)
            idxs = torch.tensor(pos_i + neg_i, dtype=torch.long)
            yield (ctx_text, idxs)


def load_cache(order_path, cache_path):
    ids = json.load(open(order_path, encoding='utf-8'))['ids']
    emb = np.load(cache_path)['embeddings']
    # model.forward's projection is a Linear layer: it needs a torch.Tensor,
    # not the numpy.ndarray that np.load returns.
    return ids, torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32))
if __name__ == '__main__':
    main()
